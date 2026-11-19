"""Finite-world pilot benchmark. Standard library only; Python >=3.9."""
import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path


FAMILIES = [
    ("loan", "loan principal", "opening principal", "payment", -1),
    ("inventory", "warehouse stock", "opening stock", "shipment received", 1),
    ("invoice", "invoice balance", "opening balance", "credit", -1),
    ("budget", "remaining budget", "opening budget", "expense", -1),
    ("attendance", "event attendance", "confirmed attendees", "registration batch", 1),
]


def worlds(spec):
    """Enumerate only worlds allowed by visible constraints, never hidden truth."""
    keys = list(spec["domains"])
    for values in itertools.product(*(spec["domains"][k] for k in keys)):
        world = dict(zip(keys, values))
        if all(world[k] == v for k, v in spec["fixed"].items()):
            yield world


def answer(spec, world):
    return spec["base"] + spec["sign"] * sum(
        amount * world[key] for key, amount in spec["amounts"].items()
    )


def valid_world(spec, world):
    return (isinstance(world, dict) and set(world) == set(spec["domains"])
            and all(type(v) is int and v in spec["domains"][k] for k, v in world.items())
            and all(world[k] == v for k, v in spec["fixed"].items()))


def verify(spec, prediction):
    """Check a witness against visible facts. Claimed answers are recomputed."""
    if not isinstance(prediction, dict):
        return False
    pair = prediction.get("worlds")
    if not isinstance(pair, list) or len(pair) != 2:
        return False
    return (all(valid_world(spec, w) for w in pair)
            and answer(spec, pair[0]) != answer(spec, pair[1]))


def oracle(spec):
    """Privileged symbolic control, NOT an LLM or a deployable text auditor."""
    by_answer = {}
    for world in worlds(spec):
        by_answer.setdefault(answer(spec, world), world)
    if not by_answer:
        raise ValueError("Inconsistent visible constraints")
    if len(by_answer) == 1:
        return {"verdict": "supported", "worlds": []}
    return {"verdict": "underdetermined", "worlds": list(by_answer.values())[:2]}


def build():
    public, private = [], []
    rng = random.Random(20261118)
    for family_index, (family, quantity, opening, event, sign) in enumerate(FAMILIES):
        # Whole families are held out. Variants and numeric siblings never cross splits.
        split = "dev" if family_index < 3 else "test"
        for index in range(4):
            scenario = f"{family}-{index}"
            base = rng.randrange(50, 150) * 100
            amounts = {"a": rng.randrange(1, 8) * 100, "b": rng.randrange(8, 15) * 100}
            truth = {"a": index % 2, "b": (index // 2) % 2}
            for variant in ("complete", "missing", "ambiguous", "distracting"):
                opaque_id = hashlib.sha256(f"pilot-v1/{scenario}/{variant}".encode()).hexdigest()[:16]
                fixed = truth.copy()
                if variant in ("missing", "ambiguous"):
                    del fixed["a"]
                spec = {"base": base, "sign": sign, "amounts": amounts,
                        "domains": {"a": [0, 1], "b": [0, 1]}, "fixed": fixed}
                documents = [
                    {"id": "D1", "text": f"Target account T has {opening} of {base} units. "
                     f"Compute its {quantity} after the two listed events. "
                     "There are no other changes, fees, or timing effects. "
                     f"Each {event} changes the target by {sign:+d} times its amount only if assigned to T. "
                     "Each event is assigned to exactly one of accounts T and U."},
                    {"id": "D2", "text": f"Event a is a {event} of {amounts['a']} units. "
                     f"Event b is a {event} of {amounts['b']} units."},
                ]
                for key, value in fixed.items():
                    documents.append({"id": f"assignment-{key}", "text":
                                      f"Event {key} is assigned to account {'T' if value else 'U'}."})
                if variant == "ambiguous":
                    documents.append({"id": "alias", "text": "Event a is assigned to the primary account. "
                                      "Primary denotes exactly one of T and U; its identity is not specified."})
                if variant == "distracting":
                    documents.extend({"id": f"noise-{j}", "text":
                        f"Unrelated account X{j} recorded {rng.randrange(10, 90)} units. "
                        "This entry does not affect T or U."} for j in range(6))
                    rng.shuffle(documents)
                public.append({"id": opaque_id, "question": f"What is the final {quantity} for T?",
                               "documents": documents,
                               "world_schema": {"a": "0 if event a is assigned to U, 1 if assigned to T",
                                                "b": "0 if event b is assigned to U, 1 if assigned to T"}})
                private.append({"id": opaque_id, "scenario": scenario, "family": family,
                                "variant": variant, "split": split, "spec": spec,
                                "hidden_world": truth, "expected_answer": answer(spec, truth),
                                "label": oracle(spec)["verdict"]})
    rng.shuffle(public)
    return public, private


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


BASE_PROMPT = """Audit whether the question has a unique answer under the supplied documents.
Do not assume an unspecified event belongs to the target account. Use only the documents.
Return one JSON object with id, verdict (supported, underdetermined, or uncertain).
Supported means the visible evidence uniquely determines the answer, not that you agree
with a hidden answer key. An uncertain verdict is an abstention.
"""
WITNESS_PROMPT = BASE_PROMPT + """
For underdetermined, include worlds: two complete assignments using the world_schema
keys and integer 0/1 values. Both must satisfy every document and yield different answers.
For other verdicts use worlds: []. Include explanation and clarification as strings.
"""


def evaluate(keys, predictions):
    expected_ids = {r["id"] for r in keys}
    ids = [p.get("id") if isinstance(p, dict) else None for p in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction IDs")
    if any(i not in expected_ids for i in ids):
        raise ValueError("Unknown prediction IDs or wrong split")
    by_id = dict(zip(ids, predictions))
    counts = Counter(total=len(keys))
    details = []
    for row in keys:
        pred = by_id.get(row["id"], {})
        verdict = pred.get("verdict", "missing")
        counts["submitted"] += row["id"] in by_id
        if verdict not in ("supported", "underdetermined", "uncertain", "missing"):
            verdict = "invalid"
        counts[verdict] += 1
        positive = row["label"] == "underdetermined"
        flagged = verdict == "underdetermined"
        checked = flagged and verify(row["spec"], pred)
        counts["positive"] += positive
        counts["negative"] += not positive
        counts["correct"] += verdict == row["label"]
        counts["true_positive"] += positive and flagged
        counts["false_positive"] += not positive and flagged
        counts["verified_witness"] += checked
        details.append({"id": row["id"], "family": row["family"], "variant": row["variant"],
                        "label": row["label"], "verdict": verdict, "verified_witness": checked})
    def ratio(n, d):
        return counts[n] / counts[d] if counts[d] else None
    return {"counts": dict(counts), "metrics": {
        "coverage": ratio("submitted", "total"), "accuracy_including_abstentions": ratio("correct", "total"),
        "underdetermined_recall": ratio("true_positive", "positive"),
        "false_alarm_rate": ratio("false_positive", "negative"),
        "witness_validity_among_flags": ratio("verified_witness", "underdetermined"),
        "verified_detection_recall": ratio("verified_witness", "positive")}, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "demo", "prompts", "evaluate", "control"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--mode", choices=["baseline", "witness"], default="witness")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        public, keys = build()
        for split in ("dev", "test"):
            subset = [k for k in keys if k["split"] == split]
            ids = {k["id"] for k in subset}
            write_jsonl(args.data / "public" / f"{split}.jsonl", [p for p in public if p["id"] in ids])
            write_jsonl(args.data / "private" / f"{split}.jsonl", subset)
        print("Built 20 scenarios / 80 variants: dev 48, test 32. Synthetic pilot only.")
        return
    keys = read_jsonl(args.data / "private" / f"{args.split}.jsonl")
    if args.command == "prompts":
        if not args.output:
            parser.error("prompts requires --output")
        public = read_jsonl(args.data / "public" / f"{args.split}.jsonl")
        write_jsonl(args.output, [{"id": p["id"], "prompt":
                    (BASE_PROMPT if args.mode == "baseline" else WITNESS_PROMPT) + json.dumps(p)} for p in public])
        print(f"Wrote {len(public)} prompts (public fields only).")
        return
    if args.command == "demo":
        row = next(k for k in keys if k["variant"] == "missing")
        prediction = oracle(row["spec"])
        result = {"notice": "Symbolic demonstration; not an LLM result", "task_id": row["id"],
                  "prediction": prediction, "answers": [answer(row["spec"], w) for w in prediction["worlds"]],
                  "verified": verify(row["spec"], prediction)}
    elif args.command == "control":
        predictions = [dict(id=k["id"], **oracle(k["spec"])) for k in keys]
        result = evaluate(keys, predictions)
        result["notice"] = "Privileged symbolic control ONLY. This validates plumbing, not model performance."
    else:
        if not args.predictions:
            parser.error("evaluate requires --predictions")
        result = evaluate(keys, read_jsonl(args.predictions))
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
