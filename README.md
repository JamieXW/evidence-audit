# Evidence Audit — pilot v0.2

Can an auditor show, with evidence a program can check, that a task's target is
not supported by the information given to the solving agent?

This is a small synthetic benchmark and a certificate checker, built on the ideas in
Abundant's [realistic targets post](https://www.abundant.ai/blog/realistic-targets/).
It is an independent project, not affiliated with Abundant. No Leader Lane data is used.

The counting argument is the post's own: in its second case study, cases with identical
inputs are grouped and the majority answer gives the best achievable score. What this
project adds is a certificate format, a checker that stays correct when the certificate
is wrong or incomplete, and a benchmark with hard negatives.

**No model experiment has been run. No research efficacy claim is made.** Every number
below comes from deterministic reference auditors with no LLM involved.

## The idea in one example

A task author has a table of 72 loan applications. The solving agent will see three
columns (`income_band`, `debt_band`, `long_term`) and must predict `approved`. A fourth
column, `underwriter_override`, is withheld. The author proposes a target of 68 of 72
correct, which is what an oracle given every column scores.

An auditor claims the target is out of reach. Its evidence is a list of collisions.
Here is one, from `python3 ceiling.py demo`:

| row | income_band | debt_band | long_term | underwriter_override (withheld) | approved |
|---|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 0 | 0 |
| 8 | 1 | 1 | 1 | 1 | 1 |
| 20 | 1 | 1 | 1 | 0 | 0 |
| 39 | 1 | 1 | 1 | 1 | 1 |
| 42 | 1 | 1 | 1 | 1 | 1 |

The agent sees five identical rows. Whatever it answers for that pattern, it gets at
least two wrong. Across the seven collision groups in this table, at least 9 rows must
be wrong, so no solver can beat 63 of 72. The target of 68 is refuted.

In the post's terms, reward is `(score - start) / (target - start)`. Here start is 37,
so a reward measured against 68 is at most 0.84 of the reward measured against 63.

## What the checker guarantees

`check_certificate` accepts a group only if its rows exist, are distinct, are identical
on every agent-visible column, carry at least two different labels, and reuse no row
from another group. It then counts the errors those rows force.

- **Sound for any input.** A wrong, partial, or adversarial certificate can only give a
  looser bound, never a false refutation. Tests fuzz this with random groups.
- **No hand-written model.** The check runs against the table itself. v0.1 needed a
  formal spec written alongside the prose; this does not.
- **One-sided.** It can refute a target. It cannot confirm one.

A certificate may declare a subset of the visible columns to match on. The checker
then reports the result as `restricted`: it limits rules that use only those columns,
and it is never counted as a refutation for every solver. Declaring a withheld column
voids the certificate.

The bound covers deterministic predictors that depend only on the agent-visible
columns. A randomized predictor obeys it in expectation, not on every draw. Row ids
are shuffled and carry no information.

## Benchmark

Five families (loan, inventory, invoice, budget, attendance), four scenarios each,
six variants per scenario: 120 tasks. Loan, inventory and invoice are development;
budget and attendance are held out. Each family has a different labelling rule, and
the withheld value changes the label in anywhere from 2 of 9 visible cells to all of them.

| Variant | Withheld column | Target | Truth |
|---|---|---|---|
| complete | none | oracle score | supported |
| irrelevant | present, label ignores it | oracle score | supported |
| redundant | present, recoverable from visible columns | oracle score | supported |
| withheld | present, label depends on it | oracle score | unsupported |
| optimistic | none | oracle score plus half the noise | unsupported |
| sparse (stress) | as withheld, plus a near-unique visible column | oracle score | unsupported |

`irrelevant` and `redundant` are the hard negatives: something is hidden, but it does
not matter. An auditor that flags every withheld column fails on them.

Truth is set by a reference solver: the best visible-only predictor under the
generator, scored on the table. A target is unsupported when that solver falls short.

`data/v02/auditor/` holds what an auditor may see: the full table and the proposed
target. `data/v02/grading/` holds variant, truth, and ceilings, and must never be sent
to a model. The auditor is on the author's side, so it does see withheld columns and
labels; what it must not see is the answer to the audit.

## Reference results (no LLM)

`python3 ceiling.py control --split all`, core variants, 100 tasks:

| Auditor | Verdict accuracy | False alarms | Certified recall | False certified refutations |
|---|---|---|---|---|
| exhaustive (group every row) | 1.00 | 0.00 | 1.00 | 0 |
| lazy (three pairs only) | 0.65 | 0.00 | 0.13 | 0 |
| sloppy (drops a visible column) | 0.72 | 0.47 | 0.00 | 0 |
| trusting (always supported) | 0.60 | 0.00 | 0.00 | 0 |
| alarmist (always unsupported) | 0.40 | 1.00 | 0.00 | 0 |
| flag any withheld column | 0.40 | 0.67 | 0.00 | 0 |

The sloppy auditor is the point of the design: it raises false alarms on 47% of
supported tasks, 98% of its groups are rejected, and it certifies nothing false.

Restricted certificates, where the auditor declares the columns it matched on:

| Auditor | Stress set: targets refuted | Core set: supported targets "refuted" |
|---|---|---|
| drops near-unique columns | 20 of 20 | 0 of 60 |
| always drops the last visible column | 20 of 20 | 28 of 60 |

Both rows pass the checker. The second is misleading on 28 tasks where the dropped
column carried signal. The checker alone reports the scope of a claim; it cannot judge
the choice of columns.

### Testing what a restriction dropped

Each task also has 144 labelled practice rows from the same generator
(`data/v02/practice/`), standing in for the training data a solver would get. For a
restricted certificate, `signal_check` fits two lookup learners on the practice rows,
one on the kept columns and one on all visible columns, and scores both on the 72
evaluation rows. A permutation test, shuffling the dropped columns among practice rows
that share kept values, gives a p-value for the gain.

If no signal is detected, the certificate is promoted: its bound becomes the restricted
ceiling plus an approximate 95% upper bound on the dropped columns' gain. A promoted
result is statistical and specific to this learner and this much data. It is never
counted as a refutation for every solver.

| Auditor | Misleading refutations flagged | False flags where nothing was cut | Promoted refutations of supported targets | Stress targets refuted after promotion |
|---|---|---|---|---|
| always drops the last visible column | 27 of 28 | 0 of 64 | 0 | 20 of 20 |
| drops near-unique columns | none to flag | 0 of 20 | 0 | 20 of 20 |

The one unflagged case has p = 0.065 and a gain of 3 rows. Its promoted bound adds
the uncertainty on that gain and stays above the target, so it refutes nothing. When
the dropped column is pure noise, the test flags 1 of 100 core tasks.

## Two-sided interval

`python3 ceiling.py interval --split all` reports three numbers per task, in correct
rows out of 72:

- **Floor:** the score on the evaluation rows of a lookup-table solver trained only on
  the practice rows, with its columns chosen by cross-validation on practice. A real
  solver reaches it, so the best achievable score is at least this.
- **Ceiling:** the exact any-solver ceiling that collisions prove.
- **Realistic estimate:** the same procedure run both ways (train on practice and score
  the evaluation rows, then the reverse, scaled to 72) and averaged. An estimate, not a
  bound.

A target above the ceiling is refuted; a target at or below the floor is witnessed,
meaning a practice-trained solver actually reached it; anything between is open.

| Truth (generator) | Witnessed | Open | Refuted |
|---|---|---|---|
| supported (60 core tasks) | 59 | 1 | 0 |
| unsupported (40 core tasks) | 0 | 0 | 40 |
| unsupported (20 stress tasks) | 0 | 20 | 0 |

No status contradicts the generator. On the stress set the any-solver ceiling is 72,
so all 20 stay open; the realistic estimate falls below the target on all 20, and the
promoted restricted bound from the signal check refutes all 20.

This replaces the generator's reference solver as the measure of a realistic score. The
two often disagree: on the `withheld` variant, the practice-trained floor differs from
the reference by -7 to +10 rows, because a single 72-row draw is noisy. In the worked
example (loan-0, `withheld`) the floor is 59, not the reference's 55, so the interval is
[59, 63] around a target of 68. Truth labels still come from the reference solver.

## Tripwire

`tripwire(task, predictions=...)` flags a submitted result above the any-solver ceiling.
A deterministic solver that uses only the visible columns cannot get there, so a score
above it means the solver saw something it should not have (a withheld column, the
labels, a grader leak) or the score is wrong. With predictions, it also counts the
groups of identical-looking rows that got different answers; beating the ceiling
requires at least one.

A solver that randomises among identical-looking rows can beat the ceiling by luck. The
tripwire bounds that chance (a Chernoff bound over every way of answering each group,
checked against exact enumeration on small cases) and raises an alarm only below 1%.

`python3 ceiling.py tripwire --split all`, core tasks:

| Solver | Within ceiling | Above, luck plausible | Alarm |
|---|---|---|---|
| honest, trained on practice rows | 100 | 0 | 0 |
| randomises close calls (5,000 runs) | 4,998 | 2 | 0 |
| copies the labels | 0 | 0 | 100 |
| uses the withheld column (20 `withheld` tasks) | 4 | 11 | 5 |

The last row is the limit. Leaking the withheld column gains only a few rows, and on 72
rows the alarm needs an excess of about 6 to 10 rows before luck drops below 1%. On the
stress set the ceiling is 72, so the tripwire cannot fire at all.

## Pairwise F1

The post's clonal-grouping case is scored by pairwise F1, not accuracy.
`check_f1_certificate` takes pairs as rows, with label 1 when two items belong to the
same cluster. Each collision group gets one call, same or different, and
`best_pairwise_f1` finds the best F1 by sorting groups by their share of truly-same
pairs and taking the best prefix. Tests check it against brute force and against every
possible rule on a toy task.

The bound holds for rules that decide each pair from that pair's own features, so its
scope is `pair_rules`, never every solver. `toy_clonal_task` shows why: 48 cells in 12
clones, where sibling clones differ only in a hidden light chain.

| Features matched on | F1 ceiling | Start: link similar cells, take connected components |
|---|---|---|
| same gene, junction mismatches | 0.575 | 0.592 |
| plus "linked by the start method" | 0.592 | 0.592 |

With only the two pair features, the start beats the "ceiling", because linking through
other cells uses information those features lack. Adding the chain feature brings it
inside the bound. The post's four features include a chain distance; this shows why that
feature is needed. The proposed target of 1.0 is refuted either way.

## Known weaknesses

1. **A short script is the strongest auditor here.** `exhaustive` is a group-by and it
   scores perfectly on the core set. On clean tables an LLM adds nothing over it. The
   contribution is the certificate format and checker, not a claim that a model is needed.
2. **Realistic features defeat the strong claim, and the weaker claim needs a reviewer.**
   In the `sparse` stress variant one near-unique visible column leaves no two rows
   identical. Nothing can be certified for every solver: 0 of 20, although the target
   is just as unreachable. Real data looks like this. The post meets the same wall:
   counting only identical inputs puts its antibody ceiling near 0.999, and its adopted
   figure comes from restricting to four features. Restricted certificates take the
   same step and recover all 20, but only as a claim about rules using the declared
   columns. A careless choice of columns passes the checker and misleads (table above).
3. **The bound is loose even when it works.** In the example above the certificate
   proves 63; the reference solver reaches 55. The gap is rows a lookup table could
   memorize. The certificate refutes a target; it does not estimate the right one.
4. **Target size is a mild tell.** The best single-threshold rule on surface features
   reaches 0.70 against a 0.60 majority rate (`python3 ceiling.py shortcuts --split all`),
   because inflated targets run slightly high. A test fails if any shortcut exceeds 0.70.
5. **Accuracy only.** The forced-error count is exact for accuracy. The post applies its
   grouping to a pairwise F1 score, and another of its cases uses average precision.
   Certificates for those metrics are not worked out here.
6. **Small and synthetic.** 20 scenarios, 5 rule families, 2 held-out families.
   One author wrote every rule.

## v0.1, kept for the record

`audit.py` is the first pilot: text tasks where an auditor produced two assignments
that fit the documents and gave different answers. It is retained because a regex with
no reasoning scores 100% on all 80 tasks, including valid witnesses. That result is
pinned as `test_known_weakness_v01_is_solved_by_a_regex`. It is why v0.2 measures
shortcut baselines before any model is run.

## Run locally

Python 3.9+; no dependencies, credentials, GPU or network.

```sh
python3 -m unittest discover -s tests
python3 ceiling.py build
python3 ceiling.py demo
python3 ceiling.py control --split all --output runs/v02-reference.json
python3 ceiling.py shortcuts --split all --output runs/v02-shortcuts.json
python3 ceiling.py interval --split all --output runs/v02-interval.json
python3 ceiling.py tripwire --split all --output runs/v02-tripwire.json
python3 ceiling.py prompts --split dev --mode baseline --output runs/v02-baseline-prompts.jsonl
python3 ceiling.py prompts --split dev --mode certificate --output runs/v02-certificate-prompts.jsonl
python3 ceiling.py evaluate --split dev --predictions runs/predictions.jsonl
```

A prediction is one JSON object per line with `id`, `verdict` (`supported`,
`unsupported`, `uncertain`), optional integer `ceiling_correct`, and, for the
certificate condition, `groups` and optionally `columns` (the subset matched on).
Score each repeat as its own file.

## Next

[docs/PROTOCOL.md](docs/PROTOCOL.md) specifies a model comparison. It has not been run
and no run is currently scheduled or funded; the prompts and scorer are ready for
anyone who wants to try it. [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) gives the
reasoning behind each decision.
