# Experiment protocol (v0.2)

Nothing here has been run. No provider or budget has been chosen.

## Question

When a model audits a task's proposed target, does asking for checkable collisions
give a more trustworthy result than asking for a verdict?

"More trustworthy" is defined before the run:

- **Primary:** false alarms. In the certificate condition only certified refutations
  count as flags, so the false-alarm rate is 0 by construction. The open question is
  the baseline's false-alarm rate on the hard negatives (`irrelevant`, `redundant`).
  If it is also near 0, certificates buy nothing on this benchmark.
- **Cost of that guarantee:** certified recall. A model that reasons correctly but
  cannot list valid row groups loses recall the baseline keeps.
- **Secondary:** invalid-group rate, certificate tightness, and the error of the
  baseline's `ceiling_correct` estimate.
- **Reported apart:** restricted certificates, where the model declares a subset of
  columns. They never count as certified refutations. Report how many refute a
  supported target and how many drop a column that carried signal.

The hypothesis can fail three ways: the baseline has no false alarms, certified recall
is far below baseline recall, or both conditions sit at ceiling.

## Baselines a model must be compared against

From `runs/v02-reference.json`, core set: a group-by script scores 1.00 with full
certified recall; always answering "supported" scores 0.60; flagging every withheld
column scores 0.40. A model result is interesting only relative to these.

## Conditions

Same model, version and sampling settings for both. One request per task, no shared
history, no tools. Prompts are `BASE_PROMPT` and `CERT_PROMPT` in `ceiling.py`,
followed by the auditor-view JSON. Three repeats per task, each scored separately.

A third condition is worth adding if budget allows: the certificate prompt with a
code-execution tool. That is closer to how a validation notebook is written, and it
is where the group-by baseline becomes the fair comparison.

## Size of a run

All 120 tasks, both conditions, one repeat: about 693,000 characters of prompt.
Three repeats: about 2.1 million characters. Table-heavy JSON tokenizes at roughly
3 characters per token, so expect in the region of 0.7 million input tokens, plus
output that is small for the baseline and a few hundred tokens per task for
certificates. A development-only run is 60% of that. Price this against the chosen
provider's current rates before agreeing a budget.

## Stages

1. Review development tasks by hand against the grading files.
2. Freeze the dataset and prompts under version control before any test-split run.
3. Run development only. Inspect failures. Fix prompts or parsing, not labels.
4. Run the held-out families once. Report raw counts per variant and per family.
5. Report the stress variant separately. It is expected to fail.

Record raw responses, model identifier, settings, timestamps, token usage and
parsing failures. Missing or unparseable responses stay in every denominator.

## Statistical unit

120 tasks are 20 scenarios from 5 rule families. Variants of a scenario share rows,
and repeats share everything. Aggregate paired differences by scenario; if adding
intervals, resample scenarios. With two held-out families, no claim about new
rule types is supported.

## Leakage controls

Send only `data/v02/auditor/` content. Never send grading files, `ceiling.py`, this
repository, or sibling variants in the same context. Task ids are opaque hashes and
row order is shuffled per variant. A tool-using auditor must run in a directory that
contains only its one task.

## Claim policy

Supported by a certified refutation: on this table, no deterministic predictor of the
agent-visible columns reaches the proposed target.

Not supported by anything here: that a target is correct; that a refuted target's
replacement should be the certified bound; that the method works on continuous or
high-cardinality features; that rescaling rewards improves training.

## Follow-ups, in order of value

1. A way to review restricted certificates. They are implemented and checked, but
   the choice of columns is unreviewed; measure how often a model's declared subset
   drops a column that carried signal (`restrictions_that_cut_signal`).
2. Ceilings for ranking and pairwise metrics.
3. Tasks where the visible/withheld split is described in prose, so the auditor has
   to work out what the agent sees. This is where a model, not a script, is needed.
4. Independently authored tasks, then a public real-world example.
