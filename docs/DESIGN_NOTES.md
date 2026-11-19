# Design notes

Why each decision was made, and the objections it has to answer.

## Why the project changed direction

v0.1 asked a model whether a short text question had a unique answer, and checked
its two "worlds" against a formal spec. Three problems surfaced before any model ran:

1. A regex labelled all 80 tasks and produced valid witnesses. The benchmark could
   not separate a good auditor from string matching.
2. The verifier needed a formal spec written by the same generator as the prose.
   Real tasks do not come with one.
3. The auditor saw only the agent's inputs and answered yes or no. In the post, the
   validator has privileged access and produces a number: the best supported score.

v0.2 keeps the central idea, a counterexample a program can check, and moves it to
where those three problems go away. The counterexample is now a pair of rows the
agent cannot distinguish, and the table is its own specification.

## Why a collision is a proof

Fix any deterministic predictor that uses only the agent-visible columns. Two rows
with identical visible values get the same prediction. If their labels differ, at
least one is wrong. For a group with label counts `c1, c2, ...`, the best the
predictor can do is match the largest count, so it errs on `size - max(c)` rows.
Summing over groups gives a floor on errors, hence a ceiling on correct rows.

Dropping rows or groups only lowers the error floor. That is why a partial or
careless certificate is still sound, and why the checker pools rows from valid groups
without trusting how the auditor arranged them.

## Why two notions of ceiling

- **Exact ceiling:** what any function of the visible columns could score on this
  table, including one that memorizes it. This is what collisions bound.
- **Reference solver:** the best visible-only predictor under the generating process.
  This is closer to "what a strong solver could realistically achieve".

Exact is always at least reference. Truth labels use the reference solver, because a
target reachable only by memorizing the evaluation rows is not a realistic target.
The certificate bounds the exact ceiling, because that is the only one provable from
the table alone. The gap between them is the method's honest limit, and the `sparse`
variant is built to make that gap total.

## Why these variants

- `irrelevant` and `redundant` exist to punish the lazy rule "something is hidden,
  so the target is wrong". Without them a false-alarm rate means nothing.
- `optimistic` has no hidden column at all. The target is wrong for a different
  reason, so "no withheld column" cannot be read as "supported".
- `sparse` is a case the method is expected to lose. It is reported separately.

## Why reference auditors instead of a symbolic oracle

v0.1's oracle read the private spec, so its perfect score said nothing. v0.2's
`exhaustive` auditor uses only the auditor view. It is a real, deployable baseline,
which is uncomfortable: it means a script solves the core set. That is stated in the
README because an evaluator would find it in a minute.

The other reference auditors show the metrics behave: `lazy` is sound but loose,
`sloppy` is confidently wrong and gets nothing certified.

## Objections and answers

**"A group-by solves this. Where is the research?"**
On clean tables, nowhere, and the README says so. The claim is narrower: unverified
audit verdicts can be confidently wrong, and a certificate format makes the wrong ones
harmless. Whether models produce wrong verdicts here is the experiment not yet run.

**"Real features are continuous. You will never see exact collisions."**
Correct, and the stress variant measures it: 0 of 20 certified. The next step is
near-collisions, which trade proof for an explicit assumption a reviewer can reject.

**"Your bound is too generous."**
Yes. It refutes targets; it does not set them. A refuted target still needs the kind
of validation work the post describes to find its replacement.

**"The auditor sees the labels. Isn't that leakage?"**
The auditor works for the task author, who owns the labels. The party that must not
see them is the solving agent. What the auditor must not see is the grading file.

**"Twenty scenarios is nothing."**
Agreed. It is enough to show the checker is sound and the benchmark is not trivially
gamed. It is not enough for any claim about models.

**"Why keep v0.1?"**
Because the regex result is the reason v0.2 has a shortcut gate, and deleting the
evidence would make the project look more deliberate than it was.

## What I would say I learned

Run the cheapest possible baseline before the expensive experiment. A regex check
removed the need for a model run that would have shown nothing.
