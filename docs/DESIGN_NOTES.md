# Design notes

The reasoning behind each design decision, and the objections each one anticipates.

## Why the design changed between versions

The first version (`audit.py`) asked a model whether a short text question had a unique
answer, and checked its two "worlds" against a formal spec. Three problems appeared
before any model was run:

1. A regular expression labelled all 80 tasks and produced valid witnesses, so the
   benchmark could not separate a good auditor from string matching.
2. The verifier needed a formal spec written by the same generator as the prose. Real
   tasks do not come with one.
3. The auditor saw only the agent's inputs and answered yes or no. In Abundant's post
   the validator has privileged access and produces a number: the best supported score.

The second version (`ceiling.py`) keeps the central idea, a counterexample a program
can check, and changes what the counterexample is. It is now a group of rows the agent
cannot distinguish, and the table is its own specification. The first version stays in
the repository, with the regex result pinned as a test, because that result is why the
second version measures shortcut baselines first.

## Why a collision is a proof

Fix any deterministic predictor that uses only the agent-visible columns. Rows with
identical visible values get the same prediction. If their labels differ, at least one
is wrong. For a group with label counts `c1, c2, ...`, the best a predictor can do is
match the largest count, so it errs on `size - max(c)` rows. Summing over groups gives
a floor on errors, and so a ceiling on correct rows.

Dropping rows or groups only lowers the error floor. That is why a partial or careless
certificate is still sound, and why the checker pools rows from valid groups without
trusting how the auditor arranged them.

This is the grouping argument from the post's second case study. The contribution here
is the certificate format and checker around it, not the argument.

## Why two notions of ceiling

- **Exact ceiling:** what any function of the visible columns could score on this table,
  including one that memorizes it. This is what collisions bound.
- **Reference solver:** the best visible-only predictor under the generating process.
  This is closer to what a strong solver could realistically achieve.

The exact ceiling is never below the reference solver's score. Truth labels use the
reference solver, because a target reachable only by memorizing the evaluation rows is
not a realistic target. The certificate bounds the exact ceiling, because that is the
only one provable from the table alone. The gap between the two is the method's limit,
and the `sparse` variant is built to make that gap total.

## Why restricted certificates are kept separate

Matching on fewer columns proves a true but narrower statement: no rule using only those
columns reaches the target. It says nothing about a solver that uses the dropped
columns. If a dropped column carries signal, the statement is true and the conclusion a
reader would draw from it is false.

So the checker labels every result with its scope, and the scorer never adds a
restricted result to the certified count. Whether a restriction mattered is measured
against the reference solver: it did if the restricted bound falls below what that
solver scores. Two reference auditors show both outcomes. One drops only near-unique
columns and is harmless here. One drops a column blindly and is not.

## Why these variants

- `irrelevant` and `redundant` punish the rule "something is hidden, so the target is
  wrong". Without them a false-alarm rate means nothing.
- `optimistic` has no hidden column. The target is wrong for a different reason, so "no
  withheld column" cannot be read as "supported".
- `sparse` is a case the method is expected to lose, and is reported separately.

## Why reference auditors instead of a symbolic oracle

The first version's oracle read the private spec, so its perfect score said nothing
about an auditor that lacks it. The second version's `exhaustive` auditor uses only the
auditor view, so it is a deployable baseline. It also solves the core set, which the
README states directly.

The other reference auditors check that the metrics behave: `lazy` is sound but loose,
and `sloppy` is confidently wrong and gets nothing certified.

## Anticipated objections

**A group-by solves this. Where is the research?**
On clean tables there is none, and the README says so. The narrower claim is that an
auditor's verdict can be wrong with confidence, and that a checked certificate cannot
turn a wrong claim into an accepted refutation for every solver. Whether models produce
wrong verdicts on this benchmark is the experiment that has not been run.

**Real features are continuous, so exact collisions will not occur.**
The stress variant measures this: 0 of 20 certified for every solver. The post meets the
same limit and answers it by restricting to a set of features. Restricted certificates
do the same and recover all 20, as a narrower claim. The checker cannot say whether a
dropped column mattered, and an auditor that drops one blindly "refutes" 28 of 60
supported targets. That judgment needs a reviewer.

**The bound is looser than the realistic score.**
Yes. It refutes targets and does not set them. Replacing a refuted target still needs
the validation work the post describes.

**The auditor sees the labels. Is that leakage?**
The auditor works for the task author, who owns the labels. The party that must not see
them is the solving agent. The auditor must not see the grading file.

**Twenty scenarios is a small sample.**
It is enough to show the checker is sound and the benchmark is not trivially gamed. It
is not enough for any claim about models.
