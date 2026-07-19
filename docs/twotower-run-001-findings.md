# Two-tower `run_001`: findings and recommended next steps

Plain-language summary of what we learned from the first full LoRA fine-tune
(`run_001`) and what to do next. For the full metric tables and technical
detail, see [`twotower-run-001-results.md`](twotower-run-001-results.md) and
[`possible-bugs.md`](possible-bugs.md) (#2, #4).

## Verdict

**The fine-tune does not beat the frozen baselines on real data.** Do not
generate a larger synthetic batch or launch `run_002` until the synthetic-data
shortcut problem is fixed. That is the decision gate from
[`two-tower-fine-tune-plan.md`](two-tower-fine-tune-plan.md) doing its job.

| Set | Pair AUC | Hard near-miss AUC | Notes |
|---|---|---|---|
| Practice set (train-dev, mostly fake) | 0.986 | — | Looks great; misleading |
| Real held-out pairs (69) | 0.578 | **0.4845** (below chance) | The number that matters |
| Frozen Voyage-4-nano | ~0.561 | — | Cached baseline |
| Frozen Voyage-4-large (prod) | ~0.573 | — | Bar to beat |

Hard near-misses below chance is the smoking gun: the model failed on exactly
the case the synthetic hard negatives were meant to teach.

## What went wrong (in simple terms)

The model looked strong on the practice set and weak on real people. The
practice set is mostly synthetic (about 80%); the held-out set is all real.

Good and bad synthetic examples are written from **two different prompts**.
A small adapter can learn “this reads like a bad-example prompt” instead of
“this person doesn’t fit the search.” Some synthetic negatives even contain
overt giveaways (phrases like “critical distinction,” “mistaken for,” “has
never”) — 9 of 240 synthetic negatives had one of these.

Pairwise “yes/no” scores kept climbing through training, while ranking quality
peaked early (epoch 2) and then drifted down. Checkpoint picking also silently
kept the final epoch instead of the best mid-run one
([`possible-bugs.md`](possible-bugs.md) #2) — small impact this run, but the
safety net is broken.

More epochs, or more of the same synthetic data, is likely to make the
shortcut worse — not better.

## Recommended improvements (do in this order)

### 1. Check if the fake data is “cheatable”

See whether you can tell a “good intro” example from a “bad intro” example
just by reading the candidate’s profile — without looking at who was
searching.

If that’s easy, the model can win on writing style from the generator, not by
learning who should actually meet whom. Build a tiny test that only sees the
candidate text and guesses good vs bad. If it scores way above chance, the
fake data has tells. Don’t train another full run until that goes away.
Re-check this every time you add new fake pairs.

### 2. Fix how the fake pairs are written

Stop good and bad examples from looking different in style. Make them differ
only in the actual matching mistake (wrong role, wrong stage, wrong city,
etc.).

In order:

1. Remove the obvious giveaways (profiles that basically say “this would be a
   bad intro”).
2. Keep the searcher’s profile and search text copied as-is; only invent the
   other person (also addresses seeker–query drift in
   [`possible-bugs.md`](possible-bugs.md) #1).
3. Better: start from a good match, then change just one thing to make it bad —
   so both sides sound the same.
4. Or: have one writing step produce both the good and the bad version
   together.

Until that’s fixed, making more of the same fake data will probably deepen the
problem.

### 3. Run a few small experiments before another full training

Don’t jump straight into another big training. First ask: does the fake data
help, hurt, or do nothing?

Suggested arms (same training recipe, one real held-out check each):

| Arm | Train on |
|---|---|
| A | Real pairs only |
| B | Real pairs + cleaned fake pairs |
| C | Real pairs + redesigned fake pairs (once ready) |

For each, ask: does it look great on the practice set but weak on real
held-out people? A big gap means it’s still cheating on style. Use the
practice set for early stopping; touch the real held-out set once per arm.

### 4. Fix training bookkeeping (only after the data is trustworthy)

Once the examples are clean:

1. Make “pick the best checkpoint” actually work — and complain loudly if it
   can’t find the per-epoch scores.
2. Prefer the version that’s best on hard near-misses and ranking, not just
   overall yes/no accuracy.
3. Train a bit more gently (fewer rounds; stop earlier when ranking stops
   improving).
4. When you have enough cases with the same searcher + one good match + one
   bad match, train in that format — closer to real “pick the right person
   from a list” use (`twotower/data.py::to_triplet_rows` is already stubbed).

### 5. Compare apples to apples

When asking “did we beat the old model?”, test both on the same people with
the same rules.

The cached baseline numbers were measured on a different (larger) set of pairs
and longer text limits, so “0.578 vs 0.573” is a rough hint, not a fair final
score. Re-score the frozen models only on the same 69 real held-out pairs,
with matched text-length settings. Wire those results into
`scripts/export_baseline_results.py` alongside
`artifacts/twotower/<run>/metrics_holdout.json`. Use that table as the only
go / no-go comparison going forward.

## What not to do yet

- Do not generate a larger synthetic batch at the current prompt settings.
- Do not launch `run_002` hoping more epochs or more of the same data will
  close the gap.
- Do not treat train-dev pair AUC as evidence of real matching quality while
  most of that slice is synthetic.

## Related docs

- [`twotower-run-001-results.md`](twotower-run-001-results.md) — full metric
  tables and caveats
- [`possible-bugs.md`](possible-bugs.md) — #1 seeker–query drift, #2 checkpoint
  selection, #3 comparison mismatch, #4 synth-artifact overfitting
- [`two-tower-fine-tune-plan.md`](two-tower-fine-tune-plan.md) — architecture,
  loss choice, decision gate
