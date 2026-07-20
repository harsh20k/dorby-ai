# Two-tower `run_001`: findings, fixes applied, and what's left

Plain-language summary of what we learned from the first full LoRA
fine-tune (`run_001`), the fixes applied since, and what's left. For full
metric tables and technical detail, see
[`twotower-run-001-results.md`](twotower-run-001-results.md) and
[`possible-bugs.md`](possible-bugs.md) (#1, #2, #3, #4).

## Verdict

**The fine-tune does not beat the frozen baselines on real data — but we
now know exactly why, we've fixed the root cause, and we've proven the fix
points in the right direction.** Do not generate a larger synthetic batch
at full scale until that's explicitly decided (see "What's left" below) —
that's the decision gate from
[`two-tower-fine-tune-plan.md`](two-tower-fine-tune-plan.md) doing its job.

| Set | Pair AUC | MRR | Hard near-miss AUC | Notes |
|---|---|---|---|---|
| Practice set (train-dev, mostly fake) | 0.986 | 0.743 | — | Looks great; misleading |
| Real held-out pairs, `run_001` (real + unfixed fake) | 0.578 | 0.283 | 0.4845 (below chance) | Doesn't beat baselines |
| **Real held-out pairs, Arm A (real-only, no fake)** | **0.579** | **0.388** | **0.500 (chance)** | **Beats `run_001` on almost everything, with 1/5th the data** |
| Frozen Voyage-4-nano (matched holdout) | 0.579 | 0.461 | 0.571 | Roughly Arm A's neighborhood |
| Frozen Voyage-4-large (prod, matched holdout) | 0.609 | 0.529 | 0.602 | Bar to beat |

The Arm A row is the newest and most important finding: training on
**111 real pairs only — a fifth of `run_001`'s 530 — with zero synthetic
data — beat `run_001` on every metric except pair AUC** (a tie). The
unfixed synthetic data wasn't neutral. It was actively making the model
worse than not using it at all.

## What went wrong (in simple terms)

The model looked strong on the practice set and weak on real people. The
practice set was mostly synthetic (about 80%); the held-out set is all
real.

Good and bad synthetic examples were written from **two different
prompts**. A small adapter learned "this reads like a bad-example prompt"
instead of "this person doesn't fit the search." We confirmed this three
separate ways:

1. Some synthetic negatives contained overt giveaways (phrases like
   "critical distinction," "mistaken for," "has never") — 9 of 240 had one.
2. A trivial classifier, shown only the candidate's own profile text (no
   seeker, no query), guessed the label with **99.2% accuracy** on the old
   synthetic data — vs. **48.7% (pure chance)** on real data. That's a
   smoking gun: the label was recoverable from surface style alone.
3. The Arm A training experiment (above): remove the synthetic data
   entirely, and the model gets *better*, not worse — direct proof the
   synthetic data was teaching the wrong thing.

We also found the actual mechanical cause of the "bad-example prompt"
smell: `synth_pipeline/llm.py` was truncating the seeker's own profile to
400 characters before showing it to the generator — but real profiles run
up to ~23,700 characters. The generator was never shown most of what it
was told to "keep the same," so it improvised, and the negative-generation
prompt lacked a style-matching instruction the positive prompt already
had.

Checkpoint picking also silently kept the final epoch instead of the best
mid-run one ([`possible-bugs.md`](possible-bugs.md) #2) — small impact on
`run_001` specifically, but the safety net was broken. Confirmed fixed:
on Arm A's training run, it correctly picked epoch 4 over the (worse)
final epoch 5.

## What we fixed (2026-07-19)

1. **Seed truncation** (`synth_pipeline/llm.py::truncate_pair_for_prompt`)
   — the seeker's profile is no longer truncated before generation.
2. **Banned meta-commentary give-aways** in `generate_neg.md` — explicit
   rule against writing sentences that narrate the mismatch.
3. **Closed the style gap** — `generate_neg.md` now has the same "match
   Boardy CRM tone" instruction `generate_pos.md` already had.
4. **Fixed checkpoint selection** (`possible-bugs.md` #2) — root cause was
   the evaluator writing results one directory deeper than the code
   looked, plus fragile parsing of a floating-point epoch number. Now
   keyed on training step count, which maps exactly to how checkpoints are
   named on disk.
5. **Rebuilt the baseline comparison on matched ground** — all four models
   (BERT, Voyage-nano, Voyage-large, twotower) are now scored on the exact
   same 69 real held-out pairs, not different-sized datasets. This
   actually made Voyage-large's numbers look *better* than the old
   comparison suggested (smaller, holdout-only candidate pool → easier
   retrieval), sharpening the gap twotower needs to close.

Validated the data fix at small scale before trusting it: generated a
36-pair pilot batch with the fixed prompts, and the "guess from style
alone" score dropped from 99.2% to 86.3% — a real, material improvement,
though not fully back down to the real-data floor (~49-53%). The residual
signal looked like small-sample topic clustering (36 examples across ~15
seed queries) rather than the old structural tell — inspected the
classifier's top words and found ordinary topic vocabulary ("health,"
"climate," "campus"), not narrative giveaways.

## What's left

**Arm C: retrain with fixed synthetic data at full scale, and see if it
beats Arm A.** Arm A already proves the architecture and training recipe
aren't the problem — real data alone, with the same recipe, roughly
matches Voyage-nano. The open question is whether *fixed* synthetic data
adds value on top of that, or whether it's still net-harmful even after
the fixes and just needs more work. This requires a full-scale synthetic
regeneration (similar size/cost to the original `batch_500_001`) — that
scale/timing decision has been deliberately deferred, not scheduled yet.

Bar to clear for Arm C: beat Arm A's numbers (not just Voyage-large's).

## What not to do yet

- Do not generate a large synthetic batch at full scale until Arm C's
  scope is explicitly decided.
- Do not treat train-dev pair AUC as evidence of real matching quality
  while a meaningful fraction of that slice is synthetic — it's proven
  unreliable as a selection/decision signal in this project.
- Do not compare twotower numbers against baseline numbers computed on a
  different population or sequence length — use
  `docs/baseline-results-holdout.md`, not `baseline-results-all.md`, for
  any twotower-vs-baseline claim.

## Related docs

- [`twotower-run-001-results.md`](twotower-run-001-results.md) — full
  metric tables, matched-population comparison, Arm A details
- [`baseline-results-holdout.md`](baseline-results-holdout.md) — the
  matched-population comparison table (source of truth for go/no-go)
- [`possible-bugs.md`](possible-bugs.md) — #1 seeker–query drift (fixed),
  #2 checkpoint selection (fixed), #3 comparison mismatch (fixed), #4
  synth-artifact overfitting (root cause fixed and confirmed via Arm A;
  full-scale validation pending)
- [`two-tower-fine-tune-plan.md`](two-tower-fine-tune-plan.md) —
  architecture, loss choice, decision gate
