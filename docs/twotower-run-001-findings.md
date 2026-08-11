# Two-tower `run_001`: findings, fixes applied, and what's left

Plain-language summary of what we learned from the first full LoRA
fine-tune (`run_001`), the fixes applied since, and what's left. For full
metric tables and technical detail, see
[`twotower-run-001-results.md`](twotower-run-001-results.md) and
[`possible-bugs.md`](possible-bugs.md) (#1, #2, #3, #4).

For what the pairs and the target actually mean — real accept/decline outcomes
on intros production already recommended, plus the <100 ms serving budget — see
[`objective.md`](objective.md). It explains why the absolute AUCs below sit near
0.6: the task is separating accepted from declined among *already-plausible*
intros, not relevant from irrelevant candidates.

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
| Real held-out pairs, judge-distilled (Arm A recipe, soft labels) | **0.604** | 0.359 | 0.552 | Best twotower pair AUC yet — see "Distillation experiment" below |
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

## Distillation experiment: LLM-judge soft labels instead of hard 0/1 (2026-07-26)

Small, one-variable side experiment run alongside the Arm A/B/C plan, not a
replacement for it. Same recipe as Arm A exactly (LoRA on voyage-4-nano,
5 epochs, 111 real-only train pairs, `ContrastiveLoss`, identical holdout) —
the only change is the training label. Instead of the hard 0/1 accept/decline
outcome, each pair's label is the naive LLM judge's (`google/gemini-3.1-flash-lite`,
see [`llm-judge-experiment.md`](llm-judge-experiment.md)) confidence-signed
score in `[0, 1]` (`baselines/llm_judge/judge.py::verdict_to_score`), used
directly as a soft `ContrastiveLoss` target. `ContrastiveLoss` linearly
interpolates between its "pull together" and "push apart" terms by the label,
so a continuous label is a valid target with no loss-function change needed.

The judge was re-run with `--split train` to score the 131 real train pairs
it hadn't seen before (its verdict cache already had the 69 holdout pairs
from earlier baseline runs) — 131 new API calls, a few cents, no other new
cost. `scripts/build_judge_soft_labels.py` turns the cache into a
`pair_id -> score` file; `twotower/train_distill.py` monkeypatches
`twotower.train.pairs_to_hf_dict` for the one call site that builds the
training dataset, so nothing else in the training/eval pipeline changed.

**Finding: distillation beat every twotower run to date on pair AUC**
(0.604 vs. Arm A's 0.579, `run_001`'s 0.578), and is now ahead of TF-IDF's
0.592 — the first twotower run to clear that bar. Hard-negative AUC also
moved off exactly-chance (0.552 vs. Arm A's 0.500). It traded away some
retrieval quality to get there — MRR dropped to 0.359 from Arm A's 0.388 —
so this is a real but partial win, not a clean sweep.

**Caveat: the checkpoint-selection safety net didn't have a real candidate
this run.** The best train-dev score was found at an early checkpoint that
`save_total_limit=3` had already pruned by the time selection ran, so
`select_best_checkpoint` fell back to the final epoch (same fallback
behavior as `possible-bugs.md` #2, not a new bug). Train-dev here is also
only 20 pairs — too small to trust as a selection signal regardless. Treat
this result as a promising lead worth carrying into Arm C, not a confirmed
win; a repeat with a higher `save_total_limit` (or holdout re-eval across
more checkpoints) would firm it up.

Rerun with:

```bash
python -m baselines.llm_judge.eval --data-dir data --variant naive --split train
python scripts/build_judge_soft_labels.py
modal run twotower/modal_train_distill.py --run-id distill_judge_001 --epochs 5
modal volume get dorby-twotower-checkpoints distill_judge_001/metrics_holdout.json \
    artifacts/twotower/distill_judge_001/metrics_holdout.json --force
```

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
