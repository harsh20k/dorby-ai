# Two-Tower `run_001`: results and why they aren't comparable to baselines yet

Full 5-epoch LoRA fine-tune of `voyage-4-nano` on `twotower/`, run on Modal
(L4 GPU). See `docs/two-tower-fine-tune-plan.md` for the architecture/loss
decision and `docs/modal-training-guide.md` for the Modal setup.

Plain-language findings and recommended next steps:
[`twotower-run-001-findings.md`](twotower-run-001-findings.md).

## Run config

| | |
|---|---|
| Model | `voyageai/voyage-4-nano`, LoRA rank 8, alpha 16, targets `q/k/v/o_proj` |
| Loss | `ContrastiveLoss` (pairwise labeled, margin 0.5) — see decision rationale below |
| Data | 660 canonical pairs → 530 train / 61 train-dev / 69 frozen holdout |
| `max_seq_length` | 4096 |
| Epochs | 5, batch size 2, grad accum 4, lr 2e-4 |
| `split_hash` | `20bbe8f293127372` |
| `data_hash` | `0cdfeb652624869a` |
| Trainable params | 983,040 / 347,435,008 (0.28%) |
| Runtime | ~754s train + eval on L4 |
| Adapter | `artifacts/twotower/run_001/adapter/` |

Loss choice: the reviewed synthetic data is independently-labeled pairs, not
same-seeker (positive, hard-negative) triples — of 91 distinct seekers with
staged synth pairs, only 5 had both a positive and a negative. That ruled
out `MultipleNegativesRankingLoss` triplets as the primary loss; pairwise
`ContrastiveLoss` consumes the labeled data directly, with a triplet path
kept available in `twotower/data.py::to_triplet_rows()` for later.

## Train-dev metrics by epoch

| Epoch | pair AUC | pair AP | best-F1 | retrieval MRR | NDCG@10 | R@10 |
|---|---|---|---|---|---|---|
| 1 | 0.890 | 0.886 | 0.857 | 0.774 | 0.830 | 1.000 |
| 2 | 0.943 | 0.933 | 0.906 | **0.781** | **0.834** | 1.000 |
| 3 | **0.989** | **0.989** | **0.952** | 0.734 | 0.789 | 0.967 |
| 4 | 0.986 | 0.986 | 0.935 | 0.727 | 0.785 | 0.967 |
| 5 (final, selected) | 0.986 | 0.987 | 0.935 | 0.743 | 0.797 | 0.967 |

Note: pair-classification metrics (AUC/AP/F1) keep climbing through epoch 5,
while retrieval metrics (MRR/NDCG/R@10) peak at epoch 2 and drift down
afterward. Consistent with the model increasingly nailing the binary
pos/neg boundary at some cost to fine-grained ranking among candidates —
worth watching if epoch count increases in later runs.

Epoch 3 (AUC 0.989) technically outscored the selected epoch 5 (AUC 0.986)
on the intended selection metric — see
[`docs/possible-bugs.md`](possible-bugs.md) #2, checkpoint selection
silently fell back to the final epoch instead of reloading epoch 3.

## Raw side-by-side vs. cached baselines (NOT a valid comparison — see below)

| | pair AUC | pair AP | best-F1 | MRR | NDCG@10 | R@10 | population / max_len |
|---|---|---|---|---|---|---|---|
| Frozen BERT | 0.470 | 0.511 | 0.676 | 0.094 | 0.099 | 0.180 | 200 pairs / 512 |
| Voyage-4-nano | 0.561 | 0.557 | 0.671 | 0.301 | 0.360 | 0.600 | 200 pairs / 8192 |
| Voyage-4-large (prod) | 0.573 | 0.571 | 0.669 | 0.310 | 0.393 | 0.700 | 200 pairs / ~8192 |
| twotower run_001 (train-dev, epoch 5) | 0.986 | 0.987 | 0.935 | 0.743 | 0.797 | 0.967 | 61 pairs / 4096 |

The gap looks huge, but this table is not a valid head-to-head. Three
uncontrolled differences, in order of severity:

1. **Different populations.** `baselines/*/eval.py::load_pairs()` reads
   `data/dataset_positive.json`/`dataset_negative.json` directly with no
   train/holdout split — the cached baseline numbers (`num_positive: 100,
   num_negative: 100`) were computed when those files held only the
   original 200 real seed pairs, combining what's now the 131-pair "train"
   split and the 69-pair frozen holdout with no separation. The twotower
   number above is 61 train-dev pairs, a different subset entirely.
2. **Train-dev, not holdout.** Train-dev is user-disjoint from training but
   drawn from the same 660-pair pool the adapter trained on — 50 of the 61
   train-dev pairs are synthetic (`train_dev_synth: 50`), generated from
   the same LLM/prompt distribution the model was fine-tuned against. The
   baselines never saw any of this data, synthetic or real. Beating a
   frozen baseline on data drawn from your own training distribution is
   expected and not evidence of beating it on real-world matching.
3. **Different `max_seq_length`.** Baselines used 8192; twotower used 4096.
   More truncation on the twotower side — if anything this should hurt its
   numbers, not inflate them, but it's still an uncontrolled variable in a
   supposed apples-to-apples table.

## Real holdout eval (2026-07-19) — the number that actually matters

Ran `twotower/eval.py --split holdout --adapter-dir
artifacts/twotower/run_001/adapter` against the frozen 69-pair real
holdout (`eval_pair_ids` — never touched by the synth generator, in
training, train-dev, or few-shot conditioning). Locally on MPS, no
retraining, same `run_001` LoRA weights.

| | pair AUC | pair AP | best-F1 | MRR | NDCG@10 | Recall@10 |
|---|---|---|---|---|---|---|
| Frozen BERT | 0.470 | 0.511 | 0.676 | 0.094 | 0.099 | 0.180 |
| Voyage-4-nano | 0.561 | 0.557 | 0.671 | 0.301 | 0.360 | 0.600 |
| Voyage-4-large (prod) | 0.573 | 0.571 | 0.669 | 0.310 | 0.393 | 0.700 |
| twotower train-dev (mostly synthetic, misleading — see above) | 0.986 | 0.987 | 0.935 | 0.743 | 0.797 | 0.967 |
| **twotower real holdout (69 pairs)** | **0.578** | **0.487** | **0.619** | **0.283** | **0.359** | **0.655** |

**Verdict: the fine-tune does not beat the frozen baselines on real data.**
Pair AUC is roughly tied with Voyage-nano (0.578 vs 0.561), below
Voyage-large (0.573 — essentially a wash) on every other metric. The
0.986-vs-0.573 gap in the earlier "raw" table was almost entirely an
artifact of evaluating on data statistically close to the model's own
training distribution, not a real capability gap.

**Worse: hard-negative slice AUC is 0.4845 — below chance.** Slice
breakdown (`neg_hardness`, lexical-Jaccard quartiles): easy negatives
AUC 0.693 (n=10), hard negatives AUC 0.4845 (n=20). On real negatives that
are lexically close to the query — exactly the case the synthetic hard
negatives were designed to teach — the fine-tuned model performs *worse
than random*. Combined with the train-dev-vs-holdout divergence, this is
strong evidence the LoRA adapter learned to detect stylistic/structural
artifacts specific to the two synthetic-generation prompts
(`generate_pos.md` vs `generate_neg.md`) rather than the intended
role/stage/side/geo/prefs matching semantics. `data/synthetic/strategy.md`
already worried about this exact failure mode ("teaching old-model blind
spots" / mode collapse) — this run is a concrete instance of it materializing
as generation-artifact overfitting instead.

Caveats on this specific holdout run, for completeness: it used the
epoch-5 checkpoint, not the true best-by-train-dev-AUC epoch 3, due to the
still-unfixed `select_best_checkpoint` bug (`docs/possible-bugs.md` #2) —
though train-dev AUC is now shown to be an unreliable selection signal
regardless, so this likely doesn't change the conclusion. `max_seq_length`
was 4096 vs. the baselines' 8192 (uncontrolled, but truncation-stats showed
only ~1.2% of texts exceed 4096, so unlikely to explain a 0.4-AUC-point
gap). Several intent slices (`customers` n=1, `hiring` n=3,
`partnerships` n=1) are too small to be meaningful.

## Next steps

- **Root-cause the shortcut-learning failure** before generating more
  synthetic data at the current settings — likely candidates: strip overt
  generator meta-commentary from synthetic profiles (9/240 synthetic
  negatives contain explicit give-away phrases like "critical distinction,"
  "has never," "mistaken for" — grep `data/dataset_negative.json` for the
  full list), and/or add a style-normalization pass so synthetic pos/neg
  text doesn't carry systematically different structural markers from the
  two generation prompts.
- Fix `select_best_checkpoint` (`docs/possible-bugs.md` #2) regardless —
  needed for any future run to be trustworthy, even though it likely wasn't
  decisive here.
- Rerun each frozen baseline restricted to the same 69-pair holdout at
  matched `max_seq_length` for a fully controlled version of the table
  above (current holdout table already uses the right population on the
  twotower side, but the baseline rows are still the old 200-pair/8192
  cached numbers).
- Extend `scripts/export_baseline_results.py`'s hardcoded `BASELINES` tuple
  to also read twotower's `metrics_holdout.json`.

**Bottom line: `run_001` does not clear the "beat voyage-4-large" bar from
`docs/two-tower-fine-tune-plan.md`'s decision gate.** Per that gate's own
rule ("if lift shows on train but not holdout, stop and diagnose — don't
scale data yet"), the next step is diagnosing the synthetic-data shortcut
problem, not generating a bigger batch or launching `run_002` with more
epochs.
