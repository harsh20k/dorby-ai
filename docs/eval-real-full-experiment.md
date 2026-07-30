# Frozen nano vs. the Arm A adapter, on all 200 real pairs

Every two-tower number published in this repo so far is measured on the frozen
**69-pair** holdout — 29 positive queries, which forced the ablation to report a
recall@1 noise floor of exactly 1/29 = 0.0345. Conclusions at that resolution
are fragile. This experiment re-measures the same two models on **all 200 real
pairs**, roughly tripling the sample.

Isolated package `eval_real_full/`. Nothing under `twotower/`,
`twotower_rrf_triplet*/`, or `baselines/` was modified; scoring goes through
`twotower.eval.evaluate_pairs` → `baselines.metrics` unchanged.

## Why this evaluation is legitimate — and why it is not general

Arm A trained on `exports/rrf_datasets/rrf_003`: 583 rows built entirely from
synthetic profiles. It saw **zero real pairs and zero real profiles**. Verified
before building the package:

* `rrf_003`'s profiles were generated with the v3+ profile-gen prompt, from
  which the `{ref_example_1}` real-profile seeding was removed in v2 — no real
  profile text conditioned generation.
* None of the 297 real contact ids appears anywhere in the `rrf_003` manifest.

So the 131 "train" pairs are unseen by Arm A in exactly the same sense as the 69
holdout pairs. **This does not hold for `twotower` runs that trained on real
pairs** (`run_001`, `arm_a_real_only`). `eval_real_full/guard.py` refuses any
adapter whose `run_meta.json` does not record an all-synthetic `rows_path`, and
is verified to reject both of those runs.

## Validation: the pipeline reproduces what it should

Before reading anything new, the holdout column must reproduce numbers already
on record. It does, exactly:

| holdout-69 | pair AUC | hard-neg | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| frozen nano, **this run** | 0.5793 | 0.5707 | 0.4610 | 0.2759 | 0.7586 |
| frozen nano, `artifacts/voyage_nano_holdout` | 0.5793 | 0.5707 | 0.4610 | 0.2759 | 0.7586 |
| Arm A v2, **this run** | 0.5983 | 0.6034 | 0.5326 | 0.3793 | 0.8621 |
| Arm A v2, ablation doc | 0.5983 | 0.6034 | 0.5326 | 0.3793 | 0.8621 |

Arm A v1 reproduces to within 0.0017 on pair AUC (0.6129 here vs 0.6112), which
is float/batching-level. The frozen row is a particularly strong check: it comes
from a completely different code path (`baselines/voyage_nano/encode.py`) and
lands digit-for-digit.

## Results

Arm A column is the mean of its two replicates; "noise" is the absolute
difference between them, the only error estimate available.

### All 200 real pairs (corpus 178 candidates)

| metric | frozen | Arm A | Δ | noise | verdict |
|---|---|---|---|---|---|
| pair AUC | 0.5593 | 0.5664 | +0.0071 | 0.0140 | **inside noise** |
| hard-neg AUC | 0.5046 | 0.5638 | **+0.0592** | 0.0160 | real |
| MRR | 0.3171 | 0.3391 | +0.0220 | 0.0099 | real, small |
| recall@1 | 0.1800 | 0.1850 | +0.0050 | 0.0100 | **inside noise** |
| recall@10 | 0.5900 | 0.6350 | +0.0450 | 0.0100 | real |

### The 131 train-split real pairs (corpus 120) — also unseen by Arm A

| metric | frozen | Arm A | Δ | noise | verdict |
|---|---|---|---|---|---|
| pair AUC | 0.5507 | 0.5475 | **−0.0032** | 0.0167 | nothing |
| hard-neg AUC | 0.4690 | 0.5411 | **+0.0721** | 0.0192 | real |
| MRR | 0.3548 | 0.3863 | +0.0315 | 0.0181 | marginal |
| recall@1 | 0.2113 | 0.2324 | +0.0211 | 0.0423 | inside noise |
| recall@10 | 0.6479 | 0.7254 | +0.0775 | 0.0423 | real |

### The 69-pair holdout (corpus 65) — for reference

| metric | frozen | Arm A | Δ | noise |
|---|---|---|---|---|
| pair AUC | 0.5793 | 0.6056 | +0.0263 | 0.0147 |
| hard-neg AUC | 0.5707 | 0.6103 | +0.0397 | 0.0138 |
| MRR | 0.4610 | 0.5422 | +0.0811 | 0.0192 |
| recall@1 | 0.2759 | 0.3966 | +0.1207 | 0.0345 |
| recall@10 | 0.7586 | 0.8621 | +0.1034 | 0.0000 |

## Verdict

**1. The holdout-69 gains do not generalise.** On the other 131 real pairs —
equally out-of-sample for Arm A — pair AUC is **−0.003** and recall@1 sits
inside noise. Pooled over all 200, pair AUC is +0.007 against a ±0.014 noise
floor and recall@1 is +0.005 against ±0.010. The "+0.026 pair AUC / +3.5 queries
at rank 1" in `docs/twotower-rrf-triplet-ablation-experiment.md` is a property of
that particular 69-pair sample, not of the model.

**2. The 69-pair holdout is an easier population.** Frozen nano scores 0.5793
there versus 0.5507 on train and 0.5593 pooled — before any fine-tuning is
involved. Any single-number comparison against it inherits that bias.

**3. What genuinely survives is hard-negative discrimination.** +0.040
(holdout), +0.059 (all 200), +0.072 (train 131). Every one clears its noise
band, the sign is consistent across three independent populations, and the
effect is *larger* on the bigger samples — the opposite of what a
sampling-artefact would do. Frozen nano is **below chance on the train subset's
hard negatives (0.4690)**; the adapter lifts it to 0.5411.

This is the metric that matters most here. Per `docs/objective.md` there is no
easy-negative population in production — every real negative is an intro
production already recommended and a human declined.

**4. Recall@10 also holds up** (+0.045 to +0.078 across subsets), while
recall@1 does not. The adapter is better at getting the right person into a
shortlist, not at putting them first.

**Revised summary:** the fine-tune does not meaningfully improve pair
classification or top-1 retrieval on real data. It reliably improves
hard-negative discrimination and top-10 recall. Narrower than the holdout
suggested, but far better evidenced.

## Caveats

- **Retrieval metrics are not comparable across subsets.** `evaluate_pairs`
  builds its corpus from the pairs it is handed, so `all` ranks against 178
  candidates and `holdout` against 65. A bigger pool is strictly harder; a lower
  MRR on `all` means the pool grew, not that the model degraded. Pair AUC has no
  corpus dependence and is comparable. Each subset's `n_candidates` is recorded
  in the output.
- The noise estimate is two replicates of one arm, not a proper interval.
- Frozen nano's row here is not identical in provenance to the published
  `baselines/` table (different encode path), though on the shared holdout
  population the two agree to four decimals.
- `data/` is gitignored real profile data, so `freeze.py` records pair ids and
  per-pair SHA-256 digests rather than copying content. `--verify` proves the
  source has not drifted; the loader fails loudly if it has.

## Reproduce

```bash
python -m eval_real_full.freeze --data-dir data            # write manifest
python -m eval_real_full.freeze --data-dir data --verify   # check for drift

modal run --detach eval_real_full/modal_eval.py --run-id real200_001
modal volume get dorby-eval-real-full-results real200_001/<config>/metrics.json \
    ./artifacts/eval_real_full/real200_001/<config>/metrics.json
```

Frozen manifest: `eval_real_full/data_frozen/real_200_manifest.json`,
combined hash `732bc16de90923a8` (200 pairs — 100 pos / 100 neg, 131 train /
69 holdout).

**Hash note.** `real200_001`'s `metrics.json` records `real_data_hash =
67b7cdf9528de7df`, from the manifest layout in use when the run executed. The
manifest was then changed to key pairs by an opaque `pair_key` —
`sha256(pair_id)[:16]` — rather than the raw contact ids, because `data/` is
gitignored precisely to keep real Boardy identifiers out of the repository and
no tracked file contained one. That reordered the digest list and so changed the
combined hash. **The 200 pairs and their per-pair content digests are
identical**; only the sort key differs. `--verify` against the current manifest
passes.

## Next steps

1. **Re-read every prior two-tower conclusion against this.** Anything claimed
   on 69 pairs alone — including "Arm A is level with Voyage-4-large" — needs
   the same 200-pair treatment before it is trusted.
2. **Score the frozen baselines on all 200 too.** Voyage-4-large and
   Qwen3-Embedding-8B are currently ranked on the same easier 69-pair sample.
3. **Optimise for hard-negative AUC**, since that is the only thing fine-tuning
   reliably moves and the only negative population production has.
