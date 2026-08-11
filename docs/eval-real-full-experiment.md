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

**Arm A column is `abl_a_batch_only_v2` alone, not a mean.** Its two runs are not
replicates — v1 shipped an epoch-5 model via the `save_total_limit=3` checkpoint
bug while v2 correctly selected epoch 4, so averaging them mixes two different
models (see `docs/twotower-rrf-triplet-ablation-experiment.md`, "Arm A has no
replicate"). The "noise" column is therefore borrowed from arms B and C, the two
genuine replicate pairs in that experiment (pair AUC 0.0017 and 0.0164); Arm A
has no error estimate of its own.

### All 200 real pairs (corpus 178 candidates)

| metric | frozen | Arm A | Δ | noise | verdict |
|---|---|---|---|---|---|
| pair AUC | 0.5593 | 0.5594 | **+0.0001** | 0.0164 | **exactly nothing** |
| hard-neg AUC | 0.5046 | 0.5558 | **+0.0512** | 0.0164 | real |
| MRR | 0.3171 | 0.3341 | +0.0170 | 0.0164 | marginal |
| recall@1 | 0.1800 | 0.1800 | **+0.0000** | 0.0345 | **exactly nothing** |
| recall@10 | 0.5900 | 0.6400 | +0.0500 | 0.0345 | real |

### The 131 train-split real pairs (corpus 120) — also unseen by Arm A

| metric | frozen | Arm A | Δ | noise | verdict |
|---|---|---|---|---|---|
| pair AUC | 0.5507 | 0.5392 | **−0.0115** | 0.0164 | nothing (slightly negative) |
| hard-neg AUC | 0.4690 | 0.5315 | **+0.0625** | 0.0164 | real |
| MRR | 0.3548 | 0.3773 | +0.0225 | 0.0164 | marginal |
| recall@1 | 0.2113 | 0.2113 | **+0.0000** | 0.0345 | exactly nothing |
| recall@10 | 0.6479 | 0.7042 | +0.0563 | 0.0345 | real |

### The 69-pair holdout (corpus 65) — for reference

| metric | frozen | Arm A | Δ | noise |
|---|---|---|---|---|
| pair AUC | 0.5793 | 0.5983 | +0.0190 | 0.0164 |
| hard-neg AUC | 0.5707 | 0.6034 | +0.0328 | 0.0164 |
| MRR | 0.4610 | 0.5326 | +0.0716 | 0.0164 |
| recall@1 | 0.2759 | 0.3793 | +0.1034 | 0.0345 |
| recall@10 | 0.7586 | 0.8621 | +0.1034 | 0.0345 |

## Verdict

**1. The holdout-69 gains do not generalise.** On the other 131 real pairs —
equally out-of-sample for Arm A — pair AUC is **−0.012** and recall@1 is
**exactly 0.0000**. Pooled over all 200, pair AUC is **+0.0001** and recall@1 is
**+0.0000** — not "small", not "inside noise", but zero to four decimal places.
The holdout's "+0.019 pair AUC / +3 queries at rank 1" is a property of that
particular 69-pair sample, not of the model.

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


## Follow-up: every model on all 200 pairs

Later runs extended this evaluation to Voyage-4-large and the Qwen3-8B family
(see `docs/twotower-qwen-bigbatch-experiment.md`). Voyage-4-large goes through
`eval_real_full/voyage_large_eval.py`, a shim exposing `encode_query` /
`encode_document` over `baselines.voyage_large.encode` so it uses the identical
`evaluate_pairs` path; its holdout row reproduces the published baseline exactly
(0.6086 / 0.5287 / 0.3448) and the whole run was free (1,163 cache hits, 0 API
calls).

| model (all 200, corpus 178) | pair AUC | hard-neg | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| qwen micro-6 | **0.5947** | 0.5608 | 0.3031 | 0.1400 | 0.6600 |
| voyage-4-large | 0.5726 | 0.5422 | 0.3102 | 0.1300 | **0.7000** |
| nano Arm A (v2) | 0.5594 | **0.5558** | 0.3341 | **0.1800** | 0.6400 |
| qwen micro-1 | 0.5604 | 0.4828 | 0.2734 | 0.1400 | 0.5600 |
| nano frozen | 0.5593 | 0.5046 | 0.3171 | 0.1800 | 0.5900 |
| qwen frozen 4096 | 0.5420 | 0.4572 | 0.2031 | 0.0400 | 0.5800 |

**voyage-4-nano beats voyage-4-large at top-1 retrieval on real data**, on all
three populations: all-200 +0.050 (18.0 vs 13.0 of 100 queries), train-131
+0.056, holdout +0.034 for Arm A. The credit is **entirely** nano's, not the
fine-tune's — frozen nano scores 0.1800 on all 200 and Arm A v2 scores 0.1800,
an identical value.

## Caveats

- **Retrieval metrics are not comparable across subsets.** `evaluate_pairs`
  builds its corpus from the pairs it is handed, so `all` ranks against 178
  candidates and `holdout` against 65. A bigger pool is strictly harder; a lower
  MRR on `all` means the pool grew, not that the model degraded. Pair AUC has no
  corpus dependence and is comparable. Each subset's `n_candidates` is recorded
  in the output.
- The noise figures are borrowed from arms B and C of the ablation (the only genuine replicate pairs); they bound run-to-run variation, not sampling error, and Arm A has no replicate of its own.
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
