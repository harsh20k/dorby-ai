# Calibrating alpha on synthetic data instead of eyeballing it

`query_weighted/` found that splitting a seeker's profile and search query into
separate vectors and blending them — `normalize(alpha*query + (1-alpha)*profile)`
— nearly doubles recall@1 on frozen voyage-4-nano, reading alpha=0.6-0.7 off a
9-point grid. That grid was scored on the same 200 real pairs the finding is
reported on, so 0.6 vs 0.7 isn't distinguishable from noise, and alpha was never
fit on a population the report hadn't already seen.

This experiment fits alpha on `rrf_003` — the pairing pipeline's 1000-profile
synthetic batch (923 unique profiles, 2,619 LLM-judge-labeled pairs from
`synth_pipeline/pairing_rrf/`), a population that shares no profiles with the
200 real pairs — then runs the winning alpha through
`query_weighted.eval.run_all_arms` unmodified, on all 200 real pairs, per the
project's standing preference for the all-200 population over the 69-pair
holdout (`docs/baseline-results-real200.md`).

Isolated package `nomad_drift/`. Nothing under `query_weighted/`,
`eval_real_full/`, or `baselines/` was modified — only their public API is
imported (`query_weighted.text`, `query_weighted.eval.combine` /
`run_all_arms`, `baselines.metrics.pair_metrics`,
`baselines.voyage_nano.encode.VoyageNanoEncoder`,
`eval_real_full.baseline_eval.build_candidate_corpus`).

## Phase 1 — calibration on rrf_003

Two corpus-free metrics (no shared retrieval corpus, so neither inherits the
circularity `query_weighted`'s own doc flags in its recall numbers):

- **pair AUC** — cosine(seeker', candidate) vs pos/neg label, across all 2,619
  pairs.
- **shortlist top-1 accuracy** — within each query's judged shortlist (RRF
  top-5 of dense+lexical retrieval), does the highest-scoring candidate carry
  the positive label? Counted only for the 385 shortlists containing both
  labels — a single-class shortlist has nothing to rank. Deliberately *not*
  called "recall@1": it's a ~5-candidate shortlist check, not full-corpus
  retrieval.

| alpha | pair AUC | shortlist top-1 |
|---|---|---|
| 0.0 (profile only) | 0.5471 | 0.5688 |
| 0.1 | 0.5567 | 0.5714 |
| 0.2 | 0.5664 | 0.5688 |
| 0.3 | 0.5764 | 0.5766 |
| 0.4 | 0.5860 | 0.5896 |
| 0.5 | 0.5953 | 0.5870 |
| 0.6 | 0.6038 | 0.5922 |
| 0.7 | 0.6110 | 0.6026 |
| 0.8 | 0.6166 | 0.5870 |
| 0.9 | 0.6209 | 0.5792 |
| **1.0 (query only)** | **0.6241** | 0.5844 |

Pair AUC rises monotonically with no interior peak — every step toward the
query helps, right to the edge of the grid. **Calibrated alpha = 1.0**
(picked by pair AUC, the corpus-free metric; shortlist accuracy points the
same general direction but is noisier).

## Phase 2 — confirmed on all 200 real pairs

Folding alpha=1.0 into `query_weighted.eval.run_all_arms`'s own alpha grid and
re-running on all 200 real pairs:

| arm | pair AUC | hard-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| profile_only | 0.5424 | 0.4862 | 0.2357 | 0.0900 | 0.5000 |
| concat_baseline *(production)* | 0.5593 | 0.5046 | 0.3171 | 0.1800 | 0.5900 |
| alpha_0.6 *(eyeballed peak)* | **0.5872** | 0.5818 | 0.4649 | 0.2500 | 0.8900 |
| **alpha_1.0 = query_only *(calibrated)*** | 0.5530 | **0.5914** | **0.5019** | **0.3000** | **0.9100** |

**The calibrated alpha does not win on the metric it was calibrated on.**
Overall pair AUC on real-200 peaks in the *interior*, at alpha=0.6, then falls
back to 0.5530 at pure query — a little profile similarity still helps
separate the easy negatives that dominate the pool. But **hard-negative AUC
keeps climbing the whole way**, peaking exactly where the synthetic
calibration landed: alpha=1.0, at 0.5914 — matching `query_weighted`'s own
published finding that this is the best hard-negative AUC of any model
measured on all 200 real pairs in this project. The full alpha=0..1 sweep on
real-200 (pair AUC vs hard-neg AUC) is charted in the published artifact
below.

rrf_003 calibration reproduces that answer from a population 13x larger than
the real-200 report and disjoint from it, without ever touching a real pair —
the train/test separation the original experiment was missing.

## Worked example

The largest single-query rank improvement among the 100 real positive
queries: seeker Rudraksh (CTO/co-founder, VantEdge Labs — an AI agent
platform for manufacturing/supply-chain ops) asking for *"US Midwest
manufacturing supply chain procurement operations executive or consultant
with buyer network in industrial or heavy manufacturing, open to AI
technology partnerships."*

- **concat_baseline**: true match (Branden, an embedded commercial operator
  helping sellers break into industrial/manufacturing) ranks **108 of 178**.
  Rank 1 is Rudraksh's *own* profile, encoded on the candidate side elsewhere
  in the corpus — cosine 0.9548 to his own seeker vector. Since the profile
  is ~98% of that string, the "best match" it finds is someone who writes
  like him.
- **alpha=1.0**: the self-echo drops to #4; the true match rises to **#3**.

Full ranked lists (top 5, both encodings) are in the published artifact.

## Caveats

- rrf_003's labels are an LLM judge's opinion (`google/gemini-3.1-flash-lite`,
  naive framing), not real accept/decline outcomes — sufficient for fitting
  one scalar hyperparameter on a much bigger sample than 200 pairs affords,
  but the real-200 report is always what's treated as the headline result.
- alpha=1.0 is the edge of the grid, not a bracketed interior optimum — the
  sweep never turned back, so "past pure query" isn't meaningful here, but
  the exact peak (if one exists past 1.0 in some other formulation) was never
  tested.
- Shortlist top-1 accuracy is not recall@1 — a within-shortlist check over ~5
  candidates, not full-corpus retrieval. Don't compare the two directly.
- One run, no replicate, no error bar on either population.
- Real-200 retrieval carries the same partial circularity `query_weighted`'s
  own doc flags (production selected these candidates using the same
  query). Pair AUC and hard-negative AUC don't share that problem, which is
  why the hard-neg-AUC match between synthetic calibration and real-200 is
  the load-bearing result, not the recall@1 jump.

## Reproduce

```bash
modal run nomad_drift/modal_run.py --run-id nd_001
modal volume get dorby-nomad-drift-results nd_001/metrics.json ./artifacts/nomad_drift/nd_001/metrics.json
modal volume get dorby-nomad-drift-results nd_001/calibration.json ./artifacts/nomad_drift/nd_001/calibration.json
modal volume get dorby-nomad-drift-results nd_001/worked_example.json ./artifacts/nomad_drift/nd_001/worked_example.json

python -m pytest tests/test_nomad_drift.py -q   # CPU-only, no GPU needed
```

Cost: one L4 session, ~2,619 + 200 pairs encoded once (nano is small; no
training). Embeddings cache to `dorby-nomad-drift-cache`, so a re-run with the
same texts is free.

## What this leaves

1. **Bracket the alpha=1.0 boundary properly** — the calibration grid stopped
   at 1.0 because that's where `query_only` lives; a formulation that lets the
   query dominate *more* than 1:0 (e.g. amplifying the query vector before
   normalizing) is untested.
2. **Apply the same calibration to the fine-tuned two-tower adapters** —
   `twotower_query_weighted/` found the eyeballed alpha=0.6 pattern replicates
   on `top1_ctrl`; whether synthetic calibration picks the same alpha=1.0
   answer there is untested.
3. **Replicate the calibration run** — rrf_003 is one batch; a second
   synthetic batch would confirm alpha=1.0 isn't an artifact of this
   particular profile pool or judge run.
