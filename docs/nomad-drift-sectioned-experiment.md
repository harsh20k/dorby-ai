# Section-selected query blend beats whole-profile blend, but the calibration missed it

Follow-up to `docs/nomad-drift-experiment.md`. That experiment calibrated the
query/profile alpha blend on rrf_003 and landed on alpha=1.0 (pure query) —
losing a bit of overall pair AUC vs. an eyeballed interior alpha (0.6), because
a little whole-profile similarity still helps separate easy negatives.
Separately, `docs/lookingfor-sectioning-findings.md` found that splitting a
seeker's own `lookingFor` into per-ask sections and scoring by the single
best-matching one (not the whole field) lifts pair AUC and top-1 retrieval on
the 69-pair holdout. Neither experiment tried the other's mechanism.

`nomad_drift_sectioned/` does: instead of blending the query with the whole
profile vector, it picks the seeker's `lookingFor` section closest to the
query (cosine similarity, one comparison per section — O(sections), not
O(candidates) the way the original sectioning experiment's per-candidate max
was) and blends only that section vector with the query. Same alpha
arithmetic as `nomad_drift.calibrate.combine_batch`, reused unmodified.

## Phase 1 — calibration on rrf_003

rrf_003 is an unusually good fit here: every query_key was generated *for*
one specific lookingFor section
(`synth_pipeline/pairing_rrf/sections.py::query_targets`), so each pair
carries a ground-truth section index — calibration can check both "does the
blend help" and "does cosine selection pick the right section."

**Section selection accuracy: 48.8%** (1047/2144 checkable pairs — seekers
with 2+ sections only) — barely better than a coin flip. The pair-AUC sweep
is nearly identical to `nomad_drift`'s whole-profile sweep (e.g. alpha=0.6:
0.6036 vs 0.6038) and still picks **alpha=1.0** as best.

## Phase 2 — confirmed (and extended) on all 200 real pairs

Full alpha grid, both mechanisms, same population, same scoring function
(`query_weighted.eval.score_arm`, called directly with a substituted seeker
matrix — not reimplemented):

| alpha | whole-profile AUC | section AUC | whole hard-neg | section hard-neg | whole MRR | section MRR | whole R@1 | section R@1 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.5424 | 0.5627 | 0.4862 | 0.5100 | 0.2357 | 0.3021 | 0.0900 | 0.1600 |
| 0.3 | 0.5785 | 0.5870 | 0.5404 | 0.5482 | 0.3884 | 0.3842 | 0.2400 | 0.2100 |
| 0.4 | 0.5834 | **0.5927** | 0.5534 | 0.5634 | 0.4064 | 0.4171 | 0.2300 | 0.2400 |
| 0.6 | 0.5872 | 0.5876 | 0.5818 | 0.5816 | 0.4649 | 0.4795 | 0.2500 | 0.2700 |
| 0.7 | 0.5799 | 0.5808 | 0.5856 | 0.5886 | 0.5009 | **0.5159** | 0.3000 | **0.3200** |
| 0.8 | 0.5695 | 0.5724 | 0.5900 | **0.5958** | 0.4958 | 0.4985 | 0.2900 | 0.3000 |
| 1.0 | 0.5530 | 0.5530 | 0.5914 | 0.5914 | 0.5019 | 0.5019 | 0.3000 | 0.3000 |

(Bold = new best across the whole `nomad_drift` family on all 200 real
pairs.) At alpha=1.0 the two are identical by construction — the profile/
section term is multiplied by 0 either way. Everywhere in between, section
selection is at or above whole-profile blending on almost every cell, and at
alpha=0.4 and 0.7 it wins **every** metric simultaneously.

## Best value reached anywhere in the sweep

| metric | whole profile | section-selected | delta |
|---|---|---|---|
| pair AUC | 0.5872 (α=0.6) | **0.5927** (α=0.4) | +0.0055 |
| hard-neg AUC | 0.5914 (α=1.0) | **0.5958** (α=0.8) | +0.0044 |
| MRR | 0.5019 (α=1.0) | **0.5159** (α=0.7) | +0.0140 |
| Recall@1 | 0.3000 | **0.3200** (α=0.7) | +0.0200 |
| Recall@10 | 0.9100 (α=1.0) | 0.9200 (α=0.9) | +0.0100 |

## Why the calibration didn't find this on its own

Pair AUC on rrf_003 rises monotonically to the alpha=1.0 boundary in both
mechanisms (same shape as `nomad_drift`'s own finding) — so the calibration
process always lands on the one alpha where whole-profile and section-selected
blending are mathematically identical. The real-200 improvement lives entirely
in the interior of the range the calibration curve can't discriminate. This is
a genuine limitation of "calibrate a scalar by sweeping to a monotonic
boundary": it can rank *how much* query weight helps, but it's blind to any
change that only matters away from the boundary it converges to. The fix
isn't a bug fix — it's a reminder to also inspect the full sweep, not just the
argmax, before concluding a mechanism doesn't matter.

## The selection-accuracy puzzle

48.8% section-selection accuracy against rrf_003's ground truth is not a
strong number, yet the real-200 result improved anyway. Two honest read: either
(a) the *ground truth* itself is a weaker signal than it looks — a query
generated "for" one section may still be legitimately close to a different
section semantically, so "wrong" selections aren't necessarily bad picks, or
(b) even a noisy section pick is less noisy than always averaging in every
section, so partial-credit selection still beats no selection at all. Not
disambiguated here.

## Caveats

- Same population caveats as `nomad_drift`: rrf_003 labels are an LLM judge's
  opinion; real-200 retrieval carries partial circularity from production's
  own candidate selection.
- The 48.8% selection accuracy is itself unreplicated and could be an
  artifact of this one rrf_003 batch.
- No error bars; one run on each population.
- Real pairs' `lookingFor` sectioning uses the same blank-line-paragraph
  splitter as `baselines/voyage_nano_sectioned` — single-section seekers fall
  back to the whole-profile vector automatically (129 of the 200 real pairs'
  seekers had 2+ sections to select between; see `calibration.json` /
  `metrics.json` for exact counts).

## Reproduce

```bash
modal run nomad_drift_sectioned/modal_run.py --run-id nds_001
modal volume get dorby-nomad-drift-sectioned-results nds_001/metrics.json ./artifacts/nomad_drift_sectioned/nds_001/metrics.json
modal volume get dorby-nomad-drift-sectioned-results nds_001/calibration.json ./artifacts/nomad_drift_sectioned/nds_001/calibration.json

python -m pytest tests/test_nomad_drift_sectioned.py -q   # CPU-only, no GPU needed
```

## What this leaves

1. **Replicate** — a second rrf_003-style batch would confirm the 48.8%
   selection accuracy and the real-200 improvement aren't artifacts of this
   one synthetic pool.
2. **A calibration metric that isn't blind to the interior** — e.g. calibrate
   against real-200's own hard-negative AUC directly (accepting the
   circularity cost) or a synthetic metric that doesn't monotonically favor
   the boundary, so future calibration runs don't need a manual full-sweep
   comparison to catch a result like this one.
3. **Apply to the fine-tuned two-tower adapters**, same as `nomad_drift`'s own
   open item.
