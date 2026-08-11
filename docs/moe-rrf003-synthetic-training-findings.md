# Training the MoE re-ranker on rrf_003's 2,619 synthetic pairs

**Status: run, and the answer is no.** Training on the synthetic batch does not
help on real pairs. Every synth-trained arm lands **below the no-model TF-IDF
floor**, and two land below chance. The real 69-pair holdout was **not spent** —
nothing came close to earning it.

Isolated package: `moe_rrf/`. Nothing in `moe_reranker/` or `baselines/` was
modified; `moe_reranker.model` and `moe_reranker.diagnostics` are imported
unchanged.

## Why this experiment existed

The previous experiment ([`docs/moe-reranker-experiment.md`](moe-reranker-experiment.md))
concluded the MMoE was untestable on 111 real training pairs — the bottleneck
looked like **data, not architecture**. `rrf_003` offers 2,619 judge-labeled pairs
(24×) and 2,773 within-seeker triplets (146×). This tests that conclusion directly.

## The evaluation design

**When training uses only synthetic pairs, every real pair outside the frozen
holdout becomes a legitimate test set.** That gives **131 real pairs** (71 pos / 60
neg) instead of 69 — a standard error of **±0.0505** instead of ±0.0709, i.e. 29%
tighter — while leaving the one-shot holdout untouched. Arms that train on real
pairs can't use that trick and fall back to seeker-disjoint 5-fold CV over the same
131.

Populations are asserted disjoint: 923 `rrf_003` contact ids against **297 real
contact ids** (129 seekers, 178 candidates, across the 200 real pairs), **zero
overlap** — every synthetic id is `cmsynth*`-prefixed.

> **Correction (2026-07-30).** This line previously read "1,217 real". That figure
> counted contact ids across all 660 rows of `data/dataset_*.json`, which includes
> the 460 promoted `batch_500_001` pairs — those are synthetic, not real, and are
> now quarantined (`data/archive/batch_500_001_quarantined/README.md`). The real
> population is 297 contacts. The disjointness conclusion is unaffected, since it
> rests on the `cmsynth*` prefix, and no experiment result changes: `moe_rrf` loads
> via `include_synth=False` and never saw those pairs.

Features are **12 text-only scalars** — the embedding channel had to be dropped,
because `rrf_003`'s cached vectors are Qwen3-Embedding-8B while the real-pair
features are voyage-nano, and mixing spaces would make a feature mean different
things per row. TF-IDF fitting follows one rule throughout: **fit on whatever
population the model trains on**, never on the population it is evaluated against.

## Results — all AUCs on real pairs

| arm | real AUC | ±fold std | n_train | trained on |
|---|---|---|---|---|
| `tfidf_realfit` | 0.5660 | — | 0 | nothing (vocabulary from real text) |
| `tfidf_synthfit` | 0.5631 | — | 0 | nothing (vocabulary from rrf_003) |
| **`logistic_real`** | **0.6398** | 0.177 | 105 | real, seeker-disjoint CV |
| `moe_real` | 0.3934 | 0.158 | 105 | real, seeker-disjoint CV |
| `logistic_synth` | 0.4270 | — | 2,619 | rrf_003, pairwise |
| `moe_synth` | 0.4066 | — | 2,619 | rrf_003, pairwise |
| `moe_synth_ws` | 0.5380 | — | 2,773 | rrf_003, **within-seeker triplets** |
| `moe_transfer` | 0.4730 | 0.145 | 105 | synth pretrain → real fine-tune, CV |

Context numbers that decide how to read the table:

| | value |
|---|---|
| Synthetic-internal AUC (did they learn the labels?) | 0.6086 – 0.6637 |
| **The judge teacher's own AUC on the same 131 real pairs** | **0.5797** |
| Judge–human agreement (accuracy) | 0.5878 |
| Synthetic label balance | 44.9% pos |
| Real label balance | 54.2% pos |

## Findings

**1. Training on rrf_003 does not help. Not marginally — it loses to using no
model at all.** The best synth-trained arm (`moe_synth_ws`, 0.5380) sits below the
TF-IDF floor (0.5660), and that floor requires no training whatsoever.

**2. Two synth-trained arms are below chance** (`moe_synth` 0.4066,
`logistic_synth` 0.4270). Below chance is not "no signal" — it means the
feature→label relationship learned from synthetic pairs is *anti-correlated* with
real human outcomes. Training on this data actively inverts the decision.

**3. The failure is not vocabulary shift.** `tfidf_synthfit` (0.5631) is within
0.003 of `tfidf_realfit` (0.5660). A vocabulary learned entirely from synthetic
profiles scores real pairs essentially as well as one learned from real profiles.
**The words transfer; the labels don't.** This rules out the most obvious
explanation and points squarely at the labeling function.

**4. It is a transfer failure, not a training failure.** The models genuinely learn
the synthetic labels — 0.6086 to 0.6637 synthetic-internal AUC. Fitting works. What
does not survive is the move to real pairs.

**5. The distillation chain loses signal at every hop, and the ceiling is low.**
The judge teacher itself only reaches **0.5797** on these 131 real pairs, so even a
*perfect* imitator would land below `logistic_real`'s 0.6398. The students come in
at 0.4066–0.5380 — **below their own teacher's ceiling**. The reason: the model
learns the judge's decision function *as expressed on synthetic profiles*, and that
is not the same function the judge applies to real profiles. Distilling on
synthetic data does not even reproduce the judge's real-pair behaviour.

**6. Within-seeker training is the one thing that worked exactly as predicted.**
Switching from pairwise to within-seeker triplets moved the MoE from 0.4066 to
**0.5380 (+0.131)** — the single largest effect in the experiment. Cancelling the
per-seeker base rate by construction is real and valuable, and the 2,773 triplets
are what made it possible. But it only rescues the model to roughly chance; it does
not lift it above the floor.

**7. Synthetic pretraining then real fine-tuning is worse than skipping the
synthetic step.** `moe_transfer` (0.4730) loses to `logistic_real` (0.6398) by a
wide margin. The pretrained initialization is not a useful prior — it is a
liability the fine-tune cannot fully undo.

**8. Plain logistic regression on ~105 real pairs is still the best model in the
table** (0.6398), beating both TF-IDF alone and every MoE variant. That is the
third independent time in this project that the MoE machinery has failed to pay for
itself.

**9. The earlier "the bottleneck is data, not architecture" conclusion is now
refuted, with a correction.** It is not raw data volume. 24× more pairs made things
*worse*. The bottleneck is **label validity** — specifically that a judge scoring
0.5797 against human outcomes cannot teach a model to beat 0.5660 lexical cosine.

## Honest caveats

- **The CV arms are extremely noisy**: fold-to-fold standard deviations of
  0.145–0.177 on 5 folds over 131 pairs. `logistic_real`'s 0.6398 and `moe_real`'s
  0.3934 differ by more than that spread, but neither is a precise estimate.
- The single-shot 131-pool arms have **SE ±0.0505**, so `moe_synth_ws`'s 0.5380 and
  the 0.5660 floor are not cleanly separable — the conclusion "does not beat the
  floor" is safe, "is worse than the floor" is not.
- **These labels were never going to be ground truth**, and the experiment was not
  designed to prove they were. The value is in quantifying by how much they fail
  and locating *where* — which turned out to be the labeling function, not the
  profile text.
- The dropped embedding channel matters. `moe_real` here (0.3934, text-only
  features) is well below the earlier experiment's 0.5536 (with the nano channel),
  so the feature set is weaker across the board. That affects the absolute levels
  but not the comparison, since every arm shares it.

## Two bugs found and fixed mid-run

Both would have produced confidently wrong conclusions.

**A broken TF-IDF reimplementation.** `moe_rrf/features.py` first rolled its own
`TfidfVectorizer` with `ngram_range=(1,1)` and `sublinear_tf=True`, against the
repo encoder's `(1,2)` and no sublinear scaling. On the 131 real pairs that scored
**0.4366 versus the repo encoder's 0.5660**, correlation only 0.52 — the lexical
feature, which is the strongest single channel in the whole feature set, was
simply wrong. Fixed by using `baselines.tfidf.encode.TfidfEncoder` unchanged, which
also keeps these numbers comparable to `docs/baseline-results-holdout.md`. Verified
by reproducing the documented holdout figure (0.5905 vs documented 0.5922).

**A stale-cache collision that silently merged two arms.**
`TfidfEncoder.encode()` keys its disk cache on `(texts, max_features,
ngram_range)` — **not on the fitted vocabulary.** This experiment deliberately
encodes the *same* real rows under *different* fits (real vocabulary vs synthetic
vocabulary), which collide on that key, so the second call returned the first
call's vectors. The symptom was `tfidf_realfit` and `tfidf_synthfit` reporting
**byte-identical 0.5660** — a coincidence too exact to be real. Fixed by calling
`vectorizer.transform` directly, bypassing the cache. CLAUDE.md flags this hazard
for `cache_name`; it applies to the content-hashed default too, which is worth
knowing for any future experiment that re-encodes identical text under a new fit.

## What this means for the plan

The distillation path recommended in the earlier review is **substantially
weakened**. Its premise was that judge labels could be generated at scale to escape
the 111-pair limit. That is true of the labels' *quantity* and false of their
*usefulness*: the teacher's own 0.5797 on real pairs is the hard ceiling, and it
sits below what plain logistic regression already achieves on 105 real pairs.

What is worth carrying forward:

- **Within-seeker training is validated** and should be kept in any future
  architecture. It was worth +0.131.
- **The judge needs to get better before more of its labels help.** More pairs at
  0.5797 teacher quality is not the lever. `structured_cot` already failed to
  improve it; a different model or a genuinely different framing would be needed.
- **The 200 real pairs remain the only trustworthy signal**, and simple models on
  them remain the strongest thing measured.

## Reproduce

```bash
# the frozen read-only copy (already imported; --verify re-checks the source hash)
PYTHONPATH=. .venv/bin/python -m moe_reranker.import_rrf --batch-id rrf_003 --verify

# the experiment
PYTHONPATH=. .venv/bin/python -m moe_rrf.experiment --run-id rrf003_001
```

Raw output: `artifacts/moe_rrf/rrf003_001/result.json`.
