# Matrix factorization on text: LSA compression and a low-rank bilinear scorer

**Package:** `bilinear_mf/` (isolated; nothing under `baselines/`, `eval_real_full/`
or `twotower/` was modified)
**Artifacts:** `artifacts/bilinear_mf/{mf_tfidf_001_lsa, mf_voyage_001_lsa, mf_tfidf_002, mf_voyage_002}/results.json`
**Tests:** `tests/test_bilinear_mf.py` (9 passing)
**Results page:** `docs/html/bilinear-mf-results.html`
([published](https://claude.ai/code/artifact/d0b86eb9-4eda-4d03-9c76-9c79201f4ebb),
built by `scripts/build_bilinear_mf_browser.py`)
**Date:** 2026-08-06

---

## The question

Recommender-system matrix factorization factors a `users × items` interaction
matrix into a learned vector per user and per item. That does not transfer here
directly: the interaction matrix is 129 seekers × 178 candidates with 200 filled
cells, and freely-learned per-contact vectors would be unidentifiable at that
density *and* useless for anyone unseen. Every deployable model in this project
has to read text.

So this experiment tests the two places factorization survives that constraint.

**Arm 1 — `lsa`: factor the text.** Truncated SVD of the `documents × terms`
matrix, then cosine in the compressed space. This is classic LSA/LSI, the literal
"text matrix factorization". Label-free, so it can be scored on any population
without leakage.

**Arm 2 — `bilinear`: factor the scoring function.** Keep a frozen encoder's
vectors and learn a low-rank correction on top:

```
score(s, c) = s·c + (A s)·(B c)  =  sᵀ (I + AᵀB) c,     A, B ∈ R^{k×d}
```

This is the *content-based* form of MF: the per-seeker and per-candidate vectors
are computed from text by `A` and `B` rather than looked up, so it generalizes to
unseen contacts. The motivation is that cosine is the special case `W = I` — it
can only reward a candidate for pointing the *same way* as the seeker. The
residual signal here is complementarity, which is a claim about *different*
directions being compatible, and that needs an off-identity `W`.

Both arms are scored through `baselines/metrics.py` unchanged.

## Protocol, and why it is shaped this way

The population is all 200 real pairs (100 positive queries, 178-candidate
corpus), per `docs/baseline-results-real200.md`.

For the trained arm that creates a leakage problem, which
`eval_real_full/guard.py` exists to police: a model trained on the 131 train
pairs cannot be scored on all 200. So the bilinear arm reports:

1. **Inner seeker-disjoint CV on the 131 train pairs only** to select
   `reduce_dim`, `rank` and `weight_decay`. The holdout is never consulted.
2. **One-shot 69-pair holdout** with the selected configuration.
3. **Seeker-disjoint 10-fold CV over all 200**, each pair scored by a model that
   never saw its seeker — the honest all-200 number.
4. **A label-permutation null**: the identical CV protocol re-run 50 times on
   shuffled labels. At 131 training pairs the question is never "is AUC above
   0.5" but "is it above what this pipeline scores on noise". `moe_reranker`
   learned that expensively (fold std 0.067 swamping every effect).

Grouping folds by *seeker*, not by pair, is load-bearing: `rrf_002`'s probes
found seeker identity alone predicts a label at 0.687 AUC, so a random pair split
would let the model memorize "this seeker declines everything" and report it as
matching skill.

## Correctness anchors

With the head disabled, this package reproduces the published frozen-cosine rows
digit-for-digit — asserted in `tests/test_bilinear_mf.py`, not just observed:

| backbone | pair AUC | MRR | R@1 | matches `baseline-results-real200.md` |
|---|---|---|---|---|
| TF-IDF, no reduction | 0.5649 | 0.1313 | 0.05 | ✅ |
| Voyage-4-large, no reduction | 0.5726 | 0.3102 | 0.13 | ✅ |

Both real bugs found while building this were caught by that check, and both were
the kind that produce a *plausible* wrong number:

- **TF-IDF fit-set drift.** Fitting the vectorizer on `seekers + candidates +
  corpus` instead of `seekers + corpus` looks harmless — every candidate text is
  already in the corpus — but IDF is a *document count*, so the duplicates shift
  every weight. All-200 AUC read 0.5572 instead of 0.5649.
- **Stale TF-IDF cache.** `TfidfEncoder`'s content-hashed key covers the texts
  being encoded but *not* the fitted vocabulary, so one cache directory shared
  across two fit sets serves the first fit's vectors to the second — the fix
  above kept "not working" until the cache was namespaced by fit set. This is the
  same defect class already recorded for `synth_pipeline/pairing`.
- **Retrieval measured in the wrong space.** The rank sweep initially reported an
  identical MRR at every `k`, which read as "compression is harmless" but
  actually meant the retrieval metric was ranking in the un-reduced backbone
  space and ignoring the model entirely. Now pinned by
  `test_reduced_space_changes_retrieval`.

---

## Arm 1 results — LSA / SVD compression

All-200 population. `evr` = fraction of variance retained. `train` is the 131-pair
split, shown because it is the only leak-free way to choose `k`.

### Voyage-4-large backbone (d=1024)

| k | evr | train AUC | **all-200 AUC** | **hard-neg AUC** | MRR | R@1 | R@10 |
|---|---|---|---|---|---|---|---|
| none (1024) | — | 0.5582 | 0.5726 | 0.5422 | 0.3102 | 0.13 | 0.70 |
| 16 | 0.443 | 0.5413 | 0.5323 | 0.5228 | 0.1943 | 0.07 | 0.49 |
| **32** | 0.593 | **0.5862** | 0.5819 | 0.5790 | 0.2737 | 0.12 | 0.68 |
| 64 | 0.757 | 0.5803 | 0.5835 | 0.5894 | 0.3003 | 0.14 | 0.68 |
| 128 | 0.906 | 0.5798 | **0.5978** | **0.5902** | **0.3118** | 0.14 | 0.69 |
| 256 | 0.993 | 0.5631 | 0.5786 | 0.5512 | 0.3106 | 0.13 | 0.70 |
| 512 | 1.000 | 0.5582 | 0.5726 | 0.5422 | 0.3102 | 0.13 | 0.70 |

### TF-IDF backbone (d=20000)

| k | evr | train AUC | **all-200 AUC** | **hard-neg AUC** | MRR | R@1 | R@10 |
|---|---|---|---|---|---|---|---|
| none (20000) | — | 0.5526 | 0.5649 | 0.5164 | 0.1313 | 0.05 | 0.26 |
| 16 | 0.157 | 0.5688 | 0.5762 | **0.6474** | 0.0699 | 0.01 | 0.18 |
| 32 | 0.252 | 0.5770 | 0.5755 | 0.6294 | 0.0923 | 0.03 | 0.23 |
| 64 | 0.397 | 0.5904 | 0.5805 | 0.6224 | 0.1066 | 0.03 | 0.24 |
| **128** | 0.625 | **0.5962** | 0.5950 | 0.6160 | 0.1002 | 0.02 | 0.22 |
| 256 | 0.941 | 0.5786 | **0.5987** | 0.5810 | 0.1378 | 0.05 | 0.29 |
| 512 | 1.000 | 0.5526 | 0.5649 | 0.5164 | 0.1313 | 0.05 | 0.26 |

### The one finding here that is robust

**SVD compression improves hard-negative discrimination on both backbones, at
every rank tested.** Voyage 0.5422 → 0.579–0.590 across k=32/64/128; TF-IDF
0.5164 → 0.616–0.647 across k=16/32/64/128. This is not a lucky `k` — it is a
plateau, on both a lexical and a neural backbone, in the same direction.

That matters more than the headline AUC because the hard slice is the only
population that exists in production: every real negative here is an intro
production already judged relevant and a human still declined. The plausible
reading is that the discarded tail dimensions carry most of the surface/topical
matching — which production has already filtered on, so within this population it
is noise — and dropping them raises the share of the remaining signal that is
about compatibility.

### The finding that is *not* yet established

Voyage at k=128 beats production Voyage-4-large on essentially everything at
once — AUC 0.5726 → 0.5978, hard-neg 0.5422 → 0.5902, MRR 0.3102 → 0.3118,
R@1 0.13 → 0.14 — and its 0.5978 would be the **best all-200 pair AUC in this
project** (previous best: twotower Qwen micro-6, 0.5947), for zero training and
zero serving cost.

**Do not report that as a result.** `k=128` was identified by reading the all-200
column. Under the only leak-free selection rule — pick `k` by train-split AUC —
Voyage selects `k=32`, and the honest gain shrinks to **+0.009 AUC, +0.037
hard-neg, −0.037 MRR**. Marginal. TF-IDF selects `k=128` and transfers better
(+0.030 AUC, +0.100 hard-neg, −0.031 MRR).

Note the selector itself is unreliable: train AUC and all-200 AUC rank `k`
differently even though the train pairs are 131 of those same 200. That is the
131-pair sample size talking, and it is the same wall every experiment in this
repo hits.

---

## Arm 2 results — low-rank bilinear scorer

Configuration selected by inner CV on train pairs only; scored by seeker-disjoint
10-fold CV over all 200, against a label-permutation null.

| | **Voyage-4-large** | **TF-IDF** |
|---|---|---|
| selected (d, rank, wd) | 32, 32, 0.001 | 32, 4, 0.001 |
| inner CV AUC (what selection saw) | 0.6218 | 0.7399 |
| **all-200 CV pair AUC** | **0.5410** | **0.6193** |
| cosine in the same space | 0.5730 | 0.5755 |
| Δ vs cosine | **−0.032** | **+0.044** |
| hard-neg AUC | 0.5794 (cosine 0.6492) | 0.6100 (cosine 0.6294) |
| easy-neg AUC | 0.4892 (cosine 0.4908) | 0.5984 (cosine 0.5604) |
| MRR | 0.098 (cosine 0.323) | 0.0585 (cosine 0.0923) |
| R@10 | 0.25 (cosine 0.67) | 0.13 (cosine 0.23) |
| fold AUC std | 0.205 | 0.158 |
| permutation null mean / p95 / max | 0.498 / 0.575 / 0.650 | 0.493 / 0.577 / 0.609 |
| p-value vs null | 0.196 | 0.020 |
| 69-pair holdout AUC | 0.4845 (cosine 0.5802) | 0.6155 (cosine 0.5672) |

**Verdict: the bilinear head does not work here.** Three independent reasons, and
they do not depend on which backbone you prefer.

**1. On the production backbone it actively hurts.** Voyage loses 0.032 all-200
AUC and 0.096 holdout AUC (0.4845 — below chance), while retrieval collapses from
0.323 MRR to 0.098. It does not clear its own null (p = 0.196).

**2. Where it appears to win, it wins on the wrong slice.** TF-IDF's +0.044 AUC
is entirely easy-negative (0.5604 → 0.5984); on hard negatives it goes *backwards*
(0.6294 → 0.6100). Since the easy/hard split is token-overlap-defined and the
hard slice is the only one that exists in production, this is the same trap
`possible-bugs.md` #3 flags for TF-IDF generally. And it clears its null only
narrowly: the observed 0.6193 sits above the null's 50-draw max of 0.6088, but
the gain over cosine (+0.044) is smaller than the null's own standard deviation
(0.058).

**3. Retrieval degrades badly in every configuration.** MRR falls on both
backbones — catastrophically on Voyage (0.323 → 0.098), and by a third on TF-IDF.
The reason is structural: the residual is fit against 200 observed pairs but
applied to a 178-candidate ranking, so it is free to distort the global geometry
in regions no training pair constrains. A pair-classification gain bought with a
retrieval loss is not a win for a retrieval system.

### The most instructive number

Inner CV said **0.7399** on TF-IDF; the honest out-of-fold number was **0.6193**.
On Voyage, 0.6218 → 0.5410. Inner CV overstated by 0.12 and 0.08 respectively,
because it reports the maximum over 64 configurations evaluated on 131 pairs.

That gap is the actual finding of this arm, and it generalizes past this
experiment: **at this data size, hyperparameter selection is itself a source of
overfitting large enough to invent a result.** Sensitivity to `reduce_dim` makes
the point concretely — with the width fixed at 128 the selected head regularized
to *numerically zero* (residual Frobenius norm 7e-08; the model correctly
concluded "keep cosine"), but adding width to the grid produced a nonzero head
that beat cosine on TF-IDF and lost on Voyage. Same code, same data, different
grid, opposite conclusions.

---

## What this says about the broader question

The bilinear arm was the interesting hypothesis — cosine cannot express
asymmetric complementarity, and 4k parameters is a far better capacity match to
131 real pairs than a LoRA adapter. It failed, and it failed for a reason that
was not about capacity: there is no protocol at 131 pairs that can reliably
*select* even a 4k-parameter model. This is the third experiment in this repo to
land there (`moe_reranker`: "not testable at 111 real training pairs";
`twotower_rrf_triplet_ablation`: "dev set too small to select").

The LSA arm is the opposite shape — no labels, no selection, nothing to overfit —
and it is the one that produced something durable: compression robustly improves
hard-negative discrimination on both backbones. It costs nothing at serving time
(a fixed `d×k` projection folded into the query encode; the ANN index gets
*smaller*), so it is compatible with the <100 ms budget in a way none of the
trained variants are.

## Next steps, in priority order

1. **Test the compression finding where it can't be selection.** Apply the
   Voyage `k=128` projection to a population this experiment never touched — the
   `rrf_003` synthetic pairs or a fresh real batch. That is the only way to
   promote "k=128 beats production Voyage on everything" from hypothesis to
   result.
2. **Try compression on the strong open-weight backbones** (Qwen3-Embedding-8B,
   BGE-en-ICL). If SVD lifts hard-neg AUC there too, the effect is a property of
   the task, not of Voyage.
3. **Do not pursue the bilinear head at the current data scale.** If the real
   labeled set grows past ~1k pairs it becomes worth revisiting; below that, the
   selection problem dominates the modeling problem.

## Reproduce

```bash
# label-free arm (TF-IDF needs no API key; Voyage reads the existing cache)
python -m bilinear_mf.run --backbone tfidf         --arms lsa --run-id mf_tfidf_001_lsa
python -m bilinear_mf.run --backbone voyage_large  --arms lsa --run-id mf_voyage_001_lsa

# trained arm, incl. the 50-draw permutation null (~1 min each, CPU)
python -m bilinear_mf.run --backbone tfidf         --arms bilinear --run-id mf_tfidf_002
python -m bilinear_mf.run --backbone voyage_large  --arms bilinear --run-id mf_voyage_002

pytest tests/test_bilinear_mf.py -q
```

Voyage runs cost **$0** — all 578 texts hit the content-hashed cache under
`artifacts/voyage_large/emb` that the published baselines already paid for
(`usage.json`: 578 hits, 0 misses).
