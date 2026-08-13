# B-data TF-IDF Accept/Reject

Isolated package: `bdata_tfidf/`.  
Source: locked `data/B-data.json` (read-only; never modified).

## Question

Does plain TF-IDF cosine similarity between seeker (`query` + profile) and
candidate profile separate Boardy **ACCEPT** vs **REJECT** intros in the new
B-dataset?

This mirrors [`baselines/tfidf/`](../baselines/tfidf/) on the original 200
pairs — unsupervised lexical scoring, not a trained classifier. Labels only
define eval groups. `PENDING` matches are dropped.

## Setup

| Item | Value |
|---|---|
| Resolved pairs | 18,304 (15,821 ACCEPT / 2,483 REJECT) |
| Seekers with ≥1 resolved match | 6,962 |
| Split | seeker-disjoint 70/30, seed 42 → `bdata_tfidf/split.json` |
| Train | 12,719 pairs / 4,873 seekers |
| Holdout | 5,585 pairs / 2,089 seekers (4,885 ACCEPT / 700 REJECT) |
| Encoder | TF-IDF `max_features=20000`, ngrams `(1,2)`, fit on **train texts only** |
| Text packing | `baselines.bert_frozen.text.seeker_to_text` / `candidate_to_text` |
| Metrics | `baselines.metrics.pair_metrics` + neg-hardness slices (read-only) |
| Retrieval | skipped (candidates have no real ids; profile-hash ids collide) |

`split_hash`: `0f050493daf9f1f8e71b6390c4bb86b6522d9121799d566aa2b6292fe678b18e`  
Source sha256: `d638dd2122710854a461463b7ecf7f2054e77aa2ce8e2bc0d3cb9d0074cd26ea`

## Reproduce

```bash
# once — freezes seeker-disjoint split (refuses overwrite unless --force)
python -m bdata_tfidf --init-split

# fit TF-IDF on train, score holdout → artifacts/bdata_tfidf/metrics.json
python -m bdata_tfidf

# unit tests
python -m pytest tests/test_bdata_tfidf_data.py -q
```

## Holdout results (2026-08-12)

| Metric | Value |
|---|---|
| Pair ROC-AUC | **0.5121** |
| Average precision | 0.8764 (dominated by yes-skew) |
| Best-F1 | 0.9331 @ threshold 0.0 (trivial always-yes) |
| Accuracy @ 0.5 | 0.1255 |
| Mean cosine ACCEPT / REJECT / gap | 0.2044 / 0.2033 / **0.0011** |
| Easy-neg pair AUC | 0.6582 (n_neg=175) |
| Hard-neg pair AUC | **0.4026** (n_neg=350, below chance) |
| Within-seeker mean AUC | 0.4970 (n=218 dual-label seekers) |
| Train-set pair AUC (sanity) | 0.5299 |

Artifact: `artifacts/bdata_tfidf/metrics.json`.

Published findings page:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-tfidf-experiment.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-tfidf-experiment.html)
(local: `docs/html/bdata-tfidf-experiment.html`). Dataset browser summary:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-summary.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-summary.html)
(local: `docs/html/bdata-summary.html`; rebuild via `python scripts/build_summary_B_data.py`).

## Reading

Lexical overlap does **not** predict accept vs decline on this dump. Mean
cosine for ACCEPT and REJECT is essentially identical; overall AUC is chance;
within-seeker ranking is chance; hard negatives score *below* chance (same
pattern as several embedding baselines on the original hard slice — keyword
overlap marks “looks related” intros that humans still reject).

High average precision / best-F1@0 are class-imbalance artifacts (~7:1
ACCEPT:REJECT on holdout), not discrimination.

## Isolation notes

- New package `bdata_tfidf/`; `baselines/tfidf/` untouched.
- Encoder copied into `bdata_tfidf/encode.py`; cosine drift pinned by
  `tests/test_bdata_tfidf_data.py::test_cosine_scores_matches_baselines_tfidf`.
- B-data lock enforced at load time (`chmod 444` required).
