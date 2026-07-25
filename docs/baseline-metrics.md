# Offline baseline metrics

Shared helpers live in [`baselines/metrics.py`](../baselines/metrics.py). All three
baselines (`bert_frozen`, `voyage_nano`, `voyage_large`) write the same JSON
shape to `artifacts/*/metrics.json`.

No-query ablations (`bert_frozen_no_query`, `voyage_nano_no_query`,
`voyage_large_no_query`) use the same metric keys; seeker packing omits
`searchQuery` (`baselines/text_no_query.py`). Metrics land under
`artifacts/*_no_query/metrics.json` and include `"seeker_text": "profile_only_no_query"`.

**What "positive" means here:** the labels are real human outcomes on intros
Boardy's production system already recommended — positive = accepted/connected,
negative = declined. So `roc_auc` below is measuring how well a score ranks
*accepted* intros above *declined* ones, among candidates production already
judged relevant. See [objective.md](objective.md); this is why absolute values
sit near 0.6 rather than the 0.8+ typical of relevance-vs-irrelevance tasks.

Live production KPIs (fleet-wide accept rate, intro success dashboards, etc.)
are **out of scope** — these are offline metrics computed on the frozen labeled
pairs only.

## Pair section

| Key | Meaning |
|-----|---------|
| `roc_auc` | How well cosine scores rank positives above negatives |
| `average_precision` | Area under the precision–recall curve |
| `best_f1` / `best_f1_threshold` / `best_f1_accuracy` | Best F1 from sweeping cosine thresholds on labeled pairs; accuracy at that threshold |
| `accuracy_at_0.5` | Accuracy if you hard-threshold cosine at 0.5 |
| `mean_cosine_*` / `std_cosine_*` / `mean_cosine_gap` | Score calibration: means, stds, pos−neg gap |
| `score_percentiles` | p10 / p50 / p90 of pos and neg cosines |

## Retrieval section

Binary relevance only (1 if the labeled match, else 0). Soft labels would unlock
graded NDCG later.

| Key | Meaning |
|-----|---------|
| `mrr` | Mean reciprocal rank of the labeled match |
| `mean_rank` / `median_rank` | Rank of the labeled match (1 = best) |
| `map` | Mean average precision; with **one** relevant item per query, MAP ≈ MRR |
| `ndcg@K` | Binary NDCG at K ∈ {1,5,10} |
| `precision@K` | With one labeled good in the corpus: `1/K` if rank≤K else `0`; mean over queries |
| `recall@K` / `top1` | Fraction of queries where the labeled match is in the top K (`top1` = `recall@1`) |

## Slices section

### Intent (`slices.intent`)

Coarse buckets from `searchQuery` (+ `lookingFor`) via keyword heuristics:
fundraise / customers / hiring / partnerships / other.

Per slice: pair AUC (if enough pos+neg), retrieval MRR, Recall@10, counts.
Slices with n&lt;5 are marked `low_n`.

### Neg hardness (`slices.neg_hardness`)

On **negative** pairs only, compute token Jaccard between seeker text and match
text:

- **easy**: overlap ≤ 25th percentile (low lexical overlap)
- **hard**: overlap ≥ median (mid–high overlap; “looks similar” but labeled bad)

Reports pair AUC (all positives + that neg bucket) and mean cosine gap.
