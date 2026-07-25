# Baseline vs. twotower — matched real-holdout comparison

Generated: `2026-07-25T13:59:36Z`

Protocol: frozen 69-pair real holdout (data/synthetic/seed_split.json eval_pair_ids) only — same population for every row, unlike baseline-results-all.md's full-dataset numbers. See docs/possible-bugs.md #3 and docs/twotower-run-001-results.md.. Sources: `artifacts/{tfidf,bert_frozen,voyage_nano,voyage_large,hybrid_tfidf_voyage}_holdout/metrics.json` + `artifacts/twotower/<run_id>_holdout_eval/metrics_holdout.json`.

Metric definitions: [baseline-metrics.md](baseline-metrics.md).

Refresh:
```bash
python scripts/export_baseline_results.py
```

## Pair

| Metric | TF-IDF (lexical) | Frozen BERT | Voyage-4-nano | Voyage-4-large (prod) | Hybrid TF-IDF+nano | Qwen3-Embedding-8B (open, Modal) | twotower run_001 | twotower arm_a_real_only |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ROC-AUC | 0.5922 | 0.4595 | 0.5793 | 0.6086 | 0.6397 | 0.6595 | 0.5784 | 0.5793 |
| Average precision | 0.5041 | 0.4054 | 0.5123 | 0.5185 | 0.5200 | 0.5335 | 0.4874 | 0.4846 |
| Best F1 | 0.6364 | 0.6154 | 0.5979 | 0.6154 | 0.6316 | 0.6479 | 0.6190 | 0.6098 |
| Best-F1 threshold | 0.1433 | 0.9179 | 0.5780 | 0.6310 | 1.0618 | 0.5566 | 0.7479 | 0.7441 |
| Accuracy @ best-F1 | 0.5362 | 0.4928 | 0.4348 | 0.6377 | 0.5942 | 0.6377 | 0.5362 | 0.5362 |
| Accuracy @ 0.5 | 0.5797 | 0.4203 | 0.4348 | 0.4348 | 0.5072 | 0.5072 | 0.4203 | 0.4203 |
| Mean cos (pos) | 0.1881 | 0.9407 | 0.6775 | 0.6372 | 1.7007 | 0.5838 | 0.8056 | 0.7765 |
| Mean cos (neg) | 0.1762 | 0.9410 | 0.6630 | 0.6178 | 1.2956 | 0.5617 | 0.7825 | 0.7649 |
| Mean cos gap | 0.0119 | -0.0003 | 0.0144 | 0.0193 | 0.4051 | 0.0221 | 0.0230 | 0.0116 |
| Std cos (pos) | 0.0367 | 0.0132 | 0.0554 | 0.0438 | 0.9061 | 0.0350 | 0.0648 | 0.0407 |
| Std cos (neg) | 0.0412 | 0.0188 | 0.0549 | 0.0516 | 1.2274 | 0.0464 | 0.0791 | 0.0498 |
| Pos p10 / p50 / p90 | 0.1528 / 0.1820 / 0.2249 | 0.9232 / 0.9423 / 0.9579 | 0.5909 / 0.6882 / 0.7395 | 0.5697 / 0.6462 / 0.6819 | 0.5694 / 1.6991 / 2.7736 | 0.5441 / 0.5802 / 0.6217 | 0.7410 / 0.8079 / 0.8756 | 0.7340 / 0.7778 / 0.8265 |
| Neg p10 / p50 / p90 | 0.1342 / 0.1703 / 0.2356 | 0.9170 / 0.9474 / 0.9611 | 0.6036 / 0.6570 / 0.7262 | 0.5663 / 0.6116 / 0.6902 | 0.0649 / 1.1099 / 2.9298 | 0.4951 / 0.5562 / 0.6202 | 0.6672 / 0.7961 / 0.8622 | 0.7287 / 0.7697 / 0.8283 |
| n pos / n neg | 29 / 40 | 29 / 40 | 29 / 40 | 29 / 40 | 29 / 40 | 29 / 40 | 29 / 40 | 29 / 40 |

## Retrieval

| Metric | TF-IDF (lexical) | Frozen BERT | Voyage-4-nano | Voyage-4-large (prod) | Hybrid TF-IDF+nano | Qwen3-Embedding-8B (open, Modal) | twotower run_001 | twotower arm_a_real_only |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| MRR | 0.2475 | 0.1371 | 0.4610 | 0.5287 | 0.4043 | 0.4040 | 0.2829 | 0.3882 |
| MAP | 0.2475 | 0.1371 | 0.4610 | 0.5287 | 0.4043 | 0.4040 | 0.2829 | 0.3882 |
| Mean rank | 14.3103 | 24.3448 | 6.7931 | 4.4483 | 7.4483 | 4.7586 | 9.6552 | 6.9655 |
| Median rank | 11.0000 | 21.0000 | 2.0000 | 2.0000 | 4.0000 | 3.0000 | 5.0000 | 5.0000 |
| Top-1 | 0.1379 | 0.0690 | 0.2759 | 0.3448 | 0.2759 | 0.1724 | 0.0690 | 0.2414 |
| R@1 | 0.1379 | 0.0690 | 0.2759 | 0.3448 | 0.2759 | 0.1724 | 0.0690 | 0.2414 |
| R@5 | 0.3448 | 0.1379 | 0.6552 | 0.8276 | 0.5862 | 0.7586 | 0.5517 | 0.5862 |
| R@10 | 0.4828 | 0.3103 | 0.7586 | 0.8621 | 0.7931 | 0.8966 | 0.6552 | 0.7931 |
| NDCG@1 | 0.1379 | 0.0690 | 0.2759 | 0.3448 | 0.2759 | 0.1724 | 0.0690 | 0.2414 |
| NDCG@5 | 0.2357 | 0.1011 | 0.4908 | 0.5929 | 0.4215 | 0.4737 | 0.3248 | 0.4093 |
| NDCG@10 | 0.2819 | 0.1558 | 0.5230 | 0.6038 | 0.4867 | 0.5174 | 0.3586 | 0.4713 |
| P@1 | 0.1379 | 0.0690 | 0.2759 | 0.3448 | 0.2759 | 0.1724 | 0.0690 | 0.2414 |
| P@5 | 0.0690 | 0.0276 | 0.1310 | 0.1655 | 0.1172 | 0.1517 | 0.1103 | 0.1172 |
| P@10 | 0.0483 | 0.0310 | 0.0759 | 0.0862 | 0.0793 | 0.0897 | 0.0655 | 0.0793 |
| n queries | 29.0000 | 29.0000 | 29.0000 | 29.0000 | 29.0000 | 29.0000 | 29.0000 | 29.0000 |

## Slices: neg hardness (pair AUC)

| Slice | TF-IDF (lexical) | Frozen BERT | Voyage-4-nano | Voyage-4-large (prod) | Hybrid TF-IDF+nano | Qwen3-Embedding-8B (open, Modal) | twotower run_001 | twotower arm_a_real_only |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| easy AUC | 0.7552 | 0.6379 | 0.6207 | 0.6000 | 0.7172 | 0.7586 | 0.6931 | 0.6552 |
| easy n_neg | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| hard AUC | 0.5017 | 0.4224 | 0.5707 | 0.6017 | 0.6034 | 0.6259 | 0.4845 | 0.5000 |
| hard n_neg | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |

## Slices: intent breakdown

| Intent | Metric | TF-IDF (lexical) | Frozen BERT | Voyage-4-nano | Voyage-4-large (prod) | Hybrid TF-IDF+nano | Qwen3-Embedding-8B (open, Modal) | twotower run_001 | twotower arm_a_real_only |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| customers | n_pairs | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| customers | n_queries | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| customers | AUC | — | — | — | — | — | — | — | — |
| customers | MRR | — | — | — | — | — | — | — | — |
| customers | R@10 | — | — | — | — | — | — | — | — |
| fundraise | n_pairs | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 60 |
| fundraise | n_queries | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| fundraise | AUC | 0.5359 | 0.4572 | 0.5683 | 0.6042 | 0.5995 | 0.6713 | 0.5822 | 0.5660 |
| fundraise | MRR | 0.2137 | 0.1333 | 0.4140 | 0.5381 | 0.3705 | 0.4228 | 0.2905 | 0.3270 |
| fundraise | R@10 | 0.4167 | 0.2083 | 0.7083 | 0.8750 | 0.7500 | 0.8750 | 0.6667 | 0.7917 |
| hiring | n_pairs | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| hiring | n_queries | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| hiring | AUC | — | — | — | — | — | — | — | — |
| hiring | MRR | — | — | — | — | — | — | — | — |
| hiring | R@10 | — | — | — | — | — | — | — | — |
| partnerships | n_pairs | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| partnerships | n_queries | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| partnerships | AUC | — | — | — | — | — | — | — | — |
| partnerships | MRR | — | — | — | — | — | — | — | — |
| partnerships | R@10 | — | — | — | — | — | — | — | — |

## Model metadata

| Field | TF-IDF (lexical) | Frozen BERT | Voyage-4-nano | Voyage-4-large (prod) | Hybrid TF-IDF+nano | Qwen3-Embedding-8B (open, Modal) | twotower run_001 | twotower arm_a_real_only |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| model_name | tfidf(max_features=20000,ngram_range=(1, 2)) | bert-base-uncased | voyageai/voyage-4-nano | voyage-4-large | hybrid_tfidf_voyage(alpha,rrf) | Qwen/Qwen3-Embedding-8B | voyageai/voyage-4-nano | voyageai/voyage-4-nano |
| device | cpu | mps | mps | — | cpu+cached | cuda | mps | mps |
| max_length | — | 512 | 4096 | — | 4096 | 8192 | 4096 | 4096 |
| output / truncate dim | — | — | 1024 | 1024 | 1024 | — | 1024 | 1024 |
