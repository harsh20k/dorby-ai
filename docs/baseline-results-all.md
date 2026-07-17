# Baseline results (all metrics)

Generated: `2026-07-17T21:45:47Z`

Protocol: same pair/retrieval/slices for all. Sources: `artifacts/{bert_frozen,voyage_nano,voyage_large}/metrics.json`.

Metric definitions: [baseline-metrics.md](baseline-metrics.md).

Refresh:
```bash
python scripts/export_baseline_results.py
```

## Pair

| Metric | Frozen BERT | Voyage-4-nano | Voyage-4-large |
|--------|--------|--------|--------|
| ROC-AUC | 0.4697 | 0.5614 | 0.5726 |
| Average precision | 0.5109 | 0.5570 | 0.5710 |
| Best F1 | 0.6758 | 0.6712 | 0.6689 |
| Best-F1 threshold | 0.9095 | 0.5780 | 0.5170 |
| Accuracy @ best-F1 | 0.5250 | 0.5200 | 0.5050 |
| Accuracy @ 0.5 | 0.5000 | 0.5050 | 0.5050 |
| Mean cos (pos) | 0.9412 | 0.6738 | 0.6292 |
| Mean cos (neg) | 0.9419 | 0.6639 | 0.6189 |
| Mean cos gap | -0.0007 | 0.0098 | 0.0103 |
| Std cos (pos) | 0.0148 | 0.0510 | 0.0466 |
| Std cos (neg) | 0.0161 | 0.0511 | 0.0445 |
| Pos p10 / p50 / p90 | 0.9199 / 0.9407 / 0.9607 | 0.5993 / 0.6798 / 0.7340 | 0.5648 / 0.6338 / 0.6832 |
| Neg p10 / p50 / p90 | 0.9179 / 0.9461 / 0.9589 | 0.6036 / 0.6603 / 0.7260 | 0.5684 / 0.6142 / 0.6778 |
| n pos / n neg | 100 / 100 | 100 / 100 | 100 / 100 |

## Retrieval

| Metric | Frozen BERT | Voyage-4-nano | Voyage-4-large |
|--------|--------|--------|--------|
| MRR | 0.0941 | 0.3007 | 0.3100 |
| MAP | 0.0941 | 0.3007 | 0.3100 |
| Mean rank | 55.2300 | 17.8000 | 12.1600 |
| Median rank | 40.0000 | 7.5000 | 5.0000 |
| Top-1 | 0.0200 | 0.1600 | 0.1300 |
| R@1 | 0.0200 | 0.1600 | 0.1300 |
| R@5 | 0.1400 | 0.4700 | 0.5600 |
| R@10 | 0.1800 | 0.6000 | 0.7000 |
| NDCG@1 | 0.0200 | 0.1600 | 0.1300 |
| NDCG@5 | 0.0865 | 0.3189 | 0.3476 |
| NDCG@10 | 0.0989 | 0.3599 | 0.3932 |
| P@1 | 0.0200 | 0.1600 | 0.1300 |
| P@5 | 0.0280 | 0.0940 | 0.1120 |
| P@10 | 0.0180 | 0.0600 | 0.0700 |
| n queries | 100.0000 | 100.0000 | 100.0000 |

## Slices: neg hardness (pair AUC)

| Slice | Frozen BERT | Voyage-4-nano | Voyage-4-large |
|-------|--------|--------|--------|
| easy AUC | 0.6508 | 0.6916 | 0.6540 |
| easy n_neg | 25 | 25 | 25 |
| hard AUC | 0.4108 | 0.5064 | 0.5422 |
| hard n_neg | 50 | 50 | 50 |

## Slices: intent breakdown

| Intent | Metric | Frozen BERT | Voyage-4-nano | Voyage-4-large |
|--------|--------|--------|--------|--------|
| customers | n_pairs | 5 | 5 | 5 |
| customers | n_queries | 3 | 3 | 3 |
| customers | AUC | 0.6667 | 0.3333 | 0.1667 |
| customers | MRR | — | — | — |
| customers | R@10 | — | — | — |
| fundraise | n_pairs | 157 | 157 | 157 |
| fundraise | n_queries | 79 | 79 | 79 |
| fundraise | AUC | 0.4245 | 0.5049 | 0.5359 |
| fundraise | MRR | 0.0727 | 0.2593 | 0.2763 |
| fundraise | R@10 | 0.1139 | 0.5443 | 0.6709 |
| hiring | n_pairs | 19 | 19 | 19 |
| hiring | n_queries | 8 | 8 | 8 |
| hiring | AUC | 0.5455 | 0.7159 | 0.6591 |
| hiring | MRR | 0.1409 | 0.5822 | 0.4537 |
| hiring | R@10 | 0.3750 | 0.7500 | 0.7500 |
| other | n_pairs | 3 | 3 | 3 |
| other | n_queries | 1 | 1 | 1 |
| other | AUC | — | — | — |
| other | MRR | — | — | — |
| other | R@10 | — | — | — |
| partnerships | n_pairs | 16 | 16 | 16 |
| partnerships | n_queries | 9 | 9 | 9 |
| partnerships | AUC | 0.5873 | 0.9365 | 0.8571 |
| partnerships | MRR | 0.1861 | 0.4660 | 0.5529 |
| partnerships | R@10 | 0.3333 | 1.0000 | 1.0000 |

## Model metadata

| Field | Frozen BERT | Voyage-4-nano | Voyage-4-large |
|-------|--------|--------|--------|
| model_name | bert-base-uncased | voyageai/voyage-4-nano | voyage-4-large |
| device | mps | mps | — |
| max_length | 512 | 8192 | — |
| output / truncate dim | — | 1024 | 1024 |
