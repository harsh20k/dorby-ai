# Baseline results (all metrics)

Generated: `2026-07-18T21:08:33Z`

Protocol: same pair/retrieval/slices for all; (no query) = profile-only seeker text (no searchQuery). Sources: `artifacts/{bert_frozen,bert_frozen_no_query,voyage_nano,voyage_nano_no_query,voyage_large,voyage_large_no_query}/metrics.json`.

Metric definitions: [baseline-metrics.md](baseline-metrics.md).

Refresh:
```bash
python scripts/export_baseline_results.py
```

## Pair

| Metric | Frozen BERT | Frozen BERT (no query) | Voyage-4-nano | Voyage-4-nano (no query) | Voyage-4-large | Voyage-4-large (no query) |
|--------|--------|--------|--------|--------|--------|--------|
| ROC-AUC | 0.4697 | 0.4699 | 0.5614 | 0.5416 | 0.5726 | 0.5252 |
| Average precision | 0.5109 | 0.5111 | 0.5570 | 0.5332 | 0.5710 | 0.5209 |
| Best F1 | 0.6758 | 0.6758 | 0.6712 | 0.6689 | 0.6689 | 0.6690 |
| Best-F1 threshold | 0.9095 | 0.9095 | 0.5780 | 0.5179 | 0.5170 | 0.5260 |
| Accuracy @ best-F1 | 0.5250 | 0.5250 | 0.5200 | 0.5050 | 0.5050 | 0.5250 |
| Accuracy @ 0.5 | 0.5000 | 0.5000 | 0.5050 | 0.5050 | 0.5050 | 0.4950 |
| Mean cos (pos) | 0.9412 | 0.9412 | 0.6738 | 0.6640 | 0.6292 | 0.6072 |
| Mean cos (neg) | 0.9419 | 0.9419 | 0.6639 | 0.6573 | 0.6189 | 0.6035 |
| Mean cos gap | -0.0007 | -0.0007 | 0.0098 | 0.0066 | 0.0103 | 0.0037 |
| Std cos (pos) | 0.0148 | 0.0148 | 0.0510 | 0.0528 | 0.0466 | 0.0506 |
| Std cos (neg) | 0.0161 | 0.0161 | 0.0511 | 0.0532 | 0.0445 | 0.0496 |
| Pos p10 / p50 / p90 | 0.9199 / 0.9407 / 0.9607 | 0.9199 / 0.9407 / 0.9607 | 0.5993 / 0.6798 / 0.7340 | 0.5921 / 0.6693 / 0.7257 | 0.5648 / 0.6338 / 0.6832 | 0.5476 / 0.6118 / 0.6690 |
| Neg p10 / p50 / p90 | 0.9179 / 0.9461 / 0.9589 | 0.9179 / 0.9461 / 0.9589 | 0.6036 / 0.6603 / 0.7260 | 0.5987 / 0.6527 / 0.7211 | 0.5684 / 0.6142 / 0.6778 | 0.5307 / 0.6013 / 0.6703 |
| n pos / n neg | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

## Retrieval

| Metric | Frozen BERT | Frozen BERT (no query) | Voyage-4-nano | Voyage-4-nano (no query) | Voyage-4-large | Voyage-4-large (no query) |
|--------|--------|--------|--------|--------|--------|--------|
| MRR | 0.0941 | 0.0941 | 0.3007 | 0.2236 | 0.3100 | 0.2503 |
| MAP | 0.0941 | 0.0941 | 0.3007 | 0.2236 | 0.3100 | 0.2503 |
| Mean rank | 55.2300 | 55.2300 | 17.8000 | 24.2800 | 12.1600 | 21.2200 |
| Median rank | 40.0000 | 40.0000 | 7.5000 | 12.0000 | 5.0000 | 9.0000 |
| Top-1 | 0.0200 | 0.0200 | 0.1600 | 0.0800 | 0.1300 | 0.1200 |
| R@1 | 0.0200 | 0.0200 | 0.1600 | 0.0800 | 0.1300 | 0.1200 |
| R@5 | 0.1400 | 0.1400 | 0.4700 | 0.3900 | 0.5600 | 0.3800 |
| R@10 | 0.1800 | 0.1800 | 0.6000 | 0.4800 | 0.7000 | 0.5500 |
| NDCG@1 | 0.0200 | 0.0200 | 0.1600 | 0.0800 | 0.1300 | 0.1200 |
| NDCG@5 | 0.0865 | 0.0865 | 0.3189 | 0.2413 | 0.3476 | 0.2527 |
| NDCG@10 | 0.0989 | 0.0989 | 0.3599 | 0.2701 | 0.3932 | 0.3076 |
| P@1 | 0.0200 | 0.0200 | 0.1600 | 0.0800 | 0.1300 | 0.1200 |
| P@5 | 0.0280 | 0.0280 | 0.0940 | 0.0780 | 0.1120 | 0.0760 |
| P@10 | 0.0180 | 0.0180 | 0.0600 | 0.0480 | 0.0700 | 0.0550 |
| n queries | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |

## Slices: neg hardness (pair AUC)

| Slice | Frozen BERT | Frozen BERT (no query) | Voyage-4-nano | Voyage-4-nano (no query) | Voyage-4-large | Voyage-4-large (no query) |
|-------|--------|--------|--------|--------|--------|--------|
| easy AUC | 0.6508 | 0.6516 | 0.6916 | 0.6712 | 0.6540 | 0.6184 |
| easy n_neg | 25 | 25 | 25 | 25 | 25 | 25 |
| hard AUC | 0.4108 | 0.4108 | 0.5064 | 0.4770 | 0.5422 | 0.4726 |
| hard n_neg | 50 | 50 | 50 | 50 | 50 | 50 |

## Slices: intent breakdown

| Intent | Metric | Frozen BERT | Frozen BERT (no query) | Voyage-4-nano | Voyage-4-nano (no query) | Voyage-4-large | Voyage-4-large (no query) |
|--------|--------|--------|--------|--------|--------|--------|--------|
| customers | n_pairs | 5 | 5 | 5 | 5 | 5 | 5 |
| customers | n_queries | 3 | 3 | 3 | 3 | 3 | 3 |
| customers | AUC | 0.6667 | 0.6667 | 0.3333 | 0.1667 | 0.1667 | 0.1667 |
| customers | MRR | — | — | — | — | — | — |
| customers | R@10 | — | — | — | — | — | — |
| fundraise | n_pairs | 157 | 157 | 157 | 157 | 157 | 157 |
| fundraise | n_queries | 79 | 79 | 79 | 79 | 79 | 79 |
| fundraise | AUC | 0.4245 | 0.4249 | 0.5049 | 0.4787 | 0.5359 | 0.4807 |
| fundraise | MRR | 0.0727 | 0.0727 | 0.2593 | 0.1748 | 0.2763 | 0.2092 |
| fundraise | R@10 | 0.1139 | 0.1139 | 0.5443 | 0.4430 | 0.6709 | 0.4937 |
| hiring | n_pairs | 19 | 19 | 19 | 19 | 19 | 19 |
| hiring | n_queries | 8 | 8 | 8 | 8 | 8 | 8 |
| hiring | AUC | 0.5455 | 0.5455 | 0.7159 | 0.6705 | 0.6591 | 0.5909 |
| hiring | MRR | 0.1409 | 0.1409 | 0.5822 | 0.4512 | 0.4537 | 0.4199 |
| hiring | R@10 | 0.3750 | 0.3750 | 0.7500 | 0.6250 | 0.7500 | 0.7500 |
| other | n_pairs | 3 | 3 | 3 | 3 | 3 | 3 |
| other | n_queries | 1 | 1 | 1 | 1 | 1 | 1 |
| other | AUC | — | — | — | — | — | — |
| other | MRR | — | — | — | — | — | — |
| other | R@10 | — | — | — | — | — | — |
| partnerships | n_pairs | 16 | 16 | 16 | 16 | 16 | 16 |
| partnerships | n_queries | 9 | 9 | 9 | 9 | 9 | 9 |
| partnerships | AUC | 0.5873 | 0.5873 | 0.9365 | 0.9683 | 0.8571 | 0.8889 |
| partnerships | MRR | 0.1861 | 0.1861 | 0.4660 | 0.4778 | 0.5529 | 0.4958 |
| partnerships | R@10 | 0.3333 | 0.3333 | 1.0000 | 0.7778 | 1.0000 | 0.8889 |

## Model metadata

| Field | Frozen BERT | Frozen BERT (no query) | Voyage-4-nano | Voyage-4-nano (no query) | Voyage-4-large | Voyage-4-large (no query) |
|-------|--------|--------|--------|--------|--------|--------|
| model_name | bert-base-uncased | bert-base-uncased | voyageai/voyage-4-nano | voyageai/voyage-4-nano | voyage-4-large | voyage-4-large |
| device | mps | mps | mps | mps | — | — |
| max_length | 512 | 512 | 8192 | 8192 | — | — |
| output / truncate dim | — | — | 1024 | 1024 | 1024 | 1024 |
