# Query-only packing: four-way comparison

Comparison page (not a new training run): frozen `voyage-4-nano` vs
`top1_ctrl` eval-time query-only swap vs `queryonly_back_look_001`, on both
candidate packings, all 200 real pairs.

Published:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/queryonly-packing-comparison.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/queryonly-packing-comparison.html)
(local: `docs/html/queryonly-packing-comparison.html`).

Numbers are copied from already-published experiments; nothing was re-scored.

| Model | Train | Eval | pair AUC | hard-neg | easy-neg | MRR | R@1 | R@5 | R@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen `voyage-4-nano` | — | query → full profile | 0.553 | 0.591 | 0.579 | 0.502 | 0.30 | 0.78 | 0.91 |
| `top1_ctrl` + query-only swap | full profile both sides | query → full profile | 0.595 | 0.646 | 0.616 | **0.508** | **0.32** | **0.80** | 0.90 |
| Frozen `voyage-4-nano` | — | query → bg+lookingFor | 0.563 | 0.437 | **0.742** | 0.476 | 0.28 | 0.76 | 0.85 |
| `queryonly_back_look_001` | query → bg+lookingFor | query → bg+lookingFor | **0.598** | **0.656** | 0.570 | 0.479 | 0.30 | 0.74 | 0.86 |

Sources: `docs/query-weighted-encoding-experiment.md`,
`artifacts/twotower_query_weighted/qw_top1_ctrl_001/top1_ctrl/metrics.json`,
`docs/twotower-queryonly-back-look-experiment.md`.
