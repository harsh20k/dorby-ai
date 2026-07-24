# Field ablation — voyage-4-nano, downstream metrics, real 69-pair holdout

Each row drops exactly one candidate-side profile field and reruns local voyage-4-nano end to end (seeker side unchanged). `delta_auc` = baseline AUC minus ablated AUC — positive means removing that field *hurt* pair classification, i.e. the model was relying on it; negative means removing it *helped* (the field was adding noise on this population, or its absence lets other fields dominate more cleanly). Sorted by delta_auc descending (most load-bearing field first).

| field | pair AUC | delta AUC vs baseline | AP | MRR | NDCG@10 |
|---|---|---|---|---|---|
| (none — baseline) | 0.5793 | — | 0.5123 | 0.4610 | 0.5230 |
| lookingFor | 0.5647 | +0.0147 | 0.4934 | 0.4022 | 0.4601 |
| background | 0.5707 | +0.0086 | 0.4699 | 0.4096 | 0.4712 |
| notes | 0.5707 | +0.0086 | 0.5038 | 0.4556 | 0.5171 |
| personalPreferences | 0.5776 | +0.0017 | 0.5121 | 0.4605 | 0.5223 |
| meetingAndSchedulingPreferences | 0.5793 | +0.0000 | 0.5123 | 0.4605 | 0.5131 |
| introPreferences | 0.5802 | -0.0009 | 0.5102 | 0.4639 | 0.5231 |
| locationAvailability | 0.5828 | -0.0034 | 0.5135 | 0.4152 | 0.4977 |
| positioning | 0.5905 | -0.0112 | 0.5395 | 0.4233 | 0.5026 |

Baseline run: `/Users/harsh/Artifacts/dorby-ai/artifacts/voyage_nano_holdout` (unablated voyage-4-nano, `--holdout-only`, 69 real pairs: 29 pos / 40 neg). Ablation runs: `/Users/harsh/Artifacts/dorby-ai/.claude/worktrees/jolly-cuddling-hopcroft/artifacts/voyage_nano_field_ablation/<field>/metrics.json`.
