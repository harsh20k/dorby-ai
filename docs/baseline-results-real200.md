# All-200 real-pair comparison

Every model in this project scored on the **same 200 real pairs** — 100
positive queries ranked against a 178-candidate corpus — rather than the
69-pair holdout used by `docs/baseline-results-holdout.md`.

Retrieval metrics are comparable **between models within one subset only**.
A larger candidate pool is strictly harder, so `all` (178 candidates) and
`holdout` (65) must never be compared to each other.

## All 200 pairs — corpus 178 candidates

Ranked by MRR.

| model | family | pair AUC | hard-neg AUC | easy-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|---|---|
| twotower top1_ctrl | fine-tuned | 0.5683 | 0.5484 | 0.6836 | 0.3550 | 0.1900 | 0.6900 |
| twotower Arm A (v2) | fine-tuned | 0.5594 | 0.5558 | 0.6416 | 0.3341 | 0.1800 | 0.6400 |
| BGE-en-ICL (open, zero-shot) | open-weight | 0.5389 | 0.5226 | 0.5928 | 0.3190 | 0.1700 | 0.6200 |
| Voyage-4-nano (frozen) | baseline | 0.5593 | 0.5046 | 0.6960 | 0.3171 | 0.1800 | 0.5900 |
| Voyage-4-large (production) | baseline | 0.5726 | 0.5422 | 0.6540 | 0.3102 | 0.1300 | 0.7000 |
| twotower Qwen micro-6 | fine-tuned | 0.5947 | 0.5608 | 0.6708 | 0.3031 | 0.1400 | 0.6600 |
| twotower top1_sharp | fine-tuned | 0.5429 | 0.4578 | 0.7164 | 0.3010 | 0.1400 | 0.6400 |
| twotower Qwen micro-1 | fine-tuned | 0.5604 | 0.4828 | 0.6980 | 0.2734 | 0.1400 | 0.5600 |
| Qwen3-Embedding-8B (open) | open-weight | 0.5529 | 0.4680 | 0.7208 | 0.2045 | 0.0500 | 0.5500 |
| TF-IDF (lexical) | baseline | 0.5649 | 0.5164 | 0.6848 | 0.1313 | 0.0500 | 0.2600 |
| E5-Mistral-7B-instruct (open) | open-weight | 0.4597 | 0.3772 | 0.6144 | 0.1159 | 0.0300 | 0.3000 |
| Frozen BERT | baseline | 0.4697 | 0.4108 | 0.6508 | 0.0941 | 0.0200 | 0.1800 |
| NV-Embed-v2 (open, non-commercial, approx.) | open-weight | 0.4841 | 0.3836 | 0.6608 | 0.0857 | 0.0400 | 0.1600 |
| zembed-1-embedding (open) | open-weight | 0.4707 | 0.4864 | 0.4816 | 0.0377 | 0.0100 | 0.0800 |

## Holdout 69 pairs (reference) — corpus 65 candidates

Ranked by MRR.

| model | family | pair AUC | hard-neg AUC | easy-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|---|---|
| twotower Qwen micro-6 | fine-tuned | 0.6810 | 0.6397 | 0.6690 | 0.5847 | 0.4483 | 0.8966 |
| twotower top1_ctrl | fine-tuned | 0.5974 | 0.6121 | 0.6241 | 0.5436 | 0.3793 | 0.8621 |
| twotower Arm A (v2) | fine-tuned | 0.5983 | 0.6034 | 0.6034 | 0.5326 | 0.3793 | 0.8621 |
| Voyage-4-large (production) | baseline | 0.6086 | 0.6017 | 0.6000 | 0.5287 | 0.3448 | 0.8621 |
| BGE-en-ICL (open, zero-shot) | open-weight | 0.5750 | 0.5862 | 0.5655 | 0.5157 | 0.3793 | 0.7586 |
| twotower Qwen micro-1 | fine-tuned | 0.6828 | 0.6172 | 0.7310 | 0.4972 | 0.3448 | 0.8621 |
| twotower top1_sharp | fine-tuned | 0.5629 | 0.5241 | 0.6586 | 0.4735 | 0.2759 | 0.7241 |
| Voyage-4-nano (frozen) | baseline | 0.5793 | 0.5707 | 0.6207 | 0.4610 | 0.2759 | 0.7586 |
| Qwen3-Embedding-8B (open) | open-weight | 0.6543 | 0.6276 | 0.7448 | 0.4097 | 0.1724 | 0.8966 |
| TF-IDF (lexical) | baseline | 0.5922 | 0.5017 | 0.7552 | 0.2475 | 0.1379 | 0.4828 |
| E5-Mistral-7B-instruct (open) | open-weight | 0.5664 | 0.4879 | 0.7138 | 0.2244 | 0.0690 | 0.5517 |
| Frozen BERT | baseline | 0.4595 | 0.4224 | 0.6379 | 0.1371 | 0.0690 | 0.3103 |
| NV-Embed-v2 (open, non-commercial, approx.) | open-weight | 0.5034 | 0.3759 | 0.6000 | 0.1092 | 0.0345 | 0.2414 |
| zembed-1-embedding (open) | open-weight | 0.5086 | 0.4776 | 0.5276 | 0.0641 | 0.0000 | 0.1379 |

## LLM judges (classification only, no retrieval metrics)

Not merged into the tables above — an LLM judge has no shared vector space to
rank a candidate corpus with, so it has no MRR/R@1/R@10, only pair AUC. See
`docs/llm-judge-experiment.md` and `docs/qwen35-judge-experiment.md` for full
writeups.

| model | pair AUC | hard-neg AUC | easy-neg AUC | gets query? |
|---|---|---|---|---|
| gemini-3.1-flash-lite (naive) | 0.6177 | — | — | no |
| Qwen3.5-4B (naive + query, self-hosted, not fine-tuned) | 0.5888 | 0.6271 | 0.5624 | yes |

---

Generated 2026-07-31T01:52:35.174284+00:00 by `python -m eval_real_full.export`.
LLM-judge rows above added 2026-08-04 by hand from
`docs/llm-judge-experiment.md` and `docs/qwen35-judge-experiment.md`.
