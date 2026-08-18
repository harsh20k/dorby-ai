# Frozen voyage-4-nano, queryonly_back_look packing, all B-data pairs

Isolated package: `bdata_queryonly_back_look_frozen/`.  
Source: locked `data/B-data.json` (read-only; never modified).  
Catalog: `data/unique_contacts_B_data.json`.  
Encoder: frozen `voyage-4-nano` — **no LoRA**.  
Packing matches `queryonly_back_look_001` / `bdata_queryonly_back_look/`:
seeker = search query only, candidate = `background` + `lookingFor`.  
No train/holdout split — every resolved pair is scored.

This is the missing control for `bdata_queryonly_back_look/` (same packing,
same 18,304 pairs, LoRA adapter). Do not edit that package.

## Question

On all resolved B-data pairs, with this packing, does frozen nano already
sit at chance — or did the LoRA adapter uniquely fail to transfer?

## Setup

| Item | Value |
|---|---|
| Model | `voyageai/voyage-4-nano` frozen (no adapter) |
| max_length / truncate_dim / batch | 4096 / 1024 / 8 |
| Device | Modal A10G (`cuda`) |
| Resolved pairs | 18,304 (15,821 ACCEPT / 2,483 REJECT; PENDING dropped) |
| Split | **none** — all resolved pairs |
| Unique people | 29,923 |
| Retrieval corpus | unique people with role `candidate` or `both` (21,168) |
| Seeker text | search query only (`query_only`; empty query falls back to full profile) |
| Candidate text | `background` + `lookingFor` only (`background_lookingfor`) |
| Hardness split | full-profile + query (same convention as the 200-pair eval) |
| Metrics | `baselines.metrics` pair + neg-hardness + within-seeker + retrieval (exact NumPy) |

Not the posbg packing (`lookingFor`+query vs positioning+background) used by
`bdata_voyage_nano_posbg/`. This package does not edit that one,
`bdata_queryonly_back_look/`, `twotower_queryonly_back_look/`,
`bdata_voyage_nano/`, `baselines/voyage_nano/`, or locked B-data.

## Reproduce

```bash
python -m pytest tests/test_bdata_queryonly_back_look_frozen.py -q

modal run --detach bdata_queryonly_back_look_frozen/modal_eval.py --batch-size 8 --gpu A10G
modal volume get dorby-bdata-queryonly-back-look-frozen allpairs \
  ./artifacts/bdata_queryonly_back_look_frozen --force
# flatten allpairs/ if the CLI nests it
```

## Results (2026-08-18)

Modal A10G app `ap-ZSKQPoXPwnX6QMQJ1PYUbe` ~34 min, clean completion:
https://modal.com/apps/harsh20k/main/ap-ZSKQPoXPwnX6QMQJ1PYUbe

Id map: 15,984 Boardy / 10,835 minted / 3,104 collision-minted. Retrieval skipped 2 ACCEPT pairs whose candidate was not in the corpus. Modal image has no chromadb; wrote `vectors/*.npy` only.

| Metric | Value |
|---|---|
| n | 18,304 (15,821 ACCEPT / 2,483 REJECT); no split |
| Pair ROC-AUC | **0.4869** |
| Average precision | 0.8606 (yes-skew artifact) |
| Accuracy @ 0.5 | 0.5640 |
| Mean cosine ACCEPT / REJECT / gap | 0.5126 / 0.5166 / **−0.0040** |
| Easy-neg pair AUC | 0.5342 (n_neg=621) |
| Hard-neg pair AUC | **0.4585** (n_neg=1,242) |
| Within-seeker mean / median AUC | 0.4873 / 0.5000 (n=804 dual-label seekers) |
| Retrieval MRR | **0.0523** |
| Retrieval mean / median rank | 393.8 / 96.0 (corpus 21,168) |
| R@1 / R@5 / R@10 / R@100 | 0.0139 / 0.0650 / 0.1194 / 0.5095 |
| NDCG@10 | 0.0566 |

Artifact: `artifacts/bdata_queryonly_back_look_frozen/metrics.json`.

## Reading vs LoRA on the matched B-data population

Retrieval is a tie. Pair AUC is chance either way; LoRA is a small bump,
not a transfer.

| Metric | Frozen (this) | LoRA B-data | Frozen 200-pair (same packing) |
|---|---:|---:|---:|
| Pair AUC | **0.4869** | 0.5075 | 0.5626 |
| Hard-neg AUC | **0.4585** | 0.4937 | 0.4374 |
| Easy-neg AUC | 0.5342 | 0.5295 | — |
| Within-seeker | 0.4873 | 0.4821 | — |
| Cosine gap | −0.0040 | +0.0020 | — |
| MRR | **0.0523** | 0.0514 | 0.4763 |
| Median rank | 96 | 96 | — |
| R@1 | **0.0139** | 0.0120 | 0.28 |
| R@10 | 0.1194 | 0.1227 | — |
| NDCG@10 | 0.0566 | 0.0567 | — |

LoRA's 200-pair hard-neg win (0.656 vs frozen 0.437) does not survive B-data
(0.494 vs 0.459 — both near/below chance). Frozen cosine is inverted
(REJECT slightly above ACCEPT). The adapter did not uniquely destroy a
working frozen encoder; frozen was already chance-or-worse on this packing
at B-data scale.

Published findings (LoRA page, frozen row added):
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-queryonly-back-look-experiment.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-queryonly-back-look-experiment.html)

## Expected R@1 if the model were unchanged

Same pool-size note as the LoRA page. 200-pair frozen R@1 is **0.28** on a
178-person corpus; B-data ranks against **21,168**.

| | 200-pair frozen | Naive same-quality on 21k | Observed |
|---|---:|---:|---:|
| R@1 | 0.28 | **0.28 × 178 / 21,168 ≈ 0.0024** | **0.0139** (~6× naive) |
| MRR | 0.476 | **0.476 × 178 / 21,168 ≈ 0.0040** | **0.0523** (~13× naive) |

LoRA's naive dilution from its 200-pair R@1 0.30 is ≈ 0.0025; observed 0.012
is ~5×. Frozen and LoRA beat dilution by about the same factor. Absolute R@1
must fall on a 119× longer list; 0.28 was never a fair B-data target. Pair
AUC has no such excuse (0.563 → 0.487).

## Comparison (quoted; not all matched-population)

| Run | Population | Packing | Pair AUC | Hard-neg | MRR | R@1 |
|---|---|---|---:|---:|---:|---:|
| **This run (frozen)** | all resolved B-data | query → bg+lookingFor | **0.4869** | **0.4585** | **0.0523** | **0.0139** |
| LoRA `queryonly_back_look_001` | same 18,304 | same | 0.5075 | 0.4937 | 0.0514 | 0.0120 |
| same packing, frozen | 200 seed pairs | same | 0.5626 | 0.4374 | 0.4763 | 0.28 |
| same packing, LoRA | 200 seed pairs | same | 0.5983 | 0.6564 | 0.4791 | 0.30 |
| `bdata_voyage_nano_posbg` | all resolved B-data | look+query → pos+bg (frozen) | 0.5185 | 0.4319 | 0.0320 | 0.0048 |

posbg is the same 18,304 pairs, different packing. Full-profile holdout rows
are a different population and packing.

## Isolation notes

- New package `bdata_queryonly_back_look_frozen/`; prior experiment packages untouched.
- Encoder copied then adapter loading removed; cosine drift pinned by
  `tests/test_bdata_queryonly_back_look_frozen.py`.
- Text packing copied from `query_weighted.text.query_only` and
  `field_pairs_sweep.text.background_lookingfor` (tests assert the copies agree).
- Modal app/volume `dorby-bdata-queryonly-back-look-frozen` (does not reuse the
  LoRA eval volume or checkpoint volume). HF cache `dorby-twotower-hf-cache`
  reused read-only.
- B-data lock enforced at load time (`chmod 444` required).
