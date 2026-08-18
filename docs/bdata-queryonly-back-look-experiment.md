# B-data queryonly_back_look_001 (all resolved pairs)

Isolated package: `bdata_queryonly_back_look/`.  
Source: locked `data/B-data.json` (read-only; never modified).  
Catalog: `data/unique_contacts_B_data.json`.  
Adapter: `queryonly_back_look_001` (LoRA on `voyage-4-nano`).  
No train/holdout split — every resolved pair is scored.

## Question

The project's best nano fine-tune on the 200 real seed pairs
(`queryonly_back_look_001`: seeker = search query only, candidate =
`background` + `lookingFor`, pair AUC 0.5983 / hard-neg 0.6564 / MRR 0.4791 /
R@1 0.30) — does that transfer to all resolved B-data pairs?

## Setup

| Item | Value |
|---|---|
| Model | `voyageai/voyage-4-nano` + LoRA `queryonly_back_look_001` |
| max_length / truncate_dim / batch | 4096 / 1024 / 8 |
| Device | Modal A10G (`cuda`) |
| Resolved pairs | 18,304 (15,821 ACCEPT / 2,483 REJECT; PENDING dropped) |
| Split | **none** — all resolved pairs |
| Unique people | 29,923 |
| Retrieval corpus | unique people with role `candidate` or `both` (21,168) |
| Seeker text | search query only (`query_only`; empty query falls back to full profile) |
| Candidate text | `background` + `lookingFor` only (`background_lookingfor`) |
| Hardness split | full-profile + query (same convention as the 200-pair eval) |
| Metrics | `baselines.metrics` pair + neg-hardness + within-seeker + retrieval (exact NumPy, not Chroma ANN) |
| Local vector DB | `artifacts/bdata_queryonly_back_look/chroma/` (candidates + queries), rebuilt from `.npy` |

Not the posbg packing (`lookingFor`+query vs positioning+background) used by
`bdata_voyage_nano_posbg/`. This package does not edit that one,
`twotower_queryonly_back_look/`, `bdata_voyage_nano/`, or locked B-data.

## Reproduce

```bash
python -m pytest tests/test_bdata_queryonly_back_look.py -q

modal run bdata_queryonly_back_look/modal_eval.py --batch-size 8 --gpu A10G
modal volume get dorby-bdata-queryonly-back-look-eval allpairs \
  ./artifacts/bdata_queryonly_back_look --force

# local Chroma cache from the pulled .npy files (no GPU)
python -m bdata_queryonly_back_look.store
```

## Results (2026-08-18)

Modal A10G app `ap-54RNu0bDrg3u8hfhlwsx85` stopped 14:44 ADT, ~35 min, clean
completion:
https://modal.com/apps/harsh20k/main/ap-54RNu0bDrg3u8hfhlwsx85

Id map: 15,984 Boardy / 10,835 minted / 3,104 collision-minted. Retrieval skipped 2 ACCEPT pairs whose candidate was not in the corpus.

| Metric | Value |
|---|---|
| n | 18,304 (15,821 ACCEPT / 2,483 REJECT); no split |
| Pair ROC-AUC | **0.5075** |
| Average precision | 0.8685 (yes-skew artifact) |
| Accuracy @ 0.5 | 0.2932 (mean cosine ~0.44, so 0.5 is not a decision point) |
| Mean cosine ACCEPT / REJECT / gap | 0.4369 / 0.4349 / **0.0020** |
| Easy-neg pair AUC | 0.5295 (n_neg=621) |
| Hard-neg pair AUC | **0.4937** (n_neg=1,242) |
| Within-seeker mean / median AUC | 0.4821 / 0.5000 (n=804 dual-label seekers) |
| Retrieval MRR | **0.0514** |
| Retrieval mean / median rank | 397.4 / 96.0 (corpus 21,168) |
| R@1 / R@5 / R@10 / R@100 | 0.0120 / 0.0678 / 0.1227 / 0.5104 |
| NDCG@10 | 0.0567 |

Artifact: `artifacts/bdata_queryonly_back_look/metrics.json`.  
Chroma: `artifacts/bdata_queryonly_back_look/chroma/` (21,168 candidates, 15,819 queries).

## Reading

The 200-pair win does not transfer. Same adapter and packing sit at chance on
B-data pair classification (AUC 0.5075, cosine gap 0.002). Within-seeker AUC
0.482 is below chance. Hard-neg AUC 0.4937 is near chance — better than frozen
posbg nano's inverted 0.4319, still not a signal.

Retrieval against the 21k pool is a bit better than frozen posbg (MRR 0.051 vs
0.032, median rank 96 vs 184, R@1 1.2% vs 0.5%) and still far from the 200-pair
MRR of 0.48. The 200-pair retrieval numbers were measured against a 178-person
corpus; this is a different ranking problem.

Published findings page:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-queryonly-back-look-experiment.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-queryonly-back-look-experiment.html)
(local: `docs/html/bdata-queryonly-back-look-experiment.html`).

## Comparison (quoted; not all matched-population)

| Run | Population | Packing | Pair AUC | Hard-neg | MRR | R@1 |
|---|---|---|---:|---:|---:|---:|
| **This run** | all resolved B-data | query → bg+lookingFor (LoRA) | **0.5075** | **0.4937** | **0.0514** | **0.0120** |
| same adapter | 200 seed pairs | same | 0.5983 | 0.6564 | 0.4791 | 0.30 |
| `bdata_voyage_nano_posbg` | all resolved B-data | look+query → pos+bg (frozen) | 0.5185 | 0.4319 | 0.0320 | 0.0048 |
| `bdata_voyage_nano` | B-data holdout | full profile (frozen) | 0.4691 | 0.3626 | — | — |
| `bdata_tfidf` | B-data holdout | full profile TF-IDF | 0.5121 | 0.4026 | — | — |
| `voyage_nano` full | 200 seed pairs | full profile (frozen) | 0.5614 | 0.5064 | — | — |

Holdout TF-IDF / frozen-nano rows are a different population **and** packing.
The matched B-data comparison is this run vs `bdata_voyage_nano_posbg` (same
18,304 pairs, different packing, LoRA vs frozen).

## Isolation notes

- New package `bdata_queryonly_back_look/`; prior experiment packages untouched.
- Encoder copied into `bdata_queryonly_back_look/encode.py`; cosine drift pinned
  by `tests/test_bdata_queryonly_back_look.py`.
- Text packing copied from `query_weighted.text.query_only` and
  `field_pairs_sweep.text.background_lookingfor` (tests assert the copies agree).
- B-data lock enforced at load time (`chmod 444` required).
- Chroma lives only under this experiment's artifacts; retrieval metrics stay
  exact NumPy.
