# B-data Voyage-4-nano posbg (all resolved pairs)

Isolated package: `bdata_voyage_nano_posbg/`.  
Source: locked `data/B-data.json` (read-only; never modified).  
Catalog: `data/unique_contacts_B_data.json` (gitignored; rebuild with
`python scripts/build_unique_contacts_B_data.py` if missing).  
No train/holdout split — every resolved pair is scored.

## Question

Does frozen Voyage-4-nano, with seeker = search query + `lookingFor` and
candidate = `positioning` + `background`, separate ACCEPT vs REJECT on all
resolved B-data pairs — and can it rank the accepted person in a
unique-candidate pool?

## Setup

| Item | Value |
|---|---|
| Model | `voyageai/voyage-4-nano` (frozen, no training) |
| max_length / truncate_dim / batch | 8192 / 1024 / 8 |
| Device | Modal A10G (`cuda`) |
| Resolved pairs | 18,304 (15,821 ACCEPT / 2,483 REJECT; PENDING dropped) |
| Split | **none** — all resolved pairs |
| Unique people | 29,923 (`data/unique_contacts_B_data.json`) |
| Retrieval corpus | unique people with role `candidate` or `both` (21,168) |
| Seeker text | `lookingFor` + this pair's search query (`seeker_look_text`) |
| Candidate text | `positioning` + `background` only (`bg_text`) |
| Ids | real Boardy `contactIds` where present and unique; else `cmb` + `identityKey[:25]` (Boardy-id reuse across positioning hashes → `minted_collision`) |
| Id map | `artifacts/bdata_voyage_nano_posbg/id_map.json` (never written to `data/`) |
| Metrics | `baselines.metrics` pair + neg-hardness + within-seeker + retrieval |

Not the full-profile packing used by `bdata_voyage_nano/` (that run was
holdout-only and skipped retrieval). This package does not edit
`bdata_voyage_nano/`, `bdata_tfidf/`, `baselines/voyage_nano/`, or locked
B-data.

## Reproduce

```bash
# unit tests (no model download)
python -m pytest tests/test_bdata_voyage_nano_posbg.py -q

# rebuild unique contacts if missing (does not write B-data.json)
python scripts/build_unique_contacts_B_data.py

# Modal GPU all-pairs eval (encode ~9k–20k unique texts + batched retrieval)
modal run bdata_voyage_nano_posbg/modal_eval.py --batch-size 8 --gpu A10G

# pull metrics + embedding caches
modal volume get dorby-bdata-voyage-nano-posbg-eval allpairs \
  ./artifacts/bdata_voyage_nano_posbg --force

# local MPS/CPU smoke (slow on tens of thousands of unique texts — prefer Modal)
python -m bdata_voyage_nano_posbg
```

## Results (2026-08-12)

Modal A10G app `ap-UqnhokCeY5vqmIz9QEEYEC` stopped 18:40 ADT, ~29 min, clean
completion:
https://modal.com/apps/harsh20k/main/ap-UqnhokCeY5vqmIz9QEEYEC

Id map: 15,984 Boardy / 10,835 minted / 3,104 collision-minted. Retrieval skipped 2 ACCEPT pairs whose candidate was not in the corpus.

| Metric | Value |
|---|---|
| n | 18,304 (15,821 ACCEPT / 2,483 REJECT); no split |
| Pair ROC-AUC | **0.5185** |
| Average precision | 0.8704 (yes-skew artifact) |
| Accuracy @ 0.5 | 0.8080 (yes-skew; 86.4% ACCEPT) |
| Mean cosine ACCEPT / REJECT / gap | 0.5802 / 0.5768 / **0.0034** |
| Easy-neg pair AUC | 0.6401 (n_neg=621) |
| Hard-neg pair AUC | **0.4319** (n_neg=1,242, below chance) |
| Within-seeker mean / median AUC | 0.5388 / 0.5000 (n=804 dual-label seekers) |
| Retrieval MRR | **0.0320** |
| Retrieval mean / median rank | 709.8 / 184.0 (corpus 21,168) |
| R@1 / R@5 / R@10 / R@100 | 0.0048 / 0.0394 / 0.0766 / 0.3755 |
| NDCG@10 | 0.0335 |

Artifact: `artifacts/bdata_voyage_nano_posbg/metrics.json`.

## Reading

Near chance on pair classification (AUC 0.5185; cosine gap 0.0034). Hard-neg
AUC is inverted (0.4319). Retrieval against the 21k unique-candidate pool is
very weak: MRR 0.032, median rank 184, R@1 0.5%. Not comparable 1:1 to
`bdata_tfidf` 0.5121 or `bdata_voyage_nano` 0.4691 — those were holdout-only
and full-profile packing.

Published findings page:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-voyage-nano-posbg-experiment.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-voyage-nano-posbg-experiment.html)
(local: `docs/html/bdata-voyage-nano-posbg-experiment.html`).

## Comparison (quoted; not matched-population)

The earlier `bdata_voyage_nano` run used **full-profile** packing on a
**holdout** slice and skipped retrieval. This run uses narrower posbg packing
on **all** resolved pairs and includes retrieval. Numbers are not directly
comparable; they are listed so the two B-data nano runs sit next to each other.

| Run | Population | Packing | Pair AUC | Hard-neg AUC |
|---|---|---|---:|---:|
| **This run** | all resolved B-data pairs | look+query / pos+bg | **0.5185** | **0.4319** |
| `bdata_voyage_nano` | B-data holdout | full profile | 0.4691 | 0.3626 |
| `bdata_tfidf` | B-data holdout | full profile TF-IDF | 0.5121 | 0.4026 |
| `voyage_nano` full | 200 seed pairs | full profile | 0.5614 | 0.5064 |

## Isolation notes

- New package `bdata_voyage_nano_posbg/`; prior experiment packages untouched.
- Encoder copied into `bdata_voyage_nano_posbg/encode.py`; cosine drift pinned
  by `tests/test_bdata_voyage_nano_posbg.py`.
- Text packing copied from `twotower_ask_offer_posbg/text.py`.
- Identity key copied from `scripts/build_unique_contacts_B_data.py` (script
  not edited); test asserts the copy still agrees.
- B-data lock enforced at load time (`chmod 444` required).
- Minted ids written only under `artifacts/bdata_voyage_nano_posbg/`.
