# B-data Voyage-4-nano Accept/Reject

Isolated package: `bdata_voyage_nano/`.  
Source: locked `data/B-data.json` (read-only; never modified).  
Split: byte-identical copy of `bdata_tfidf/split.json` (matched population).

## Question

Does frozen Voyage-4-nano cosine similarity separate Boardy **ACCEPT** vs
**REJECT** on the new B-dataset — and how does that compare to the same
model on the old 200-pair seed set?

## Setup

| Item | Value |
|---|---|
| Model | `voyageai/voyage-4-nano` (frozen, no training) |
| max_length / truncate_dim / batch | 8192 / 1024 / 8 |
| Device | Modal A10G (`cuda`) |
| Resolved pairs | 18,304 (15,821 ACCEPT / 2,483 REJECT) |
| Split | seeker-disjoint 70/30, seed 42 → `bdata_voyage_nano/split.json` |
| Holdout | 5,585 pairs / 2,089 seekers (4,885 ACCEPT / 700 REJECT) |
| Text packing | `baselines.bert_frozen.text` seeker/candidate (read-only) |
| Metrics | `baselines.metrics` pair + neg-hardness + within-seeker |
| Retrieval | skipped (candidate ids are profile hashes) |

`split_hash`: `0f050493daf9f1f8e71b6390c4bb86b6522d9121799d566aa2b6292fe678b18e`  
Source sha256: `d638dd2122710854a461463b7ecf7f2054e77aa2ce8e2bc0d3cb9d0074cd26ea`

## Reproduce

```bash
# unit tests (no model download)
python -m pytest tests/test_bdata_voyage_nano_data.py -q

# Modal GPU holdout eval (~15–20 min encode on A10G)
modal run bdata_voyage_nano/modal_eval.py --batch-size 8 --gpu A10G

# pull metrics if needed
modal volume get dorby-bdata-voyage-nano-eval holdout/metrics.json \
  ./artifacts/bdata_voyage_nano/metrics.json --force

# local MPS/CPU (slow on ~9k unique long texts — prefer Modal)
python -m bdata_voyage_nano
```

## Holdout results (2026-08-12)

| Metric | Value |
|---|---|
| Pair ROC-AUC | **0.4691** |
| Average precision | 0.8621 (yes-skew artifact) |
| Mean cosine ACCEPT / REJECT / gap | 0.6572 / 0.6626 / **−0.0055** |
| Easy-neg pair AUC | 0.6267 (n_neg=175) |
| Hard-neg pair AUC | **0.3626** (n_neg=350, below chance) |
| Within-seeker mean AUC | 0.5024 (n=218 dual-label seekers) |

Artifact: `artifacts/bdata_voyage_nano/metrics.json`.

## Comparison

| Run | Population | Pair AUC | Hard-neg AUC |
|---|---|---:|---:|
| **This run** (voyage-4-nano) | B-data matched holdout | **0.4691** | **0.3626** |
| `bdata_tfidf` | same B-data holdout | 0.5121 | 0.4026 |
| `voyage_nano` full | 200 seed pairs | 0.5614 | 0.5064 |
| `voyage_nano` holdout | 69 seed holdout | 0.5793 | 0.5707 |

## Reading

Frozen Voyage-4-nano does **not** predict accept vs decline on B-data. Overall
AUC is *below* chance; REJECT mean cosine is slightly *higher* than ACCEPT;
within-seeker ranking is chance; hard negatives are badly inverted (0.36).

On the old 200-pair seed set the same model sat in the mid-0.56–0.58 band —
weak but above chance. On B-data it is worse than that seed run *and* worse
than plain TF-IDF on the identical holdout. Whatever signal nano had on the
small balanced seed set does not transfer to this larger, yes-skewed
production dump.

Published findings page:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-voyage-nano-experiment.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-voyage-nano-experiment.html)
(local: `docs/html/bdata-voyage-nano-experiment.html`).

## Isolation notes

- New package `bdata_voyage_nano/`; `baselines/voyage_nano/` and
  `bdata_tfidf/` untouched.
- Encoder copied into `bdata_voyage_nano/encode.py`; cosine drift pinned by
  `tests/test_bdata_voyage_nano_data.py`.
- Split is a byte-identical copy of `bdata_tfidf/split.json` (asserted in
  tests) for matched-population comparison with the TF-IDF B-data run.
- B-data lock enforced at load time (`chmod 444` required).
