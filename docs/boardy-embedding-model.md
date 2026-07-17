# Boardy production embedding model

**Source:** Prof. Ga Wu (2026-07-17) — Boardy uses
[`voyageai/voyage-4-large`](https://huggingface.co/voyageai/voyage-4-large)
([Voyage text embeddings docs](https://docs.voyageai.com/docs/embeddings)).

## Model card (summary)

| Property | Value |
|----------|--------|
| Model id | `voyage-4-large` |
| Provider | Voyage AI (MongoDB) |
| Role | General-purpose / multilingual **retrieval** embeddings |
| Context length | **32,000** tokens |
| Embedding dim | **1024** default; also 256, 512, 2048 (Matryoshka) |
| Typical access | Voyage **API** (`POST /v1/embeddings`), not a local BERT `AutoModel` load |
| Retrieval hint | `input_type`: `query` vs `document` |

## Why this matters vs our frozen BERT baseline

This repo’s first offline baseline (`baselines/bert_frozen`) used
`bert-base-uncased`:

| | Our baseline | Boardy (reported) |
|--|--|--|
| Family | Classic BERT (MLM encoder) | Voyage retrieval embedder |
| Context | 512 tokens (hard truncate) | 32k tokens |
| Dims | 768 | 1024 (default) |
| Fit for long dossiers | Poor — seeker text often ~2k tokens | Can fit nearly full profiles |
| Serving | Local HF weights on MPS/CPU | Usually API (+ key) |

Implication: pair/retrieval metrics from frozen BERT (~chance AUC on hard
negatives) are a **weak proxy** for Boardy’s live system. Re-benchmark with
Voyage before claiming lift over “their” approach.

## Local offline baseline: Voyage-4-nano (done)

Open-weight cousin [`voyageai/voyage-4-nano`](https://huggingface.co/voyageai/voyage-4-nano)
(Apache 2.0, shared Voyage-4 embedding space, 32k context). Implemented under
`baselines/voyage_nano/` via `sentence-transformers` + `encode_query` /
`encode_document` (no Voyage API key). Same pair AUC / retrieval protocol and
field-tagged text as frozen BERT.

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m baselines.voyage_nano.eval \
  --data-dir data \
  --model voyageai/voyage-4-nano \
  --batch-size 4 \
  --max-length 8192 \
  --truncate-dim 1024
```

Defaults: `max_length=8192` (of 32k) for MPS memory; `truncate_dim=1024` to
align with Boardy large’s reported default. Cache + metrics:
`artifacts/voyage_nano/`.

## API baseline: Voyage-4-large (done)

Production model via Voyage API under `baselines/voyage_large/`. Same pair AUC /
retrieval protocol and field-tagged text as frozen BERT / nano. Uses
`input_type=query` (seekers) and `input_type=document` (candidates/corpus),
`output_dimension=1024`, truncation on. Aggressive per-text disk cache + dedupe
under `artifacts/voyage_large/`; rate-limit knobs default to ~2.5M TPM / 1500 RPM
(headroom under 3M TPM / 2k RPM). Free allowance: first **200M tokens**/account
for `voyage-4-large`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
export VOYAGE_API_KEY=pa-...

python -m baselines.voyage_large.eval \
  --data-dir data \
  --model voyage-4-large \
  --output-dimension 1024
```

Writes `artifacts/voyage_large/metrics.json` (+ `usage.json`, embedding cache).

## Offline results (2026-07-17)

Same protocol on `data/dataset_{positive,negative}.json` (100/100 pairs;
retrieval vs ~178 unique matches). Sources: `artifacts/*/metrics.json`.

| Baseline | ROC-AUC | AP | MRR | Top-1 | R@5 | R@10 |
|----------|---------|-----|-----|-------|-----|------|
| Frozen BERT (`bert-base-uncased`, 512) | 0.47 | 0.51 | 0.09 | 0.02 | 0.14 | 0.18 |
| Voyage-4-nano (local, dim 1024) | 0.56 | 0.56 | 0.30 | **0.16** | 0.47 | 0.60 |
| Voyage-4-large (API, dim 1024) | **0.57** | **0.57** | **0.31** | 0.13 | **0.56** | **0.70** |

**Takeaway:** On this labeled set, **`voyage-4-large` performs very close to
`voyage-4-nano`** (shared Voyage-4 embedding space). Large edges nano slightly
on AUC / MRR / R@10; nano is slightly higher on Top-1. Both crush frozen BERT.
For iteration, nano is enough locally; large remains the Boardy-faithful
production reference (~691k API tokens for the first full run; cache thereafter).
See [nano-vs-large-similarity.md](nano-vs-large-similarity.md) for why they
match (shared space, hard-neg ceiling, length, n=100).

Pair discrimination is still weak for all three (cosine gaps tiny) — hard
negatives look similar in embedding space. Headroom for two-tower / fine-tune /
rerank remains large even vs Boardy’s model family.

## Open questions for Boardy / course staff

1. Do they call Voyage via API with `input_type=query` for seekers and
   `document` for candidates?
2. Which `output_dimension` (256 / 512 / 1024 / 2048)?
3. Full dossier text vs curated fields (`searchQuery` + `lookingFor` + …)?
4. Any stages after embedding (filters, rerank, LLM, human connector)?
5. Can the course get API access / rate limits for a fair offline baseline?

## Project follow-ups

- [x] Add local Voyage-4-nano baseline eval parallel to `baselines/bert_frozen`
      (sentence-transformers, cache embeddings, same pair AUC / Recall@K).
- [x] API `voyage-4-large` client baseline (`baselines/voyage_large/`, needs
      `VOYAGE_API_KEY`; free 200M tokens + TPM/RPM throttles).
- [x] Update README with Voyage-nano + Voyage-large API run commands (BERT kept
      as weaker control).
- [ ] Decide whether two-tower / MoE work should beat Voyage offline, or only
      beat frozen BERT.

## References

- Hugging Face (large): https://huggingface.co/voyageai/voyage-4-large
- Hugging Face (nano): https://huggingface.co/voyageai/voyage-4-nano
- Voyage embeddings: https://docs.voyageai.com/docs/embeddings
- Voyage-4 announcement: https://blog.voyageai.com/2026/01/15/voyage-4
