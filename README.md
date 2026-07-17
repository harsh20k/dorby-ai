# Dorby AI

RecSys course project (Prof. Ga Wu) for industry partner **Boardy AI**. The goal
is to explore multiple approaches to improve Boardy AI's recommendation
performance, starting with:

- **Two-tower model** trained on their dataset
- Try **Mixture of Experts** architecture
- Possibly try - **Student - Teacher** architecture

Frozen Voyage is the baseline; official Voyage fine-tune is enterprise/sales-only
(not self-serve). `voyage-4-large` is closed (no DIY FT); `voyage-4-nano` has open
weights (SentenceTransformers / LoRA OK) but ~200 hard pairs overfit easily.
Grow hard labels + two-tower first; revisit Voyage FT / student–teacher
(large → small) when data scales (~1k+) and frozen baselines plateau.
Student–teacher = optional distillation for cost/latency, not a substitute for
hard intro labels.

## Overview

Boardy’s production embeddings are **Voyage `voyage-4-large`** (32k context),
not classic BERT — see [docs/boardy-embedding-model.md](docs/boardy-embedding-model.md).
This repo still includes a frozen `bert-base-uncased` offline control; we will
benchmark against Voyage and against two-tower / MoE / student–teacher variants.

### Baseline snapshot (offline, 2026-07-17)

| Baseline | ROC-AUC | MRR | Top-1 | R@10 |
|----------|---------|-----|-------|------|
| Frozen BERT | 0.47 | 0.09 | 0.02 | 0.18 |
| Voyage-4-nano (local) | 0.56 | 0.30 | 0.16 | 0.60 |
| Voyage-4-large (API) | 0.57 | 0.31 | 0.13 | 0.70 |

**Large ≈ nano** on this dataset (shared Voyage-4 space); both far above BERT.
Details in the docs page above.

## Data prep

Dedupe pair datasets into a unique-user catalog (canonical `userContactFile`
by highest `userContactFileVersion`):

```bash
python scripts/build_unique_users.py --data-dir data
# writes data/unique_users.json
```

Browse the unique-user catalog in a self-contained HTML page (JSON embedded, no server needed):

```bash
python scripts/build_unique_users_browser.py
# writes data/unique_users_browser.html — open in any browser
```

Per-field BERT token counts (`bert-base-uncased`, `add_special_tokens=False`) plus a filterable/sortable table browser (includes a top summary table: min / mean / median / max per field across all users):

```bash
# use project venv so transformers + cached tokenizer are available
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/build_unique_users_token_counts.py
# writes data/unique_users_token_counts.json
#      and data/unique_users_token_counts_browser.html — open in any browser

# rebuild HTML only (no tokenizer) after template changes:
.venv/bin/python scripts/build_unique_users_token_counts.py --html-only
```

## Baseline eval (frozen BERT)

Offline bi-encoder baseline: `bert-base-uncased`, mean-pool + L2, cosine
similarity. Runs on Apple MPS when available (else CPU).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# expects data/dataset_positive.json and data/dataset_negative.json
python -m baselines.bert_frozen.eval \
  --data-dir data \
  --model bert-base-uncased \
  --batch-size 16 \
  --max-length 512
```

Writes embeddings cache + `artifacts/bert_frozen/metrics.json` (pair ROC-AUC /
AP / best-F1, retrieval MRR + NDCG/Precision/Recall@K, intent + neg-hardness
slices). See [docs/baseline-metrics.md](docs/baseline-metrics.md). Full metrics:
[docs/baseline-results-all.md](docs/baseline-results-all.md).

## Baseline eval (Voyage-4-nano)

Local open-weight cousin of Boardy’s API `voyage-4-large`, same Voyage-4 embedding
space. Uses `sentence-transformers` with `encode_query` / `encode_document`,
shared field-tagged text from `baselines.bert_frozen.text`. Default
`max_length=8192` (model supports 32k; capped for MPS memory) and
`truncate_dim=1024` (Boardy large default).

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

Writes embeddings cache + `artifacts/voyage_nano/metrics.json` (same expanded
metrics as frozen BERT; see [docs/baseline-metrics.md](docs/baseline-metrics.md)).
Full metrics: [docs/baseline-results-all.md](docs/baseline-results-all.md).
First run downloads HF weights.

Note: `voyage-4-nano` remote code needs `transformers>=4.51,<5` (transformers 5.x
currently fails with `config_class` None on load). Cold MPS encode ~40+ min;
re-runs hit `artifacts/voyage_nano/` cache.

## Baseline eval (Voyage-4-large API)

Boardy production model via Voyage API (`voyage-4-large`, `output_dimension=1024`,
`input_type=query` for seekers / `document` for candidates). Requires
`VOYAGE_API_KEY`. Per-text disk cache under `artifacts/voyage_large/` so re-runs
cost ~0 tokens. Free tier: first **200M tokens**/account for voyage-4-large;
throttle defaults leave headroom under 3M TPM / 2k RPM.

```bash
source .venv/bin/activate
pip install -r requirements.txt
export VOYAGE_API_KEY=pa-...   # https://docs.voyageai.com/docs/api-key-and-installation

python -m baselines.voyage_large.eval \
  --data-dir data \
  --model voyage-4-large \
  --output-dimension 1024
```

Optional: `--batch-size 16`, `--tpm-limit 2500000`, `--rpm-limit 1500` (or env
`VOYAGE_TPM_LIMIT` / `VOYAGE_RPM_LIMIT`). Writes `artifacts/voyage_large/metrics.json`
+ per-embedding cache (same expanded metrics as BERT/nano; see
[docs/baseline-metrics.md](docs/baseline-metrics.md)). Full metrics:
[docs/baseline-results-all.md](docs/baseline-results-all.md).
