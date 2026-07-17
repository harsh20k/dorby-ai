# Dorby AI

RecSys course project (Prof. Ga Wu) for industry partner **Boardy AI**. The goal
is to explore multiple approaches to improve Boardy AI's recommendation
performance, starting with:

- **Two-tower model** trained on their dataset
- Try **Mixture of Experts** architecture
- Possibly try - **Student - Teacher** architecture 

## Overview

Boardy’s production embeddings are **Voyage `voyage-4-large`** (32k context),
not classic BERT — see [docs/boardy-embedding-model.md](docs/boardy-embedding-model.md).
This repo still includes a frozen `bert-base-uncased` offline control; we will
benchmark against Voyage and against two-tower / MoE / student–teacher variants.

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
AP, retrieval MRR + Recall@1/5/10).

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

Writes embeddings cache + `artifacts/voyage_nano/metrics.json` (same pair /
retrieval protocol as frozen BERT). First run downloads HF weights.

Note: `voyage-4-nano` remote code needs `transformers>=4.51,<5` (transformers 5.x
currently fails with `config_class` None on load). Cold MPS encode ~40+ min;
re-runs hit `artifacts/voyage_nano/` cache.
