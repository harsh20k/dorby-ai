# Dorby AI

RecSys course project (Prof. Ga Wu) for industry partner **Boardy AI**. The goal
is to explore multiple approaches to improve Boardy AI's recommendation
performance, starting with:

- **Two-tower model** trained on their dataset
- Try **Mixture of Experts** architecture
- Possibly try - **Student - Teacher** architecture 

## Overview

Boardy AI currently relies on generic (non-fine-tuned) BERT embeddings for
recommendations. This project benchmarks that baseline against a two-tower
retrieval architecture and task-specific trained embeddings, evaluating
whichever combination yields the best offline/online recommendation metrics.

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