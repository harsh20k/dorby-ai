"""Encode section and candidate texts with Qwen3-Embedding-8B on a Modal GPU.

**This is the only paid step in the experiment, and it is not required to run
anything.** The whole pipeline runs locally and for free on ``--encoder tfidf``;
this exists to swap in the one open-weight model measured to beat Voyage-4-large
on this task (pair ROC-AUC 0.6595 vs 0.6086, see
``docs/hf-embedding-baseline-findings.md``).

Run it only after deciding the local result justifies the spend::

    modal run moe_sectioned/modal_encode.py
    modal volume get dorby-moe-sectioned-emb qwen3 ./artifacts/moe_sectioned/embeddings
    PYTHONPATH=. python -m moe_sectioned.experiment --encoder qwen3 --run-id sec_002

Sizing note from the existing baseline runs: an A10G (24 GB) OOMs on any 7-8B
model, so this pins A100-40GB. Texts are deduplicated before encoding — a
candidate profile is repeated once per section, so dedup removes roughly 80% of
the work.

The output is content-addressed: ``index.json`` maps SHA-256 of each exact text
to its row in ``vectors.npy``. That means a changed section splitter cannot
silently reuse stale vectors — the hash misses and ``Qwen3Backend`` raises
rather than serving the wrong embedding.
"""

from __future__ import annotations

import modal

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
VOLUME = "dorby-moe-sectioned-emb"
HF_CACHE = "dorby-hf-cache"

app = modal.App("dorby-moe-sectioned-encode")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0",
        "transformers>=4.51,<5",
        "torch",
        "numpy",
    )
)

emb_volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_volume = modal.Volume.from_name(HF_CACHE, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/emb": emb_volume, "/root/.cache/huggingface": hf_volume},
    timeout=60 * 60,
)
def encode_texts(texts: list[str], batch_size: int = 2) -> dict:
    """Encode unique texts, write vectors.npy + index.json into the volume."""
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    from sentence_transformers import SentenceTransformer

    uniq = list(dict.fromkeys(texts))  # order-preserving dedup
    print(f"{len(texts)} texts -> {len(uniq)} unique")

    model = SentenceTransformer(MODEL_ID, trust_remote_code=True, device="cuda")
    vecs = model.encode(
        uniq,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    out = Path("/emb/qwen3")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vecs)
    (out / "index.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "dim": int(vecs.shape[1]),
                "hash_to_row": {
                    hashlib.sha256(t.encode("utf-8")).hexdigest(): i
                    for i, t in enumerate(uniq)
                },
            }
        )
    )
    emb_volume.commit()
    return {"unique_texts": len(uniq), "dim": int(vecs.shape[1])}


@app.local_entrypoint()
def main(data_dir: str = "data", max_sections: int = 8, min_section_chars: int = 40):
    """Collect every text the experiment will need, then encode them once."""
    import sys

    sys.path.insert(0, ".")
    from baselines.bert_frozen.text import candidate_to_text

    from moe_sectioned.data import load_real
    from moe_sectioned.sections import sections_for_pair

    from pathlib import Path

    pool, holdout = load_real(Path(data_dir))
    texts: list[str] = []
    for pop in (pool, holdout):
        for pair in pop.rows:
            texts.append(candidate_to_text(pair.get("matchContactFile") or {}))
            for s in sections_for_pair(
                pair, min_chars=min_section_chars, max_sections=max_sections
            ):
                texts.append(s.text)

    print(f"collected {len(texts)} texts ({len(set(texts))} unique)")
    print(encode_texts.remote(texts))
    print(
        f"\nnow: modal volume get {VOLUME} qwen3 "
        "./artifacts/moe_sectioned/embeddings"
    )
