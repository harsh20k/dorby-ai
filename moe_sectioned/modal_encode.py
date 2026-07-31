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
    # Hard cost cap, not a performance guess. A100-40GB is ~$2.50/hr, so a 20
    # minute ceiling bounds a runaway run at roughly $0.85 — well inside the $3
    # budget this was authorized under. Expected work is ~830 unique texts,
    # which should finish in 4-15 min including the one-time 16 GB model pull.
    timeout=20 * 60,
)
def encode_texts(
    section_texts: list[str],
    candidate_texts: list[str],
    batch_size: int = 2,
    max_seq_length: int = 4096,
) -> dict:
    """Encode unique texts, write vectors.npy + index.json into the volume.

    **Loads in bfloat16, and that is not an optimization — it is required.** The
    first attempt at this run left dtype to the default and OOMed on an
    A100-40GB: an 8B-parameter model in fp32 is ~32 GB of weights before a single
    activation, and the traceback showed 36.68 GiB allocated with 942 MiB free.
    bf16 halves the weights to ~16 GB and leaves room to actually compute.
    ``baselines/hf_embedding/encode.py`` already did this; this script did not,
    which is exactly the kind of thing copying a working loader would have
    avoided.

    ``max_seq_length`` matters for the same reason. Qwen3 accepts 32k tokens, and
    attention memory grows with the square of the sequence, so leaving it
    unbounded lets one long candidate profile blow up the batch. Our longest text
    is ~8.6k characters (~2.2k tokens), so 4096 truncates nothing in practice.
    """
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    # Qwen3-Embedding is **asymmetric**: the registry in
    # baselines/hf_embedding/models.py sets query_prompt_name="query" for it, and
    # the query side must carry that instruction prefix while documents must not.
    # Here the seeker's ask is the query and the candidate profile is the
    # document. The first version of this script encoded both with no prompt at
    # all; on that cache the sectioned model scored 0.5378 mean AUC across five
    # seeds versus TF-IDF's 0.6467, which is not a fair test of the model.
    sec_uniq = list(dict.fromkeys(section_texts))
    cand_uniq = list(dict.fromkeys(candidate_texts))
    print(
        f"sections {len(section_texts)} -> {len(sec_uniq)} unique | "
        f"candidates {len(candidate_texts)} -> {len(cand_uniq)} unique"
    )

    model = SentenceTransformer(
        MODEL_ID,
        trust_remote_code=True,
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = max_seq_length
    def _run(items: list[str], **kw) -> "np.ndarray":
        return model.encode(
            items,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            **kw,
        ).astype("float32")

    sec_vecs = _run(sec_uniq, prompt_name="query")
    cand_vecs = _run(cand_uniq)
    vecs = np.concatenate([sec_vecs, cand_vecs], axis=0)

    out = Path("/emb/qwen3")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vecs)
    (out / "index.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "dim": int(vecs.shape[1]),
                # Keyed on role + text, because the same string encoded as a
                # query and as a document is now two different vectors.
                "hash_to_row": {
                    **{
                        hashlib.sha256(f"query\x00{t}".encode("utf-8")).hexdigest(): i
                        for i, t in enumerate(sec_uniq)
                    },
                    **{
                        hashlib.sha256(f"document\x00{t}".encode("utf-8")).hexdigest():
                        len(sec_uniq) + i
                        for i, t in enumerate(cand_uniq)
                    },
                },
            }
        )
    )
    emb_volume.commit()
    return {
        "unique_sections": len(sec_uniq),
        "unique_candidates": len(cand_uniq),
        "dim": int(vecs.shape[1]),
    }


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
    sections: list[str] = []
    candidates: list[str] = []
    for pop in (pool, holdout):
        for pair in pop.rows:
            candidates.append(candidate_to_text(pair.get("matchContactFile") or {}))
            for sec in sections_for_pair(
                pair, min_chars=min_section_chars, max_sections=max_sections
            ):
                sections.append(sec.text)

    print(f"collected {len(sections)} sections, {len(candidates)} candidates")
    print(encode_texts.remote(sections, candidates))
    print(
        f"\nnow: modal volume get {VOLUME} qwen3 "
        "./artifacts/moe_sectioned/embeddings"
    )
