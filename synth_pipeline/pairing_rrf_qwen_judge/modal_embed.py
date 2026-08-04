"""Modal GPU embedding for the RRF pairing pipeline.

Qwen3-Embedding-8B is ~16GB in bf16 — it does not fit comfortably on local MPS,
and an A10G (24GB) OOMs on any 7-8B model, so the default here is A100-40GB, the
same lesson recorded in ``docs/hf-embedding-baseline-findings.md``.

Both sides go over in a single call. At batch scale the payload is small — a few
hundred vectors at 4096 dims is single-digit megabytes — so returning arrays
directly avoids the volume-download dance that ``modal volume get`` makes awkward
for whole directories on this CLI version.

Invoked automatically by ``run.py`` when ``--embed-backend modal`` is set; the
local path stays the default so small models and smoke tests need no GPU.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-pairing-rrf-embed"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared with the other baselines

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "accelerate>=0.30",
        "numpy>=1.26.0",
        "einops",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("baselines", "synth_pipeline")
)

hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": hf_cache},
    timeout=60 * 60,
)
def embed_remote(
    seeker_texts: list[str],
    candidate_texts: list[str],
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    batch_size: int = 2,
    max_length: int = 4096,
    truncate_dim: int | None = None,
) -> tuple[bytes, bytes]:
    """Encode both sides and return the two arrays as ``.npy`` bytes."""
    import io

    import numpy as np

    from baselines.hf_embedding.encode import get_encoder_class

    encoder_cls = get_encoder_class(model_name)
    encoder = encoder_cls(
        model_name,
        device="cuda",
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir="/tmp/enc_cache",
    )
    print(f"encoding {len(seeker_texts)} seeker + {len(candidate_texts)} candidate texts")

    seeker = encoder.encode(seeker_texts, role="query", batch_size=batch_size)
    cand = encoder.encode(candidate_texts, role="document", batch_size=batch_size)
    hf_cache.commit()

    def to_bytes(arr: "np.ndarray") -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr.astype(np.float32))
        return buf.getvalue()

    print(f"done: seeker {seeker.shape}, candidate {cand.shape}")
    return to_bytes(seeker), to_bytes(cand)


def embed_via_modal(
    seeker_texts: list[str],
    candidate_texts: list[str],
    *,
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    batch_size: int = 2,
    max_length: int = 4096,
    truncate_dim: int | None = None,
):
    """Call the remote function from local code and return two NumPy arrays."""
    import io

    import numpy as np

    with app.run():
        seeker_bytes, cand_bytes = embed_remote.remote(
            seeker_texts,
            candidate_texts,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            truncate_dim=truncate_dim,
        )
    return np.load(io.BytesIO(seeker_bytes)), np.load(io.BytesIO(cand_bytes))


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": hf_cache},
    timeout=60 * 60,
)
def embed_isolated_remote(
    texts: list[str],
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    batch_size: int = 2,
    max_length: int = 4096,
    truncate_dim: int | None = None,
) -> bytes:
    """Encode a flat list of isolated field/section texts, all as ``role="document"``.

    These rows are points in profile space, not retrieval queries — same
    asymmetry choice ``voyage_nano_field_isolation/modal_embed_space.py`` made
    for the real-holdout version of this experiment.
    """
    import io

    import numpy as np

    from baselines.hf_embedding.encode import get_encoder_class

    encoder_cls = get_encoder_class(model_name)
    encoder = encoder_cls(
        model_name,
        device="cuda",
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir="/tmp/enc_cache",
    )
    print(f"encoding {len(texts)} isolated field/section texts")
    vecs = encoder.encode(texts, role="document", batch_size=batch_size)
    hf_cache.commit()

    buf = io.BytesIO()
    np.save(buf, np.asarray(vecs, dtype=np.float32))
    print(f"done: {vecs.shape}")
    return buf.getvalue()


def embed_isolated_via_modal(
    texts: list[str],
    *,
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    batch_size: int = 2,
    max_length: int = 4096,
    truncate_dim: int | None = None,
):
    """Call the remote isolated-embedding function and return one NumPy array."""
    import io

    import numpy as np

    with app.run():
        vec_bytes = embed_isolated_remote.remote(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            truncate_dim=truncate_dim,
        )
    return np.load(io.BytesIO(vec_bytes))


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-Embedding-8B") -> None:
    """Smoke-test the GPU path without running the pipeline."""
    seeker, cand = embed_remote.remote(
        ["a founder raising a seed round"], ["an early-stage fintech investor"],
        model_name=model,
    )
    import io

    import numpy as np

    s = np.load(io.BytesIO(seeker))
    c = np.load(io.BytesIO(cand))
    print(f"seeker {s.shape} candidate {c.shape} cosine={float(s[0] @ c[0]):.4f}")
