"""Per-model quirks for open-weight sentence-transformers-compatible embedders.

Modern open-weight embedding models mostly load fine through a single generic
SentenceTransformer(...).encode(...) call, but differ on a few axes: whether
they need trust_remote_code, what sentence-transformers `prompt_name` selects
asymmetric query encoding (vs. plain document encoding with no prompt), and
whether they support Matryoshka output truncation. This registry holds those
per-model quirks; unregistered model ids still run, with generic defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    trust_remote_code: bool = False
    query_prompt_name: str | None = None
    supports_truncate_dim: bool = False
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Qwen/Qwen3-Embedding-8B": ModelSpec(
        trust_remote_code=True,
        query_prompt_name="query",
        supports_truncate_dim=True,
        notes="Apache 2.0. ~16GB weights in bf16; use a GPU with >=24GB VRAM.",
    ),
    "Qwen/Qwen3-Embedding-4B": ModelSpec(
        trust_remote_code=True,
        query_prompt_name="query",
        supports_truncate_dim=True,
        notes="Apache 2.0.",
    ),
    "Qwen/Qwen3-Embedding-0.6B": ModelSpec(
        trust_remote_code=True,
        query_prompt_name="query",
        supports_truncate_dim=True,
        notes="Apache 2.0. Small enough to smoke-test locally on MPS/CPU.",
    ),
    "Alibaba-NLP/gte-Qwen2-7B-instruct": ModelSpec(
        trust_remote_code=True,
        query_prompt_name="query",
        supports_truncate_dim=False,
        notes="Apache 2.0.",
    ),
    "BAAI/bge-m3": ModelSpec(
        trust_remote_code=False,
        query_prompt_name=None,
        supports_truncate_dim=False,
        notes="MIT. Symmetric encoding, no query/document prompt distinction.",
    ),
    "intfloat/e5-mistral-7b-instruct": ModelSpec(
        trust_remote_code=False,
        query_prompt_name="query",
        supports_truncate_dim=False,
        notes="MIT.",
    ),
    "Snowflake/snowflake-arctic-embed-m": ModelSpec(
        trust_remote_code=False,
        query_prompt_name="query",
        supports_truncate_dim=False,
        notes="Apache 2.0. Small enough to smoke-test locally on MPS/CPU.",
    ),
    "mixedbread-ai/mxbai-embed-large-v1": ModelSpec(
        trust_remote_code=False,
        query_prompt_name="query",
        supports_truncate_dim=False,
        notes="Apache 2.0.",
    ),
}

_DEFAULT_SPEC = ModelSpec()


def get_model_spec(model_name: str) -> ModelSpec:
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        print(
            f"warning: {model_name!r} is not in MODEL_REGISTRY — running with generic "
            "defaults (no query prompt, trust_remote_code=False, no truncate_dim). "
            "Add an entry to models.py if this model needs special handling."
        )
        return _DEFAULT_SPEC
    return spec


def slugify(model_name: str) -> str:
    """cm5.../Qwen3-Embedding-8B -> qwen_qwen3-embedding-8b, safe for a dir name."""
    return model_name.strip().lower().replace("/", "_").replace(" ", "-")
