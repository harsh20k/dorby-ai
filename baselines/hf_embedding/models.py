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
    loader: str = "sentence_transformers"  # or "flagembedding_icl"
    uses_encode_methods: bool = False  # calls model.encode_query()/.encode_document()
    # instead of model.encode(texts, prompt_name=...) — same convention as
    # voyage-4-nano; some sentence-transformers-compatible models (e.g.
    # zeroentropy/zembed-1-embedding) expose asymmetric encoding this way instead.
    requires_legacy_transformers: bool = False  # pins transformers==4.44.2 +
    # sentence-transformers==3.0.1 on Modal — needed by trust_remote_code models
    # whose custom modeling file predates transformers' Cache API refactor
    # (e.g. calls Cache.get_usable_length(), removed in newer transformers).
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
    "nvidia/NV-Embed-v2": ModelSpec(
        trust_remote_code=True,
        query_prompt_name=None,
        supports_truncate_dim=False,
        requires_legacy_transformers=True,
        notes=(
            "CC-BY-NC-4.0 — NON-COMMERCIAL ONLY, benchmark reference only, cannot "
            "ship in production. ~15GB weights; needs >=24GB VRAM, prefer A100-40GB. "
            "requires_legacy_transformers=True: NVIDIA's custom modeling code calls "
            "Cache.get_usable_length(), removed from transformers>=4.51 during the "
            "Cache API refactor — confirmed via AttributeError on transformers 4.57. "
            "Runs on a pinned transformers==4.44.2 + sentence-transformers==3.0.1 "
            "Modal image instead of the shared default image. query_prompt_name=None: "
            "NV-Embed-v2 ships no sentence-transformers `prompts` dict at all (empty "
            "keys, confirmed via ValueError) — it expects a manually-prepended "
            "`instruction=` string via its own custom .encode(), not ST's prompt_name. "
            "We run it symmetrically (no query/document distinction) as an "
            "approximation; this understates its true instruction-tuned performance."
        ),
    ),
    "BAAI/bge-en-icl": ModelSpec(
        trust_remote_code=True,
        query_prompt_name="query",
        supports_truncate_dim=False,
        loader="flagembedding_icl",
        notes=(
            "MIT. 7B, Mistral-7B-based. Not sentence-transformers-loadable — "
            "confirmed via ValueError('Unrecognized processing class') — this model "
            "is designed for BAAI's own FlagEmbedding library (FlagICLModel), which "
            "also supports prepending in-context few-shot examples to the query "
            "(we run zero-shot, no examples). loader='flagembedding_icl' routes to "
            "FlagICLEncoder in encode.py instead of HFEmbeddingEncoder."
        ),
    ),
    "zeroentropy/zembed-1-embedding": ModelSpec(
        trust_remote_code=True,
        query_prompt_name=None,
        supports_truncate_dim=True,
        uses_encode_methods=True,
        notes=(
            "Apache 2.0. 4B params, Qwen3-4B base. ~8GB weights in bf16 — fits "
            "comfortably on A10G, no need for A100. Matryoshka dims "
            "2560/1280/640/320/160/80/40. uses_encode_methods=True: calls "
            "model.encode_query()/.encode_document() (same convention as "
            "voyage-4-nano), not ST's prompt_name."
        ),
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
