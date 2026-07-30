"""Score one model on the real pairs, across all / train / holdout subsets.

Scoring reuses ``twotower.eval.evaluate_pairs`` **unmodified**, which in turn
reuses ``baselines.metrics``. Nothing about how "good" is measured changes here,
so every number is directly comparable to the ablation's holdout tables.

Corpus-size caveat — read before comparing retrieval across subsets
-------------------------------------------------------------------
``evaluate_pairs`` builds its retrieval corpus from the pairs it is handed. The
three subsets therefore rank against pools of different sizes (~178 candidates
for ``all`` vs ~60 for ``holdout``), and a bigger pool is strictly harder. So:

  * **model vs. model on the same subset is exact** — identical queries,
    identical corpus. That is the comparison this experiment exists to make.
  * **subset vs. subset is not comparable on any retrieval metric.** A lower
    MRR on ``all`` than on ``holdout`` means the pool grew, not that the model
    got worse.
  * pair AUC has no corpus dependence, so it is comparable across subsets,
    though the populations still differ.

Each subset's ``n_candidates`` is recorded in the output for this reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from twotower.eval import evaluate_pairs, load_model_for_eval

from eval_real_full.data import Subset, load_real_pairs
from eval_real_full.guard import assert_trained_without_real_pairs

DEFAULT_SUBSETS: tuple[Subset, ...] = ("all", "train", "holdout")


def _load_model_with_dtype(
    *,
    model_name: str,
    adapter_dir: Path | None,
    device: str,
    max_seq_length: int,
    truncate_dim: int,
    torch_dtype: str,
):
    """Load an embedding model in an explicit weight dtype, then apply the adapter.

    ``twotower.eval.load_model_for_eval`` never sets ``torch_dtype``, so it loads
    fp32 — fine for voyage-4-nano (347M, ~1.4GB) but 32GB for an 8B model, which
    does not fit the GPU this eval otherwise runs on. Rather than edit that
    shared function (it is the exact call path every prior published number went
    through), this mirrors it with a dtype argument.

    Prompts are deliberately *not* passed: both models in scope register the
    correct ``query``/``document`` prompts in their own
    ``config_sentence_transformers.json``, byte-identical to what training used
    — verified for voyage-4-nano and Qwen3-Embedding-8B. Passing them here would
    be a second, divergent source of truth.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if torch_dtype not in dtypes:
        raise KeyError(f"unknown torch_dtype {torch_dtype!r}; choices: {sorted(dtypes)}")

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        truncate_dim=truncate_dim,
        device=device,
        model_kwargs={"torch_dtype": dtypes[torch_dtype]},
    )
    model.max_seq_length = max_seq_length
    if adapter_dir is not None:
        model.load_adapter(str(adapter_dir))
    return model


def run_eval(
    *,
    data_dir: Path,
    split_path: Path,
    model_name: str = "voyageai/voyage-4-nano",
    adapter_dir: Path | None = None,
    label: str = "frozen",
    subsets: Sequence[Subset] = DEFAULT_SUBSETS,
    batch_size: int = 8,
    max_length: int = 4096,
    truncate_dim: int = 1024,
    device: str = "cuda",
    torch_dtype: str | None = None,
) -> dict[str, Any]:
    """Evaluate one model configuration over each requested subset.

    ``torch_dtype=None`` keeps the exact shared ``load_model_for_eval`` path that
    every prior published number used (fp32). Pass ``"bfloat16"`` only for
    backbones too large to load fp32 — it changes numerics slightly, so it is
    recorded in the output and must not be mixed within a comparison.
    """
    adapter_meta: dict[str, Any] | None = None
    if adapter_dir is not None:
        # Fails loudly rather than producing a flattering leaked number.
        adapter_meta = assert_trained_without_real_pairs(adapter_dir)

    if torch_dtype is None:
        model = load_model_for_eval(
            model_name=model_name,
            adapter_dir=adapter_dir,
            device=device,
            max_seq_length=max_length,
            truncate_dim=truncate_dim,
        )
    else:
        model = _load_model_with_dtype(
            model_name=model_name,
            adapter_dir=adapter_dir,
            device=device,
            max_seq_length=max_length,
            truncate_dim=truncate_dim,
            torch_dtype=torch_dtype,
        )

    out: dict[str, Any] = {
        "label": label,
        "model_name": model_name,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "adapter_provenance": {
            "rows_path": adapter_meta.get("rows_path") if adapter_meta else None,
            "negatives_per_anchor": adapter_meta.get("negatives_per_anchor") if adapter_meta else None,
            "train_batch_size": (adapter_meta.get("config", {}) or {}).get("train_batch_size")
            if adapter_meta
            else None,
        },
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "torch_dtype": torch_dtype or "float32",
        "subsets": {},
    }

    for subset in subsets:
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        print(
            f"\n=== {label} | subset={subset} | n={len(ps.pairs)} "
            f"(pos={ps.n_pos}, neg={ps.n_neg}) | corpus={ps.n_candidates} candidates ==="
        )
        metrics = evaluate_pairs(
            model,
            ps.pairs,
            batch_size=batch_size,
            model_name=model_name,
            device=device,
            max_length=max_length,
            truncate_dim=truncate_dim,
        )
        metrics["subset"] = subset
        metrics["n_candidates"] = ps.n_candidates
        metrics["real_data_hash"] = ps.combined_hash
        out["subsets"][subset] = metrics
        print(
            f"    pair AUC {metrics['pair']['roc_auc']:.4f} | "
            f"MRR {metrics['retrieval']['mrr']:.4f} | "
            f"R@1 {metrics['retrieval']['recall@1']:.4f}"
        )
    return out


def write_metrics(metrics: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {path}")
    return path
