"""Score Voyage-4-large on all 200 real pairs, via the shared metric path.

Voyage-4-large is an API model, so it has no ``SentenceTransformer`` and cannot
go through ``eval_real_full/modal_eval.py``. But the metrics must stay identical
to every other model's, or the comparison is meaningless.

The solution is a thin adapter: ``_VoyageLargeShim`` exposes the two methods
``twotower.eval.evaluate_pairs`` actually calls — ``encode_query`` and
``encode_document`` — and delegates to ``baselines.voyage_large.encode``
unchanged. ``evaluate_pairs`` is then used exactly as-is, so pair/retrieval/slice
metrics come from the same ``baselines.metrics`` code as the nano and Qwen runs.

Runs locally (API calls, no GPU) and reuses the existing on-disk embedding cache
under ``artifacts/voyage_large/emb/``, so most texts cost nothing.

    python -m eval_real_full.voyage_large_eval --data-dir data

Note ``output_dimension=1024`` matches the existing published Voyage-large
baseline exactly, so this run is comparable to it. Voyage-4-large's native width
is larger; 1024 is a deliberate, pre-existing choice, not introduced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from baselines.voyage_large.encode import VoyageLargeEncoder, estimate_tokens
from twotower.eval import evaluate_pairs

from eval_real_full.data import Subset, load_real_pairs
from eval_real_full.eval import DEFAULT_SUBSETS, write_metrics


class _VoyageLargeShim:
    """Duck-types the two methods ``evaluate_pairs`` needs onto the API encoder."""

    def __init__(self, encoder: VoyageLargeEncoder) -> None:
        self._encoder = encoder

    def _encode(self, texts: Sequence[str], input_type: str, batch_size: int | None) -> np.ndarray:
        return self._encoder.encode(
            list(texts),
            input_type=input_type,  # type: ignore[arg-type]
            batch_size=batch_size,
            show_progress=True,
            label=input_type,
        )

    # evaluate_pairs passes normalize_embeddings/convert_to_numpy/show_progress_bar;
    # the Voyage encoder already returns L2-normalised float32 numpy, and
    # encode_role re-normalises anyway, so they are accepted and ignored.
    def encode_query(self, texts, batch_size=None, **_: Any) -> np.ndarray:
        return self._encode(texts, "query", batch_size)

    def encode_document(self, texts, batch_size=None, **_: Any) -> np.ndarray:
        return self._encode(texts, "document", batch_size)


def run(
    *,
    data_dir: Path,
    split_path: Path,
    artifacts_dir: Path,
    out_dir: Path,
    model_name: str = "voyage-4-large",
    output_dimension: int = 1024,
    batch_size: int = 16,
    subsets: Sequence[Subset] = DEFAULT_SUBSETS,
) -> dict[str, Any]:
    encoder = VoyageLargeEncoder(
        model_name=model_name,
        output_dimension=output_dimension,
        truncation=True,
        cache_dir=artifacts_dir,
        batch_size=batch_size,
    )
    model = _VoyageLargeShim(encoder)

    out: dict[str, Any] = {
        "label": "voyage_large",
        "model_name": model_name,
        "adapter_dir": None,
        "output_dimension": output_dimension,
        "batch_size": batch_size,
        "subsets": {},
    }

    for subset in subsets:
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        texts = (
            [p.seeker_text for p in ps.pairs]
            + [p.candidate_text for p in ps.pairs]
        )
        print(
            f"\n=== voyage_large | subset={subset} | n={len(ps.pairs)} "
            f"(pos={ps.n_pos}, neg={ps.n_neg}) | corpus={ps.n_candidates} | "
            f"~{estimate_tokens(texts):,} tokens pre-cache ==="
        )
        metrics = evaluate_pairs(
            model,  # type: ignore[arg-type]
            ps.pairs,
            batch_size=batch_size,
            model_name=model_name,
            device="api",
            max_length=32000,
            truncate_dim=output_dimension,
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

    out["encoder_stats"] = encoder.stats()
    print(f"\nencoder stats: {out['encoder_stats']}")
    write_metrics(out, out_dir)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/voyage_large"))
    p.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/eval_real_full/real200_voyage_large")
    )
    p.add_argument("--output-dimension", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(argv)
    split_path = args.split_path or args.data_dir / "synthetic" / "seed_split.json"

    run(
        data_dir=args.data_dir,
        split_path=split_path,
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
        output_dimension=args.output_dimension,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
