"""Local CLI for the baseline all-200 eval (companion to modal_baseline_eval.py).

Exists for one specific reason: **TF-IDF cannot be scored on Modal and stay
comparable to its published row.**

``baselines/tfidf`` fits with ``max_features=20000``, and on this data the
vocabulary lands at exactly 20000 — the truncation binds. sklearn then drops
features by document frequency, and the tie-break at that cutoff is not stable
across environments. The result is that the same code, same data, same
scikit-learn 1.9.0 and numpy 2.4.6 produce holdout AUC 0.5922 / MRR 0.2475 /
R@1 0.1379 in this repo's venv (matching ``artifacts/tfidf_holdout``
digit-for-digit) but 0.5914 / 0.2653 / 0.1724 in a Modal container. Pair AUC
barely moves; the rankings flip, because a different vocabulary slice changes
which terms carry IDF weight. No ties exist in the score matrix, so this is not
argsort tie-breaking — it is a genuinely different fitted vectorizer.

Neural baselines have no such problem: frozen BERT reproduced its published
holdout row exactly on Modal, so those runs belong there.

    python -m eval_real_full.run_baseline --config tfidf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval_real_full.baseline_eval import run_baseline_eval, write_metrics

CONFIGS: dict[str, dict] = {
    "tfidf": {"kind": "tfidf", "model": None, "device": "cpu"},
    "bert": {"kind": "bert", "model": "bert-base-uncased", "device": "cpu", "max_length": 512},
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local baseline eval on the real pairs")
    p.add_argument("--config", choices=sorted(CONFIGS), default="tfidf")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--subsets", default="all,train,holdout")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/eval_real_full/real200_baselines_local"),
    )
    args = p.parse_args(argv)

    spec = CONFIGS[args.config]
    metrics = run_baseline_eval(
        kind=spec["kind"],
        data_dir=args.data_dir,
        split_path=args.split_path or args.data_dir / "synthetic" / "seed_split.json",
        label=args.config,
        model_name=spec["model"],
        subsets=tuple(s.strip() for s in args.subsets.split(",") if s.strip()),
        max_length=spec.get("max_length", 8192),
        device=spec["device"],
        cache_dir=Path("/tmp") / f"eval_real_full_local_{args.config}",
    )
    metrics["library_versions"] = _library_versions()
    write_metrics(metrics, args.out_dir / args.config)
    return 0


def _library_versions() -> dict:
    import importlib

    out = {}
    for mod in ("sklearn", "numpy", "scipy", "torch"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:
            out[mod] = None
    out["python"] = sys.version.split()[0]
    return out


if __name__ == "__main__":
    sys.exit(main())
