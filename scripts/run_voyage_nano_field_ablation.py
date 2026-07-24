#!/usr/bin/env python3
"""Run the downstream-metric field-ablation experiment: for each of the 8
candidate-profile fields, drop it, rerun voyage-4-nano (local) on the frozen
69-pair real holdout, and compare pair AUC / retrieval MRR against the
unablated baseline. The field whose removal costs the most is the field
voyage-4-nano leans on most for actually matching candidates — not just for
representing them in embedding space (see docs/experiment-graphs-index.md
for the embedding-distance version of this question).

Reuses the existing artifacts/voyage_nano_holdout/ baseline run (unablated,
--holdout-only) as both the comparison point and the source of the
seeker-side embedding cache, so only the candidate-side + corpus encode is
new work per field (8 short local MPS encodes, not 9 full ones).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.text_field_ablation import PROFILE_FIELDS  # noqa: E402

DEFAULT_DATA_DIR = Path("/Users/harsh/Artifacts/dorby-ai/data")
DEFAULT_BASELINE_DIR = Path("/Users/harsh/Artifacts/dorby-ai/artifacts/voyage_nano_holdout")
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "voyage_nano_field_ablation"
DEFAULT_DOC_OUT = ROOT / "docs" / "field-ablation-voyage-nano.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    parser.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--truncate-dim", type=int, default=1024)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a field if artifacts/voyage_nano_field_ablation/<field>/metrics.json "
        "already exists (resume after a partial/killed run)",
    )
    args = parser.parse_args()

    baseline = json.loads((args.baseline_dir / "metrics.json").read_text())
    baseline_auc = baseline["pair"]["roc_auc"]
    baseline_ap = baseline["pair"]["average_precision"]
    baseline_mrr = baseline["retrieval"]["mrr"]
    baseline_ndcg10 = baseline["retrieval"]["ndcg@10"]

    rows = [
        {
            "field": "(none — baseline)",
            "roc_auc": baseline_auc,
            "ap": baseline_ap,
            "mrr": baseline_mrr,
            "ndcg@10": baseline_ndcg10,
            "delta_auc": 0.0,
        }
    ]

    for field in PROFILE_FIELDS:
        artifacts_dir = args.out_root / field
        metrics_path = artifacts_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"\n=== skipping field (already done): {field} ===")
            metrics = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "field": field,
                    "roc_auc": metrics["pair"]["roc_auc"],
                    "ap": metrics["pair"]["average_precision"],
                    "mrr": metrics["retrieval"]["mrr"],
                    "ndcg@10": metrics["retrieval"]["ndcg@10"],
                    "delta_auc": baseline_auc - metrics["pair"]["roc_auc"],
                }
            )
            continue

        print(f"\n=== ablating field: {field} (subprocess) ===")
        # Each field runs as its own subprocess (not an in-process loop): the prior
        # in-process version reloaded the sentence-transformers model on MPS 8x in
        # one long-lived process without releasing the previous copy, which built up
        # memory pressure over the run (batches went from ~1-2s to 40-50s each by
        # field 5-6) and eventually got the process killed outright by macOS before
        # finishing. A fresh subprocess per field guarantees full memory release
        # between fields regardless of what sentence-transformers/torch does or
        # doesn't clean up internally.
        cmd = [
            sys.executable,
            "-m",
            "baselines.voyage_nano_field_ablation.eval",
            "--data-dir",
            str(args.data_dir),
            "--ablate-field",
            field,
            "--artifacts-dir",
            str(artifacts_dir),
            "--baseline-cache-dir",
            str(args.baseline_dir),
            "--model",
            args.model,
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--truncate-dim",
            str(args.truncate_dim),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "field": field,
                "roc_auc": metrics["pair"]["roc_auc"],
                "ap": metrics["pair"]["average_precision"],
                "mrr": metrics["retrieval"]["mrr"],
                "ndcg@10": metrics["retrieval"]["ndcg@10"],
                "delta_auc": baseline_auc - metrics["pair"]["roc_auc"],
            }
        )

    # sort field rows (excluding baseline) by AUC drop, largest first
    baseline_row, field_rows = rows[0], rows[1:]
    field_rows.sort(key=lambda r: r["delta_auc"], reverse=True)

    lines = [
        "# Field ablation — voyage-4-nano, downstream metrics, real 69-pair holdout",
        "",
        "Each row drops exactly one candidate-side profile field and reruns local "
        "voyage-4-nano end to end (seeker side unchanged). `delta_auc` = baseline "
        "AUC minus ablated AUC — positive means removing that field *hurt* pair "
        "classification, i.e. the model was relying on it; negative means removing "
        "it *helped* (the field was adding noise on this population, or its "
        "absence lets other fields dominate more cleanly). Sorted by delta_auc "
        "descending (most load-bearing field first).",
        "",
        "| field | pair AUC | delta AUC vs baseline | AP | MRR | NDCG@10 |",
        "|---|---|---|---|---|---|",
    ]
    b = baseline_row
    lines.append(
        f"| {b['field']} | {b['roc_auc']:.4f} | — | {b['ap']:.4f} | {b['mrr']:.4f} | {b['ndcg@10']:.4f} |"
    )
    for r in field_rows:
        lines.append(
            f"| {r['field']} | {r['roc_auc']:.4f} | {r['delta_auc']:+.4f} | "
            f"{r['ap']:.4f} | {r['mrr']:.4f} | {r['ndcg@10']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"Baseline run: `{args.baseline_dir}` (unablated voyage-4-nano, "
        "`--holdout-only`, 69 real pairs: 29 pos / 40 neg). Ablation runs: "
        f"`{args.out_root}/<field>/metrics.json`."
    )
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.doc_out}")

    print("\n=== ranked by AUC drop (most load-bearing first) ===")
    for r in field_rows:
        print(f"{r['field']:>32s}  delta_auc={r['delta_auc']:+.4f}  auc={r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
