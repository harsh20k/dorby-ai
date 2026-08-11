"""Copy a `pairing_rrf` batch into the MoE experiment's own namespace.

**Read-only on the source, always.** The source batch under
``artifacts/pairing_rrf/<batch_id>/`` is opened, never written. This module
records a SHA-256 over the staged pairs so a later run can prove the source did
not change underneath it, and ``verify`` re-checks it on demand.

Why copy rather than read in place: the RRF batches belong to the synthetic-data
generation track, which is still being iterated on. An experiment that reads them
live would silently change its own inputs whenever that track re-runs. The copy
freezes what this experiment was actually trained on.

    PYTHONPATH=. .venv/bin/python -m moe_reranker.import_rrf --batch-id rrf_003
    PYTHONPATH=. .venv/bin/python -m moe_reranker.import_rrf --batch-id rrf_003 --verify

**What these labels are, and are not.** They are one LLM judge's opinion
(`google/gemini-3.1-flash-lite`, naive framing), not real human accept/decline
outcomes. On the real holdout that judge scores 0.6358 pair AUC and 0.5942
accuracy on the hard slice, so roughly 4 in 10 of these labels are wrong on hard
pairs. That makes the batch usable as an **auxiliary task** and as within-seeker
ranking structure — never as a substitute for the 200 real pairs, and never as
something to promote into ``data/dataset_*.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_ROOT = Path("artifacts/pairing_rrf")
DEFAULT_DEST_ROOT = Path("artifacts/moe_reranker/data")


def _staged_files(batch_dir: Path) -> list[Path]:
    staged = batch_dir / "staged"
    if not staged.is_dir():
        raise FileNotFoundError(f"no staged/ directory under {batch_dir}")
    return sorted(staged.glob("*.json"))


def _digest(files: list[Path]) -> str:
    """SHA-256 over file names + bytes, order-stable."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_seeker: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in pairs:
        by_seeker[p["userContactId"]][0 if p["label"] == "pos" else 1] += 1
    both = [s for s, (a, b) in by_seeker.items() if a and b]
    return {
        "n_pairs": len(pairs),
        "n_pos": sum(1 for p in pairs if p["label"] == "pos"),
        "n_neg": sum(1 for p in pairs if p["label"] == "neg"),
        "n_seekers": len(by_seeker),
        "n_seekers_with_both_classes": len(both),
        "within_seeker_triplets": sum(
            by_seeker[s][0] * by_seeker[s][1] for s in both
        ),
    }


def import_batch(
    batch_id: str,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    dest_root: Path = DEFAULT_DEST_ROOT,
) -> dict[str, Any]:
    src = source_root / batch_id
    files = _staged_files(src)
    if not files:
        raise RuntimeError(f"{src}/staged contains no pairs")

    source_manifest: dict[str, Any] = {}
    mpath = src / "manifest.json"
    if mpath.exists():
        source_manifest = json.loads(mpath.read_text())

    pairs: list[dict[str, Any]] = []
    for f in files:
        env = json.loads(f.read_text())
        pair = env.get("pair") or {}
        if not pair.get("userContactId") or not pair.get("matchContactId"):
            continue
        pairs.append(
            {
                "pair_key": f.stem,
                "label": env.get("label"),
                "userContactId": pair["userContactId"],
                "matchContactId": pair["matchContactId"],
                "searchQuery": pair.get("searchQuery"),
                "userContactFile": pair.get("userContactFile"),
                "matchContactFile": pair.get("matchContactFile"),
            }
        )

    summary = _summarize(pairs)
    record = {
        "batch_id": batch_id,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src),
        "source_staged_files": len(files),
        "source_sha256": _digest(files),
        "labeler": source_manifest.get("labeler"),
        "source_counts": source_manifest.get("counts"),
        "summary": summary,
        "labels_are": (
            "LLM judge opinion, NOT real human accept/decline outcomes. Usable as "
            "an auxiliary task and for within-seeker ranking structure; never as a "
            "substitute for the 200 real pairs, never promoted into data/dataset_*.json."
        ),
    }

    dest = dest_root / batch_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "pairs.json").write_text(json.dumps(pairs))
    (dest / "provenance.json").write_text(json.dumps(record, indent=2))
    return record


def verify(
    batch_id: str,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    dest_root: Path = DEFAULT_DEST_ROOT,
) -> bool:
    """True if the source batch still hashes to what was recorded at import."""
    prov = json.loads((dest_root / batch_id / "provenance.json").read_text())
    current = _digest(_staged_files(source_root / batch_id))
    ok = current == prov["source_sha256"]
    print(
        f"{batch_id}: source {'UNCHANGED' if ok else 'CHANGED SINCE IMPORT'}\n"
        f"  recorded {prov['source_sha256'][:16]}...\n"
        f"  current  {current[:16]}..."
    )
    return ok


def load_pairs(
    batch_id: str, *, dest_root: Path = DEFAULT_DEST_ROOT
) -> list[dict[str, Any]]:
    """Read the experiment's frozen copy. Never touches the source."""
    return json.loads((dest_root / batch_id / "pairs.json").read_text())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-id", default="rrf_003")
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    p.add_argument("--verify", action="store_true", help="Only re-check the digest")
    a = p.parse_args(argv)

    if a.verify:
        return 0 if verify(a.batch_id, source_root=a.source_root, dest_root=a.dest_root) else 1

    rec = import_batch(a.batch_id, source_root=a.source_root, dest_root=a.dest_root)
    s = rec["summary"]
    print(f"imported {a.batch_id} -> {a.dest_root / a.batch_id}")
    print(f"  {s['n_pairs']} pairs ({s['n_pos']} pos / {s['n_neg']} neg)")
    print(
        f"  {s['n_seekers']} seekers, {s['n_seekers_with_both_classes']} with both "
        f"classes -> {s['within_seeker_triplets']} within-seeker triplets"
    )
    print(f"  source sha256 {rec['source_sha256'][:16]}... (source not modified)")
    lab = rec.get("labeler") or {}
    if lab:
        print(f"  labeler: {lab.get('model')} / {lab.get('framing')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
