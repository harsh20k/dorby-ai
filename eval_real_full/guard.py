"""Refuse to score adapters that trained on real pairs.

Evaluating on all 200 real pairs is only sound for a model that never saw any of
them. That is true of the ``rrf_*`` runs, which trained purely on synthetic
``rrf_003`` rows, and false of the original ``twotower`` runs (``run_001``,
``arm_a_real_only``), which trained on real train pairs.

Without this guard the difference is invisible at the call site — both are just
an ``adapter/`` directory — and the failure would be a *flattering* number, the
worst kind. So the allowlist is explicit and keyed on the training-rows path
recorded in the run's own ``run_meta.json``, not on a name convention.
"""

from __future__ import annotations

import json
from pathlib import Path

# Substrings of `run_meta.json:rows_path` that identify all-synthetic training.
SYNTHETIC_ONLY_ROW_SOURCES = ("rrf_003_multineg", "rrf_003_triplets")


class RealPairLeakError(RuntimeError):
    pass


def assert_trained_without_real_pairs(adapter_dir: Path) -> dict:
    """Verify the run that produced ``adapter_dir`` never trained on real pairs.

    Returns the run's metadata on success; raises ``RealPairLeakError`` if the
    run trained on real pairs, or if provenance is missing or ambiguous.
    """
    adapter_dir = Path(adapter_dir)
    meta_path = adapter_dir.parent / "run_meta.json"
    if not meta_path.exists():
        raise RealPairLeakError(
            f"no run_meta.json beside {adapter_dir} — cannot prove this adapter "
            f"avoided real pairs, so refusing to score it on the full real set."
        )
    meta = json.loads(meta_path.read_text())
    rows_path = str(meta.get("rows_path", ""))
    if not rows_path:
        raise RealPairLeakError(
            f"{meta_path} records no rows_path; this looks like a `twotower` run "
            f"trained on real pairs. Score it on the 69-pair holdout instead."
        )
    if not any(token in rows_path for token in SYNTHETIC_ONLY_ROW_SOURCES):
        raise RealPairLeakError(
            f"rows_path={rows_path!r} is not a known all-synthetic source "
            f"{SYNTHETIC_ONLY_ROW_SOURCES}. If this run really saw no real pair, "
            f"add its source to SYNTHETIC_ONLY_ROW_SOURCES deliberately."
        )
    if meta.get("config", {}).get("include_synth") and "data_dir" in meta.get("config", {}):
        # include_synth only governs build_split_bundle's train pool, which these
        # runs never touch (they train from rows_path). Recorded, not fatal.
        pass
    return meta
