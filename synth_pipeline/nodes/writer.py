"""staging writer: persist accepted pairs + update batch manifest."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synth_pipeline.config import PipelineConfig
from synth_pipeline.state import PairState

# Guards the manifest.json read-modify-write cycle in staging_node, which is
# invoked from worker threads when batches run concurrently (run_batch.py).
_MANIFEST_LOCK = threading.Lock()


def batch_dir(cfg: PipelineConfig, batch_id: str) -> Path:
    return cfg.artifacts_dir / batch_id


def manifest_path(cfg: PipelineConfig, batch_id: str) -> Path:
    return batch_dir(cfg, batch_id) / "manifest.json"


def init_manifest(cfg: PipelineConfig, batch_id: str, *, split_hash: str) -> dict[str, Any]:
    path = manifest_path(cfg, batch_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    batch_dir(cfg, batch_id).mkdir(parents=True, exist_ok=True)
    (batch_dir(cfg, batch_id) / "staged").mkdir(exist_ok=True)
    (batch_dir(cfg, batch_id) / "dropped").mkdir(exist_ok=True)
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_hash": split_hash,
        "generate_model": cfg.generate_model,
        "judge_model": cfg.judge_model,
        "prompt_version": cfg.prompt_version,
        "counts": {
            "attempts": 0,
            "filtered_out": 0,
            "judge_rejected": 0,
            "staged": 0,
            "dropped": 0,
            "approved": 0,
        },
        "records": [],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _save_manifest(cfg: PipelineConfig, batch_id: str, manifest: dict[str, Any]) -> None:
    manifest_path(cfg, batch_id).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def staging_node(state: PairState, cfg: PipelineConfig) -> dict[str, Any]:
    with _MANIFEST_LOCK:
        return _staging_node_locked(state, cfg)


def _staging_node_locked(state: PairState, cfg: PipelineConfig) -> dict[str, Any]:
    batch_id = state["batch_id"]
    manifest = init_manifest(cfg, batch_id, split_hash=state["split_hash"])
    manifest["counts"]["attempts"] = manifest["counts"].get("attempts", 0) + 1

    record = {
        "seed_pair_id": state["seed_pair_id"],
        "seed_user_id": state["seed_user_id"],
        "label": state["label"],
        "failure_mode": state.get("failure_mode"),
        "retry_count": state.get("retry_count", 0),
        "status": state.get("status"),
        "drop_reason": state.get("drop_reason"),
        "qc": {
            "filter_passed": state.get("qc", {}).get("filter_passed"),
            "filter_reject_reason": state.get("qc", {}).get("filter_reject_reason"),
            "would_be_good_intro": state.get("qc", {}).get("would_be_good_intro"),
            "is_easy_negative": state.get("qc", {}).get("is_easy_negative"),
            "judge_verdict": state.get("qc", {}).get("judge_verdict"),
            "query_match_jaccard": state.get("qc", {}).get("query_match_jaccard"),
        },
        "metadata": {
            "generate_model": state.get("metadata", {}).get("generate_model"),
            "judge_model": state.get("metadata", {}).get("judge_model"),
            "prompt_version": state.get("metadata", {}).get("prompt_version"),
            "prompt_refs": state.get("metadata", {}).get("prompt_refs"),
            "attempt_index": state.get("metadata", {}).get("attempt_index"),
        },
    }

    status = state.get("status")
    if status == "judged" and state.get("pair"):
        out_name = (
            f"{state['label']}_{state['metadata']['assigned_match_contact_id']}.json"
        )
        out_path = batch_dir(cfg, batch_id) / "staged" / out_name
        envelope = {
            "pair": state["pair"],
            "label": state["label"],
            "failure_mode": state.get("failure_mode"),
            "seed_pair_id": state["seed_pair_id"],
            "seed_user_id": state["seed_user_id"],
            "batch_id": batch_id,
            "split_hash": state["split_hash"],
            "metadata": state.get("metadata"),
            "qc": state.get("qc"),
            "human_review": None,
        }
        out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        record["staged_path"] = str(out_path.relative_to(cfg.artifacts_dir))
        record["status"] = "staged"
        manifest["counts"]["staged"] += 1
        manifest["records"].append(record)
        _save_manifest(cfg, batch_id, manifest)
        return {"status": "staged"}

    # Dropped path (filter exhausted or judge reject)
    reason = state.get("drop_reason") or "unknown"
    if state.get("qc", {}).get("filter_passed") is False:
        manifest["counts"]["filtered_out"] += 1
    elif str(reason).startswith("judge:"):
        manifest["counts"]["judge_rejected"] += 1
    manifest["counts"]["dropped"] += 1

    drop_name = (
        f"{state['label']}_{state['metadata'].get('assigned_match_contact_id', 'x')}"
        f"_r{state.get('retry_count', 0)}.json"
    )
    drop_path = batch_dir(cfg, batch_id) / "dropped" / drop_name
    drop_path.write_text(
        json.dumps(
            {
                "pair": state.get("pair"),
                "label": state["label"],
                "failure_mode": state.get("failure_mode"),
                "drop_reason": reason,
                "qc": state.get("qc"),
                "metadata": state.get("metadata"),
                "seed_pair_id": state["seed_pair_id"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record["dropped_path"] = str(drop_path.relative_to(cfg.artifacts_dir))
    record["status"] = "dropped"
    manifest["records"].append(record)
    _save_manifest(cfg, batch_id, manifest)
    return {"status": "dropped", "drop_reason": reason}
