"""semantic_judge: independent model, binary pass/reject."""

from __future__ import annotations

import json
from typing import Any

from synth_pipeline.config import PipelineConfig
from synth_pipeline.llm import complete_json, load_prompt, truncate_pair_for_prompt
from synth_pipeline.state import PairState


def _dry_run_verdict(state: PairState) -> dict[str, Any]:
    return {
        "verdict": "pass",
        "reason": "dry-run auto-pass",
        "axes": {
            "role": "ok",
            "side": "ok",
            "stage": "ok",
            "geo": "n/a",
            "prefs": "n/a",
        },
        "is_easy_negative": False,
    }


def judge_node(state: PairState, cfg: PipelineConfig) -> dict[str, Any]:
    pair = state.get("pair")
    if not isinstance(pair, dict):
        return {
            "status": "dropped",
            "drop_reason": "missing_pair_for_judge",
            "qc": {**state.get("qc", {}), "judge_verdict": "reject"},
        }

    system = load_prompt("judge.md")
    user = json.dumps(
        {
            "label": state["label"],
            "failure_mode": state.get("failure_mode"),
            "pair": truncate_pair_for_prompt(pair, max_field=600),
        },
        indent=2,
    )
    result = complete_json(
        system=system,
        user=user,
        model=cfg.judge_model,
        temperature=cfg.judge_temperature,
        dry_run=cfg.dry_run,
        dry_run_payload=_dry_run_verdict(state),
    )
    verdict = str(result.get("verdict", "reject")).lower().strip()
    qc = {
        **state.get("qc", {}),
        "judge_raw": result,
        "judge_verdict": verdict,
        "judge_model": cfg.judge_model,
    }
    meta = dict(state.get("metadata") or {})
    meta["judge_model"] = cfg.judge_model

    if verdict != "pass":
        return {
            "qc": qc,
            "metadata": meta,
            "status": "dropped",
            "drop_reason": f"judge:{result.get('reason', 'reject')}",
        }

    if state["label"] == "neg" and result.get("is_easy_negative") is True:
        return {
            "qc": qc,
            "metadata": meta,
            "status": "dropped",
            "drop_reason": "judge:easy_negative",
        }

    return {
        "qc": qc,
        "metadata": meta,
        "status": "judged",
        "drop_reason": None,
    }
