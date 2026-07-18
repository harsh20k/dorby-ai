"""semantic_judge: independent model scores intro quality; code computes pass/reject."""

from __future__ import annotations

import json
from typing import Any

from synth_pipeline.config import PipelineConfig
from synth_pipeline.llm import complete_json, truncate_pair_for_prompt
from synth_pipeline.prompts import hub_run_metadata, load_prompt, prompt_version_label
from synth_pipeline.state import PairState


def compute_verdict(label: str, judge_output: dict[str, Any]) -> str:
    """Map objective intro quality + label → pass/reject.

    LLM must not decide pass/reject; it only reports whether the intro would
    be good. Legacy ``verdict`` keys in judge_output are ignored.
    """
    good = bool(judge_output.get("would_be_good_intro"))
    easy = bool(judge_output.get("is_easy_negative"))
    if label == "pos":
        return "pass" if good else "reject"
    return "pass" if (not good and not easy) else "reject"


def _dry_run_judge_output(state: PairState) -> dict[str, Any]:
    """Dry-run payload: quality that yields pass for the current label."""
    label = state["label"]
    return {
        "would_be_good_intro": label == "pos",
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

    loaded = load_prompt("judge")
    system = loaded.text
    judge_ref = loaded.ref
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
        dry_run_payload=_dry_run_judge_output(state),
        cfg=cfg,
        run_metadata=hub_run_metadata(judge_ref),
        run_tags=[f"prompt:{judge_ref.version_label}"],
    )

    # Ignore legacy LLM ``verdict`` if present — decision is code-side only.
    would_be_good = bool(result.get("would_be_good_intro"))
    is_easy = bool(result.get("is_easy_negative"))
    verdict = compute_verdict(state["label"], result)

    qc = {
        **state.get("qc", {}),
        "would_be_good_intro": would_be_good,
        "is_easy_negative": is_easy,
        "judge_verdict": verdict,
        "judge_raw": result,
        "judge_model": cfg.judge_model,
    }
    meta = dict(state.get("metadata") or {})
    meta["judge_model"] = cfg.judge_model
    prompt_refs = dict(meta.get("prompt_refs") or {})
    prompt_refs["judge"] = judge_ref.to_dict()
    meta["prompt_refs"] = prompt_refs
    meta["prompt_version"] = prompt_version_label(prompt_refs)

    if verdict != "pass":
        reason = result.get("reason", "reject")
        if state["label"] == "neg" and is_easy:
            drop = "judge:easy_negative"
        elif state["label"] == "neg" and would_be_good:
            drop = f"judge:would_be_good_intro:{reason}"
        else:
            drop = f"judge:{reason}"
        return {
            "qc": qc,
            "metadata": meta,
            "status": "dropped",
            "drop_reason": drop,
        }

    return {
        "qc": qc,
        "metadata": meta,
        "status": "judged",
        "drop_reason": None,
    }


if __name__ == "__main__":
    cases = [
        ("pos", {"would_be_good_intro": True, "is_easy_negative": False}, "pass"),
        ("pos", {"would_be_good_intro": False, "is_easy_negative": False}, "reject"),
        ("neg", {"would_be_good_intro": False, "is_easy_negative": False}, "pass"),
        ("neg", {"would_be_good_intro": True, "is_easy_negative": False}, "reject"),
        ("neg", {"would_be_good_intro": False, "is_easy_negative": True}, "reject"),
        # legacy verdict key must not affect decision
        (
            "pos",
            {"would_be_good_intro": False, "verdict": "pass", "is_easy_negative": False},
            "reject",
        ),
    ]
    failed = 0
    for label, out, expected in cases:
        got = compute_verdict(label, out)
        ok = got == expected
        print(f"{'OK' if ok else 'FAIL'} label={label} → {got} (expected {expected})")
        failed += int(not ok)
    raise SystemExit(failed)
