"""generate: single-label hard pos or hard neg via LLM."""

from __future__ import annotations

import copy
import json
from typing import Any

from synth_pipeline.config import PROFILE_KEYS, PipelineConfig
from synth_pipeline.llm import complete_json, load_prompt, truncate_pair_for_prompt
from synth_pipeline.state import PairState


def _dry_run_pair(state: PairState) -> dict[str, Any]:
    seed = state["seed_pair"]
    user_id = state["metadata"]["assigned_user_contact_id"]
    match_id = state["metadata"]["assigned_match_contact_id"]
    user_file = copy.deepcopy(seed["userContactFile"])
    match_file = copy.deepcopy(seed["matchContactFile"])

    # Nudge text so leakage/dedup filters don't trip on exact seed copies.
    suffix = f" [synth {state['batch_id']} {state['label']}]"
    if isinstance(match_file.get("positioning"), str):
        match_file["positioning"] = match_file["positioning"] + suffix
    else:
        match_file["positioning"] = "Synthetic match profile" + suffix
    if state["label"] == "neg" and state.get("failure_mode"):
        note = f"Hard-neg failure mode: {state['failure_mode']}."
        prev = match_file.get("notes")
        match_file["notes"] = f"{prev}\n{note}" if isinstance(prev, str) else note

    return {
        "userContactId": user_id,
        "matchContactId": match_id,
        "userContactFileVersion": 1,
        "matchContactFileVersion": 1,
        "searchQuery": seed["searchQuery"],
        "userContactFile": user_file,
        "matchContactFile": match_file,
    }


def _force_ids(pair: dict[str, Any], state: PairState) -> dict[str, Any]:
    pair = dict(pair)
    pair["userContactId"] = state["metadata"]["assigned_user_contact_id"]
    pair["matchContactId"] = state["metadata"]["assigned_match_contact_id"]
    # Normalize nested keys / nulls
    for side in ("userContactFile", "matchContactFile"):
        prof = pair.get(side)
        if not isinstance(prof, dict):
            continue
        for key in PROFILE_KEYS:
            prof.setdefault(key, None)
        pair[side] = prof
    if not isinstance(pair.get("userContactFileVersion"), int):
        pair["userContactFileVersion"] = 1
    if not isinstance(pair.get("matchContactFileVersion"), int):
        pair["matchContactFileVersion"] = 1
    if not pair.get("searchQuery"):
        pair["searchQuery"] = state["seed_pair"]["searchQuery"]
    return pair


def generate_node(state: PairState, cfg: PipelineConfig) -> dict[str, Any]:
    label = state["label"]
    if label == "pos":
        system = load_prompt("generate_pos.md")
    else:
        system = load_prompt("generate_neg.md").format(
            failure_mode=state.get("failure_mode") or "wrong_role"
        )

    payload = {
        "label": label,
        "failure_mode": state.get("failure_mode"),
        "assigned_ids": {
            "userContactId": state["metadata"]["assigned_user_contact_id"],
            "matchContactId": state["metadata"]["assigned_match_contact_id"],
            "userContactFileVersion": 1,
            "matchContactFileVersion": 1,
        },
        "seed": truncate_pair_for_prompt(state["seed_pair"]),
        "few_shot": [truncate_pair_for_prompt(p) for p in state["few_shot_pairs"]],
        "instructions": (
            "Return only the pair JSON. Use assigned_ids exactly. "
            "Do not invent other top-level keys."
        ),
    }
    user = json.dumps(payload, indent=2)

    raw = complete_json(
        system=system,
        user=user,
        model=cfg.generate_model,
        temperature=cfg.temperature,
        dry_run=cfg.dry_run,
        dry_run_payload=_dry_run_pair(state),
    )
    pair = _force_ids(raw, state)
    meta = dict(state.get("metadata") or {})
    meta["attempt_index"] = state.get("retry_count", 0)
    meta["generate_model"] = cfg.generate_model
    return {"pair": pair, "metadata": meta, "status": "pending", "drop_reason": None}
