"""heuristic_filter: schema, IDs, leakage, easy-neg, dedup — no LLM judge."""

from __future__ import annotations

from typing import Any

from synth_pipeline.config import PipelineConfig
from synth_pipeline.ids import is_valid_contact_id
from synth_pipeline.schema import (
    core_fields_nonempty,
    jaccard,
    tokenize,
    validate_pair_schema,
)
from synth_pipeline.state import PairState

# Long shared substrings imply copying seed PII/companies.
_MIN_LEAK_LEN = 48
_EASY_NEG_JACCARD = 0.02


def _long_substrings(text: str, n: int = _MIN_LEAK_LEN) -> set[str]:
    text = " ".join(text.split())
    if len(text) < n:
        return set()
    return {text[i : i + n].lower() for i in range(0, len(text) - n + 1, n // 2)}


def _profile_blob(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("positioning", "lookingFor", "background", "notes", "locationAvailability"):
        val = profile.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    return "\n".join(parts)


def heuristic_filter_node(state: PairState, cfg: PipelineConfig) -> dict[str, Any]:
    pair = state.get("pair")
    errors = validate_pair_schema(pair)
    qc: dict[str, Any] = {"filter_errors": errors}

    if errors:
        return _reject(state, qc, "schema:" + errors[0], max_retries=cfg.max_retries)

    assert isinstance(pair, dict)
    if pair["userContactId"] == pair["matchContactId"]:
        return _reject(state, qc, "self_match", max_retries=cfg.max_retries)

    if not is_valid_contact_id(pair["userContactId"]) or not is_valid_contact_id(
        pair["matchContactId"]
    ):
        return _reject(state, qc, "bad_id_format", max_retries=cfg.max_retries)

    # Must use assigned synth IDs (prevents model inventing seed IDs).
    if pair["userContactId"] != state["metadata"]["assigned_user_contact_id"]:
        return _reject(state, qc, "user_id_not_assigned", max_retries=cfg.max_retries)
    if pair["matchContactId"] != state["metadata"]["assigned_match_contact_id"]:
        return _reject(state, qc, "match_id_not_assigned", max_retries=cfg.max_retries)

    user_file = pair["userContactFile"]
    match_file = pair["matchContactFile"]
    if not core_fields_nonempty(user_file) or not core_fields_nonempty(match_file):
        return _reject(state, qc, "core_fields_empty", max_retries=cfg.max_retries)

    seed = state["seed_pair"]
    seed_match_blob = _profile_blob(seed["matchContactFile"])
    new_match_blob = _profile_blob(match_file)
    seed_subs = _long_substrings(seed_match_blob)
    new_subs = _long_substrings(new_match_blob)
    leaked = seed_subs & new_subs
    # Dry-run mutates seed match lightly; ignore when dry_run.
    if leaked and not state["metadata"].get("dry_run"):
        qc["leakage_samples"] = list(sorted(leaked))[:3]
        return _reject(state, qc, "seed_match_leakage", max_retries=cfg.max_retries)

    # Reject exact seed contact IDs reused anywhere.
    seed_ids = {seed["userContactId"], seed["matchContactId"]}
    if pair["userContactId"] in seed_ids or pair["matchContactId"] in seed_ids:
        return _reject(state, qc, "reused_seed_id", max_retries=cfg.max_retries)

    q_tok = tokenize(pair["searchQuery"])
    m_tok = tokenize(new_match_blob)
    overlap = jaccard(q_tok, m_tok)
    qc["query_match_jaccard"] = overlap

    if (
        state["label"] == "neg"
        and overlap < _EASY_NEG_JACCARD
        and not state["metadata"].get("dry_run")
    ):
        return _reject(state, qc, "easy_neg_low_overlap", max_retries=cfg.max_retries)

    # Near-dup of seed match (high jaccard) — not useful diversity.
    seed_tok = tokenize(seed_match_blob)
    near = jaccard(seed_tok, m_tok)
    qc["seed_match_jaccard"] = near
    if near > 0.92 and not state["metadata"].get("dry_run"):
        return _reject(state, qc, "near_dup_seed_match", max_retries=cfg.max_retries)

    qc["filter_passed"] = True
    qc["filter_reject_reason"] = None
    return {
        "qc": {**state.get("qc", {}), **qc},
        "status": "filtered",
        "drop_reason": None,
    }


def _reject(
    state: PairState,
    qc: dict[str, Any],
    reason: str,
    *,
    max_retries: int,
) -> dict[str, Any]:
    qc = {**qc, "filter_passed": False, "filter_reject_reason": reason}
    retry = int(state.get("retry_count", 0)) + 1
    from synth_pipeline.ids import new_contact_id

    meta = dict(state.get("metadata") or {})
    can_retry = retry <= max_retries
    if can_retry:
        uid = new_contact_id()
        mid = new_contact_id()
        while mid == uid:
            mid = new_contact_id()
        meta["assigned_user_contact_id"] = uid
        meta["assigned_match_contact_id"] = mid

    return {
        "qc": {**state.get("qc", {}), **qc},
        # Cap the stored count at max_retries even though the drop happens on
        # the (max_retries + 1)th failure — retry_count reads as "retries used".
        "retry_count": min(retry, max_retries),
        "metadata": meta,
        "status": "pending" if can_retry else "dropped",
        "drop_reason": reason,
        "pair": state.get("pair"),
    }
