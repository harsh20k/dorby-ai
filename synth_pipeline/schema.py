"""Pair / profile schema checks shared by filter and LLM parsers."""

from __future__ import annotations

from typing import Any

from synth_pipeline.config import PAIR_KEYS, PROFILE_KEYS
from synth_pipeline.ids import is_valid_contact_id


def _is_str_or_none(value: object) -> bool:
    return value is None or isinstance(value, str)


def validate_profile(profile: object, *, side: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(profile, dict):
        return [f"{side} must be an object"]
    for key in PROFILE_KEYS:
        if key not in profile:
            errors.append(f"{side}.{key} missing")
        elif not _is_str_or_none(profile[key]):
            errors.append(f"{side}.{key} must be string or null")
    return errors


def validate_pair_schema(pair: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(pair, dict):
        return ["pair must be an object"]
    for key in PAIR_KEYS:
        if key not in pair:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors

    for id_key in ("userContactId", "matchContactId"):
        if not is_valid_contact_id(pair.get(id_key)):
            errors.append(f"{id_key} must be 25-char cm… id")

    for ver_key in ("userContactFileVersion", "matchContactFileVersion"):
        ver = pair.get(ver_key)
        if not isinstance(ver, int) or isinstance(ver, bool) or ver < 1:
            errors.append(f"{ver_key} must be a positive int")

    if not isinstance(pair.get("searchQuery"), str) or not pair["searchQuery"].strip():
        errors.append("searchQuery must be a non-empty string")

    errors.extend(validate_profile(pair.get("userContactFile"), side="userContactFile"))
    errors.extend(validate_profile(pair.get("matchContactFile"), side="matchContactFile"))
    return errors


def core_fields_nonempty(profile: dict[str, Any]) -> bool:
    for key in ("positioning", "lookingFor", "background"):
        val = profile.get(key)
        if not isinstance(val, str) or not val.strip():
            return False
    return True


def tokenize(text: str) -> set[str]:
    return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
