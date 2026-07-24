"""Load a standalone profile-generation run into pairable records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth_pipeline.config import ID_LENGTH, PROFILE_KEYS
from synth_pipeline.schema import core_fields_nonempty

# Distinct from the `cmsynth` used by batch_500_001 so the two generations stay
# separable by grep, while still matching every existing startswith("cmsynth")
# filter (build_real_pairs_graph.is_synthetic, twotower.data._is_synth_pair, ...).
PAIRING_ID_PREFIX = "cmsynthp"


@dataclass(frozen=True)
class SynthProfile:
    contact_id: str
    profile_id: int
    archetype: str
    profile: dict[str, Any]
    source_run: str

    def as_contact_file(self) -> dict[str, Any]:
        return dict(self.profile)


def _extract_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Project onto exactly PROFILE_KEYS.

    An allowlist, deliberately, rather than deleting `reasoning`. That field is
    discarded chain-of-thought explaining why the persona hangs together — the
    same meta-commentary class that caused the label leak in possible-bugs #4.
    It must never reach a prompt, an envelope, or the graph payload, and an
    allowlist keeps that true even if the generator's schema grows new fields.
    """
    return {key: raw.get(key) for key in PROFILE_KEYS}


def contact_id_for(source_run: str, profile_id: int) -> str:
    """Deterministic 25-char `cmsynthp…` id derived from (run, profile id).

    Deliberately not random: a pairing batch is an experiment that must be
    re-runnable. Random ids would mint a fresh identity for the same profile on
    every run, invalidating the query checkpoint and making two runs over the
    same profile pool incomparable. Hex digits are a subset of the id alphabet,
    so these still satisfy `is_valid_contact_id`.
    """
    digest = hashlib.sha256(f"{source_run}:{profile_id}".encode("utf-8")).hexdigest()
    return PAIRING_ID_PREFIX + digest[: ID_LENGTH - len(PAIRING_ID_PREFIX)]


def load_profile_run(
    run_dir: Path,
    *,
    limit: int | None = None,
) -> list[SynthProfile]:
    """Read `<run_dir>/profiles/*_ok.json` into SynthProfile records."""
    profiles_dir = Path(run_dir) / "profiles"
    if not profiles_dir.is_dir():
        raise FileNotFoundError(f"no profiles/ directory under {run_dir}")

    run_name = Path(run_dir).name
    out: list[SynthProfile] = []
    for path in sorted(profiles_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("success") or not isinstance(raw.get("profile"), dict):
            continue

        profile = _extract_profile(raw["profile"])
        if not core_fields_nonempty(profile):
            continue

        contact_id = contact_id_for(run_name, int(raw["id"]))

        out.append(
            SynthProfile(
                contact_id=contact_id,
                profile_id=int(raw["id"]),
                archetype=str(raw.get("archetype") or "unknown"),
                profile=profile,
                source_run=Path(run_dir).name,
            )
        )
        if limit is not None and len(out) >= limit:
            break

    return out
