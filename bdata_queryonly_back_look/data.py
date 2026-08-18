"""Load locked B-data.json into ACCEPT/REJECT pairs with minted candidate ids."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import candidate_to_text as full_candidate_to_text
from baselines.bert_frozen.text import seeker_to_text as full_seeker_to_text

from bdata_queryonly_back_look.config import DEFAULT_CONFIG, ExperimentConfig
from bdata_queryonly_back_look.text import background_lookingfor, query_only

RESOLVED_STATUSES = frozenset({"ACCEPT", "REJECT"})
CORPUS_ROLES = frozenset({"candidate", "both"})
MINTED_PREFIX = "cmb"
MINTED_HEX_LEN = 25
EMPTY_CONTACT_ID = "cmb" + ("0" * MINTED_HEX_LEN)

# Copied from scripts/build_unique_contacts_B_data.py (read-only; do not edit that script).
PROFILE_FIELDS = (
    "positioning",
    "background",
    "lookingFor",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
)


@dataclass(frozen=True)
class IdAssignment:
    contact_id: str
    source: str  # boardy | minted | minted_collision
    role: str
    identity_key: str


@dataclass(frozen=True)
class BPair:
    pair_id: str
    seeker_id: str
    match_contact_id: str
    label: str
    query: str
    seeker_text: str
    cand_text: str
    hardness_seeker_text: str
    hardness_cand_text: str
    match_type: str
    row_index: int
    match_index: int
    identity_key: str | None


def assert_source_locked(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing B-data source: {path}")
    if path.stat().st_mode & 0o222:
        raise PermissionError(
            f"Refusing to read writable B-data at {path}. "
            "Lock it first (chmod 444) — source is immutable for this experiment."
        )


def source_provenance(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _strip(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def identity_key(contact_file: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return (field, hex digest) or None if the profile is entirely empty.

    Copied from scripts/build_unique_contacts_B_data.py.
    """
    cf = contact_file if isinstance(contact_file, dict) else {}
    for field in PROFILE_FIELDS:
        text = _strip(cf.get(field))
        if text:
            digest = hashlib.sha256(f"{field}\0{text}".encode()).hexdigest()
            return field, digest
    return None


def mint_id(identity_key_hex: str) -> str:
    return f"{MINTED_PREFIX}{identity_key_hex[:MINTED_HEX_LEN]}"


def load_unique_contacts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing unique contacts at {path}. Rebuild with: "
            "python scripts/build_unique_contacts_B_data.py"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a JSON array")
    return rows


def build_id_map(contacts: list[dict[str, Any]]) -> dict[str, IdAssignment]:
    """Keep first sorted Boardy contactId if present; else mint cmb+identityKey[:25]."""
    out: dict[str, IdAssignment] = {}
    used: dict[str, str] = {}
    for c in contacts:
        key = str(c.get("identityKey") or "")
        if not key:
            continue
        ids = [str(x) for x in (c.get("contactIds") or []) if x]
        if ids:
            assigned = sorted(ids)[0]
            source = "boardy"
            prev = used.get(assigned)
            if prev is not None and prev != key:
                assigned = mint_id(key)
                source = "minted_collision"
        else:
            assigned = mint_id(key)
            source = "minted"
        prev = used.get(assigned)
        if prev is not None and prev != key:
            raise ValueError(f"id collision: {assigned} for {key} and {prev}")
        used[assigned] = key
        out[key] = IdAssignment(
            contact_id=assigned,
            source=source,
            role=str(c.get("role") or ""),
            identity_key=key,
        )
    return out


def write_id_map(id_map: dict[str, IdAssignment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rule": (
            "first sorted Boardy contactId if present and unique; "
            "else cmb + identityKey[:25] (minted_collision if Boardy id reused)"
        ),
        "n": len(id_map),
        "n_boardy": sum(1 for v in id_map.values() if v.source == "boardy"),
        "n_minted": sum(1 for v in id_map.values() if v.source == "minted"),
        "n_minted_collision": sum(
            1 for v in id_map.values() if v.source == "minted_collision"
        ),
        "ids": {
            k: {"id": v.contact_id, "source": v.source, "role": v.role}
            for k, v in id_map.items()
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def id_map_stats(id_map: dict[str, IdAssignment]) -> dict[str, Any]:
    return {
        "n": len(id_map),
        "n_boardy": sum(1 for v in id_map.values() if v.source == "boardy"),
        "n_minted": sum(1 for v in id_map.values() if v.source == "minted"),
        "n_minted_collision": sum(
            1 for v in id_map.values() if v.source == "minted_collision"
        ),
        "n_corpus": sum(1 for v in id_map.values() if v.role in CORPUS_ROLES),
        "rule": (
            "first sorted Boardy contactId if present and unique; "
            "else cmb + identityKey[:25] (minted_collision if Boardy id reused)"
        ),
    }


def retrieval_corpus(
    contacts: list[dict[str, Any]],
    id_map: dict[str, IdAssignment],
) -> tuple[list[str], list[str]]:
    """Unique people who appeared as a match (role candidate or both)."""
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for c in contacts:
        if str(c.get("role") or "") not in CORPUS_ROLES:
            continue
        key = str(c.get("identityKey") or "")
        assignment = id_map.get(key)
        if assignment is None:
            continue
        cid = assignment.contact_id
        if cid in seen:
            continue
        seen.add(cid)
        cf = c.get("contactFile") if isinstance(c.get("contactFile"), dict) else {}
        ids.append(cid)
        texts.append(background_lookingfor(cf))
    return ids, texts


def expand_resolved_pairs(
    rows: list[dict[str, Any]],
    id_lookup: dict[str, str] | None = None,
) -> list[BPair]:
    """One pair per (seeker, query, match) with status ACCEPT or REJECT."""
    lookup = id_lookup or {}
    pairs: list[BPair] = []
    for row_i, row in enumerate(rows):
        seeker_id = str(row.get("contactId") or "")
        if not seeker_id:
            continue
        query = str(row.get("query") or "")
        seeker_file = row.get("contactFile") if isinstance(row.get("contactFile"), dict) else {}
        seeker_text = query_only(seeker_file, query)
        hardness_seeker = full_seeker_to_text(seeker_file, query)
        matches = row.get("matches") if isinstance(row.get("matches"), list) else []
        for match_i, match in enumerate(matches):
            if not isinstance(match, dict):
                continue
            status = str(match.get("status") or "")
            if status not in RESOLVED_STATUSES:
                continue
            cand_file = (
                match.get("contactFile")
                if isinstance(match.get("contactFile"), dict)
                else {}
            )
            ident = identity_key(cand_file)
            if ident is None:
                digest = None
                mid = EMPTY_CONTACT_ID
            else:
                _field, digest = ident
                mid = lookup.get(digest) or mint_id(digest)
            pair_id = f"{seeker_id}:{row_i}:{match_i}:{mid[:16]}"
            pairs.append(
                BPair(
                    pair_id=pair_id,
                    seeker_id=seeker_id,
                    match_contact_id=mid,
                    label=status,
                    query=query,
                    seeker_text=seeker_text,
                    cand_text=background_lookingfor(cand_file),
                    hardness_seeker_text=hardness_seeker,
                    hardness_cand_text=full_candidate_to_text(cand_file),
                    match_type=str(match.get("matchType") or "UNKNOWN"),
                    row_index=row_i,
                    match_index=match_i,
                    identity_key=digest,
                )
            )
    return pairs


def load_pairs(
    cfg: ExperimentConfig | None = None,
) -> tuple[list[BPair], dict[str, IdAssignment], list[dict[str, Any]], dict[str, Any]]:
    cfg = cfg or DEFAULT_CONFIG
    assert_source_locked(cfg.source_path)
    prov = source_provenance(cfg.source_path)
    rows = json.loads(cfg.source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{cfg.source_path} must be a JSON array")
    contacts = load_unique_contacts(cfg.unique_contacts_path)
    id_map = build_id_map(contacts)
    lookup = {k: v.contact_id for k, v in id_map.items()}
    pairs = expand_resolved_pairs(rows, lookup)
    meta = {
        "provenance": prov,
        "n_rows": len(rows),
        "n_resolved_pairs": len(pairs),
        "n_accept": sum(1 for p in pairs if p.label == "ACCEPT"),
        "n_reject": sum(1 for p in pairs if p.label == "REJECT"),
        "n_unique_seekers": len({p.seeker_id for p in pairs}),
        "n_unique_contacts": len(contacts),
        "id_map": id_map_stats(id_map),
        "population": "all resolved pairs (no train/holdout split)",
    }
    return pairs, id_map, contacts, meta


def within_seeker_dual_label_groups(
    pairs: list[BPair],
) -> dict[str, dict[str, list[BPair]]]:
    by: dict[str, dict[str, list[BPair]]] = {}
    for p in pairs:
        bucket = by.setdefault(p.seeker_id, {"ACCEPT": [], "REJECT": []})
        bucket[p.label].append(p)
    return {
        sid: g
        for sid, g in by.items()
        if g["ACCEPT"] and g["REJECT"]
    }
