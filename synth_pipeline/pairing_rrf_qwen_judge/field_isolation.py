"""Isolate each profile field (and each ``lookingFor`` ask) into its own embedding text.

Sibling to ``baselines/voyage_nano_field_isolation/modal_embed_space.py``, which ran
this recipe over the real 69-pair holdout in voyage-4-nano space. This module runs
the same recipe over an ``rrf_*`` batch's seeker + candidate pool in Qwen3 space, so
the two experiments stay comparable.

"Isolated" is the key word: a field-alone or section-alone row contains *only that
field's text* (``"field: value"``), nothing else — unlike ``sections.py``'s
per-``lookingFor`` seeker vectors, which swap in one ask but keep every other field
present. Isolation removes the rest-of-profile context entirely so a field's own
vector isn't dragged toward wherever the rest of the profile sits.

Row kinds, per contact:
  * ``whole``        — already embedded by ``embed.py``; not reproduced here.
  * ``field``         — one row per non-empty ``PROFILE_FIELDS`` entry, alone.
  * ``section_alone`` — one row per ``lookingFor`` paragraph, alone (only when
    ``lookingFor`` splits into more than one paragraph — the single-paragraph
    case is already covered by the ``field`` row for ``lookingFor``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from baselines.bert_frozen.text import PROFILE_FIELDS, _nonempty
from baselines.voyage_nano_sectioned.text import split_looking_for_sections


@dataclass(frozen=True)
class IsolationRow:
    contact_id: str
    role: str  # "seeker" | "candidate"
    kind: str  # "field" | "section_alone"
    field: str
    section_index: int | None  # None for "field" rows
    text: str


def build_rows(
    seekers: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> list[IsolationRow]:
    """One row per isolated field/section, for every seeker and candidate.

    Sorted by contact id (then field, then section index) so a re-run produces
    byte-identical row order — the same determinism guarantee ``embed.py``'s
    ``build_plan`` makes for the whole/section vectors.
    """
    rows: list[IsolationRow] = []
    for role, pool in (("seeker", seekers), ("candidate", candidates)):
        for cid in sorted(pool):
            profile = pool[cid]
            for f in PROFILE_FIELDS:
                value = profile.get(f)
                if not _nonempty(value):
                    continue
                rows.append(
                    IsolationRow(
                        contact_id=cid, role=role, kind="field", field=f,
                        section_index=None, text=f"{f}: {value.strip()}",
                    )
                )
            looking_for = (profile.get("lookingFor") or "").strip()
            if looking_for:
                sections = split_looking_for_sections(looking_for)
                if len(sections) > 1:
                    for i, section in enumerate(sections):
                        rows.append(
                            IsolationRow(
                                contact_id=cid, role=role, kind="section_alone",
                                field="lookingFor", section_index=i,
                                text=f"lookingFor: {section}",
                            )
                        )
    return rows


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def persist(out_dir: Path, rows: list[IsolationRow], vecs: np.ndarray, *, model_name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "isolated_vectors.npy", vecs.astype(np.float32))
    meta = {
        "model": model_name,
        "dim": int(vecs.shape[1]) if vecs.size else None,
        "n_rows": len(rows),
        "rows": [
            {
                "contact_id": r.contact_id,
                "role": r.role,
                "kind": r.kind,
                "field": r.field,
                "section_index": r.section_index,
                "text_sha256": _sha(r.text),
                "text": r.text,
                "row": i,
            }
            for i, r in enumerate(rows)
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_persisted(out_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    out_dir = Path(out_dir)
    vecs = np.load(out_dir / "isolated_vectors.npy")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return vecs, meta
