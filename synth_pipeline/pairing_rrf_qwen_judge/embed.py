"""Embed both sides with an open-weight HF model and persist the vectors as files.

Files first, database second. Every vector the GPU produces is written here as a
plain ``.npy`` array plus a JSON manifest naming which profile and which section
it came from. ``store.py`` then builds Chroma *from* those files. In that order a
corrupt index or a library upgrade costs a rebuild rather than another GPU run —
and the last profile batch was lost precisely because a single copy lived in a
gitignored directory.

Two encoding asymmetries, both deliberate:

* Seekers are encoded with the model's **query** prompt, candidates as plain
  documents — Qwen3-Embedding declares ``query_prompt_name="query"`` in
  ``baselines/hf_embedding/models.py``, the same convention the Voyage baselines
  already follow.
* Seekers get N+1 vectors (whole profile + one per ``lookingFor`` section);
  candidates get exactly one, unsectioned. See ``sections.py`` for why.

The ``searchQuery`` text reaches neither side. It re-enters only in the lexical
recall channel and at the judge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from synth_pipeline.pairing_rrf_qwen_judge.sections import (
    SeekerVector,
    candidate_text,
    seeker_vectors,
)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _content_cache_name(prefix: str, texts: Sequence[str], model: str) -> str:
    """Cache key bound to the exact texts.

    ``HFEmbeddingEncoder.encode`` returns a cached array whenever ``cache_name``
    exists *without* re-checking that the input texts still match — the same
    footgun documented for the pairing pipeline's batch cache. Hashing the
    payload into the key makes a stale hit impossible.
    """
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return f"{prefix}_{h.hexdigest()[:16]}"


@dataclass
class VectorRecord:
    """One row of the persisted manifest — matches one row of the .npy array."""

    key: str
    contact_id: str
    role: str  # "seeker" | "candidate"
    section_index: int | None  # None for candidates; -1 = whole profile
    section_text: str | None
    text_sha256: str
    row: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "contact_id": self.contact_id,
            "role": self.role,
            "section_index": self.section_index,
            "section_text": self.section_text,
            "text_sha256": self.text_sha256,
            "row": self.row,
        }


@dataclass
class EmbeddingPlan:
    """Every text to be embedded, in the exact row order of the output arrays."""

    seeker_vectors: list[SeekerVector]
    seeker_texts: list[str]
    candidate_ids: list[str]
    candidate_texts: list[str]

    @property
    def n_seeker(self) -> int:
        return len(self.seeker_texts)

    @property
    def n_candidate(self) -> int:
        return len(self.candidate_texts)


def build_plan(
    seekers: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> EmbeddingPlan:
    """Expand seekers into N+1 vectors each; candidates stay one apiece.

    Both mappings are ``contact_id -> profile``. Iteration order is sorted by
    contact id so a re-run produces byte-identical row ordering.
    """
    svecs: list[SeekerVector] = []
    for contact_id in sorted(seekers):
        svecs.extend(seeker_vectors(contact_id, seekers[contact_id]))

    cand_ids = sorted(candidates)
    return EmbeddingPlan(
        seeker_vectors=svecs,
        seeker_texts=[v.text for v in svecs],
        candidate_ids=cand_ids,
        candidate_texts=[candidate_text(candidates[c]) for c in cand_ids],
    )


def embed_plan(
    plan: EmbeddingPlan,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 2,
    max_length: int = 4096,
    truncate_dim: int | None = None,
    device: str | None = None,
    cache_dir: Path | None = None,
    backend: str = "local",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode both sides. Returns ``(seeker_matrix, candidate_matrix)``, L2-normalized.

    ``backend="modal"`` runs on an A100 instead of locally — required for any
    7-8B model, which will not fit on local MPS.
    """
    if backend == "modal":
        from synth_pipeline.pairing_rrf_qwen_judge.modal_embed import embed_via_modal

        return embed_via_modal(
            plan.seeker_texts,
            plan.candidate_texts,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            truncate_dim=truncate_dim,
        )

    from baselines.hf_embedding.encode import get_encoder_class

    encoder_cls = get_encoder_class(model_name)
    encoder = encoder_cls(
        model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=cache_dir,
    )
    seeker_mat = encoder.encode(
        plan.seeker_texts,
        role="query",
        batch_size=batch_size,
        cache_name=_content_cache_name("rrf_seeker", plan.seeker_texts, model_name),
    )
    cand_mat = encoder.encode(
        plan.candidate_texts,
        role="document",
        batch_size=batch_size,
        cache_name=_content_cache_name("rrf_cand", plan.candidate_texts, model_name),
    )
    return seeker_mat, cand_mat


def persist(
    out_dir: Path,
    plan: EmbeddingPlan,
    seeker_mat: np.ndarray,
    cand_mat: np.ndarray,
    *,
    model_name: str,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write arrays + manifest. This is the copy that must survive."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "seeker_vectors.npy", seeker_mat.astype(np.float32))
    np.save(out_dir / "candidate_vectors.npy", cand_mat.astype(np.float32))

    seeker_records = [
        VectorRecord(
            key=v.key,
            contact_id=v.contact_id,
            role="seeker",
            section_index=v.section_index,
            section_text=v.section_text,
            text_sha256=_sha(v.text),
            row=i,
        )
        for i, v in enumerate(plan.seeker_vectors)
    ]
    candidate_records = [
        VectorRecord(
            key=cid,
            contact_id=cid,
            role="candidate",
            section_index=None,
            section_text=None,
            text_sha256=_sha(plan.candidate_texts[i]),
            row=i,
        )
        for i, cid in enumerate(plan.candidate_ids)
    ]

    manifest = {
        "model": model_name,
        "dim": int(seeker_mat.shape[1]) if seeker_mat.size else None,
        "n_seeker_vectors": len(seeker_records),
        "n_candidate_vectors": len(candidate_records),
        "n_seekers": len({r.contact_id for r in seeker_records}),
        "seeker": [r.to_dict() for r in seeker_records],
        "candidate": [r.to_dict() for r in candidate_records],
        **(extra_meta or {}),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return out_dir


def load_persisted(out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read back what ``persist`` wrote — the source of truth for everything downstream."""
    out_dir = Path(out_dir)
    seeker_mat = np.load(out_dir / "seeker_vectors.npy")
    cand_mat = np.load(out_dir / "candidate_vectors.npy")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return seeker_mat, cand_mat, manifest
