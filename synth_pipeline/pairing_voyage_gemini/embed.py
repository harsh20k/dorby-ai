"""Embed both sides with Voyage-4-large (API) and persist the vectors as files.

Files first, database second, same discipline as
``synth_pipeline.pairing_rrf_qwen_judge.embed``. Two differences from that
module, both deliberate:

* **Model**: Voyage-4-large via the API (``baselines.voyage_large.encode``,
  imported read-only), not Qwen3-Embedding-8B on Modal — no GPU step, no batch
  chunking, just the existing disk-cached, rate-limited encoder.
* **Rows**: one vector per *query target* on the seeker side (``positioning`` +
  that section's ``searchQuery``), not one whole-profile vector plus N section
  vectors. A seeker with 3 lookingFor sections gets exactly 3 seeker rows here,
  matching the 3 queries already generated for them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synth_pipeline.pairing_voyage_gemini.sections import (
    QueryTarget,
    candidate_text,
    seeker_query_text,
)

DEFAULT_MODEL = "voyage-4-large"
DEFAULT_OUTPUT_DIMENSION = 1024


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class VectorRecord:
    key: str
    contact_id: str
    role: str  # "seeker" | "candidate"
    section_index: int | None  # None for candidates
    text_sha256: str
    row: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "contact_id": self.contact_id,
            "role": self.role,
            "section_index": self.section_index,
            "text_sha256": self.text_sha256,
            "row": self.row,
        }


@dataclass
class EmbeddingPlan:
    """Every text to be embedded, in the exact row order of the output arrays."""

    query_targets: list[QueryTarget]
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
    targets: list[QueryTarget],
    queries: dict[str, str],
) -> EmbeddingPlan:
    """One seeker row per query target that has generated text, in target order
    (already sorted by contact id from ``run.py``); candidates stay one apiece,
    sorted by contact id for a byte-identical re-run."""
    kept_targets: list[QueryTarget] = []
    seeker_texts: list[str] = []
    for t in targets:
        query_text = queries.get(t.key, "")
        if not query_text.strip():
            continue
        kept_targets.append(t)
        seeker_texts.append(seeker_query_text(seekers[t.contact_id], query_text))

    cand_ids = sorted(candidates)
    return EmbeddingPlan(
        query_targets=kept_targets,
        seeker_texts=seeker_texts,
        candidate_ids=cand_ids,
        candidate_texts=[candidate_text(candidates[c]) for c in cand_ids],
    )


def _encode_chunked(encoder, texts: list[str], *, input_type: str, label: str) -> np.ndarray:
    """Call ``encoder.encode`` in chunks that stay under its own
    ``MAX_ESTIMATED_TOKENS`` safety cap (5M) — this batch's 5,506 candidate
    texts alone estimate ~5.7M tokens in one call, over the cap. Chunking here
    rather than raising the cap in ``baselines/voyage_large/encode.py``, which
    other callers share and whose limit is deliberate (isolation rule: don't
    edit shared code, work around it locally). The encoder's own disk cache
    still dedupes/short-circuits across chunks."""
    from baselines.voyage_large.encode import MAX_ESTIMATED_TOKENS, estimate_tokens

    if not texts:
        return np.zeros((0, encoder.output_dimension), dtype=np.float32)

    budget = int(MAX_ESTIMATED_TOKENS * 0.8)  # headroom under the cap
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for t in texts:
        t_tokens = estimate_tokens([t])
        if cur and cur_tokens + t_tokens > budget:
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += t_tokens
    if cur:
        chunks.append(cur)

    mats = [
        encoder.encode(chunk, input_type=input_type, label=f"{label} chunk {i + 1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ]
    return np.concatenate(mats, axis=0)


def embed_plan(
    plan: EmbeddingPlan,
    *,
    model_name: str = DEFAULT_MODEL,
    output_dimension: int = DEFAULT_OUTPUT_DIMENSION,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode both sides. Returns ``(seeker_matrix, candidate_matrix)``, L2-normalized."""
    from baselines.voyage_large.encode import VoyageLargeEncoder

    encoder = VoyageLargeEncoder(
        model_name=model_name,
        output_dimension=output_dimension,
        cache_dir=cache_dir,
    )
    seeker_mat = _encode_chunked(encoder, plan.seeker_texts, input_type="query", label="seeker")
    cand_mat = _encode_chunked(encoder, plan.candidate_texts, input_type="document", label="candidate")
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "seeker_vectors.npy", seeker_mat.astype(np.float32))
    np.save(out_dir / "candidate_vectors.npy", cand_mat.astype(np.float32))

    seeker_records = [
        VectorRecord(
            key=t.key,
            contact_id=t.contact_id,
            role="seeker",
            section_index=t.section_index,
            text_sha256=_sha(plan.seeker_texts[i]),
            row=i,
        )
        for i, t in enumerate(plan.query_targets)
    ]
    candidate_records = [
        VectorRecord(
            key=cid,
            contact_id=cid,
            role="candidate",
            section_index=None,
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
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def load_persisted(out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    out_dir = Path(out_dir)
    seeker_mat = np.load(out_dir / "seeker_vectors.npy")
    cand_mat = np.load(out_dir / "candidate_vectors.npy")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return seeker_mat, cand_mat, manifest
