"""Calibrate alpha for query + best-matching-section blend, using rrf_003's ground truth.

Every rrf_003 query was generated *for* one specific lookingFor section
(``synth_pipeline/pairing_rrf/sections.py::query_targets``), so this module can
check not just "does the blend help" but "does cosine-similarity selection
actually pick the section the query was written for."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nomad_drift.calibrate import ALPHAS, RRF_BATCH_DEFAULT, AlignedArrays, pick_best_alpha, sweep
from query_weighted import text as qtext
from synth_pipeline.pairing_rrf.sections import seeker_vectors


@dataclass(frozen=True)
class SectionedRrfRecord:
    pair_key: str
    seeker_id: str
    candidate_id: str
    query_key: str
    label: str
    user_contact_file: dict[str, Any]
    match_contact_file: dict[str, Any]
    search_query: str
    true_section_index: int  # the section this query_key was generated for


def load_sectioned_records(batch_dir: Path = RRF_BATCH_DEFAULT) -> list[SectionedRrfRecord]:
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    records: list[SectionedRrfRecord] = []
    for rec in manifest["records"]:
        staged = json.loads((batch_dir / rec["path"]).read_text())
        pair = staged["pair"]
        records.append(
            SectionedRrfRecord(
                pair_key=rec["pair_key"],
                seeker_id=rec["seeker_id"],
                candidate_id=rec["candidate_id"],
                query_key=rec["query_key"],
                label=rec["label"],
                user_contact_file=pair["userContactFile"],
                match_contact_file=pair["matchContactFile"],
                search_query=pair["searchQuery"],
                true_section_index=staged["metadata"]["section_index"],
            )
        )
    return records


@dataclass(frozen=True)
class SectionEncoded:
    records: list[SectionedRrfRecord]
    query_by_key: dict[str, np.ndarray]
    candidate_by_id: dict[str, np.ndarray]
    sections_by_seeker: dict[str, list[tuple[int, np.ndarray]]]  # (section_index, vector)


def encode_sectioned_calibration_set(
    encoder: Any, records: list[SectionedRrfRecord], batch_size: int = 4
) -> SectionEncoded:
    query_texts: dict[str, str] = {}
    candidate_texts: dict[str, str] = {}
    seeker_profiles: dict[str, dict[str, Any]] = {}
    for r in records:
        query_texts.setdefault(r.query_key, qtext.query_only(r.user_contact_file, r.search_query))
        candidate_texts.setdefault(r.candidate_id, qtext.candidate_to_text(r.match_contact_file))
        seeker_profiles.setdefault(r.seeker_id, r.user_contact_file)

    section_texts: list[str] = []
    section_meta: list[tuple[str, int]] = []  # (seeker_id, section_index) aligned to section_texts
    seeker_slice: dict[str, tuple[int, int]] = {}
    for seeker_id, profile in seeker_profiles.items():
        vecs = seeker_vectors(seeker_id, profile)
        per_section = [v for v in vecs if not v.is_whole] or vecs  # single-section seekers fall back to whole
        start = len(section_texts)
        for v in per_section:
            section_texts.append(v.text)
            section_meta.append((seeker_id, v.section_index))
        seeker_slice[seeker_id] = (start, len(section_texts))

    query_keys = list(query_texts)
    cand_ids = list(candidate_texts)

    print(f"  encoding {len(query_keys)} unique queries")
    query_mat = encoder.encode(
        [query_texts[q] for q in query_keys], role="query", batch_size=batch_size, show_progress=False
    )
    print(f"  encoding {len(cand_ids)} unique candidates")
    cand_mat = encoder.encode(
        [candidate_texts[c] for c in cand_ids], role="document", batch_size=batch_size, show_progress=False
    )
    print(f"  encoding {len(section_texts)} seeker lookingFor sections ({len(seeker_profiles)} seekers)")
    section_mat = encoder.encode(section_texts, role="query", batch_size=batch_size, show_progress=False)

    sections_by_seeker: dict[str, list[tuple[int, np.ndarray]]] = {}
    for seeker_id, (start, end) in seeker_slice.items():
        sections_by_seeker[seeker_id] = [(section_meta[i][1], section_mat[i]) for i in range(start, end)]

    return SectionEncoded(
        records=records,
        query_by_key={q: query_mat[i] for i, q in enumerate(query_keys)},
        candidate_by_id={c: cand_mat[i] for i, c in enumerate(cand_ids)},
        sections_by_seeker=sections_by_seeker,
    )


def select_and_align(enc: SectionEncoded) -> tuple[AlignedArrays, dict[str, Any]]:
    """Pick each record's best-matching section by cosine to the query.

    Reports selection accuracy against rrf_003's ground-truth section index,
    counted only where a seeker has >=2 sections (otherwise there is only one
    thing to pick and the check is vacuous).
    """
    selected: list[np.ndarray] = []
    query_vecs: list[np.ndarray] = []
    cand_vecs: list[np.ndarray] = []
    labels: list[str] = []
    query_keys: list[str] = []

    n_checkable = 0
    n_correct = 0

    for r in enc.records:
        qv = enc.query_by_key[r.query_key]
        sections = enc.sections_by_seeker[r.seeker_id]
        sims = [float(np.dot(qv, sv)) for _, sv in sections]
        best_local = int(np.argmax(sims))
        best_section_index, best_vec = sections[best_local]

        if len(sections) >= 2:
            n_checkable += 1
            if best_section_index == r.true_section_index:
                n_correct += 1

        selected.append(best_vec)
        query_vecs.append(qv)
        cand_vecs.append(enc.candidate_by_id[r.candidate_id])
        labels.append(r.label)
        query_keys.append(r.query_key)

    arrays = AlignedArrays(
        profile=np.stack(selected),
        query=np.stack(query_vecs),
        candidate=np.stack(cand_vecs),
        labels=labels,
        query_keys=query_keys,
    )
    selection_stats = {
        "n_checkable": n_checkable,
        "n_correct": n_correct,
        "selection_accuracy": (n_correct / n_checkable) if n_checkable else None,
    }
    return arrays, selection_stats


def run_sectioned_calibration(
    encoder: Any, batch_dir: Path = RRF_BATCH_DEFAULT, batch_size: int = 4
) -> dict[str, Any]:
    records = load_sectioned_records(batch_dir)
    print(f"loaded {len(records)} rrf_003 pairs")
    enc = encode_sectioned_calibration_set(encoder, records, batch_size=batch_size)
    arrays, selection_stats = select_and_align(enc)
    acc = selection_stats["selection_accuracy"]
    acc_s = f"{acc:.4f}" if acc is not None else "n/a"
    print(
        f"  section selection accuracy: {acc_s} "
        f"({selection_stats['n_correct']}/{selection_stats['n_checkable']})"
    )

    results = sweep(arrays, ALPHAS)
    best_alpha = pick_best_alpha(results)
    for r in results:
        marker = "  <-- best" if r["alpha"] == best_alpha else ""
        sl = r["shortlist_top1_accuracy"]
        sl_s = f"{sl:.4f}" if sl is not None else "n/a"
        print(f"  alpha={r['alpha']:.1f}  pairAUC={r['pair_auc']:.4f}  shortlist_top1={sl_s}{marker}")

    return {
        "batch_dir": str(batch_dir),
        "n_records": len(records),
        "n_unique_seekers": len({r.seeker_id for r in records}),
        "selection_stats": selection_stats,
        "sweep": results,
        "best_alpha": best_alpha,
    }
