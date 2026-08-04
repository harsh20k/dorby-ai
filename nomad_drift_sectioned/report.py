"""Score the section-selected alpha blend on all 200 real pairs.

Reuses ``query_weighted.eval.encode_everything``/``score_arm`` directly — this
module only builds one extra seeker representation (best-section-selected,
blended with the query) and hands it to the exact scoring function that
produced every other row in this project's tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval_real_full.data import Subset, load_real_pairs
from nomad_drift.calibrate import combine_batch
from query_weighted.eval import DEFAULT_SUBSETS, Encoded, encode_everything, score_arm
from synth_pipeline.pairing_rrf.sections import seeker_vectors


def _reconstruct_raw_by_pair_index(enc: Encoded) -> list[dict[str, Any]]:
    """Undo encode_everything's pos/neg split back into enc.pair_ids order."""
    raw: list[dict[str, Any] | None] = [None] * len(enc.pair_ids)
    for k, i in enumerate(enc.pos_index):
        raw[i] = enc.positives[k]
    for k, i in enumerate(enc.neg_index):
        raw[i] = enc.negatives[k]
    assert all(r is not None for r in raw), "every row must be either pos or neg"
    return raw  # type: ignore[return-value]


def build_section_selected_seeker_matrix(encoder: Any, enc: Encoded, batch_size: int = 4) -> np.ndarray:
    """For every real pair, the seeker's lookingFor section closest to that pair's query."""
    raw = _reconstruct_raw_by_pair_index(enc)

    seeker_profiles: dict[str, dict[str, Any]] = {}
    for r in raw:
        seeker_profiles.setdefault(r["userContactId"], r["userContactFile"])

    section_texts: list[str] = []
    seeker_slice: dict[str, tuple[int, int]] = {}
    for seeker_id, profile in seeker_profiles.items():
        vecs = seeker_vectors(seeker_id, profile)
        per_section = [v for v in vecs if not v.is_whole] or vecs
        start = len(section_texts)
        section_texts.extend(v.text for v in per_section)
        seeker_slice[seeker_id] = (start, len(section_texts))

    print(f"  encoding {len(section_texts)} seeker lookingFor sections ({len(seeker_profiles)} seekers)")
    section_mat = encoder.encode(section_texts, role="query", batch_size=batch_size, show_progress=False)
    sections_by_seeker = {sid: section_mat[a:b] for sid, (a, b) in seeker_slice.items()}

    query_mat = enc.seeker["query_only"]
    selected = np.zeros_like(query_mat)
    for i, r in enumerate(raw):
        candidates = sections_by_seeker[r["userContactId"]]
        sims = candidates @ query_mat[i]
        selected[i] = candidates[int(np.argmax(sims))]
    return selected


@dataclass(frozen=True)
class SectionReportEncoded:
    enc: Encoded
    section_selected: np.ndarray  # (n_pairs, d), aligned to enc.pair_ids


def build_section_report_encoding(
    encoder: Any, data_dir: Path, split_path: Path, batch_size: int = 4
) -> SectionReportEncoded:
    enc = encode_everything(
        encoder,
        data_dir,
        split_path,
        batch_size=batch_size,
        arms=["concat_baseline", "profile_only", "query_only"],
    )
    section_vecs = build_section_selected_seeker_matrix(encoder, enc, batch_size=batch_size)
    return SectionReportEncoded(enc=enc, section_selected=section_vecs)


def score_alphas(
    sre: SectionReportEncoded,
    alphas: Sequence[float],
    data_dir: Path,
    split_path: Path,
    subsets: Sequence[Subset] = DEFAULT_SUBSETS,
) -> dict[str, Any]:
    subset_ids: dict[str, set[str]] = {}
    for subset in subsets:
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        subset_ids[subset] = {p.pair_id for p in ps.pairs}

    out: dict[str, Any] = {"arms": {}}
    for alpha in alphas:
        blended = combine_batch(sre.enc.seeker["query_only"], sre.section_selected, alpha)
        name = f"section_alpha_{alpha:.1f}"
        out["arms"][name] = {subset: score_arm(sre.enc, blended, subset_ids[subset]) for subset in subsets}
        a = out["arms"][name]["all"]
        print(
            f"{name:22s} all-200  AUC={a['pair']['roc_auc']:.4f} "
            f"hardneg={a['slices']['neg_hardness']['hard']['pair_auc']:.4f} "
            f"MRR={a['retrieval']['mrr']:.4f} R@1={a['retrieval']['recall@1']:.4f} "
            f"R@10={a['retrieval']['recall@10']:.4f}"
        )
    return out
