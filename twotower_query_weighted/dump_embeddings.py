"""Dump raw profile/query/candidate vectors for the topology visualization.

Not a scoring path — ``eval.py`` already answered the accuracy question. This
exists only to materialize the actual embedding vectors (which ``metrics.json``
never stores) for both the frozen encoder and the fine-tuned ``top1_ctrl``
adapter, so a 3D graph can be built comparing them.

One node per unique seeker (profile_only embedding), one node per unique
candidate (candidate embedding), one node per *pair* for the query (query_only
embedding) — a seeker can issue more than one query, so the query is keyed by
pair id, not seeker id.

Frozen embeddings are pulled from the same ``dorby-query-weighted-cache``
volume ``query_weighted/modal_eval.py`` already populated (content-hash keyed,
so this is a cache hit, not a re-encode) via ``VoyageNanoEncoder`` — read-only
reuse of that package's public API, no file there is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_real_full.data import load_real_pairs
from query_weighted import text as qtext


def _unique_nodes(raw: list[dict], id_key: str, file_key: str, builder) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for r in raw:
        nid = r[id_key]
        if nid in seen:
            continue
        seen.add(nid)
        ids.append(nid)
        texts.append(builder(r[file_key]))
    return ids, texts


def build_dump(
    frozen_encoder,
    finetuned_model,
    data_dir: Path,
    split_path: Path,
    *,
    batch_size: int = 4,
) -> dict[str, Any]:
    from twotower.eval import encode_role

    ps = load_real_pairs(data_dir, split_path, subset="all", verify=True)
    ordered = list(ps.pairs)
    raw = [p.pair for p in ordered]
    pair_ids = [p.pair_id for p in ordered]
    labels = [p.label for p in ordered]

    seeker_ids, seeker_texts = _unique_nodes(
        raw, "userContactId", "userContactFile", qtext.profile_only
    )
    cand_ids, cand_texts = _unique_nodes(
        raw, "matchContactId", "matchContactFile", qtext.candidate_to_text
    )
    query_texts = [qtext.query_only(p["userContactFile"], p["searchQuery"]) for p in raw]

    out: dict[str, Any] = {
        "pair_ids": pair_ids,
        "labels": labels,
        "seeker_id_by_pair": [p["userContactId"] for p in raw],
        "candidate_id_by_pair": [p["matchContactId"] for p in raw],
        "seeker_ids": seeker_ids,
        "candidate_ids": cand_ids,
        "models": {},
    }

    print(f"  frozen: encoding {len(seeker_texts)} seekers, {len(query_texts)} queries, {len(cand_texts)} candidates")
    out["models"]["frozen"] = {
        "seeker": frozen_encoder.encode(seeker_texts, role="query", batch_size=batch_size, show_progress=False).tolist(),
        "query": frozen_encoder.encode(query_texts, role="query", batch_size=batch_size, show_progress=False).tolist(),
        "candidate": frozen_encoder.encode(cand_texts, role="document", batch_size=batch_size, show_progress=False).tolist(),
    }

    print(f"  top1_ctrl: encoding {len(seeker_texts)} seekers, {len(query_texts)} queries, {len(cand_texts)} candidates")
    out["models"]["top1_ctrl"] = {
        "seeker": encode_role(finetuned_model, seeker_texts, role="query", batch_size=batch_size).tolist(),
        "query": encode_role(finetuned_model, query_texts, role="query", batch_size=batch_size).tolist(),
        "candidate": encode_role(finetuned_model, cand_texts, role="document", batch_size=batch_size).tolist(),
    }
    return out


def write_dump(dump: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump))
    print(f"\nwrote {out_path}")
    return out_path
