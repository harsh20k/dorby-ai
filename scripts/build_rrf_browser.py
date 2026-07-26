#!/usr/bin/env python3
"""Self-contained HTML browser for an ``artifacts/pairing_rrf/<batch>`` batch.

Three tabs, one file, zero network requests:

**Pairs** — every labeled pair with its retrieval provenance and the judge's
written reasoning, plus the leakage/circularity probes up front. These labels are
one model's opinion, so the useful question is not "is this pair right" but "is
the label predictable from something that is not matching semantics".

**Topology** — the batch's contact graph beside the 200 real pairs. Two numbers
drive it and both are easier to see than to read: real data is 0.673 edges per
node across 97 disconnected components, this batch is 3.022 across exactly one.

**Embeddings** — the Qwen3 vectors projected to 3D. Seeker profile anchors with
their ``lookingFor`` asks tethered to them, candidates, and the pos/neg pair
edges drawn between profile points.

The existing browsers do not fit this data: ``build_synth_browser.py`` expects
the LangGraph pipeline's staged/dropped layout with ``failure_mode`` negatives
and a human-review workflow, and ``build_profile_browser.py`` only understands
unlabeled profiles.

    python scripts/build_rrf_browser.py --batch-id rrf_002
    open artifacts/pairing_rrf/rrf_002/_browser.html

Degrades rather than fails: without ``data/`` (gitignored) the topology tab shows
one pane; without ``embeddings/`` or numpy/sklearn the embeddings tab is hidden
and the first two tabs still build under bare system Python.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PROFILE_FIELDS = [
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
]

# The real-pane graph is the only place profile text is duplicated (those 297
# contacts are not in this batch's contact table), so it is truncated hard —
# untruncated it alone costs 2.7 MB.
REAL_TRUNCATE = {"positioning": 260, "background": 320, "lookingFor": 420}
SECTION_CHARS = 420
QUERY_CHARS = 200


def truncate(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------- loading


def load_batch(batch_dir: Path) -> dict[str, Any]:
    staged = sorted((batch_dir / "staged").glob("*.json"))
    if not staged:
        raise SystemExit(f"no staged pairs in {batch_dir}")
    pairs = [json.loads(p.read_text(encoding="utf-8")) for p in staged]

    summary = {}
    summary_path = batch_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {"pairs": pairs, "summary": summary}


# ------------------------------------------------------------ diagnostics


def _auc(y: list[int], score: list[float]) -> float | None:
    """Rank-based ROC-AUC with tie handling. Kept dependency-free on purpose —
    the first two tabs must build even without the project venv active."""
    pairs = sorted(zip(score, y))
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos_rank_sum = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def diagnostics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    y = [1 if p["label"] == "pos" else 0 for p in pairs]
    seekers = [p["pair"]["userContactId"] for p in pairs]
    cands = [p["pair"]["matchContactId"] for p in pairs]
    rrf = [(p.get("metadata", {}).get("fusion") or {}).get("rrf_score") or 0.0 for p in pairs]

    def loo(nodes: list[str]) -> list[float]:
        """Leave-one-out positive rate per node — a text-free identity predictor.
        If this alone separates the classes, the label is partly 'who is in the
        pair' rather than 'do these two fit'."""
        tot: dict[str, int] = defaultdict(int)
        pos: dict[str, int] = defaultdict(int)
        for n, lab in zip(nodes, y):
            tot[n] += 1
            pos[n] += lab
        prior = sum(y) / len(y)
        return [
            (pos[n] - lab) / (tot[n] - 1) if tot[n] > 1 else prior
            for n, lab in zip(nodes, y)
        ]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(seekers):
        groups[s].append(i)
    mixed = [g for g in groups.values() if 0 < sum(y[i] for i in g) < len(g)]
    within = None
    if mixed:
        per, weights = [], []
        for g in mixed:
            a = _auc([y[i] for i in g], [rrf[i] for i in g])
            if a is not None:
                per.append(a)
                weights.append(len(g))
        if per:
            within = sum(a * w for a, w in zip(per, weights)) / sum(weights)

    seek_stats = []
    for s, idx in groups.items():
        n_pos = sum(y[i] for i in idx)
        seek_stats.append({"id": s, "n": len(idx), "pos": n_pos, "rate": n_pos / len(idx)})
    seek_stats.sort(key=lambda d: (-d["rate"], -d["n"]))

    cand_count: dict[str, int] = defaultdict(int)
    for c in cands:
        cand_count[c] += 1

    triplets = sum(
        sum(y[i] for i in idx) * (len(idx) - sum(y[i] for i in idx)) for idx in groups.values()
    )

    return {
        "n_pairs": len(pairs),
        "n_pos": sum(y),
        "n_neg": len(y) - sum(y),
        "pos_frac": sum(y) / len(y),
        "auc_seeker_identity": _auc(y, loo(seekers)),
        "auc_candidate_identity": _auc(y, loo(cands)),
        "auc_rrf_pooled": _auc(y, rrf),
        "auc_rrf_within_seeker": within,
        "n_seekers": len(groups),
        "n_seekers_mixed": len(mixed),
        "n_seekers_all_neg": sum(1 for s in seek_stats if s["rate"] == 0),
        "n_candidates": len(cand_count),
        "max_candidate_reuse": max(cand_count.values()) if cand_count else 0,
        "triplets": triplets,
        "seeker_stats": seek_stats,
    }


# -------------------------------------------------------- tab 1: pairs


def contact_table(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One entry per distinct contact, replacing the per-pair profile copies.

    There are 91 contacts across 275 pairs, so embedding profiles on each pair
    duplicated every one of them 5-7 times — that alone was 3.8 MB. All three
    tabs read this table instead.
    """
    out: dict[str, dict[str, Any]] = {}

    def touch(cid: str, profile: dict[str, Any], role: str, is_pos: bool) -> None:
        entry = out.get(cid)
        if entry is None:
            entry = {k: profile.get(k) for k in PROFILE_FIELDS}
            entry.update({"roles": set(), "pairCount": 0, "posCount": 0})
            out[cid] = entry
        entry["roles"].add(role)
        entry["pairCount"] += 1
        entry["posCount"] += int(is_pos)

    for p in pairs:
        pair, is_pos = p["pair"], p["label"] == "pos"
        touch(pair["userContactId"], pair["userContactFile"], "seeker", is_pos)
        touch(pair["matchContactId"], pair["matchContactFile"], "candidate", is_pos)

    for entry in out.values():
        roles = entry.pop("roles")
        entry["role"] = "both" if len(roles) == 2 else next(iter(roles))
    return out


def compact_pair(p: dict[str, Any], idx: int) -> dict[str, Any]:
    """Pair row without profiles — the browser resolves ids against ``contacts``."""
    pair = p["pair"]
    qc = p.get("qc") or {}
    fusion = (p.get("metadata") or {}).get("fusion") or {}
    return {
        "i": idx,
        "label": p["label"],
        "seeker_id": pair["userContactId"],
        "cand_id": pair["matchContactId"],
        "query": pair.get("searchQuery", ""),
        "section": (p.get("metadata") or {}).get("section_index"),
        "reasoning": qc.get("judge_reasoning", ""),
        "confidence": qc.get("judge_confidence_unused"),
        "rrf": fusion.get("rrf_score"),
        "dense_rank": fusion.get("dense_rank"),
        "lex_rank": fusion.get("lexical_rank"),
        "dense_score": fusion.get("dense_score"),
        "both": bool(fusion.get("found_by_both")),
    }


# ----------------------------------------------------- tab 2: topology
#
# build_graph / graph_stats / suggest_physics below are deliberate duplicates of
# scripts/build_real_pairs_graph.py:54 / :216 / :229. `scripts/` has no
# __init__.py, so importing across it needs a sys.path hack; repo convention
# (CLAUDE.md, on synth_pipeline/pairing/bedrock.py) prefers a small copy.


def is_synthetic(contact_id: str) -> bool:
    return contact_id.startswith("cmsynth")


def build_graph(
    pos_pairs: list[dict],
    neg_pairs: list[dict],
    *,
    include: Callable[[str], bool] = lambda cid: True,
    truncate_profile: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Contacts to nodes, seeker->candidate pairs to labeled directed edges.

    Mirrors build_real_pairs_graph.py:54, plus ``truncate_profile`` — the
    reference stores all eight fields verbatim, which this browser cannot afford.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def touch_node(contact_id: str, file: dict[str, Any] | None, role: str) -> None:
        node = nodes.setdefault(
            contact_id, {"id": contact_id, "profile": {}, "roles": set(), "pairCount": 0}
        )
        node["roles"].add(role)
        node["pairCount"] += 1
        if not node["profile"] and file:
            if truncate_profile is None:
                node["profile"] = {k: file.get(k) for k in PROFILE_FIELDS}
            else:
                node["profile"] = {
                    k: truncate(file.get(k), lim) for k, lim in truncate_profile.items()
                }

    for label, pairs in (("pos", pos_pairs), ("neg", neg_pairs)):
        for p in pairs:
            seeker_id, cand_id = p["userContactId"], p["matchContactId"]
            if not (include(seeker_id) and include(cand_id)):
                continue
            touch_node(seeker_id, p.get("userContactFile"), "seeker")
            touch_node(cand_id, p.get("matchContactFile"), "candidate")
            edges.append(
                {
                    "source": seeker_id,
                    "target": cand_id,
                    "label": label,
                    "searchQuery": truncate(p.get("searchQuery"), QUERY_CHARS),
                }
            )

    for node in nodes.values():
        roles = node.pop("roles")
        node["role"] = "both" if len(roles) == 2 else next(iter(roles))
    return {"nodes": list(nodes.values()), "edges": edges}


def _components(node_ids: list[str], edges: list[dict[str, Any]]) -> int:
    """Connected components, union-find. 97 vs 1 is the sharpest single
    statement of how differently shaped the two graphs are."""
    parent = {n: n for n in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        a, b = find(e["source"]), find(e["target"])
        if a != b:
            parent[a] = b
    return len({find(n) for n in node_ids})


def graph_stats(graph: dict[str, Any]) -> dict[str, Any]:
    """build_real_pairs_graph.py:216, plus ``components``."""
    n_nodes, n_edges = len(graph["nodes"]), len(graph["edges"])
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "pos": sum(1 for e in graph["edges"] if e["label"] == "pos"),
        "neg": sum(1 for e in graph["edges"] if e["label"] == "neg"),
        "both_role": sum(1 for n in graph["nodes"] if n["role"] == "both"),
        "mean_degree": round(n_edges / n_nodes, 3) if n_nodes else 0.0,
        "components": _components([n["id"] for n in graph["nodes"]], graph["edges"]),
    }


def suggest_physics(stats: dict[str, Any]) -> dict[str, float]:
    """build_real_pairs_graph.py:229, verbatim. A dense batch needs more
    repulsion than the sparse real graph or it collapses into a ball."""
    ratio = max(1.0, stats["mean_degree"] / 1.35) if stats["nodes"] else 1.0
    return {"repulsion": round(4200 * ratio, 1), "springLen": round(90 * (ratio**0.5), 1)}


def load_real_pairs(pos_path: Path, neg_path: Path) -> tuple[list[dict], list[dict]] | None:
    """The canonical datasets, real contacts only. ``data/`` is gitignored, so
    returning None on a fresh checkout is expected, not an error."""
    try:
        pos = json.loads(pos_path.read_text(encoding="utf-8"))
        neg = json.loads(neg_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return None
    return pos, neg


def topology_payload(
    pairs: list[dict[str, Any]],
    contacts: dict[str, dict[str, Any]],
    real: tuple[list[dict], list[dict]] | None,
) -> dict[str, Any]:
    synth_graph = {
        "nodes": [{"id": cid, "role": c["role"], "pairCount": c["pairCount"]}
                  for cid, c in contacts.items()],
        "edges": [
            {"source": p["pair"]["userContactId"], "target": p["pair"]["matchContactId"],
             "label": p["label"], "searchQuery": p["pair"].get("searchQuery")}
            for p in pairs
        ],
    }
    synth_stats = graph_stats(synth_graph)
    # Nodes and edges are omitted: the browser rebuilds them from `contacts` and
    # `pairs`, which it already holds. Shipping them again costs ~600 KB.
    out: dict[str, Any] = {
        "synth": {"stats": synth_stats, "physics": suggest_physics(synth_stats)},
        "real": None,
    }

    if real is not None:
        real_only = lambda cid: not is_synthetic(cid)  # noqa: E731
        rg = build_graph(real[0], real[1], include=real_only, truncate_profile=REAL_TRUNCATE)
        rstats = graph_stats(rg)
        out["real"] = {
            "nodes": rg["nodes"],
            "edges": rg["edges"],
            "stats": rstats,
            "physics": suggest_physics(rstats),
        }
    return out


# --------------------------------------------------- tab 3: embeddings


def embedding_payload(batch_dir: Path, pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Project the batch's vectors to 3D and measure the geometry.

    numpy/sklearn are imported here rather than at module scope so the first two
    tabs keep building without the project venv (see ``_auc``).
    """
    emb_dir = batch_dir / "embeddings"
    manifest_path = emb_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:
        print(f"  embeddings tab skipped: {exc}", file=sys.stderr)
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeker_vecs = np.load(emb_dir / "seeker_vectors.npy")
    cand_vecs = np.load(emb_dir / "candidate_vectors.npy")

    # Optional: isolated field/ask vectors from scripts/embed_rrf_field_isolation.py
    # (a separate, later embedding pass -- not every batch has run it).
    iso_dir = batch_dir / "field_isolation"
    iso_vecs: Any = None
    iso_meta: dict[str, Any] | None = None
    if (iso_dir / "isolated_vectors.npy").exists() and (iso_dir / "meta.json").exists():
        iso_vecs = np.load(iso_dir / "isolated_vectors.npy")
        iso_meta = json.loads((iso_dir / "meta.json").read_text(encoding="utf-8"))

    # One joint PCA over every vector, so a section's (or isolated field's) offset
    # from its own anchor is on the same scale as the gap between two people.
    # Fitting the parts separately would make cross-part distances meaningless.
    parts = [seeker_vecs, cand_vecs]
    if iso_vecs is not None:
        parts.append(iso_vecs)
    stacked = np.vstack(parts)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(stacked)
    coords = coords / (np.abs(coords).max() or 1.0)
    n_seek_rows = len(seeker_vecs)
    n_cand_rows = len(cand_vecs)
    evr = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    xyz = lambda i: [round(float(v), 5) for v in coords[i]]  # noqa: E731

    anchors: dict[str, int] = {}
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest["seeker"]:
        cid, row = entry["contact_id"], entry["row"]
        if entry["section_index"] == -1:
            anchors[cid] = row
        else:
            sections[cid].append(
                {"i": entry["section_index"], "row": row,
                 "text": truncate(entry.get("section_text"), SECTION_CHARS)}
            )
    cand_rows = {e["contact_id"]: e["row"] for e in manifest["candidate"]}

    iso_by_contact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    iso_offset = n_seek_rows + n_cand_rows
    if iso_meta is not None:
        for r in iso_meta["rows"]:
            iso_by_contact[r["contact_id"]].append(
                {
                    "field": r["field"],
                    "i": r["section_index"],
                    "kind": r["kind"],
                    "text": truncate(r.get("text"), SECTION_CHARS),
                    "row": r["row"],
                }
            )

    def field_entries(cid: str, whole_vec) -> list[dict[str, Any]]:
        out = []
        for r in sorted(iso_by_contact.get(cid, []), key=lambda d: (d["field"], d["i"] or 0)):
            iso_row = r["row"]
            cos = float(whole_vec @ iso_vecs[iso_row])
            out.append(
                {
                    "field": r["field"], "i": r["i"], "kind": r["kind"],
                    "text": r["text"], "cos": round(cos, 4),
                    "xyz": xyz(iso_offset + iso_row),
                }
            )
        return out

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [pairs, positives]
    for p in pairs:
        for cid in (p["pair"]["userContactId"], p["pair"]["matchContactId"]):
            counts[cid][0] += 1
            counts[cid][1] += int(p["label"] == "pos")

    people: list[dict[str, Any]] = []
    section_cosines: list[float] = []
    for cid, row in anchors.items():
        secs = []
        for s in sorted(sections[cid], key=lambda d: d["i"]):
            cos = float(seeker_vecs[row] @ seeker_vecs[s["row"]])
            section_cosines.append(cos)
            secs.append({"i": s["i"], "text": s["text"], "cos": round(cos, 4),
                         "xyz": xyz(s["row"])})
        people.append({"id": cid, "role": "seeker", "pairCount": counts[cid][0],
                       "posCount": counts[cid][1], "whole": xyz(row), "sections": secs,
                       "fields": field_entries(cid, seeker_vecs[row])})
    for cid, row in cand_rows.items():
        people.append({"id": cid, "role": "candidate", "pairCount": counts[cid][0],
                       "posCount": counts[cid][1], "whole": xyz(n_seek_rows + row),
                       "sections": [], "fields": field_entries(cid, cand_vecs[row])})

    def mean(vals) -> float | None:
        vals = [float(v) for v in vals]
        return round(sum(vals) / len(vals), 4) if vals else None

    def upper_mean(mat) -> float | None:
        if len(mat) < 2:
            return None
        gram = mat @ mat.T
        iu = np.triu_indices(len(mat), 1)
        return round(float(gram[iu].mean()), 4)

    within: list[float] = []
    for cid in anchors:
        rows = [s["row"] for s in sections[cid]]
        if len(rows) > 1:
            m = seeker_vecs[rows]
            gram = m @ m.T
            iu = np.triu_indices(len(rows), 1)
            within.extend(gram[iu].tolist())

    # The headline: if accepted and declined pairs sit at the same cosine, the
    # dense channel's geometry does not separate the judge's label at all.
    pair_cos = {"pos": [], "neg": []}
    for p in pairs:
        s_row = anchors.get(p["pair"]["userContactId"])
        c_row = cand_rows.get(p["pair"]["matchContactId"])
        if s_row is None or c_row is None:
            continue
        pair_cos[p["label"]].append(float(seeker_vecs[s_row] @ cand_vecs[c_row]))

    anchor_mat = seeker_vecs[[anchors[c] for c in anchors]] if anchors else seeker_vecs[:0]
    staged_cands = {p["pair"]["matchContactId"] for p in pairs}

    field_cosines = [f["cos"] for p in people for f in p["fields"]]

    return {
        "model": manifest.get("model"),
        "dim": manifest.get("dim"),
        "evr": evr,
        "evrSum": round(sum(evr), 4),
        "people": people,
        "hasFields": iso_meta is not None,
        "isoModel": (iso_meta or {}).get("model"),
        "stats": {
            "nSeekers": len(anchors),
            "nCandidates": len(cand_rows),
            "nSections": sum(len(v) for v in sections.values()),
            "nFields": sum(len(v) for v in iso_by_contact.values()),
            "nOrphanCandidates": len(set(cand_rows) - staged_cands),
            "wholeSectionCos": mean(section_cosines),
            "sectionSectionCos": mean(within),
            "wholeFieldCos": mean(field_cosines),
            "acrossSeekerCos": upper_mean(anchor_mat),
            "acrossCandCos": upper_mean(cand_vecs),
            "pairCosPos": mean(pair_cos["pos"]),
            "pairCosNeg": mean(pair_cos["neg"]),
        },
    }


# ---------------------------------------------------------------- payload


def build_payload(
    data: dict[str, Any],
    batch_id: str,
    batch_dir: Path,
    real: tuple[list[dict], list[dict]] | None,
) -> dict[str, Any]:
    pairs = data["pairs"]
    contacts = contact_table(pairs)
    return {
        "batch_id": batch_id,
        "labeler": (pairs[0].get("metadata") or {}).get("labeler") or {},
        "summary": data["summary"],
        "diag": diagnostics(pairs),
        "contacts": contacts,
        "pairs": [compact_pair(p, i) for i, p in enumerate(pairs)],
        "topology": topology_payload(pairs, contacts, real),
        "embed": embedding_payload(batch_dir, pairs),
    }


# --------------------------------------------------------------- template

_CSS = r"""
:root{
  --ink:#12161c; --ink-2:#3d4653; --ink-3:#6b7686;
  --bg:#f7f8fa; --panel:#ffffff; --line:#dfe3ea; --line-2:#eceff4;
  --accent:#3b4ea8; --accent-soft:#e6e9f6;
  --pos:#1a7a5e; --pos-bg:#dff2ea; --neg:#a63a4a; --neg-bg:#fbe4e8;
  --seeker:#3b4ea8; --cand:#b8862c; --both:#7a4fa3; --sect:#b8547c; --muted:#9aa3b0;
  --f-positioning:#3f9e6d; --f-background:#c2762f; --f-lookingfor:#d4463f; --f-notes:#4f83c9;
  --f-locationavailability:#2f9bb8; --f-intropreferences:#a68b1f; --f-personalpreferences:#5a5ac9;
  --f-meetingandschedulingpreferences:#8a5a3f;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{
    --ink:#e6eaf1; --ink-2:#a7b1c0; --ink-3:#78838f;
    --bg:#10131a; --panel:#171b24; --line:#272d3a; --line-2:#1e232d;
    --accent:#8fa2e8; --accent-soft:#1e2540;
    --pos:#5fd3a8; --pos-bg:#12332a; --neg:#f08fa0; --neg-bg:#361c23;
    --seeker:#8fa2e8; --cand:#dcab5e; --both:#b98fd8; --sect:#e69ab8; --muted:#6b7686;
    --f-positioning:#6fd9a8; --f-background:#e6a15e; --f-lookingfor:#f08f86; --f-notes:#8fb8e6;
    --f-locationavailability:#6fd4e6; --f-intropreferences:#d9c26f; --f-personalpreferences:#9a9af0;
    --f-meetingandschedulingpreferences:#c99a7a;
  }
}
:root[data-theme="dark"]{
  --ink:#e6eaf1; --ink-2:#a7b1c0; --ink-3:#78838f;
  --bg:#10131a; --panel:#171b24; --line:#272d3a; --line-2:#1e232d;
  --accent:#8fa2e8; --accent-soft:#1e2540;
  --pos:#5fd3a8; --pos-bg:#12332a; --neg:#f08fa0; --neg-bg:#361c23;
  --seeker:#8fa2e8; --cand:#dcab5e; --both:#b98fd8; --sect:#e69ab8; --muted:#6b7686;
    --f-positioning:#6fd9a8; --f-background:#e6a15e; --f-lookingfor:#f08f86; --f-notes:#8fb8e6;
    --f-locationavailability:#6fd4e6; --f-intropreferences:#d9c26f; --f-personalpreferences:#9a9af0;
    --f-meetingandschedulingpreferences:#c99a7a;
}
:root[data-theme="light"]{
  --ink:#12161c; --ink-2:#3d4653; --ink-3:#6b7686;
  --bg:#f7f8fa; --panel:#ffffff; --line:#dfe3ea; --line-2:#eceff4;
  --accent:#3b4ea8; --accent-soft:#e6e9f6;
  --pos:#1a7a5e; --pos-bg:#dff2ea; --neg:#a63a4a; --neg-bg:#fbe4e8;
  --seeker:#3b4ea8; --cand:#b8862c; --both:#7a4fa3; --sect:#b8547c; --muted:#9aa3b0;
  --f-positioning:#3f9e6d; --f-background:#c2762f; --f-lookingfor:#d4463f; --f-notes:#4f83c9;
  --f-locationavailability:#2f9bb8; --f-intropreferences:#a68b1f; --f-personalpreferences:#5a5ac9;
  --f-meetingandschedulingpreferences:#8a5a3f;
}
*{box-sizing:border-box}
[hidden]{display:none !important}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);
     -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 0}
.fullbleed{max-width:1600px;margin:0 auto;padding:0 20px 70px}
h1{font-size:22px;margin:0 0 2px;letter-spacing:-.01em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3);
   margin:0 0 12px;font-weight:600}
.sub{color:var(--ink-3);font-size:13px;font-family:var(--mono);margin-bottom:18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
       padding:18px 20px;margin-bottom:18px}

.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:20px;
      position:sticky;top:0;z-index:20;background:var(--bg);padding-top:4px}
.tabs button{font:15px var(--sans);border:0;border-bottom:2px solid transparent;
  border-radius:0;background:none;padding:9px 15px;color:var(--ink-3);cursor:pointer}
.tabs button[aria-selected=true]{color:var(--accent);border-bottom-color:var(--accent)}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:14px}
.stat .v{font:600 21px/1.1 var(--mono);font-variant-numeric:tabular-nums}
.stat .k{font-size:11px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em;margin-top:3px}

table.diag{width:100%;border-collapse:collapse;font-size:14px}
table.diag td{padding:7px 0;border-bottom:1px solid var(--line-2);vertical-align:top}
table.diag td:last-child{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums;white-space:nowrap;padding-left:14px}
table.diag tr:last-child td{border-bottom:0}
.note{font-size:13px;color:var(--ink-2);margin-top:14px;padding-top:14px;
      border-top:1px solid var(--line-2)}
.note b{color:var(--ink)}
.bar{height:6px;border-radius:3px;background:var(--line-2);overflow:hidden;margin-top:6px}
.bar i{display:block;height:100%;background:var(--accent)}

.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:16px}
input[type=search],select{font:14px var(--sans);color:var(--ink);background:var(--panel);
  border:1px solid var(--line);border-radius:7px;padding:7px 10px}
input[type=search]{flex:1;min-width:200px}
input[type=search]:focus,select:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
button{font:14px var(--sans);color:var(--ink-2);background:var(--panel);
  border:1px solid var(--line);border-radius:7px;padding:7px 12px;cursor:pointer}
.controls button[aria-pressed=true],.topo-toolbar button[aria-pressed=true],
.rail button[aria-pressed=true]{background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}
.count{font:13px var(--mono);color:var(--ink-3);margin-left:auto}

.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      margin-bottom:10px;overflow:hidden}
.card>summary{padding:13px 16px;cursor:pointer;display:flex;gap:11px;align-items:baseline;
  list-style:none;flex-wrap:wrap}
.card>summary::-webkit-details-marker{display:none}
.card[open]>summary{border-bottom:1px solid var(--line-2)}
.pill{font:600 11px/1 var(--mono);letter-spacing:.05em;padding:4px 7px;border-radius:4px;
      text-transform:uppercase;flex:none}
.pill.pos{background:var(--pos-bg);color:var(--pos)}
.pill.neg{background:var(--neg-bg);color:var(--neg)}
.qtext{flex:1;min-width:240px;font-size:14px}
.chips{display:flex;gap:5px;flex-wrap:wrap}
.chip{font:11px/1 var(--mono);color:var(--ink-3);border:1px solid var(--line);
      border-radius:4px;padding:4px 6px;font-variant-numeric:tabular-nums}
.chip.on{color:var(--accent);border-color:var(--accent)}
.body{padding:16px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
.side h3{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);
         margin:0 0 7px;font-weight:600}
.side{border:1px solid var(--line-2);border-radius:8px;padding:12px}
.fld{margin-bottom:9px;font-size:13.5px}
.fld .n{font:10px/1 var(--mono);text-transform:uppercase;letter-spacing:.06em;
        color:var(--ink-3);display:block;margin-bottom:2px}
.reason{margin-top:14px;padding:12px 14px;border-radius:8px;background:var(--accent-soft);
        font-size:13.5px;color:var(--ink-2)}
.reason b{color:var(--ink);font-size:11px;text-transform:uppercase;letter-spacing:.06em;
          display:block;margin-bottom:5px}
code{font-family:var(--mono);font-size:12px;color:var(--ink-3)}
.theme{position:fixed;top:14px;right:16px;z-index:30}

/* ---- topology ---- */
.delta{background:var(--accent-soft);color:var(--accent);border-radius:8px;
  padding:11px 15px;margin-bottom:14px;font-size:14px}
.delta b{font-family:var(--mono);font-variant-numeric:tabular-nums}
.topo-panes{display:grid;grid-template-columns:1fr 1fr;
  border:1px solid var(--line);border-radius:10px;overflow:hidden}
.topo-panes.single{grid-template-columns:1fr}
.pane{position:relative;min-width:0;display:flex;flex-direction:column;
  height:min(72vh,760px);background:var(--panel)}
.pane + .pane{border-left:1px solid var(--line)}
@media(max-width:900px){
  .topo-panes{grid-template-columns:1fr}
  .pane{height:60vh}
  .pane + .pane{border-left:0;border-top:1px solid var(--line)}
}
.pane header{padding:12px 14px 0}
.pane .title{font-weight:600;font-size:15px}
.pane .pstats{font:12px var(--mono);color:var(--ink-3);margin-top:3px;
  font-variant-numeric:tabular-nums}
.topo-toolbar{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:10px 14px}
.topo-toolbar label{font-size:12.5px;color:var(--ink-2);display:inline-flex;
  gap:4px;align-items:center;cursor:pointer}
.topo-toolbar input[type=search]{min-width:90px;padding:5px 8px;font-size:13px}
.topo-toolbar button{padding:5px 10px;font-size:13px}
svg.graph{flex:1;width:100%;min-height:0;cursor:grab;display:block;touch-action:none}
svg.graph.panning{cursor:grabbing}
.node:focus-visible circle{outline:2px solid var(--accent);outline-offset:2px}
.tooltip{position:absolute;display:none;pointer-events:none;z-index:8;max-width:300px;
  background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:8px 10px;
  font-size:12.5px;color:var(--ink-2);box-shadow:0 6px 20px rgba(0,0,0,.14)}
.tooltip strong{color:var(--ink)}
.busy{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  background:var(--panel);color:var(--ink-3);font:13px var(--mono);z-index:9}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--ink-3);
  padding:0 14px 10px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;
  vertical-align:-1px}
.topo-detail{position:fixed;top:0;right:0;width:380px;height:100vh;overflow:auto;z-index:25;
  background:var(--panel);border-left:1px solid var(--line);padding:20px}
@media(max-width:900px){.topo-detail{width:100%;height:70vh;top:auto;bottom:0;
  border-left:0;border-top:1px solid var(--line)}}
.topo-detail .close{position:absolute;top:14px;right:16px}
.edgelist{list-style:none;padding:0;margin:6px 0 0;font-size:13px}
.edgelist li{padding:6px 9px;margin-bottom:5px;border-radius:6px;border-left:3px solid}
.edgelist li.pos{border-color:var(--pos);background:var(--pos-bg)}
.edgelist li.neg{border-color:var(--neg);background:var(--neg-bg)}

/* ---- embeddings ---- */
.embed-grid{display:grid;grid-template-columns:1fr 300px;gap:16px}
@media(max-width:1000px){.embed-grid{grid-template-columns:1fr}}
#stage{position:relative;border:1px solid var(--line);border-radius:10px;
  background:var(--panel);height:min(74vh,780px);overflow:hidden}
#c{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
#c.dragging{cursor:grabbing}
.tip{position:absolute;display:none;pointer-events:none;z-index:8;max-width:320px;
  background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:9px 11px;
  font-size:12.5px;color:var(--ink-2);box-shadow:0 6px 20px rgba(0,0,0,.14)}
.tip.on{display:block}
.tip .who{font:11px var(--mono);color:var(--ink-3)}
.tip .hd{font-weight:600;color:var(--ink);margin:2px 0 4px}
.tip .meta{margin-top:6px;font:11.5px var(--mono);color:var(--ink-3);
  font-variant-numeric:tabular-nums}
.rail{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;
  align-self:start}
.rail h3{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);
  margin:0 0 9px;font-weight:600}
.rail .row{display:flex;justify-content:space-between;gap:10px;font-size:13px;
  padding:5px 0;border-bottom:1px solid var(--line-2)}
.rail .row:last-of-type{border-bottom:0}
.rail .row b{font-family:var(--mono);font-weight:600;font-variant-numeric:tabular-nums}
.rail .grp{margin-bottom:16px}
.warn{background:var(--neg-bg);color:var(--neg);border-radius:7px;padding:10px 12px;
  font-size:12.5px;margin-bottom:14px}
.ctlrow{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-bottom:12px}
.ctlrow label{font-size:12.5px;color:var(--ink-2);display:inline-flex;gap:4px;
  align-items:center;cursor:pointer}
input[type=range]{width:120px;accent-color:var(--accent)}
.findings{margin-top:22px;padding-top:18px;border-top:1px solid var(--line-2)}
.findings h3{font-size:15px;margin:0 0 8px}
.findings p{font-size:13.5px;color:var(--ink-2);line-height:1.55;max-width:860px}
.findtbl{width:100%;max-width:860px;border-collapse:collapse;font-size:13px;margin:12px 0 18px}
.findtbl th{text-align:left;font-weight:600;color:var(--ink-2);font-size:11.5px;
  text-transform:uppercase;letter-spacing:.02em;padding:5px 10px 5px 0;border-bottom:1px solid var(--line-2)}
.findtbl td{padding:5px 10px 5px 0;border-bottom:1px solid var(--line-2);font-family:var(--mono);
  font-variant-numeric:tabular-nums}
.findtbl td:first-child{font-family:inherit}
.findings .caveat{background:var(--neg-bg);color:var(--neg);border-radius:7px;padding:10px 12px;
  font-size:13px;max-width:860px;margin-top:6px}
.fieldctl{display:inline-flex;flex-wrap:wrap;gap:9px;align-items:center}
.fieldctl label{font-size:12.5px;color:var(--ink-2);display:inline-flex;gap:4px;
  align-items:center;cursor:pointer}
.fieldctl label i.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.miss{padding:40px;text-align:center;color:var(--ink-3);font-size:14px}
"""

_MARKUP = r"""
<button class="theme" id="theme" title="Toggle theme">&#9680;</button>

<div class="wrap">
  <h1>RRF pair batch <span id="bid"></span></h1>
  <div class="sub" id="prov"></div>

  <nav class="tabs" role="tablist" aria-label="Views">
    <button role="tab" id="tab-pairs" aria-controls="panel-pairs" aria-selected="true">Pairs</button>
    <button role="tab" id="tab-topo"  aria-controls="panel-topo"  aria-selected="false">Topology</button>
    <button role="tab" id="tab-embed" aria-controls="panel-embed" aria-selected="false">Embeddings</button>
  </nav>

  <section role="tabpanel" id="panel-pairs" aria-labelledby="tab-pairs">
    <div class="panel">
      <h2>Batch</h2>
      <div class="stats" id="stats"></div>
    </div>
    <div class="panel">
      <h2>Is this trainable? &mdash; leakage &amp; circularity probes</h2>
      <table class="diag" id="diagtbl"></table>
      <div class="note" id="diagnote"></div>
    </div>
    <div class="panel">
      <h2>Per-seeker positive rate</h2>
      <div id="seekbars"></div>
    </div>
    <div class="controls">
      <input type="search" id="q" placeholder="Search query, reasoning, profile text, contact id&hellip;">
      <button id="f-all" aria-pressed="true">All</button>
      <button id="f-pos" aria-pressed="false">Positive</button>
      <button id="f-neg" aria-pressed="false">Negative</button>
      <button id="f-both" aria-pressed="false">Found by both channels</button>
      <select id="sort">
        <option value="rrf">Sort: RRF score &darr;</option>
        <option value="dense">Sort: dense rank &uarr;</option>
        <option value="seeker">Sort: seeker</option>
      </select>
      <span class="count" id="count"></span>
    </div>
    <div id="list"></div>
    <div style="height:60px"></div>
  </section>
</div>

<section role="tabpanel" id="panel-topo" aria-labelledby="tab-topo" class="fullbleed" hidden>
  <div class="delta" id="topoDelta"></div>
  <div class="topo-panes" id="panes"></div>
  <aside class="topo-detail" id="topoDetail" hidden></aside>
</section>

<section role="tabpanel" id="panel-embed" aria-labelledby="tab-embed" class="fullbleed" hidden>
  <div class="embed-grid">
    <div>
      <div class="ctlrow" id="embedCtl">
        <label><input type="checkbox" id="showSeek" checked> Seeker profiles</label>
        <label><input type="checkbox" id="showCand" checked> Candidate profiles</label>
        <label id="asksCtl"><input type="checkbox" id="showSections" checked> Asks</label>
        <span id="fieldsCtl" class="fieldctl"></span>
        <label><input type="checkbox" id="showPairs" checked> Match lines</label>
        <button id="e-all" aria-pressed="true">All</button>
        <button id="e-pos" aria-pressed="false">Positive</button>
        <button id="e-neg" aria-pressed="false">Negative</button>
        <label id="ampCtl">Spread apart
          <input type="range" id="amp" min="1" max="12" step="0.5" value="1">
          <span id="ampVal" style="font:12px var(--mono)">1.0&times;</span>
        </label>
        <button id="e-reset">Reset view</button>
      </div>
      <div id="stage"><canvas id="c"></canvas><div class="tip" id="tip"></div></div>
    </div>
    <div class="rail" id="embedRail"></div>
  </div>
  <div id="embedFindings"></div>
</section>
"""

# --- JS -------------------------------------------------------------------
# Four blocks, one <script>: they share DATA and the small helpers below, so
# splitting into separate script elements would mean duplicating those or
# hanging them off window.

_JS_SHELL = r"""
const DATA = __PAYLOAD__;
const CONTACTS = DATA.contacts;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const f3 = v => v == null ? "—" : (+v).toFixed(3);
const css = getComputedStyle(document.documentElement);
const cvar = n => css.getPropertyValue(n).trim();
const trunc = (s, n) => !s ? "" : (s.length > n ? s.slice(0, n - 1) + "…" : s);
const P_FIELDS = ["positioning","background","lookingFor","locationAvailability",
                  "introPreferences","personalPreferences","meetingAndSchedulingPreferences","notes"];
const flat = v => Array.isArray(v) ? v.map(flat).join(" · ")
  : (v && typeof v === "object") ? Object.values(v).map(flat).join(" · ") : String(v ?? "");
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Seeded PRNG. The reference graph seeds initial positions with Math.random(),
   so every reload produces a different layout — which turns "the right pane is a
   solid blob" into something that reads as a rendering accident, and makes a
   physics change indistinguishable from seed drift. */
function mulberry32(seed){
  return function(){
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

$("#bid").textContent = DATA.batch_id;
const L = DATA.labeler || {};
$("#prov").textContent = [L.model, L.framing && L.framing + " framing",
  (L.prompt_ref||{}).identifier, DATA.pairs.length + " pairs"].filter(Boolean).join("  ·  ");

if (!DATA.embed) $("#tab-embed").hidden = true;

const TABS = ["pairs","topo","embed"];
const INIT = {};
const BUILD = {};              // filled in by the tab blocks below
const ONSHOW = {};

function showTab(key){
  if (!TABS.includes(key)) key = "pairs";
  if (key === "embed" && !DATA.embed) key = "pairs";
  for (const k of TABS){
    const btn = $("#tab-"+k);
    if (btn) btn.setAttribute("aria-selected", String(k === key));
    $("#panel-"+k).hidden = (k !== key);
  }
  history.replaceState(null, "", "#" + key);
  if (!BUILD[key]) return;
  /* Force a synchronous reflow now that the panel is visible. A canvas inside
     display:none measures 0x0 and its draw() guard then no-ops forever; an SVG
     fitView() reads a 0x0 rect and computes a NaN viewBox. */
  void $("#panel-"+key).offsetHeight;
  if (!INIT[key]) INIT[key] = BUILD[key]();
  else if (ONSHOW[key]) ONSHOW[key](INIT[key]);
}
for (const k of TABS){
  const btn = $("#tab-"+k);
  if (btn) btn.addEventListener("click", () => showTab(k));
}
$("#theme").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
};
"""

_JS_PAIRS = r"""
/* ---------------- tab 1: pairs ---------------- */
const d = DATA.diag;
$("#stats").innerHTML = [
  ["pairs", d.n_pairs], ["positive", d.n_pos], ["negative", d.n_neg],
  ["positive rate", (d.pos_frac*100).toFixed(1) + "%"],
  ["seekers", d.n_seekers], ["candidates", d.n_candidates],
  ["triplets", d.triplets],
].map(([k,v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

/* Each row is a predictor that should NOT work well: if the label is guessable
   without comparing the two people, that is what a model would learn. */
$("#diagtbl").innerHTML = [
  ["Seeker identity alone → label", d.auc_seeker_identity, "leave-one-out positive rate, no text at all"],
  ["Candidate identity alone → label", d.auc_candidate_identity, "same, on the offered side"],
  ["RRF score → label (pooled)", d.auc_rrf_pooled, "how much the retriever already agrees with the judge"],
  ["RRF score → label (within seeker)", d.auc_rrf_within_seeker, "base rate removed — the real pair-level signal"],
].map(([k,v,sub]) => `<tr><td>${esc(k)}<br><code>${esc(sub)}</code></td><td>${f3(v)}</td></tr>`).join("");

$("#diagnote").innerHTML =
  `<b>How to read this.</b> 0.500 means the predictor is useless, which is what you want ` +
  `from the identity rows. Seeker identity scores <b>${f3(d.auc_seeker_identity)}</b> here: ` +
  `${d.n_seekers_all_neg} of ${d.n_seekers} seekers were rejected on every candidate, so a ` +
  `chunk of the label is <i>who asked</i> rather than <i>whether these two fit</i>. ` +
  `The defence is training within a seeker — ${d.n_seekers_mixed} seekers have both a ` +
  `positive and a negative, yielding <b>${d.triplets}</b> (anchor, +, −) triplets, and ` +
  `pair-level signal does survive that normalisation (${f3(d.auc_rrf_within_seeker)}). ` +
  `One candidate appears in ${d.max_candidate_reuse} pairs.`;

$("#seekbars").innerHTML = d.seeker_stats.map(s =>
  `<div style="margin-bottom:9px">
     <div style="display:flex;justify-content:space-between;font:12px var(--mono);color:var(--ink-3)">
       <span>${esc(s.id.slice(0,20))}…</span><span>${s.pos}/${s.n}</span></div>
     <div class="bar"><i style="width:${(s.rate*100).toFixed(1)}%"></i></div>
   </div>`).join("");

/* Flatten each contact's profile once, not once per pair on every keystroke. */
const HAY = {};
for (const [id, c] of Object.entries(CONTACTS)) HAY[id] = (id + " " + flat(c)).toLowerCase();
for (const p of DATA.pairs) {
  p._h = ((p.query || "") + " " + (p.reasoning || "") + " " +
          (HAY[p.seeker_id] || "") + " " + (HAY[p.cand_id] || "")).toLowerCase();
}

function profileHtml(p, title, id){
  const body = P_FIELDS.filter(k => p && p[k] != null && flat(p[k]).trim())
    .map(k => `<div class="fld"><span class="n">${k}</span>${esc(flat(p[k]))}</div>`).join("");
  return `<div class="side"><h3>${title} <code>${esc(id.slice(0,14))}…</code></h3>${body}</div>`;
}

function cardHtml(p){
  const chips = [
    p.rrf != null ? `<span class="chip">rrf ${f3(p.rrf)}</span>` : "",
    p.dense_rank != null ? `<span class="chip on">dense #${p.dense_rank}</span>` : `<span class="chip">dense —</span>`,
    p.lex_rank != null ? `<span class="chip on">bm25 #${p.lex_rank}</span>` : `<span class="chip">bm25 —</span>`,
    p.section != null ? `<span class="chip">§${p.section}</span>` : "",
  ].join("");
  return `<details class="card">
    <summary>
      <span class="pill ${p.label}">${p.label}</span>
      <span class="qtext">${esc(p.query)}</span>
      <span class="chips">${chips}</span>
    </summary>
    <div class="body">
      <div class="cols">
        ${profileHtml(CONTACTS[p.seeker_id], "Seeker", p.seeker_id)}
        ${profileHtml(CONTACTS[p.cand_id], "Candidate", p.cand_id)}
      </div>
      <div class="reason"><b>Judge reasoning</b>${esc(p.reasoning)}</div>
    </div>
  </details>`;
}

let filter = "all", onlyBoth = false;
function render(){
  const term = $("#q").value.toLowerCase().trim();
  const mode = $("#sort").value;
  let rows = DATA.pairs.filter(p => {
    if (filter !== "all" && p.label !== filter) return false;
    if (onlyBoth && !p.both) return false;
    return !term || p._h.includes(term);
  });
  rows.sort((a,b) =>
    mode === "rrf"   ? (b.rrf ?? 0) - (a.rrf ?? 0) :
    mode === "dense" ? (a.dense_rank ?? 99) - (b.dense_rank ?? 99) :
                       a.seeker_id.localeCompare(b.seeker_id) || (b.rrf ?? 0) - (a.rrf ?? 0));
  $("#count").textContent = `${rows.length} of ${DATA.pairs.length}`;
  $("#list").innerHTML = rows.map(cardHtml).join("");
}
for (const [id, val] of [["f-all","all"],["f-pos","pos"],["f-neg","neg"]]) {
  $("#"+id).onclick = () => {
    filter = val;
    ["f-all","f-pos","f-neg"].forEach(x => $("#"+x).setAttribute("aria-pressed", String(x === id)));
    render();
  };
}
$("#f-both").onclick = e => {
  onlyBoth = !onlyBoth;
  e.currentTarget.setAttribute("aria-pressed", String(onlyBoth));
  render();
};
$("#q").oninput = render;
$("#sort").onchange = render;
render();
"""

_JS_TOPO = r"""
/* ---------------- tab 2: topology ----------------
   Physics and rendering ported from scripts/build_real_pairs_graph.py:689-1141.
   Dropped from the original: the polarize force and the similarity-matrix force
   (a label-driven left/right split would fabricate structure in a tab whose only
   job is an honest density comparison), and archetype coloring (this batch does
   not carry archetypes). Added: a seeded PRNG for reproducible layouts. */
const svgNS = "http://www.w3.org/2000/svg";
const ROLE_COLOR = { seeker: "var(--seeker)", candidate: "var(--cand)", both: "var(--both)" };

function paneTemplate(){
  const el = document.createElement("div");
  el.className = "pane";
  el.innerHTML = `
    <header><div class="title"></div><div class="pstats"></div></header>
    <div class="topo-toolbar">
      <label><input type="checkbox" class="showPos" checked> positive</label>
      <label><input type="checkbox" class="showNeg" checked> negative</label>
      <input type="search" class="search" placeholder="filter&hellip;">
      <button class="fit-btn">Fit</button>
    </div>
    <div class="legend">
      <span><i class="dot" style="background:var(--seeker)"></i>seeker</span>
      <span><i class="dot" style="background:var(--cand)"></i>candidate</span>
      <span><i class="dot" style="background:var(--both)"></i>both</span>
      <span><i class="dot" style="background:var(--pos)"></i>accepted</span>
      <span><i class="dot" style="background:var(--neg)"></i>declined</span>
    </div>
    <svg class="graph"></svg>
    <div class="tooltip"></div>`;
  return el;
}

function openTopoPanel(n, spec, edges){
  const p = CONTACTS[n.id] || n.profile || {};
  const out = edges.filter(e => e.source === n.id);
  const inc = edges.filter(e => e.target === n.id);
  const li = (e, other, arrow) =>
    `<li class="${e.label}">${arrow} <code>${esc(other.slice(0,16))}…</code><br>${esc(e.searchQuery || "")}</li>`;
  const fld = (k, label) => p[k] ? `<div class="fld"><span class="n">${label}</span>${esc(flat(p[k]))}</div>` : "";
  const jump = spec.key === "synth"
    ? `<button id="topoJump" style="margin-top:12px">See all ${n.pairCount} pairs →</button>` : "";
  const el = $("#topoDetail");
  el.innerHTML = `
    <button class="close" id="topoClose">✕</button>
    <h2 style="margin-bottom:6px">${esc(trunc(String(p.positioning || n.id).replace(/\s+/g," "), 70))}</h2>
    <code>${esc(n.id)}</code>
    <div style="margin:10px 0 14px"><span class="chip on">${n.role}</span>
      <span class="chip">${n.pairCount} pair edges</span>
      <span class="chip">${esc(spec.title)}</span></div>
    ${fld("positioning","Positioning")}${fld("background","Background")}
    ${fld("lookingFor","Looking for")}${fld("notes","Notes")}
    <h3 style="font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);margin:16px 0 0">
      As seeker (${out.length})</h3>
    <ul class="edgelist">${out.map(e => li(e, e.target, "→")).join("") || "<li>none</li>"}</ul>
    <h3 style="font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);margin:16px 0 0">
      As candidate (${inc.length})</h3>
    <ul class="edgelist">${inc.map(e => li(e, e.source, "←")).join("") || "<li>none</li>"}</ul>
    ${jump}`;
  el.hidden = false;
  $("#topoClose").onclick = () => { el.hidden = true; };
  const j = $("#topoJump");
  if (j) j.onclick = () => { el.hidden = true; showTab("pairs"); $("#q").value = n.id; render(); };
}

function createGraph(root, spec){
  const rand = mulberry32(spec.seed);
  const nodes = spec.nodes.map(n => ({ ...n, x: (rand()-0.5)*900, y: (rand()-0.5)*900, vx: 0, vy: 0 }));
  const nodeById = new Map(nodes.map(n => [n.id, n]));
  const edges = spec.edges.filter(e => nodeById.has(e.source) && nodeById.has(e.target));

  // connected components, so separate clusters repel each other harder
  const compOf = new Map();
  {
    const adj = new Map(nodes.map(n => [n.id, []]));
    for (const e of edges){ adj.get(e.source).push(e.target); adj.get(e.target).push(e.source); }
    let compId = 0;
    for (const n of nodes){
      if (compOf.has(n.id)) continue;
      const stack = [n.id];
      compOf.set(n.id, compId);
      while (stack.length){
        const cur = stack.pop();
        for (const nb of adj.get(cur)) if (!compOf.has(nb)){ compOf.set(nb, compId); stack.push(nb); }
      }
      compId++;
    }
  }

  const P = spec.physics || {};
  const REPULSION = P.repulsion ?? 4200, SPRING_LEN = P.springLen ?? 90;
  const CROSS_COMPONENT_BOOST = 3.2, SPRING = 0.012, CENTER = 0.0006;
  const DAMP = 0.85, ALPHA_DECAY = 0.996, ALPHA_MIN = 0.02;
  let alpha = 0, simRunning = false;

  function tickPhysics(strength){
    for (const n of nodes){ n.fx = 0; n.fy = 0; }
    for (let i = 0; i < nodes.length; i++){
      for (let j = i + 1; j < nodes.length; j++){
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx*dx + dy*dy + 0.01, dd = Math.sqrt(d2);
        const boost = compOf.get(a.id) === compOf.get(b.id) ? 1 : CROSS_COMPONENT_BOOST;
        const f = (REPULSION * boost) / d2;
        dx /= dd; dy /= dd;
        a.fx += dx*f; a.fy += dy*f; b.fx -= dx*f; b.fy -= dy*f;
      }
    }
    for (const e of edges){
      const a = nodeById.get(e.source), b = nodeById.get(e.target);
      let dx = b.x - a.x, dy = b.y - a.y;
      const dd = Math.sqrt(dx*dx + dy*dy) || 0.01;
      const f = (dd - SPRING_LEN) * SPRING;
      dx /= dd; dy /= dd;
      a.fx += dx*f; a.fy += dy*f; b.fx -= dx*f; b.fy -= dy*f;
    }
    for (const n of nodes){
      n.fx -= n.x * CENTER; n.fy -= n.y * CENTER;
      if (n.pinned){ n.vx = 0; n.vy = 0; continue; }
      n.vx = (n.vx + n.fx*strength) * DAMP;
      n.vy = (n.vy + n.fy*strength) * DAMP;
      n.x += n.vx; n.y += n.vy;
    }
  }
  function runSimulationLoop(){
    if (!simRunning) return;
    tickPhysics(alpha);
    if (!$("#panel-topo").hidden) render();
    alpha *= ALPHA_DECAY;
    if (alpha < ALPHA_MIN){ simRunning = false; fitView(); return; }
    requestAnimationFrame(runSimulationLoop);
  }
  function reheat(amount = 0.7){
    alpha = Math.max(alpha, amount);
    if (!simRunning){ simRunning = true; requestAnimationFrame(runSimulationLoop); }
  }

  const svg = root.querySelector("svg.graph");
  let viewBox = { x: -600, y: -600, w: 1200, h: 1200 };
  const applyViewBox = () => svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
  applyViewBox();

  /* Marker ids must be unique document-wide: url(#id) resolves to the FIRST
     match, so a shared id renders pane 2 with pane 1's arrowheads. */
  const posMarker = `topo-arrow-pos-${spec.key}`, negMarker = `topo-arrow-neg-${spec.key}`;
  svg.innerHTML = `
    <defs>
      <marker id="${posMarker}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--pos)"></path></marker>
      <marker id="${negMarker}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--neg)"></path></marker>
    </defs>
    <g class="edgeLayer"></g><g class="nodeLayer"></g>`;
  const edgeLayer = svg.querySelector(".edgeLayer"), nodeLayer = svg.querySelector(".nodeLayer");

  /* A->B and B->A drawn straight overlap exactly, making the picture understate
     the edge count. Fan each group onto its own arc. */
  const groupIndex = new Map();
  for (const e of edges){
    const key = [e.source, e.target].sort().join("|");
    const list = groupIndex.get(key) || [];
    list.push(e); groupIndex.set(key, list);
  }
  const curveOf = new Map();
  for (const list of groupIndex.values()){
    const m = list.length;
    list.forEach((e, k) => curveOf.set(e, m === 1 ? 0 : (k - (m-1)/2) * 18));
  }
  function edgePath(e){
    const a = nodeById.get(e.source), b = nodeById.get(e.target);
    const curve = curveOf.get(e) || 0;
    if (!curve) return `M${a.x},${a.y} L${b.x},${b.y}`;
    const mx = (a.x+b.x)/2, my = (a.y+b.y)/2;
    let dx = b.x-a.x, dy = b.y-a.y;
    const dd = Math.sqrt(dx*dx+dy*dy) || 1;
    return `M${a.x},${a.y} Q${mx - dy/dd*curve},${my + dx/dd*curve} ${b.x},${b.y}`;
  }

  const tooltip = root.querySelector(".tooltip");
  function moveTooltip(ev){
    const rect = root.getBoundingClientRect();
    tooltip.style.left = (ev.clientX - rect.left + 14) + "px";
    tooltip.style.top = (ev.clientY - rect.top + 14) + "px";
  }
  function showTooltip(ev, edge, node){
    if (edge) tooltip.innerHTML =
      `<strong>${edge.label === "pos" ? "Accepted" : "Declined"}</strong> · ` +
      `${esc(edge.source.slice(0,12))}… → ${esc(edge.target.slice(0,12))}…` +
      `<br>${esc(edge.searchQuery || "")}`;
    else if (node) tooltip.innerHTML =
      `<strong>${esc(node.id.slice(0,18))}…</strong><br>${node.role} · ${node.pairCount} pair edges`;
    tooltip.style.display = "block";
    moveTooltip(ev);
  }
  const hideTooltip = () => { tooltip.style.display = "none"; };

  const edgeEls = edges.map(e => {
    const path = document.createElementNS(svgNS, "path");
    path.style.stroke = e.label === "pos" ? "var(--pos)" : "var(--neg)";
    path.setAttribute("fill", "none");
    path.setAttribute("stroke-width", "2.4");
    path.setAttribute("stroke-opacity", "0.55");
    path.setAttribute("marker-end", `url(#${e.label === "pos" ? posMarker : negMarker})`);
    path.style.pointerEvents = "none";
    edgeLayer.appendChild(path);
    const hit = document.createElementNS(svgNS, "path");   // wide invisible hover target
    hit.setAttribute("stroke", "transparent");
    hit.setAttribute("fill", "none");
    hit.setAttribute("stroke-width", "14");
    hit.style.cursor = "pointer";
    hit.addEventListener("mouseenter", ev => showTooltip(ev, e));
    hit.addEventListener("mousemove", moveTooltip);
    hit.addEventListener("mouseleave", hideTooltip);
    edgeLayer.appendChild(hit);
    return { e, path, hit };
  });

  let draggingNode = null;
  const nodeEls = nodes.map(n => {
    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("class", "node");
    g.setAttribute("tabindex", "0");
    g.setAttribute("role", "button");
    g.setAttribute("aria-label", n.id);
    g.style.cursor = "pointer";
    const circle = document.createElementNS(svgNS, "circle");
    circle.setAttribute("r", 5 + Math.min(n.pairCount, 8));
    circle.style.fill = ROLE_COLOR[n.role] || "var(--muted)";
    circle.style.stroke = "var(--panel)";
    circle.setAttribute("stroke-width", "1.2");
    g.appendChild(circle);
    g.addEventListener("click", ev => { ev.stopPropagation(); openTopoPanel(n, spec, edges); });
    g.addEventListener("keydown", ev => {
      if (ev.key === "Enter" || ev.key === " "){ ev.preventDefault(); openTopoPanel(n, spec, edges); }
    });
    g.addEventListener("mouseenter", ev => showTooltip(ev, null, n));
    g.addEventListener("mousemove", moveTooltip);
    g.addEventListener("mouseleave", hideTooltip);
    g.addEventListener("mousedown", ev => {
      draggingNode = n; n.pinned = true; n.vx = 0; n.vy = 0; reheat(); ev.stopPropagation();
    });
    nodeLayer.appendChild(g);
    return { n, g };
  });

  root.querySelector(".title").textContent = spec.title;
  root.querySelector(".pstats").textContent = spec.stats;

  const showPos = root.querySelector(".showPos"), showNeg = root.querySelector(".showNeg");
  const search = root.querySelector(".search");
  function passesFilter(n){
    const term = search.value.trim().toLowerCase();
    if (!term) return true;
    const p = CONTACTS[n.id] || n.profile || {};
    return (n.id + " " + flat(p)).toLowerCase().includes(term);
  }
  showPos.addEventListener("change", render2);
  showNeg.addEventListener("change", render2);
  search.addEventListener("input", render2);

  function render2(){
    for (const { e, path, hit } of edgeEls){
      const dpath = edgePath(e);
      path.setAttribute("d", dpath); hit.setAttribute("d", dpath);
      const visible = (e.label === "pos" ? showPos.checked : showNeg.checked)
        && passesFilter(nodeById.get(e.source)) && passesFilter(nodeById.get(e.target));
      path.style.display = visible ? "" : "none";
      hit.style.display = visible ? "" : "none";
    }
    for (const { n, g } of nodeEls){
      g.setAttribute("transform", `translate(${n.x},${n.y})`);
      g.style.display = passesFilter(n) ? "" : "none";
    }
  }
  const render = render2;

  /* With the viewBox tracking the node bbox, only the RATIO of REPULSION to
     SPRING matters — which is what lets two very differently sized graphs share
     one code path. */
  function fitView(pad = 90){
    if (!nodes.length) return;
    const rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;   // hidden tab: a NaN viewBox otherwise
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes){
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x > maxX) maxX = n.x;
      if (n.y > maxY) maxY = n.y;
    }
    const w = Math.max(maxX-minX, 1) + pad*2, h = Math.max(maxY-minY, 1) + pad*2;
    const aspect = rect.width / rect.height;
    let vw = w, vh = h;
    if (w/h > aspect) vh = w/aspect; else vw = h*aspect;
    viewBox = { x: (minX+maxX)/2 - vw/2, y: (minY+maxY)/2 - vh/2, w: vw, h: vh };
    applyViewBox();
  }
  root.querySelector(".fit-btn").addEventListener("click", () => fitView());

  let panning = false, panStart = null, viewStart = null;
  svg.addEventListener("mousedown", ev => {
    if (ev.target.closest(".node")) return;
    panning = true; svg.classList.add("panning");
    panStart = { x: ev.clientX, y: ev.clientY }; viewStart = { ...viewBox };
  });
  // One delegated listener per pane, not one per node.
  root.addEventListener("mousemove", ev => {
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    const scale = viewBox.w / rect.width;
    if (draggingNode){
      draggingNode.x += ev.movementX * scale;
      draggingNode.y += ev.movementY * scale;
      if (!simRunning) render2();
      return;
    }
    if (!panning) return;
    viewBox.x = viewStart.x - (ev.clientX - panStart.x) * scale;
    viewBox.y = viewStart.y - (ev.clientY - panStart.y) * scale;
    applyViewBox();
  });
  window.addEventListener("mouseup", () => {
    if (draggingNode){ draggingNode.pinned = false; draggingNode = null; reheat(); }
    panning = false; svg.classList.remove("panning");
  });
  svg.addEventListener("wheel", ev => {
    ev.preventDefault();
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    const mx = viewBox.x + (ev.clientX - rect.left) / rect.width * viewBox.w;
    const my = viewBox.y + (ev.clientY - rect.top) / rect.height * viewBox.h;
    const factor = ev.deltaY > 0 ? 1.1 : 0.9;
    viewBox.x = mx - (mx - viewBox.x) * factor;
    viewBox.y = my - (my - viewBox.y) * factor;
    viewBox.w *= factor; viewBox.h *= factor;
    applyViewBox();
  }, { passive: false });

  function start(){
    if (REDUCED_MOTION || nodes.length < 60){
      for (let it = 0; it < 400; it++) tickPhysics(1);
      render2();
      requestAnimationFrame(() => fitView());
    } else {
      render2();
      reheat(1);
    }
  }
  return { start, render: render2, fitView };
}

BUILD.topo = function(){
  const T = DATA.topology;
  const fmt = s => `${s.nodes} contacts · ${s.edges} pairs (${s.pos} pos / ${s.neg} neg) ` +
                   `· ${s.mean_degree} edges/node · ${s.components} component${s.components === 1 ? "" : "s"}`;

  // The synth graph is rebuilt here rather than shipped: its nodes are the
  // contact table and its edges are the pair list, both already in the payload.
  const synthNodes = Object.entries(CONTACTS).map(([id, c]) =>
    ({ id, role: c.role, pairCount: c.pairCount }));
  const synthEdges = DATA.pairs.map(p =>
    ({ source: p.seeker_id, target: p.cand_id, label: p.label, searchQuery: p.query }));

  const specs = [];
  if (T.real) specs.push({ key: "real", seed: 0x9E3779B9, title: "Real pairs",
    stats: fmt(T.real.stats), nodes: T.real.nodes, edges: T.real.edges, physics: T.real.physics });
  specs.push({ key: "synth", seed: 0x85EBCA6B, title: DATA.batch_id,
    stats: fmt(T.synth.stats), nodes: synthNodes, edges: synthEdges, physics: T.synth.physics });

  const s = T.synth.stats;
  $("#topoDelta").innerHTML = T.real
    ? `<b>${(s.mean_degree / T.real.stats.mean_degree).toFixed(1)}×</b> denser than real data ` +
      `(<b>${s.mean_degree}</b> vs <b>${T.real.stats.mean_degree}</b> edges/node), and real data's ` +
      `<b>${T.real.stats.components}</b> separate components collapse to <b>${s.components}</b>. ` +
      `Every seeker in this batch is disjoint from every candidate — the graph is bipartite, ` +
      `where real data has ${T.real.stats.both_role} contacts appearing on both sides.`
    : `<b>data/dataset_positive.json</b> is not present in this checkout, so the real-pair ` +
      `comparison is unavailable. Showing ${DATA.batch_id} alone: <b>${s.mean_degree}</b> edges/node ` +
      `across <b>${s.components}</b> component${s.components === 1 ? "" : "s"} (real data: 0.673, 97).`;

  const panesEl = $("#panes");
  if (specs.length === 1) panesEl.classList.add("single");
  const panes = specs.map(spec => {
    const root = paneTemplate();
    panesEl.appendChild(root);
    const g = createGraph(root, spec);
    if (REDUCED_MOTION){
      const busy = document.createElement("div");
      busy.className = "busy";
      busy.textContent = "solving layout…";
      root.appendChild(busy);
      requestAnimationFrame(() => { g.start(); busy.remove(); });
    } else g.start();
    return g;
  });
  return { panes };
};
ONSHOW.topo = h => h.panes.forEach(p => p.fitView());
"""

_JS_EMBED = r"""
const FIELD_ORDER = ["positioning","background","lookingFor","notes","locationAvailability",
                      "introPreferences","personalPreferences","meetingAndSchedulingPreferences"];
const FIELD_LABEL = {positioning:"Positioning", background:"Background", lookingFor:"LookingFor (isolated)",
  notes:"Notes", locationAvailability:"Location", introPreferences:"Intro prefs",
  personalPreferences:"Personal prefs", meetingAndSchedulingPreferences:"Scheduling prefs"};
const fieldColorVar = f => "--f-" + f.toLowerCase();

/* ---------------- tab 3: embeddings ----------------
   Projection, pointer handling and the painter's-algorithm draw are ported from
   scripts/build_field_isolation_embedding_space_3d.py:589-826. Simplified: that
   view has two tether levels (whole -> field -> section) because it embeds each
   profile field separately; this batch has only whole profiles and lookingFor
   sections, so there is exactly one. */
BUILD.embed = function(){
  const E = DATA.embed, S = E.stats;
  const canvas = $("#c"), ctx = canvas.getContext("2d");
  const stage = $("#stage"), tip = $("#tip");
  const nodeById = new Map(E.people.map(n => [n.id, n]));
  let W = 0, H = 0, dpr = 1;
  let yaw = 0.6, pitch = -0.28, zoom = 1, focus = null, hover = null, amp = 1;
  let edgeFilter = "all", hits = [];

  const row = (k, v, title) =>
    `<div class="row"${title ? ` title="${esc(title)}"` : ""}><span>${k}</span><b>${v}</b></div>`;
  const sepPos = S.pairCosPos, sepNeg = S.pairCosNeg;
  const gap = (sepPos != null && sepNeg != null) ? (sepPos - sepNeg) : null;
  $("#embedRail").innerHTML =
    `<div class="warn">This 3D shadow keeps <b>${(E.evrSum*100).toFixed(1)}%</b> of
      ${E.dim} dimensions (PCA ${E.evr.map(v => (v*100).toFixed(1) + "%").join(" + ")}).
      Two points looking close is weak evidence — read the cosines, not the picture.</div>
     <div class="grp"><h3>Does geometry see the label?</h3>
       ${row("accepted pairs, mean cos", f3(sepPos))}
       ${row("declined pairs, mean cos", f3(sepNeg))}
       ${row("separation", gap == null ? "—" : (gap >= 0 ? "+" : "") + gap.toFixed(4))}
       <div class="note" style="margin-top:9px;padding-top:9px;font-size:12px">
         If these two are equal, the dense channel's geometry does not separate the
         judge's verdict at all — the same question the Pairs tab asks in rank space
         (within-seeker RRF ${f3(DATA.diag.auc_rrf_within_seeker)}).</div></div>
     <div class="grp"><h3>Spread</h3>
       ${row("profile ↔ own asks", f3(S.wholeSectionCos))}
       ${row("ask ↔ ask, same person", f3(S.sectionSectionCos))}
       ${E.hasFields ? row("profile ↔ own isolated fields", f3(S.wholeFieldCos),
          "each field embedded completely alone -- no other field's text present") : ""}
       ${row("seeker ↔ seeker", f3(S.acrossSeekerCos))}
       ${row("candidate ↔ candidate", f3(S.acrossCandCos))}</div>
     <div class="grp"><h3>Counts</h3>
       ${row("seekers", S.nSeekers)}${row("candidates", S.nCandidates)}
       ${row("asks", S.nSections)}
       ${E.hasFields ? row("isolated fields", S.nFields) : ""}
       ${S.nOrphanCandidates ? row("never staged", S.nOrphanCandidates,
          "embedded and retrieved, but no pair reached the judge") : ""}
       ${row("model", "")}</div>
     <div style="font:11px var(--mono);color:var(--ink-3);word-break:break-all">${esc(E.model)}</div>`;

  if (E.hasFields){
    const compareRows = [
      ["field-alone vs. own whole profile", "0.705", "0.600"],
      ["ask-alone vs. own whole profile", "0.684", "0.593"],
      ["own fields vs. each other", "—", "0.437"],
      ["own asks vs. each other", "—", "0.571"],
      ["whole vs. whole, different people", "0.579", "0.489"],
      ["constellation ratio (own-field spread ÷ inter-person gap)", "~229%", "~110%"],
    ];
    const perField = [
      ["positioning", "0.890 / 0.549", "0.770 / 0.477"],
      ["background", "0.850 / 0.580", "0.672 / 0.520"],
      ["lookingFor", "0.852 / 0.595", "0.733 / 0.544"],
      ["notes", "0.700 / 0.563", "0.666 / 0.513"],
      ["introPreferences", "0.731 / 0.680", "0.714 / 0.649"],
      ["locationAvailability", "0.450 / 0.676", "0.465 / 0.680"],
      ["personalPreferences", "0.366 / 0.756", "0.376 / 0.797"],
      ["meetingAndSchedulingPreferences", "0.391 / 0.751", "0.402 / 0.731"],
    ];
    const tbl = (head, rows) =>
      `<table class="findtbl"><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead>` +
      `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    $("#embedFindings").innerHTML =
      `<div class="findings">
         <h3>Original (real holdout) vs. this synthetic batch — field-isolation findings</h3>
         <p>Isolating a field moves it a lot more than swapping one in and out ever did. A lone
         field scores well below its owner's whole profile, and well above a random stranger's
         whole profile — loosely related to its owner, but not indistinguishable from anyone
         else's. That pattern holds in both the original real-pair holdout (voyage-4-nano) and
         this synthetic batch (Qwen3-Embedding-8B), though the two used different embedding
         models, so only the gaps between rows — not the raw numbers — are comparable across
         the two columns.</p>
         ${tbl(["comparing", "original (real, 115 contacts)", "this batch (synthetic, 92 contacts)"], compareRows)}
         <p><b>Do same-named fields cluster by topic, or by person?</b> Per-field, "vs. own whole
         profile" then "vs. same field, other people" (cosine on raw vectors). When the second
         number is close to or above the first, that field reads as generic/boilerplate once
         isolated — carrying almost no person-identifying signal on its own.</p>
         ${tbl(["field", "original: own / other-people", "this batch: own / other-people"], perField)}
         <p>The ranking is the same in both datasets: <code>positioning</code>/<code>background</code>/
         <code>lookingFor</code>/<code>notes</code>/<code>introPreferences</code> stay person-specific
         even alone; <code>locationAvailability</code>, <code>personalPreferences</code>, and
         <code>meetingAndSchedulingPreferences</code> invert and read as generic logistics. That match
         suggests the synthetic profile generator reproduces the same "which fields carry identity"
         structure as real Boardy profiles — a real structural agreement, not a coincidence of
         similar-looking numbers.</p>
         <div class="caveat"><b>This is a profile-shape check only — it says nothing about pairing/
         labeling quality.</b> It only checked whether individual fake profiles read like real ones.
         It did not check whether the accept/decline label between two people is correct — a batch
         can have very realistic-looking people in it and still pair them up wrong. That question is
         answered separately in <code>docs/rrf-pairing-pipeline.md</code>, and more weakly: 12 of this
         batch's 40 seekers were rejected on every candidate they were shown (a red flag for how those
         labels were assigned), and the judge model only agreed with itself 59.4% of the time on the
         hardest pairs.</div>
       </div>`;
  }

  const showSeek = $("#showSeek"), showCand = $("#showCand");
  const showSections = $("#showSections"), showPairs = $("#showPairs");
  const ampEl = $("#amp"), ampVal = $("#ampVal");
  if (!S.nSections){ $("#asksCtl").hidden = true; $("#ampCtl").hidden = true; }

  // One checkbox per profile field actually present in this batch's isolated
  // vectors, each its own color -- built at render time since which fields a
  // synthetic batch's profiles populate isn't fixed ahead of time.
  const presentFields = new Set();
  for (const n of E.people) for (const f of (n.fields || [])) presentFields.add(f.field);
  const fieldOn = {};
  const fieldsCtl = $("#fieldsCtl");
  if (!E.hasFields || !presentFields.size){
    fieldsCtl.hidden = true;
  } else {
    fieldsCtl.innerHTML = FIELD_ORDER.filter(f => presentFields.has(f)).map(f => {
      fieldOn[f] = false;
      return `<label><input type="checkbox" data-field="${f}"> ` +
             `<i class="dot" style="background:var(${fieldColorVar(f)})"></i>${esc(FIELD_LABEL[f] || f)}</label>`;
    }).join("");
    fieldsCtl.querySelectorAll("input[data-field]").forEach(el => {
      el.addEventListener("change", () => { fieldOn[el.dataset.field] = el.checked; draw(); });
    });
  }

  function resize(){
    if (!canvas.clientWidth || !canvas.clientHeight) return;  // hidden tab
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W*dpr; canvas.height = H*dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  new ResizeObserver(resize).observe(stage);

  function project(p){
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = p[0]*cy - p[2]*sy, z1 = p[0]*sy + p[2]*cy;
    const y1 = p[1]*cp - z1*sp,   z2 = p[1]*sp + z1*cp;
    const scale = Math.min(W, H) * 0.40 * zoom;
    const persp = 2.6 / (2.6 + z2);
    return { x: W/2 + x1*scale*persp, y: H/2 + y1*scale*persp, z: z2, s: persp };
  }
  const offsetPoint = (anchor, xyz) => [
    anchor[0] + (xyz[0]-anchor[0])*amp,
    anchor[1] + (xyz[1]-anchor[1])*amp,
    anchor[2] + (xyz[2]-anchor[2])*amp,
  ];

  let dragging = false, lx = 0, ly = 0, moved = false;
  canvas.addEventListener("mousedown", e => {
    dragging = true; moved = false; lx = e.clientX; ly = e.clientY;
    canvas.classList.add("dragging");
  });
  window.addEventListener("mouseup", () => { dragging = false; canvas.classList.remove("dragging"); });
  window.addEventListener("mousemove", e => {
    if (!dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    yaw += dx*0.006; pitch = Math.max(-1.45, Math.min(1.45, pitch + dy*0.006));
    lx = e.clientX; ly = e.clientY;
    draw();
  });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    zoom = Math.max(0.35, Math.min(9, zoom * (e.deltaY > 0 ? 0.9 : 1.11)));
    draw();
  }, { passive: false });
  canvas.addEventListener("mousemove", e => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 13*13;
    for (const h of hits){
      const dd = (h.x-mx)*(h.x-mx) + (h.y-my)*(h.y-my);
      if (dd < bd){ bd = dd; best = h; }
    }
    if (best !== hover){ hover = best; draw(); }
    if (best) showTip(best, mx, my); else tip.classList.remove("on");
  });
  canvas.addEventListener("mouseleave", () => { hover = null; tip.classList.remove("on"); draw(); });
  canvas.addEventListener("click", () => {
    if (moved) return;
    focus = (hover && hover.node && focus !== hover.node.id) ? hover.node.id : null;
    draw();
  });

  function showTip(h, mx, my){
    const n = h.node, c = CONTACTS[n.id];
    let hd, body, meta;
    if (h.kind === "section"){
      const s = n.sections[h.si];
      hd = `Ask ${h.si + 1} of ${n.sections.length}`;
      body = esc(s.text || "");
      meta = `${f3(s.cos)} similar to this person's whole profile`;
    } else if (h.kind === "field"){
      const f = n.fields[h.fi];
      hd = f.kind === "section_alone" ? `Isolated ask (${f.field})` : `Isolated field: ${f.field}`;
      body = esc(f.text || "");
      meta = `${f3(f.cos)} similar to this person's whole profile · no other field present`;
    } else {
      hd = n.role === "seeker" ? "Seeker profile" : "Candidate profile";
      body = esc(trunc((c && c.positioning) || "", 240));
      meta = n.pairCount
        ? `${n.pairCount} pairs · ${n.posCount} accepted / ${n.pairCount - n.posCount} declined`
        : "retrieved but never staged";
      if (n.role === "seeker" && n.sections.length) meta += ` · ${n.sections.length} asks`;
    }
    tip.innerHTML = `<div class="who">${esc(n.id.slice(0,20))}… · ${n.role}</div>` +
                    `<div class="hd">${hd}</div><div>${body}</div><div class="meta">${meta}</div>`;
    tip.classList.add("on");
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = (mx + tw + 26 > W ? mx - tw - 14 : mx + 14) + "px";
    tip.style.top  = (my + th + 26 > H ? my - th - 14 : my + 14) + "px";
  }

  function draw(){
    if (!W || !H) return;
    ctx.clearRect(0, 0, W, H);
    hits = [];
    const dim = id => focus && focus !== id;
    const items = [];
    const drawSections = showSections.checked;

    for (const n of E.people){
      const visible = n.role === "seeker" ? showSeek.checked : showCand.checked;
      const wp = project(n.whole);
      const faded = dim(n.id);
      const col = n.role === "seeker" ? cvar("--seeker") : cvar("--cand");
      if (visible) items.push({ t:"node", kind:"whole", node:n, x:wp.x, y:wp.y, z:wp.z, s:wp.s, col, faded });
      if (drawSections && visible && n.sections.length){
        for (let si = 0; si < n.sections.length; si++){
          const p = offsetPoint(n.whole, n.sections[si].xyz);
          const sp = project(p);
          items.push({ t:"tether", x0:wp.x, y0:wp.y, x:sp.x, y:sp.y, z:(wp.z+sp.z)/2, col:cvar("--sect"), faded });
          items.push({ t:"node", kind:"section", node:n, si, x:sp.x, y:sp.y, z:sp.z, s:sp.s, col:cvar("--sect"), faded });
        }
      }
      if (visible && n.fields && n.fields.length){
        for (let fi = 0; fi < n.fields.length; fi++){
          const fEntry = n.fields[fi];
          if (!fieldOn[fEntry.field]) continue;
          const p = offsetPoint(n.whole, fEntry.xyz);
          const fp = project(p);
          const col = cvar(fieldColorVar(fEntry.field));
          items.push({ t:"tether", x0:wp.x, y0:wp.y, x:fp.x, y:fp.y, z:(wp.z+fp.z)/2, col, faded });
          items.push({ t:"node", kind:"field", node:n, fi, x:fp.x, y:fp.y, z:fp.z, s:fp.s, col, faded });
        }
      }
    }

    if (showPairs.checked){
      for (const e of DATA.pairs){
        if (edgeFilter !== "all" && e.label !== edgeFilter) continue;
        const a = nodeById.get(e.seeker_id), b = nodeById.get(e.cand_id);
        if (!a || !b) continue;
        const faded = focus ? (focus !== e.seeker_id && focus !== e.cand_id) : false;
        const lit = !!(hover && hover.node && (hover.node.id === e.seeker_id || hover.node.id === e.cand_id));
        const pa = project(a.whole), pb = project(b.whole);
        items.push({ t:"pair", x0:pa.x, y0:pa.y, x:pb.x, y:pb.y, z:(pa.z+pb.z)/2,
                     col: cvar(e.label === "pos" ? "--pos" : "--neg"), faded, lit });
      }
    }

    items.sort((a, b) => b.z - a.z);   // painter's algorithm

    for (const it of items){
      if (it.t === "tether"){
        ctx.save();
        ctx.globalAlpha = it.faded ? 0.05 : 0.32;
        ctx.strokeStyle = it.col; ctx.lineWidth = 1; ctx.setLineDash([1.5, 3]);
        ctx.beginPath(); ctx.moveTo(it.x0, it.y0); ctx.lineTo(it.x, it.y); ctx.stroke();
        ctx.restore();
      } else if (it.t === "pair"){
        ctx.save();
        ctx.globalAlpha = it.faded ? 0.04 : (it.lit ? 0.95 : 0.5);
        ctx.strokeStyle = it.col; ctx.lineWidth = it.lit ? 2.2 : 1.2;
        ctx.beginPath(); ctx.moveTo(it.x0, it.y0); ctx.lineTo(it.x, it.y); ctx.stroke();
        ctx.restore();
      } else {
        const isWhole = it.kind === "whole";
        const on = hover && hover.node === it.node && hover.kind === it.kind &&
                   (isWhole || hover.si === it.si) && (isWhole || hover.fi === it.fi);
        const r = (isWhole ? 5.2 : 2.2) * it.s * (on ? 1.7 : 1);
        ctx.save();
        ctx.globalAlpha = it.faded ? 0.07 : (isWhole ? 0.95 : 0.6);
        if (isWhole){
          ctx.fillStyle = it.col;
          ctx.beginPath(); ctx.arc(it.x, it.y, r, 0, Math.PI*2); ctx.fill();
          ctx.globalAlpha = it.faded ? 0.1 : 1;
          ctx.strokeStyle = cvar("--panel"); ctx.lineWidth = 1.2; ctx.stroke();
        } else {
          ctx.strokeStyle = it.col; ctx.lineWidth = 1.3;
          ctx.beginPath(); ctx.arc(it.x, it.y, r, 0, Math.PI*2); ctx.stroke();
        }
        ctx.restore();
        hits.push({ x: it.x, y: it.y, kind: it.kind, node: it.node, si: it.si, fi: it.fi });
      }
    }
  }

  ampEl.addEventListener("input", () => {
    amp = parseFloat(ampEl.value);
    ampVal.textContent = amp.toFixed(1) + "×";
    draw();
  });
  for (const el of [showSeek, showCand, showSections, showPairs]) el.addEventListener("change", draw);
  for (const [id, val] of [["e-all","all"],["e-pos","pos"],["e-neg","neg"]]){
    $("#"+id).onclick = () => {
      edgeFilter = val;
      ["e-all","e-pos","e-neg"].forEach(x => $("#"+x).setAttribute("aria-pressed", String(x === id)));
      draw();
    };
  }
  $("#e-reset").onclick = () => {
    yaw = 0.6; pitch = -0.28; zoom = 1; focus = null; amp = 1;
    ampEl.value = "1"; ampVal.textContent = "1.0×";
    showSeek.checked = showCand.checked = showSections.checked = showPairs.checked = true;
    edgeFilter = "all";
    ["e-all","e-pos","e-neg"].forEach(x => $("#"+x).setAttribute("aria-pressed", String(x === "e-all")));
    draw();
  };

  // Canvas has no CSS cascade — it must be repainted by hand on a theme change.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  new MutationObserver(draw).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] });

  resize();
  return { resize };
};
ONSHOW.embed = h => h.resize();
"""

_JS_BOOT = r"""
showTab((location.hash || "#pairs").slice(1));
"""

_TEMPLATE = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>RRF batch __BATCH__</title>\n<style>"
    + _CSS
    + "</style>\n</head>\n<body>\n"
    + _MARKUP
    + "\n<script>\n"
    + _JS_SHELL + _JS_PAIRS + _JS_TOPO + _JS_EMBED + _JS_BOOT
    + "\n</script>\n</body>\n</html>\n"
)


def build_html(payload: dict[str, Any], batch_id: str) -> str:
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    # .replace() only — the JS and CSS are dense with {} and %, so f-strings and
    # str.format are both unusable here.
    return _TEMPLATE.replace("__PAYLOAD__", blob).replace("__BATCH__", html.escape(batch_id))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-id", default="rrf_002")
    ap.add_argument("--batch-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--real-pos", type=Path, default=REPO_ROOT / "data" / "dataset_positive.json")
    ap.add_argument("--real-neg", type=Path, default=REPO_ROOT / "data" / "dataset_negative.json")
    ap.add_argument("--no-real", action="store_true",
                    help="skip the real-pair comparison pane in the topology tab")
    args = ap.parse_args(argv)

    batch_dir = args.batch_dir or (REPO_ROOT / "artifacts" / "pairing_rrf" / args.batch_id)
    if not batch_dir.is_dir():
        raise SystemExit(f"no such batch directory: {batch_dir}")

    data = load_batch(batch_dir)
    real = None if args.no_real else load_real_pairs(args.real_pos, args.real_neg)
    payload = build_payload(data, args.batch_id, batch_dir, real)

    out = args.out or (batch_dir / "_browser.html")
    out.write_text(build_html(payload, args.batch_id), encoding="utf-8")

    d = payload["diag"]
    print(f"{d['n_pairs']} pairs  ({d['n_pos']} pos / {d['n_neg']} neg), "
          f"{len(payload['contacts'])} contacts")
    print(f"  probes: seeker-identity AUC {d['auc_seeker_identity']:.3f}  "
          f"within-seeker RRF AUC {d['auc_rrf_within_seeker']:.3f}  triplets {d['triplets']}")
    t = payload["topology"]
    if t["real"]:
        r = t["real"]["stats"]
        print(f"  topology: real {r['nodes']}n/{r['edges']}e  {r['mean_degree']}/node  "
              f"{r['components']} components")
    else:
        print("  topology: real pairs unavailable — synthetic pane only")
    s = t["synth"]["stats"]
    print(f"            synth {s['nodes']}n/{s['edges']}e  {s['mean_degree']}/node  "
          f"{s['components']} component(s)")
    e = payload["embed"]
    if e:
        st = e["stats"]
        fields_note = f" + {st['nFields']} isolated fields" if e.get("hasFields") else ""
        print(f"  embeddings: {st['nSeekers']} seekers + {st['nCandidates']} candidates "
              f"+ {st['nSections']} asks{fields_note}, PCA3 keeps {e['evrSum']*100:.1f}%")
    else:
        print("  embeddings: unavailable — tab hidden")
    print(f"→ {out}  ({out.stat().st_size/1048576:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
