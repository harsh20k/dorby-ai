"""Orchestrates the Voyage-4-large + Gemini pairing run. Plain function chain, no graph.

    reuse queries.json (rrf_qwen_full_001) → embed (Voyage-4-large, API)
             → persist vectors → vector store → dense-only top-10 recall
             → dedupe → judge (gemini-3.1-flash-lite, direct Google API) → labels

No fusion phase — a query's dense top-10 *is* its shortlist, since there is
only one recall channel in this batch. Query generation is also skipped
entirely: this batch reuses the queries already generated and paid for in
``rrf_qwen_full_001`` (same 9,659-profile pool, same seeker/candidate split
inputs), read only, never regenerated.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from synth_pipeline.pairing.profiles import SynthProfile, load_profile_run
from synth_pipeline.pairing.select import split_seekers_candidates
from synth_pipeline.pairing_voyage_gemini import embed as embed_mod
from synth_pipeline.pairing_voyage_gemini import label as label_mod
from synth_pipeline.pairing_voyage_gemini import recall, store
from synth_pipeline.pairing_voyage_gemini.judge import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL
from synth_pipeline.pairing_voyage_gemini.judge import Judge
from synth_pipeline.pairing_voyage_gemini.sections import query_targets

DEFAULT_QUERIES_SOURCE = Path("artifacts/pairing_rrf_qwen_judge/rrf_qwen_full_001/queries.json")


@dataclass
class RunConfig:
    profile_run: Path
    batch_id: str
    data_dir: Path
    artifacts_dir: Path
    queries_source: Path = DEFAULT_QUERIES_SOURCE
    embed_model: str = embed_mod.DEFAULT_MODEL
    embed_output_dimension: int = embed_mod.DEFAULT_OUTPUT_DIMENSION
    seeker_frac: float = 0.43
    recall_k: int = 10
    seed: int = 42
    limit: int | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_concurrency: int = 4
    use_chroma: bool = True
    dedupe_pairs: bool = True
    skip_judge: bool = False

    def batch_dir(self) -> Path:
        return Path(self.artifacts_dir) / "pairing_voyage_gemini" / self.batch_id


@dataclass
class PhaseTimings:
    phases: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        self.phases[name] = round(seconds, 2)


def _profile_map(profiles: list[SynthProfile]) -> dict[str, dict[str, Any]]:
    return {p.contact_id: p.profile for p in profiles}


def _contact_files(profiles: list[SynthProfile]) -> dict[str, dict[str, Any]]:
    return {p.contact_id: p.as_contact_file() for p in profiles}


def _load_reused_queries(path: Path) -> dict[str, str]:
    """Read-only import of a prior batch's checkpoint, written by
    ``synth_pipeline.pairing_rrf_qwen_judge.query_gen``: a wrapper object with
    ``{"queries": {key: text}, "model_id": ..., "usage": ..., "failures": [...]}``."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"queries source not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = raw.get("queries", raw)  # tolerate a bare flat map too
    out: dict[str, str] = {}
    for key, value in queries.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, dict):
            out[key] = str(value.get("query") or value.get("text") or "")
    return out


def run(cfg: RunConfig) -> dict[str, Any]:
    t0 = time.time()
    timings = PhaseTimings()
    out_dir = cfg.batch_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. profiles + disjoint split (same inputs as rrf_qwen_full_001) --
    step = time.time()
    profiles = load_profile_run(Path(cfg.profile_run), limit=cfg.limit)
    if not profiles:
        raise SystemExit(f"no usable profiles under {cfg.profile_run}")
    seekers, candidates = split_seekers_candidates(
        profiles, seeker_frac=cfg.seeker_frac, seed=cfg.seed
    )
    seeker_profiles = _profile_map(seekers)
    candidate_profiles = _profile_map(candidates)
    seeker_files = _contact_files(seekers)
    candidate_files = _contact_files(candidates)
    timings.record("load_and_split", time.time() - step)
    print(f"profiles={len(profiles)} seekers={len(seekers)} candidates={len(candidates)}")

    # ---- 2. reuse queries from rrf_qwen_full_001, no generation ------------
    step = time.time()
    targets = [t for cid in sorted(seeker_profiles) for t in query_targets(cid, seeker_profiles[cid])]
    queries = _load_reused_queries(cfg.queries_source)
    n_found = sum(1 for t in targets if queries.get(t.key, "").strip())
    timings.record("reuse_queries", time.time() - step)
    print(f"query targets: {len(targets)}, matched in {cfg.queries_source}: {n_found}")

    # ---- 3. embed both sides, persist arrays first -------------------------
    step = time.time()
    embed_dir = out_dir / "embeddings"
    plan = embed_mod.build_plan(seeker_profiles, candidate_profiles, targets, queries)
    if (embed_dir / "manifest.json").exists():
        print(f"embeddings already persisted at {embed_dir}, skipping re-embed")
        seeker_mat, cand_mat, _ = embed_mod.load_persisted(embed_dir)
    else:
        print(
            f"embedding {plan.n_seeker} seeker vectors (1 per query) "
            f"and {plan.n_candidate} candidate vectors with {cfg.embed_model}"
        )
        seeker_mat, cand_mat = embed_mod.embed_plan(
            plan,
            model_name=cfg.embed_model,
            output_dimension=cfg.embed_output_dimension,
            cache_dir=embed_dir / "_cache",
        )
        embed_mod.persist(
            embed_dir,
            plan,
            seeker_mat,
            cand_mat,
            model_name=cfg.embed_model,
            extra_meta={"batch_id": cfg.batch_id, "profile_run": Path(cfg.profile_run).name},
        )
    timings.record("embedding", time.time() - step)

    # ---- 4. vector store ---------------------------------------------------
    step = time.time()
    vstore = store.open_store(
        out_dir / "chroma",
        plan.candidate_ids,
        cand_mat,
        use_chroma=cfg.use_chroma,
        metadatas=[{"contact_id": cid, "role": "candidate"} for cid in plan.candidate_ids],
    )
    _, _, manifest = embed_mod.load_persisted(embed_dir)
    seeker_rows = recall.seeker_row_index(manifest)
    timings.record("index_build", time.time() - step)
    print(f"vector store: {getattr(vstore, 'backend', '?')} with {vstore.size} candidates")

    # ---- 5. dense-only recall = shortlist directly --------------------------
    step = time.time()
    recalls = recall.recall_all(
        targets,
        queries,
        store=vstore,
        seeker_matrix=seeker_mat,
        seeker_rows=seeker_rows,
        k=cfg.recall_k,
    )
    timings.record("recall", time.time() - step)
    print(f"queries with recall: {len(recalls)}")

    # ---- 6. dedupe (seeker, candidate) pairs across all their queries ------
    step = time.time()
    seen: set[tuple[str, str]] = set()
    shortlist_rows: list[dict[str, Any]] = []
    for qr in recalls:
        for hit in qr.hits:
            pair_key_tuple = (qr.target.contact_id, hit.candidate_id)
            if cfg.dedupe_pairs and pair_key_tuple in seen:
                continue
            seen.add(pair_key_tuple)
            shortlist_rows.append(
                {
                    "seeker_id": qr.target.contact_id,
                    "section_index": qr.target.section_index,
                    "query_key": qr.target.key,
                    "query_text": qr.query_text,
                    "candidate_id": hit.candidate_id,
                    "similarity": round(hit.similarity, 6),
                }
            )
    n_pairs = len(shortlist_rows)
    (out_dir / "shortlist.json").write_text(json.dumps(shortlist_rows, indent=2), encoding="utf-8")
    timings.record("dedupe", time.time() - step)
    print(f"shortlisted pairs to judge: {n_pairs}")

    if cfg.skip_judge:
        print("--skip-judge set; stopping before any Gemini API spend.")
        return _write_run_summary(out_dir, cfg, timings, n_pairs, judged=None, judge_usage=None, t0=t0)

    # ---- 7. judge -----------------------------------------------------------
    step = time.time()
    judge = Judge(cache_dir=out_dir / "judge_cache", model=cfg.judge_model)
    items = []
    lookup: dict[str, dict[str, Any]] = {}
    for row in shortlist_rows:
        key = f"{row['query_key']}__{row['candidate_id']}"
        items.append((key, seeker_files[row["seeker_id"]], row["query_text"], candidate_files[row["candidate_id"]]))
        lookup[key] = row

    done = {"n": 0}

    def _progress(_res) -> None:
        done["n"] += 1
        if done["n"] % 25 == 0:
            print(f"  judged {done['n']}/{len(items)} — ${judge.usage.cost_usd:.4f} so far")

    results = judge.judge_many(items, concurrency=cfg.judge_concurrency, on_result=_progress)
    timings.record("judging", time.time() - step)

    judged: list[label_mod.JudgedPair] = []
    errors = []
    for res in results:
        if res.error or not res.verdict:
            errors.append({"pair_key": res.pair_key, "error": res.error})
            continue
        row = lookup[res.pair_key]
        judged.append(
            label_mod.JudgedPair(
                seeker_id=row["seeker_id"],
                candidate_id=row["candidate_id"],
                query_key=row["query_key"],
                query_text=row["query_text"],
                section_index=row["section_index"],
                seeker_profile=seeker_files[row["seeker_id"]],
                candidate_profile=candidate_files[row["candidate_id"]],
                verdict=res.verdict,
                dense={"similarity": row["similarity"]},
            )
        )
    if errors:
        (out_dir / "judge_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"!! {len(errors)} judge calls failed — see judge_errors.json")

    # ---- 8. labels + manifest ----------------------------------------------
    step = time.time()
    labeler_meta = {
        "kind": "llm_judge",
        "model": judge.model,
        "framing": "focused",
        "backend": "google_direct_api",
        "prompt_ref": judge.prompt_ref,
        "votes_per_pair": 1,
        "deadband": False,
        "all200_pair_auc": 0.6451,
        "all200_decision_accuracy": 0.5950,
    }
    label_manifest = label_mod.write_batch(
        out_dir,
        judged,
        batch_id=cfg.batch_id,
        split_hash=_split_hash(cfg.data_dir),
        labeler_meta=labeler_meta,
        manifest_extra={
            "balance": label_mod.label_balance(judged),
            "profile_run": Path(cfg.profile_run).name,
            "queries_source": str(cfg.queries_source),
        },
    )
    timings.record("labeling", time.time() - step)
    print(f"labels: {label_manifest['counts']}")

    return _write_run_summary(out_dir, cfg, timings, n_pairs, judged=judged, judge_usage=judge.usage, t0=t0)


def _split_hash(data_dir: Path) -> str:
    path = Path(data_dir) / "synthetic" / "seed_split.json"
    if not path.exists():
        return "unknown"
    return str(json.loads(path.read_text(encoding="utf-8")).get("split_hash", "unknown"))


def _write_run_summary(
    out_dir: Path,
    cfg: RunConfig,
    timings: PhaseTimings,
    n_pairs: int,
    *,
    judged: list[label_mod.JudgedPair] | None,
    judge_usage: Any,
    t0: float,
) -> dict[str, Any]:
    cost = {
        "gemini_judge": judge_usage.to_dict() if judge_usage else None,
        "measured_judge_cost_usd": round(judge_usage.cost_usd, 6) if judge_usage else 0.0,
        "note": (
            "Judge cost is computed from the token counts Gemini's usageMetadata "
            "reports, at published gemini-3.1-flash-lite pricing. Voyage-4-large "
            "embedding cost is tracked separately in artifacts/voyage_large/usage.json."
        ),
    }
    summary = {
        "batch_id": cfg.batch_id,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "pairs_shortlisted": n_pairs,
        "pairs_labeled": len(judged) if judged is not None else 0,
        "balance": label_mod.label_balance(judged) if judged else None,
        "cost": cost,
        "timings_seconds": timings.phases,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (Path(out_dir) / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
