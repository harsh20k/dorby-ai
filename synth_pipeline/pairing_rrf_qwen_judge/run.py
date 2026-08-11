"""Orchestrates the RRF pairing run. Plain function chain, no graph.

    profiles → disjoint split → queries (Bedrock) → embed (Qwen3, GPU)
             → persist vectors → vector store → dense + lexical recall
             → weighted RRF → shortlist → judge (flash-lite) → labels

Every phase writes its output before the next begins, so a failure costs one
phase rather than the run, and the expensive artifacts — profiles, queries,
embeddings — are never regenerated to recover from a later crash.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from synth_pipeline.pairing.profiles import SynthProfile, load_profile_run
from synth_pipeline.pairing.select import split_seekers_candidates
from synth_pipeline.pairing_rrf_qwen_judge import embed as embed_mod
from synth_pipeline.pairing_rrf_qwen_judge import fuse as fuse_mod
from synth_pipeline.pairing_rrf_qwen_judge import label as label_mod
from synth_pipeline.pairing_rrf_qwen_judge import query_gen, recall, store
from synth_pipeline.pairing_rrf_qwen_judge.judge import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL
from synth_pipeline.pairing_rrf_qwen_judge.judge import Judge
from synth_pipeline.pairing_rrf_qwen_judge.sections import query_targets

DEFAULT_BEDROCK_MODEL = "google.gemma-3-27b-it"  # query generation model, unchanged
DEFAULT_JUDGE_AWS_PROFILE = "tf_provisioner"


@dataclass
class RunConfig:
    profile_run: Path
    batch_id: str
    data_dir: Path
    artifacts_dir: Path
    embed_model: str = embed_mod.DEFAULT_MODEL
    bedrock_model: str = DEFAULT_BEDROCK_MODEL
    region: str = "us-east-1"
    seeker_frac: float = 0.43
    recall_k: int = 10
    top_k: int = 5
    max_pairs_per_seeker: int | None = None
    min_dense_similarity: float | None = None
    dense_weight: float = fuse_mod.DENSE_WEIGHT
    lexical_weight: float = fuse_mod.LEXICAL_WEIGHT
    rrf_k: int = fuse_mod.RRF_K
    seed: int = 42
    limit: int | None = None
    judge_concurrency: int = 4
    query_gen_concurrency: int = 1
    use_chroma: bool = True
    embed_batch_size: int = 2
    embed_max_length: int = 4096
    embed_device: str | None = None
    embed_backend: str = "local"
    dedupe_pairs: bool = True
    skip_judge: bool = False

    def batch_dir(self) -> Path:
        return Path(self.artifacts_dir) / "pairing_rrf_qwen_judge" / self.batch_id


@dataclass
class PhaseTimings:
    phases: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        self.phases[name] = round(seconds, 2)


def _profile_map(profiles: list[SynthProfile]) -> dict[str, dict[str, Any]]:
    return {p.contact_id: p.profile for p in profiles}


def _contact_files(profiles: list[SynthProfile]) -> dict[str, dict[str, Any]]:
    return {p.contact_id: p.as_contact_file() for p in profiles}


def run(cfg: RunConfig) -> dict[str, Any]:
    t0 = time.time()
    timings = PhaseTimings()
    out_dir = cfg.batch_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. profiles + disjoint split -------------------------------------
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

    # ---- 2. one query per lookingFor section -------------------------------
    step = time.time()
    targets = [t for cid in sorted(seeker_profiles) for t in query_targets(cid, seeker_profiles[cid])]
    print(f"query targets (sections): {len(targets)}")

    from synth_pipeline.pairing.bedrock import make_client

    client = make_client(cfg.region)
    style_examples = _style_examples(cfg.data_dir)
    qres = query_gen.generate_queries(
        targets,
        seeker_profiles,
        client=client,
        model_id=cfg.bedrock_model,
        style_examples=style_examples,
        checkpoint_path=out_dir / "queries.json",
        concurrency=cfg.query_gen_concurrency,
    )
    timings.record("query_generation", time.time() - step)
    print(f"queries written: {len(qres.queries)} (failures: {len(qres.failures)})")

    # ---- 3. embed both sides, persist arrays first -------------------------
    step = time.time()
    embed_dir = out_dir / "embeddings"
    plan = embed_mod.build_plan(seeker_profiles, candidate_profiles)
    if (embed_dir / "manifest.json").exists():
        # Resume: a prior run (or an out-of-band precompute, e.g. splitting the
        # embedding workload across two Modal accounts to run in parallel)
        # already wrote embeddings here. Trust them rather than re-spending GPU
        # time — embed_mod.persist()/load_persisted() round-trip losslessly.
        print(f"embeddings already persisted at {embed_dir}, skipping re-embed")
        seeker_mat, cand_mat, _ = embed_mod.load_persisted(embed_dir)
    else:
        print(
            f"embedding {plan.n_seeker} seeker vectors (N+1 per seeker) "
            f"and {plan.n_candidate} candidate vectors with {cfg.embed_model}"
        )
        seeker_mat, cand_mat = embed_mod.embed_plan(
            plan,
            model_name=cfg.embed_model,
            batch_size=cfg.embed_batch_size,
            max_length=cfg.embed_max_length,
            device=cfg.embed_device,
            cache_dir=embed_dir / "_cache",
            backend=cfg.embed_backend,
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
    candidate_index = {cid: i for i, cid in enumerate(plan.candidate_ids)}
    timings.record("index_build", time.time() - step)
    print(f"vector store: {getattr(vstore, 'backend', '?')} with {vstore.size} candidates")

    # ---- 5. two recall channels -------------------------------------------
    step = time.time()
    bm25 = recall.BM25Index(plan.candidate_ids, plan.candidate_texts)
    recalls = recall.recall_all(
        targets,
        qres.queries,
        store=vstore,
        seeker_matrix=seeker_mat,
        seeker_rows=seeker_rows,
        bm25=bm25,
        k=cfg.recall_k,
    )
    timings.record("recall", time.time() - step)
    print(f"queries with recall: {len(recalls)}")

    # ---- 6. weighted RRF + shortlist ---------------------------------------
    step = time.time()
    shortlists = []
    for qr in recalls:
        rows = [seeker_rows[k] for k in (
            f"{qr.target.contact_id}::whole",
            f"{qr.target.contact_id}::s{qr.target.section_index}",
        ) if k in seeker_rows]
        shortlists.append(
            fuse_mod.build_shortlist(
                qr,
                seeker_vectors=seeker_mat[rows] if rows else np.zeros((0, seeker_mat.shape[1]), np.float32),
                candidate_index=candidate_index,
                candidate_matrix=cand_mat,
                top_k=cfg.top_k,
                min_dense_similarity=cfg.min_dense_similarity,
                dense_weight=cfg.dense_weight,
                lexical_weight=cfg.lexical_weight,
                rrf_k=cfg.rrf_k,
            )
        )
    if cfg.dedupe_pairs:
        shortlists = fuse_mod.deduplicate_pairs(shortlists)
    shortlists = fuse_mod.apply_seeker_budget(
        shortlists, max_pairs_per_seeker=cfg.max_pairs_per_seeker
    )
    n_pairs = sum(len(s.candidates) for s in shortlists)
    (out_dir / "shortlists.json").write_text(
        json.dumps(
            [
                {
                    "seeker_id": s.seeker_id,
                    "section_index": s.section_index,
                    "query_key": s.query_key,
                    "query_text": s.query_text,
                    "candidates": [c.to_dict() for c in s.candidates],
                    "dropped": s.dropped,
                }
                for s in shortlists
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    timings.record("fusion", time.time() - step)
    print(f"shortlisted pairs to judge: {n_pairs}")

    if cfg.skip_judge:
        print("--skip-judge set; stopping before any Bedrock judge spend.")
        return _write_run_summary(
            out_dir, cfg, timings, qres, n_pairs, judged=None, judge_usage=None, t0=t0
        )

    # ---- 7. judge -----------------------------------------------------------
    step = time.time()
    judge = Judge(cache_dir=out_dir / "judge_cache", aws_profile=DEFAULT_JUDGE_AWS_PROFILE)
    items = []
    lookup: dict[str, tuple[Any, Any]] = {}
    for s in shortlists:
        for cand in s.candidates:
            key = f"{s.query_key}__{cand.candidate_id}"
            items.append(
                (
                    key,
                    seeker_files[s.seeker_id],
                    s.query_text,
                    candidate_files[cand.candidate_id],
                )
            )
            lookup[key] = (s, cand)

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
        s, cand = lookup[res.pair_key]
        judged.append(
            label_mod.JudgedPair(
                seeker_id=s.seeker_id,
                candidate_id=cand.candidate_id,
                query_key=s.query_key,
                query_text=s.query_text,
                section_index=s.section_index,
                seeker_profile=seeker_files[s.seeker_id],
                candidate_profile=candidate_files[cand.candidate_id],
                verdict=res.verdict,
                fusion=cand.to_dict(),
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
        "framing": "naive",
        "prompt_ref": judge.prompt_ref.to_dict(),
        "votes_per_pair": 1,
        "deadband": False,
        "holdout_pair_auc": 0.6358,
        "holdout_hard_neg_auc": 0.6466,
        "holdout_accuracy_at_decision_point": 0.5942,
    }
    manifest = label_mod.write_batch(
        out_dir,
        judged,
        batch_id=cfg.batch_id,
        split_hash=_split_hash(cfg.data_dir),
        labeler_meta=labeler_meta,
        manifest_extra={
            "balance": label_mod.label_balance(judged),
            "profile_run": Path(cfg.profile_run).name,
        },
    )
    timings.record("labeling", time.time() - step)
    print(f"labels: {manifest['counts']}")

    return _write_run_summary(
        out_dir, cfg, timings, qres, n_pairs, judged=judged, judge_usage=judge.usage, t0=t0
    )


def _style_examples(data_dir: Path, k: int = 3) -> list[str]:
    """Real queries used purely as tone anchors, train users only."""
    from synth_pipeline.pairing.query import load_style_examples

    try:
        return load_style_examples(Path(data_dir), k=k)
    except Exception as exc:  # noqa: BLE001
        print(f"style examples unavailable ({exc}); continuing without them")
        return []


def _split_hash(data_dir: Path) -> str:
    path = Path(data_dir) / "synthetic" / "seed_split.json"
    if not path.exists():
        return "unknown"
    return str(json.loads(path.read_text(encoding="utf-8")).get("split_hash", "unknown"))


def _write_run_summary(
    out_dir: Path,
    cfg: RunConfig,
    timings: PhaseTimings,
    qres: query_gen.QueryGenResult,
    n_pairs: int,
    *,
    judged: list[label_mod.JudgedPair] | None,
    judge_usage: Any,
    t0: float,
) -> dict[str, Any]:
    cost = {
        "bedrock_query_generation": qres.usage.to_dict(),
        "openrouter_judge": judge_usage.to_dict() if judge_usage else None,
        "measured_judge_cost_usd": round(judge_usage.cost_usd, 6) if judge_usage else 0.0,
        "note": (
            "Judge cost is the exact amount OpenRouter billed, read from each "
            "response's usage payload. Bedrock token counts are real; convert to "
            "dollars with scripts/estimate_bedrock_cost.py."
        ),
    }
    summary = {
        "batch_id": cfg.batch_id,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "pairs_shortlisted": n_pairs,
        "pairs_labeled": len(judged) if judged is not None else 0,
        "calls_per_labeled_pair": (
            round(judge_usage.calls / len(judged), 3)
            if judged and judge_usage and len(judged)
            else None
        ),
        "balance": label_mod.label_balance(judged) if judged else None,
        "cost": cost,
        "timings_seconds": timings.phases,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (Path(out_dir) / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
