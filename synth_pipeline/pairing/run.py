"""Orchestrate profiles -> queries -> candidates -> scores -> labeled batch.

Phases run to completion in order rather than per-pair: selection needs the whole
TF-IDF matrix over the profile set, and scoring batches far better than it streams.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from synth_pipeline.pairing import label as label_mod
from synth_pipeline.pairing import query as query_mod
from synth_pipeline.pairing import stage as stage_mod
from synth_pipeline.pairing.bedrock import DEFAULT_MODEL_ID, make_client
from synth_pipeline.pairing.profiles import load_profile_run
from synth_pipeline.pairing.select import select_candidates


def run_pairing(
    *,
    profile_run: Path,
    batch_id: str,
    data_dir: Path,
    artifacts_dir: Path,
    split_path: Path | None = None,
    queries_per_profile: int = 2,
    k_per_query: int = 5,
    model_id: str = DEFAULT_MODEL_ID,
    region: str = "us-east-1",
    deadband_margin: float = 0.25,
    label_mode: str = "quantile",
    pos_frac: float = 0.3,
    neg_frac: float = 0.3,
    fusion_mode: str = "alpha",
    concurrency: int = 4,
    limit: int | None = None,
    refresh_queries: bool = False,
    seed: int = 42,
    log=print,
) -> dict[str, Any]:
    split_path = split_path or (Path(data_dir) / "synthetic" / "seed_split.json")
    out_dir = stage_mod.batch_dir(artifacts_dir, batch_id)

    # --- 1. profiles ---
    profiles = load_profile_run(profile_run, limit=limit)
    log(f"[1/5] loaded {len(profiles)} usable profiles from {Path(profile_run).name}")
    if len(profiles) < 2:
        raise RuntimeError("need at least 2 usable profiles to form pairs")

    # --- 2. queries (the only LLM calls) ---
    usage_total = {"inputTokens": 0, "outputTokens": 0, "calls": 0}
    checkpoint = out_dir / "queries.json"
    known = set()
    queries: dict[str, list[str]] = {}

    # Contact ids are deterministic, so a checkpoint from an earlier run of this
    # batch is still valid — reuse it rather than re-billing the same calls.
    if checkpoint.exists() and not refresh_queries:
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        known = {p.contact_id for p in profiles}
        queries = {k: v for k, v in cached.items() if k in known}
        log(f"[2/5] reusing {sum(len(v) for v in queries.values())} cached queries "
            f"({checkpoint.relative_to(artifacts_dir)})")

    todo = [p for p in profiles if not queries.get(p.contact_id)]
    if todo:
        client = make_client(region)
        style_examples = query_mod.load_style_examples(data_dir, split_path=split_path, k=3)
        log(f"[2/5] generating {queries_per_profile} queries for "
            f"{len(todo)} profile(s) via {model_id}")

        def _one(p):
            qs, usage = query_mod.generate_queries(
                client, p, model_id=model_id,
                style_examples=style_examples, n=queries_per_profile,
            )
            return p.contact_id, qs, usage

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for cid, qs, usage in pool.map(_one, todo):
                queries[cid] = qs
                usage_total["inputTokens"] += usage.get("inputTokens") or 0
                usage_total["outputTokens"] += usage.get("outputTokens") or 0
                usage_total["calls"] += 1
        log(f"      {usage_total['inputTokens']} in / "
            f"{usage_total['outputTokens']} out tokens")
    log(f"      {sum(len(v) for v in queries.values())} queries total")

    # --- 3. candidate selection (pure) ---
    candidates = select_candidates(profiles, queries, k_per_query=k_per_query)
    log(f"[3/5] selected {len(candidates)} unique (seeker, candidate) pairs")
    if not candidates:
        raise RuntimeError("no candidate pairs selected")

    # --- 4. score + label ---
    log("[4/5] fitting hybrid TF-IDF+Voyage-nano scorer on real train pairs...")
    scorer = label_mod.fit_scorer(
        Path(data_dir), Path(split_path),
        cache_dir=out_dir / "_scorer_cache",
        fusion_mode=fusion_mode, seed=seed,
    )
    log(f"      fit on {scorer.n_fit} real pairs, fit AUC={scorer.fusion.fit_auc:.4f}")

    seeker_texts, cand_texts = label_mod.pair_texts(candidates)
    # Content-hash the cache key: both encoders return a cached array whenever the
    # cache_name file exists, WITHOUT checking that the input texts still match.
    # A fixed key would silently serve stale embeddings after a query regen.
    texts_key = hashlib.sha256(
        "\n".join(seeker_texts + cand_texts).encode("utf-8")
    ).hexdigest()[:12]
    scores = scorer.score(
        seeker_texts, cand_texts, cache_prefix=f"batch_{batch_id}_{texts_key}"
    )
    log(f"      batch scores: min={scores.min():.3f} med={np.median(scores):.3f} "
        f"max={scores.max():.3f}")

    if label_mode == "quantile":
        thresholds = label_mod.quantile_thresholds(
            scores, pos_frac=pos_frac, neg_frac=neg_frac
        )
    else:
        thresholds = label_mod.compute_thresholds(scorer, margin=deadband_margin)
        real_lo, real_hi = float(scorer.fit_scores.min()), float(scorer.fit_scores.max())
        if scores.min() > real_hi or scores.max() < real_lo:
            log("      WARNING: batch scores lie entirely outside the real-pair "
                f"range [{real_lo:.3f}, {real_hi:.3f}] — an absolute threshold "
                "cannot transfer here; consider --label-mode quantile")
    log(f"      [{thresholds.mode}] deadband: <={thresholds.lower:.4f} neg | "
        f">={thresholds.upper:.4f} pos")

    labels = label_mod.label_scores(scores, thresholds)

    # --- 5. stage ---
    envelopes = [
        stage_mod.build_envelope(
            c, batch_id=batch_id, label=lab, score=float(sc),
            split_hash=scorer.meta.get("split_hash", ""),
            scorer_meta=scorer.meta.get("fusion", {}),
            drop_reason=None if lab else "deadband",
        )
        for c, lab, sc in zip(candidates, labels, scores)
    ]

    manifest = stage_mod.write_batch(
        artifacts_dir, batch_id,
        envelopes=envelopes,
        profiles=profiles,
        queries=queries,
        config={
            "profile_run": str(profile_run),
            "queries_per_profile": queries_per_profile,
            "k_per_query": k_per_query,
            "model_id": model_id,
            "region": region,
            "deadband_margin": deadband_margin,
            "label_mode": label_mode,
            "pos_frac": pos_frac,
            "neg_frac": neg_frac,
            "fusion_mode": fusion_mode,
            "seed": seed,
        },
        thresholds=asdict(thresholds),
        scorer_meta=scorer.meta,
        usage=usage_total,
    )
    counts = manifest["counts"]
    log(f"[5/5] wrote {counts['pos']} pos / {counts['neg']} neg / "
        f"{counts['excluded']} excluded -> "
        f"{stage_mod.batch_dir(artifacts_dir, batch_id)}")
    return manifest
