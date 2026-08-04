#!/usr/bin/env python3
"""Split the RRF pairing embedding workload across two Modal accounts, run
both halves concurrently, merge, and persist exactly what
synth_pipeline.pairing_rrf_qwen_judge.run.py's own embedding step would have
written — so a normal pipeline run picks it up via the resume check in
run.py and skips re-embedding.

One-off operational speedup, not a pipeline stage: this run's embedding
workload (~22.7k vectors, ~35-40 min single-account) is large enough that
running two Modal accounts side by side roughly halves wall-clock time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile-run", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="batch dir, e.g. artifacts/pairing_rrf_qwen_judge/<batch_id>")
    p.add_argument("--seeker-frac", type=float, default=0.43)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--account-a-token-id", required=True)
    p.add_argument("--account-a-token-secret", required=True)
    p.add_argument("--account-b-token-id", required=True)
    p.add_argument("--account-b-token-secret", required=True)
    args = p.parse_args()

    from synth_pipeline.pairing.profiles import load_profile_run
    from synth_pipeline.pairing.select import split_seekers_candidates
    from synth_pipeline.pairing_rrf_qwen_judge import embed as embed_mod

    profiles = load_profile_run(args.profile_run)
    seekers, candidates = split_seekers_candidates(profiles, seeker_frac=args.seeker_frac, seed=args.seed)
    seeker_profiles = {sp.contact_id: sp.profile for sp in seekers}
    candidate_profiles = {sp.contact_id: sp.profile for sp in candidates}

    plan = embed_mod.build_plan(seeker_profiles, candidate_profiles)
    n_s, n_c = plan.n_seeker, plan.n_candidate
    print(f"plan: {n_s} seeker vectors, {n_c} candidate vectors")

    s_mid, c_mid = n_s // 2, n_c // 2
    halves = [
        {"seeker_texts": plan.seeker_texts[:s_mid], "candidate_texts": plan.candidate_texts[:c_mid]},
        {"seeker_texts": plan.seeker_texts[s_mid:], "candidate_texts": plan.candidate_texts[c_mid:]},
    ]
    tmp_dir = args.out_dir / "embeddings" / "_two_account_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tokens = [
        (args.account_a_token_id, args.account_a_token_secret),
        (args.account_b_token_id, args.account_b_token_secret),
    ]
    procs = []
    for i, (half, (tok_id, tok_secret)) in enumerate(zip(halves, tokens)):
        input_path = tmp_dir / f"half_{i}_input.json"
        output_path = tmp_dir / f"half_{i}_output.npz"
        input_path.write_text(json.dumps(half), encoding="utf-8")
        env = dict(os.environ)
        env["MODAL_TOKEN_ID"] = tok_id
        env["MODAL_TOKEN_SECRET"] = tok_secret
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "embed_two_accounts_worker.py"),
            "--input", str(input_path),
            "--output", str(output_path.with_suffix("")),  # np.savez appends .npz
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--max-length", str(args.max_length),
        ]
        log_path = tmp_dir / f"half_{i}.log"
        print(f"launching half {i} ({len(half['seeker_texts'])} seeker + "
              f"{len(half['candidate_texts'])} candidate texts) -> {log_path}")
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        procs.append((proc, log_f, output_path))

    exit_codes = []
    for proc, log_f, _ in procs:
        code = proc.wait()
        log_f.close()
        exit_codes.append(code)

    for i, code in enumerate(exit_codes):
        print(f"half {i} exit code: {code}")
    if any(code != 0 for code in exit_codes):
        raise SystemExit("one or both halves failed; check logs under " + str(tmp_dir))

    import numpy as np

    seeker_parts, cand_parts = [], []
    for _, _, output_path in procs:
        data = np.load(output_path)
        seeker_parts.append(data["seeker"])
        cand_parts.append(data["candidate"])

    seeker_mat = np.concatenate(seeker_parts, axis=0)
    cand_mat = np.concatenate(cand_parts, axis=0)
    assert seeker_mat.shape[0] == n_s, f"seeker row mismatch: {seeker_mat.shape[0]} != {n_s}"
    assert cand_mat.shape[0] == n_c, f"candidate row mismatch: {cand_mat.shape[0]} != {n_c}"
    print(f"merged: seeker {seeker_mat.shape}, candidate {cand_mat.shape}")

    embed_mod.persist(
        args.out_dir / "embeddings",
        plan,
        seeker_mat,
        cand_mat,
        model_name=args.model,
        extra_meta={"profile_run": args.profile_run.name, "two_account_split": True},
    )
    print(f"persisted to {args.out_dir / 'embeddings'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
