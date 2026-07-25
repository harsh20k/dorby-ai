"""LLM-judge experiment: can an LLM predict accept/decline from two profiles alone?

Feeds an LLM both complete profiles of a real pair — no ``searchQuery``, no
embeddings, no training — asks for a yes/no match call plus a confidence, and
scores it against the real human outcome on the 200 real seed pairs.

    python -m baselines.llm_judge.eval --data-dir data
    python -m baselines.llm_judge.eval --split holdout --model anthropic/claude-sonnet-4.5

Reports two views of the same run:
  * ``decision`` — the model's literal yes/no (accuracy / P / R / F1), the
    thing actually asked for, which no embedding baseline can produce.
  * ``pair`` — ``baselines/metrics.py::pair_metrics`` over the confidence-signed
    score, so ROC-AUC/AP are directly comparable to every other baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from baselines.bert_frozen.text import profile_to_text
from baselines.llm_judge.judge import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    VerdictCache,
    judge_all,
    load_env_file,
    make_bedrock_call_fn,
    make_openrouter_call_fn,
    model_slug,
    prompt_hash,
    verdict_to_score,
)
from baselines.llm_judge.metrics import (
    calibration_buckets,
    decision_metrics,
    intent_pair_auc,
)
from baselines.llm_judge.prompt import SYSTEM_PROMPTS, build_user_prompt
from baselines.llm_judge.real_pairs import load_real_pairs, pair_id
from baselines.metrics import neg_hardness_slice_metrics, pair_metrics


def build_requests(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    variant: str,
    max_field: int | None,
) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    """Return (requests, pos_keys, neg_keys) where a request is (key, system, user)."""
    system = SYSTEM_PROMPTS[variant]
    requests: list[tuple[str, str, str]] = []
    pos_keys: list[str] = []
    neg_keys: list[str] = []

    for label, records, keys in (("pos", positives, pos_keys), ("neg", negatives, neg_keys)):
        for record in records:
            user = build_user_prompt(
                record["userContactFile"],
                record["matchContactFile"],
                max_field=max_field,
            )
            key = f"{pair_id(record, label)}|{prompt_hash(system, user)}"
            keys.append(key)
            requests.append((key, system, user))
    return requests, pos_keys, neg_keys


def run_eval(
    *,
    data_dir: Path,
    split: str,
    variant: str,
    model: str,
    temperature: float,
    max_field: int | None,
    workers: int,
    max_attempts: int,
    max_failures: int,
    artifacts_dir: Path,
    split_path: Path | None,
    make_call_fn: Callable[[str, str], Any],
) -> dict[str, Any]:
    print(f"model:   {model}")
    print(f"variant: {variant} (searchQuery withheld from the model)")
    print(f"split:   {split}")

    positives, negatives = load_real_pairs(data_dir, split=split, split_path=split_path)
    print(f"loaded {len(positives)} real positives, {len(negatives)} real negatives")

    requests, pos_keys, neg_keys = build_requests(
        positives, negatives, variant=variant, max_field=max_field
    )
    prompt_chars = [len(u) for _, _, u in requests]
    print(
        f"prompt size (chars): mean {int(np.mean(prompt_chars))}, "
        f"max {max(prompt_chars)}"
        + ("" if max_field is None else f" (fields truncated at {max_field})")
    )

    cache = VerdictCache(artifacts_dir / "verdicts.json")
    print(f"verdict cache: {len(cache)} entries at {cache.path}")

    def progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            print(f"  judged {done}/{total}", flush=True)

    verdicts, errors = judge_all(
        requests,
        make_call_fn=make_call_fn,
        cache=cache,
        workers=workers,
        max_attempts=max_attempts,
        on_progress=progress,
    )

    if errors:
        print(f"\n{len(errors)} pair(s) failed after {max_attempts} attempts:")
        for key, msg in list(errors.items())[:5]:
            print(f"  {key.split('|')[0]}: {msg}")
        if len(errors) > max_failures:
            raise RuntimeError(
                f"{len(errors)} failures exceeds --max-failures {max_failures}; "
                "refusing to report metrics on a population this incomplete. "
                "Re-run to retry only the missing pairs (successes are cached)."
            )
        print("  → excluded from metrics (see failed_pairs in metrics.json)")

    # Dropping a failed pair changes the pos/neg balance, so the surviving
    # counts are recorded explicitly rather than assumed to be 100/100.
    pos_kept = [(k, verdicts[k]) for k in pos_keys if k in verdicts]
    neg_kept = [(k, verdicts[k]) for k in neg_keys if k in verdicts]
    pos_records = [r for k, r in zip(pos_keys, positives) if k in verdicts]
    neg_records = [r for k, r in zip(neg_keys, negatives) if k in verdicts]

    pos_verdicts = [v for _, v in pos_kept]
    neg_verdicts = [v for _, v in neg_kept]
    if not pos_verdicts or not neg_verdicts:
        raise RuntimeError("no usable verdicts on one side; cannot compute metrics")

    pos_scores = np.array([verdict_to_score(v) for v in pos_verdicts], dtype=np.float64)
    neg_scores = np.array([verdict_to_score(v) for v in neg_verdicts], dtype=np.float64)

    # Negative-hardness slicing measures lexical overlap between the two texts
    # the model was actually shown — both profiles, no query.
    neg_seeker_texts = [profile_to_text(r["userContactFile"]) for r in neg_records]
    neg_cand_texts = [profile_to_text(r["matchContactFile"]) for r in neg_records]

    return {
        "model_name": model,
        "experiment": "llm_judge",
        "prompt_variant": variant,
        "uses_search_query": False,
        "split": split,
        "temperature": temperature,
        "max_field": max_field,
        "num_pairs_requested": len(requests),
        "num_pairs_scored": len(pos_verdicts) + len(neg_verdicts),
        "failed_pairs": {k.split("|")[0]: v for k, v in errors.items()},
        "decision": decision_metrics(pos_verdicts, neg_verdicts),
        "calibration": calibration_buckets(pos_verdicts, neg_verdicts),
        "pair": pair_metrics(pos_scores, neg_scores),
        # No "retrieval" key: an LLM judge has no shared vector space to rank a
        # candidate corpus in. Absence is deliberate, not an oversight.
        "slices": {
            "intent": intent_pair_auc(pos_records, neg_records, pos_scores, neg_scores),
            "neg_hardness": neg_hardness_slice_metrics(
                neg_scores, neg_seeker_texts, neg_cand_texts, pos_scores=pos_scores
            ),
        },
    }


def print_report(m: dict[str, Any]) -> None:
    d = m["decision"]
    p = m["pair"]
    print(f"\n=== LLM judge: {m['model_name']} ({m['prompt_variant']}) ===")
    print(f"pairs scored: {m['num_pairs_scored']}/{m['num_pairs_requested']}")

    print("\n--- Decision (the model's own yes/no) ---")
    print(f"Accuracy:   {d['accuracy']:.4f}   (chance = 0.5)")
    print(f"Precision:  {d['precision']:.4f}")
    print(f"Recall:     {d['recall']:.4f}")
    print(f"F1:         {d['f1']:.4f}")
    c = d["confusion"]
    print(
        f"Confusion:  TP={c['true_positive']} FP={c['false_positive']} "
        f"TN={c['true_negative']} FN={c['false_negative']}"
    )
    print(
        f"Says 'yes': {d['yes_rate']:.1%} overall "
        f"({d['yes_rate_on_positives']:.1%} of accepted, "
        f"{d['yes_rate_on_negatives']:.1%} of declined)"
    )
    mc_ok, mc_bad = d["mean_confidence_when_correct"], d["mean_confidence_when_wrong"]
    print(
        "Confidence: "
        f"{d['mean_confidence']:.1f} mean, "
        f"{'n/a' if mc_ok is None else format(mc_ok, '.1f')} when right, "
        f"{'n/a' if mc_bad is None else format(mc_bad, '.1f')} when wrong"
    )

    print("\n--- Pair metrics (confidence-signed score, comparable to baselines) ---")
    print(f"ROC-AUC:            {p['roc_auc']:.4f}")
    print(f"Average Precision:  {p['average_precision']:.4f}")
    print(f"Best-F1:            {p['best_f1']:.4f} @ {p['best_f1_threshold']:.4f}")
    print(f"Mean score gap:     {p['mean_cosine_gap']:+.4f}")

    print("\n--- Calibration ---")
    for b in m["calibration"]:
        acc = b["accuracy"]
        print(
            f"  conf {b['confidence_range']:>14}: n={b['n']:>3}  "
            f"acc={'n/a' if acc is None else format(acc, '.4f')}"
        )

    hard = m["slices"]["neg_hardness"]
    print("\n--- Negative-hardness slices ---")
    for name in ("easy", "hard"):
        s = hard.get(name) or {}
        auc = s.get("pair_auc")
        print(
            f"  {name:>4}-neg: n={s.get('n_negatives', '?')}  "
            f"pair_auc={'n/a' if auc is None else format(auc, '.4f')}"
        )

    print("\n--- Intent slices (pair AUC) ---")
    for bucket, s in m["slices"]["intent"].items():
        auc = s["pair_auc"]
        flag = "  (low n)" if s["low_n"] else ""
        print(
            f"  {bucket:>12}: n={s['n_pairs']:>3}  "
            f"pair_auc={'n/a' if auc is None else format(auc, '.4f')}{flag}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-judge experiment on real pairs (no searchQuery given to the model)"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--split",
        choices=["all", "train", "holdout"],
        default="all",
        help="'all' = the 200 real seed pairs; 'holdout' = the frozen 69 used by "
        "docs/baseline-results-holdout.md.",
    )
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument(
        "--backend",
        choices=["openrouter", "bedrock"],
        default="openrouter",
        help="'openrouter' uses OPENROUTER_API_KEY; 'bedrock' uses --aws-profile/--aws-region "
        "credentials and boto3's Converse API (see baselines/llm_judge/bedrock_backend.py).",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"OpenRouter model id (default {DEFAULT_MODEL}) or Bedrock model id "
        "(e.g. google.gemma-3-27b-it) depending on --backend.",
    )
    p.add_argument("--base-url", default=None, help=f"OpenRouter only; default {DEFAULT_BASE_URL}")
    p.add_argument("--aws-profile", default="tf_provisioner", help="Bedrock only.")
    p.add_argument("--aws-region", default="us-east-1", help="Bedrock only.")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Output token cap. The verdict schema needs a few hundred at most — "
        "left high, OpenRouter's credit preflight check reserves against the "
        "model's absolute ceiling instead of real usage (see docs/llm-judge-experiment.md).",
    )
    p.add_argument(
        "--variant",
        choices=sorted(SYSTEM_PROMPTS),
        default="naive",
        help="'naive' asks plainly if it's a good match; 'calibrated' also states "
        "that production already deemed the pair relevant and the base rate is 50/50.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--max-field",
        type=int,
        default=None,
        help="Truncate each profile field to N chars. Default: no truncation "
        "(complete profiles, which is the point of the experiment).",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Abort instead of reporting metrics if more pairs than this fail.",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to a .env holding OPENROUTER_API_KEY. Required when running "
        "from a git worktree, which has no .env of its own.",
    )
    p.add_argument("--artifacts-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(args.env_file)

    if args.backend == "openrouter":
        model = args.model or DEFAULT_MODEL
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            print(
                "error: no OPENROUTER_API_KEY (or OPENAI_API_KEY) in the environment. "
                "Pass --env-file /path/to/.env.",
                file=sys.stderr,
            )
            return 2
        base_url = args.base_url or DEFAULT_BASE_URL

        def make_call_fn(system: str, user: str):
            return make_openrouter_call_fn(
                system=system,
                user=user,
                model=model,
                temperature=args.temperature,
                api_key=api_key,
                base_url=base_url,
                max_tokens=args.max_tokens,
            )

    else:
        if not args.model:
            print("error: --model is required with --backend bedrock", file=sys.stderr)
            return 2
        model = args.model
        from baselines.llm_judge.bedrock_backend import make_client

        try:
            client = make_client(profile=args.aws_profile, region=args.aws_region)
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not create a Bedrock client: {exc}", file=sys.stderr)
            return 2

        def make_call_fn(system: str, user: str):
            return make_bedrock_call_fn(
                client=client,
                system=system,
                user=user,
                model_id=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

    # The split is deliberately *not* part of the artifacts path: verdict cache
    # keys are pair-identity + prompt-hash, so they are split-independent, and
    # sharing one cache dir per (model, variant) means `--split holdout` after
    # `--split all` is a pure cache hit instead of re-paying for 69 pairs.
    # Only metrics.json is split-scoped.
    artifacts_dir = args.artifacts_dir or Path("artifacts/llm_judge") / (
        f"{args.backend}_{model_slug(model)}_{args.variant}"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        split=args.split,
        variant=args.variant,
        model=model,
        temperature=args.temperature,
        max_field=args.max_field,
        workers=args.workers,
        max_attempts=args.max_attempts,
        max_failures=args.max_failures,
        artifacts_dir=artifacts_dir,
        split_path=args.split_path,
        make_call_fn=make_call_fn,
    )
    metrics["backend"] = args.backend
    print_report(metrics)

    out_path = artifacts_dir / f"metrics_{args.split}.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
