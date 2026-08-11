"""LLM-judge experiment: focused prompt + seeker `background` field, direct Google API only.

Isolated variation of baselines/llm_judge_with_pos_look_pos_back_look — same
prompt, query, and candidate fields; the only change is the seeker also gets
`background`. Only the direct Google Generative Language API backend is
supported here (not OpenRouter/Bedrock), since the focused-prompt experiment
already established the direct API is the more trustworthy measurement.

    python -m baselines.llm_judge_with_pos_look_back_pos_back_look.eval --data-dir data --env-file .env
    python -m baselines.llm_judge_with_pos_look_back_pos_back_look.eval --split holdout --env-file .env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from baselines.llm_judge.judge import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    VerdictCache,
    judge_all,
    load_env_file,
    model_slug,
    prompt_hash,
    verdict_to_score,
)
from baselines.llm_judge.metrics import calibration_buckets, decision_metrics, intent_pair_auc
from baselines.llm_judge.real_pairs import load_real_pairs, pair_id
from baselines.llm_judge_with_pos_look_back_pos_back_look.prompt import (
    SYSTEM_PROMPTS,
    build_user_prompt,
    candidate_to_text,
    seeker_to_text,
)
from baselines.llm_judge_with_pos_look_pos_back_look.google_backend import call_google_verdict
from baselines.metrics import neg_hardness_slice_metrics, pair_metrics

VARIANT = "seeker_background"


def build_requests(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    system = SYSTEM_PROMPTS[VARIANT]
    requests: list[tuple[str, str, str]] = []
    pos_keys: list[str] = []
    neg_keys: list[str] = []

    for label, records, keys in (("pos", positives, pos_keys), ("neg", negatives, neg_keys)):
        for record in records:
            user = build_user_prompt(
                record["userContactFile"],
                record["matchContactFile"],
                record.get("searchQuery", ""),
            )
            key = f"{pair_id(record, label)}|{prompt_hash(system, user)}"
            keys.append(key)
            requests.append((key, system, user))
    return requests, pos_keys, neg_keys


def run_eval(
    *,
    data_dir: Path,
    split: str,
    model: str,
    temperature: float,
    workers: int,
    max_attempts: int,
    max_failures: int,
    artifacts_dir: Path,
    split_path: Path | None,
    make_call_fn: Callable[[str, str], Any],
) -> dict[str, Any]:
    print(f"model:   {model}")
    print(f"variant: {VARIANT} (searchQuery given; seeker=positioning+lookingFor+background)")
    print(f"split:   {split}")

    positives, negatives = load_real_pairs(data_dir, split=split, split_path=split_path)
    print(f"loaded {len(positives)} real positives, {len(negatives)} real negatives")

    requests, pos_keys, neg_keys = build_requests(positives, negatives)
    prompt_chars = [len(u) for _, _, u in requests]
    print(f"prompt size (chars): mean {int(np.mean(prompt_chars))}, max {max(prompt_chars)}")

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
                "refusing to report metrics on a population this incomplete."
            )
        print("  → excluded from metrics (see failed_pairs in metrics.json)")

    pos_records = [r for k, r in zip(pos_keys, positives) if k in verdicts]
    neg_records = [r for k, r in zip(neg_keys, negatives) if k in verdicts]
    pos_verdicts = [verdicts[k] for k in pos_keys if k in verdicts]
    neg_verdicts = [verdicts[k] for k in neg_keys if k in verdicts]
    if not pos_verdicts or not neg_verdicts:
        raise RuntimeError("no usable verdicts on one side; cannot compute metrics")

    pos_scores = np.array([verdict_to_score(v) for v in pos_verdicts], dtype=np.float64)
    neg_scores = np.array([verdict_to_score(v) for v in neg_verdicts], dtype=np.float64)

    neg_seeker_texts = [seeker_to_text(r["userContactFile"]) for r in neg_records]
    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in neg_records]

    return {
        "model_name": model,
        "experiment": "llm_judge_with_pos_look_back_pos_back_look",
        "prompt_variant": VARIANT,
        "uses_search_query": True,
        "seeker_fields": ["positioning", "lookingFor", "background"],
        "candidate_fields": ["positioning", "background", "lookingFor"],
        "split": split,
        "temperature": temperature,
        "num_pairs_requested": len(requests),
        "num_pairs_scored": len(pos_verdicts) + len(neg_verdicts),
        "failed_pairs": {k.split("|")[0]: v for k, v in errors.items()},
        "decision": decision_metrics(pos_verdicts, neg_verdicts),
        "calibration": calibration_buckets(pos_verdicts, neg_verdicts),
        "pair": pair_metrics(pos_scores, neg_scores),
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
    print(f"\n=== LLM judge (seeker+background): {m['model_name']} ===")
    print(f"pairs scored: {m['num_pairs_scored']}/{m['num_pairs_requested']}")

    print("\n--- Decision (the model's own yes/no) ---")
    print(f"Accuracy:   {d['accuracy']:.4f}   (chance = 0.5)")
    print(f"Precision:  {d['precision']:.4f}")
    print(f"Recall:     {d['recall']:.4f}")
    print(f"F1:         {d['f1']:.4f}")

    print("\n--- Pair metrics ---")
    print(f"ROC-AUC:            {p['roc_auc']:.4f}")
    print(f"Average Precision:  {p['average_precision']:.4f}")
    print(f"Best-F1:            {p['best_f1']:.4f} @ {p['best_f1_threshold']:.4f}")

    hard = m["slices"]["neg_hardness"]
    print("\n--- Negative-hardness slices ---")
    for name in ("easy", "hard"):
        s = hard.get(name) or {}
        auc = s.get("pair_auc")
        print(f"  {name:>4}-neg: n={s.get('n_negatives', '?')}  pair_auc={'n/a' if auc is None else format(auc, '.4f')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split", choices=["all", "train", "holdout"], default="all")
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--max-failures", type=int, default=5)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(args.env_file)

    model = args.model
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not api_key:
        print(
            "error: no GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment. "
            "Pass --env-file /path/to/.env.",
            file=sys.stderr,
        )
        return 2

    def make_call_fn(system: str, user: str):
        def call() -> dict[str, Any]:
            return call_google_verdict(
                api_key=api_key,
                model=model,
                system=system,
                user=user,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

        return call

    artifacts_dir = args.artifacts_dir or Path(
        "artifacts/llm_judge_with_pos_look_back_pos_back_look"
    ) / f"google_{model_slug(model)}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        split=args.split,
        model=model,
        temperature=args.temperature,
        workers=args.workers,
        max_attempts=args.max_attempts,
        max_failures=args.max_failures,
        artifacts_dir=artifacts_dir,
        split_path=args.split_path,
        make_call_fn=make_call_fn,
    )
    metrics["backend"] = "google"
    print_report(metrics)

    out_path = artifacts_dir / f"metrics_{args.split}.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
