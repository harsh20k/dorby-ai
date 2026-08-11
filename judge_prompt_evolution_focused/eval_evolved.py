"""Eval the automatic-prompt-optimization evolved judge prompt on all 200 real pairs.

Isolated from `baselines/llm_judge/` — this file only *imports* that package
(read-only) and writes its own artifacts under
`artifacts/judge_prompt_evolution_focused/<run-id>/eval/`, never into
`artifacts/llm_judge/`.

    python -m judge_prompt_evolution.eval_evolved
    python -m judge_prompt_evolution.eval_evolved --run-id evo_001

Reference to beat (focused seed prompt, gemini-3.1-flash-lite via the direct
Google API, all-200 split, from `docs/llm-judge-focused-prompt-experiment.md`):
pair ROC-AUC 0.6451, decision accuracy 0.5950, F1 0.6197.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from baselines.bert_frozen.text import profile_to_text
from baselines.llm_judge.judge import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    VerdictCache,
    judge_all,
    load_env_file,
    make_openrouter_call_fn,
    prompt_hash,
    verdict_to_score,
)
from baselines.llm_judge.metrics import calibration_buckets, decision_metrics, intent_pair_auc
from judge_prompt_evolution_focused.focused_prompt import build_user_prompt
from baselines.llm_judge.real_pairs import load_real_pairs, pair_id
from baselines.metrics import neg_hardness_slice_metrics, pair_metrics

DEFAULT_MAX_TOKENS = 600
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _make_gemini_call_fn(*, api_key: str, model: str, temperature: float, max_tokens: int,
                          base_url: str = DEFAULT_GEMINI_BASE_URL):
    """Direct Google Gemini API call (not via OpenRouter) — raw REST via
    urllib, no new SDK dependency, matching this repo's existing lightweight
    HTTP-call pattern (baselines/llm_judge/judge.py's OpenRouter path).
    Isolated here since no other module in this repo talks to Gemini
    directly; everything else routes it through OpenRouter."""
    from synth_pipeline.llm import parse_json_object

    def call(system: str, user: str) -> dict[str, Any]:
        url = f"{base_url}/models/{model}:generateContent?key={api_key}"
        body = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError(f"no candidates in Gemini response: {payload!r}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        return parse_json_object(text)

    return call


def _make_bedrock_reasoning_safe_call_fn(*, client, model_id: str, temperature: float, max_tokens: int):
    """Like ``baselines.llm_judge.judge.make_bedrock_call_fn``, but doesn't
    assume ``content[0]`` is the text block. Reasoning models (MiniMax M2.5
    confirmed) return ``content = [{"reasoningContent": ...}, {"text": ...}]``
    — ``bedrock_backend.call_bedrock_verdict`` indexes ``content[0]["text"]``
    directly and KeyErrors on this shape. Fixed locally (not in the shared
    ``baselines/llm_judge/bedrock_backend.py``) per the isolation rule: this
    duplicates just the response-parsing step, reusing everything else
    (client construction, the Converse call itself, structured-output
    fallback) from the original module."""
    from baselines.llm_judge.bedrock_backend import _structured_output_unsupported, VERDICT_SCHEMA
    from synth_pipeline.llm import parse_json_object

    def _first_text_block(content: list[dict]) -> str:
        for block in content:
            if "text" in block:
                return block["text"]
        raise ValueError(f"no text block in Bedrock response content: {content!r}")

    def call(system: str, user: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": [{"text": user}]}]
        inference_config = {"maxTokens": max_tokens, "temperature": temperature}
        try:
            resp = client.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": system}],
                inferenceConfig=inference_config,
                outputConfig={
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                "schema": json.dumps(VERDICT_SCHEMA),
                                "name": "verdict",
                                "description": "Match verdict with confidence and reasoning.",
                            }
                        },
                    }
                },
            )
            text = _first_text_block(resp["output"]["message"]["content"])
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            if not _structured_output_unsupported(exc):
                raise

        plain_system = (
            system
            + '\n\nRespond with only a single JSON object matching this shape, '
            'no other text: {"reasoning": string, "match": "yes"|"no", "confidence": integer 0-100}'
        )
        resp = client.converse(
            modelId=model_id,
            messages=messages,
            system=[{"text": plain_system}],
            inferenceConfig=inference_config,
        )
        text = _first_text_block(resp["output"]["message"]["content"])
        return parse_json_object(text)

    return call

# The focused seed prompt this evolved prompt started from, same model
# (gemini-3.1-flash-lite via the direct Google API), same all-200-pair split —
# see docs/llm-judge-focused-prompt-experiment.md.
SEED_REFERENCE = {
    "pair_roc_auc": 0.6451,
    "decision_accuracy": 0.5950,
    "decision_f1": 0.6197,
}


def load_evolved_prompt(summary_path: Path) -> tuple[str, dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    prompt = summary["final_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"final_prompt missing/empty in {summary_path}")
    return prompt, summary


def load_prompt_from_iteration(iteration_path: Path) -> tuple[str, dict[str, Any]]:
    """Load ``prompt_after`` from a specific iteration file (e.g. a
    pre-summarize round) instead of a run's final summary.json."""
    record = json.loads(iteration_path.read_text())
    prompt = record["prompt_after"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"prompt_after missing/empty in {iteration_path}")
    fake_summary = {
        "optimizer_model": record.get("optimizer_model"),
        "leakage_warning": None,
        "source_iteration_file": str(iteration_path),
    }
    return prompt, fake_summary


def build_requests(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    system: str,
) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    """Return (requests, pos_keys, neg_keys) where a request is (key, system, user)."""
    requests: list[tuple[str, str, str]] = []
    pos_keys: list[str] = []
    neg_keys: list[str] = []

    for label, records, keys in (("pos", positives, pos_keys), ("neg", negatives, neg_keys)):
        for record in records:
            # Focused packing: trimmed fields + searchQuery — the same text
            # the optimizer saw as examples, and the same text the published
            # 0.6451 focused-prompt number was measured on. Differs from
            # judge_prompt_evolution/eval_evolved.py, which packs complete
            # profiles and withholds the query.
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
    system_prompt: str,
    model: str,
    temperature: float,
    workers: int,
    max_attempts: int,
    max_failures: int,
    artifacts_dir: Path,
    make_call_fn,
) -> dict[str, Any]:
    print(f"model:  {model}")
    print("split:  all (200 real pairs)")
    print(f"prompt: evolved, {len(system_prompt)} chars "
          "(searchQuery INCLUDED + trimmed fields, as in the focused seed experiment)")

    positives, negatives = load_real_pairs(data_dir, split="all")
    print(f"loaded {len(positives)} real positives, {len(negatives)} real negatives")

    requests, pos_keys, neg_keys = build_requests(positives, negatives, system=system_prompt)
    prompt_chars = [len(u) for _, _, u in requests]
    print(f"user-prompt size (chars): mean {int(np.mean(prompt_chars))}, max {max(prompt_chars)}")

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
        print("  -> excluded from metrics (see failed_pairs in metrics)")

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

    neg_seeker_texts = [profile_to_text(r["userContactFile"]) for r in neg_records]
    neg_cand_texts = [profile_to_text(r["matchContactFile"]) for r in neg_records]

    return {
        "model_name": model,
        "experiment": "judge_prompt_evolution_focused/eval_evolved",
        "prompt_source": "evolved (automatic prompt optimization, final round)",
        "uses_search_query": True,
        "split": "all",
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
    print(f"\n=== Evolved judge prompt: {m['model_name']} ===")
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

    print("\n--- Pair metrics (confidence-signed score) ---")
    print(f"ROC-AUC:            {p['roc_auc']:.4f}")
    print(f"Average Precision:  {p['average_precision']:.4f}")
    print(f"Best-F1:            {p['best_f1']:.4f} @ {p['best_f1_threshold']:.4f}")

    hard = m["slices"]["neg_hardness"]
    print("\n--- Negative-hardness slices ---")
    for name in ("easy", "hard"):
        s = hard.get(name) or {}
        auc = s.get("pair_auc")
        print(
            f"  {name:>4}-neg: n={s.get('n_negatives', '?')}  "
            f"pair_auc={'n/a' if auc is None else format(auc, '.4f')}"
        )

    print("\n=== Comparison against seed naive prompt (all-200 split) ===")
    ref = SEED_REFERENCE
    auc_delta = p["roc_auc"] - ref["pair_roc_auc"]
    acc_delta = d["accuracy"] - ref["decision_accuracy"]
    f1_delta = d["f1"] - ref["decision_f1"]
    print(f"  pair ROC-AUC:  evolved {p['roc_auc']:.4f}  vs seed {ref['pair_roc_auc']:.4f}  (delta {auc_delta:+.4f})")
    print(f"  accuracy:      evolved {d['accuracy']:.4f}  vs seed {ref['decision_accuracy']:.4f}  (delta {acc_delta:+.4f})")
    print(f"  F1:            evolved {d['f1']:.4f}  vs seed {ref['decision_f1']:.4f}  (delta {f1_delta:+.4f})")
    verdict = "BEAT" if auc_delta > 0 and acc_delta >= 0 else ("LOST TO" if auc_delta < 0 else "MATCHED")
    print(f"  => evolved prompt {verdict} the seed on pair AUC")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval the evolved (auto-optimized) judge prompt on all 200 real pairs"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--run-id", default="evo_focused_001", help="evolution run id under artifacts/judge_prompt_evolution_focused/")
    p.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Explicit path to summary.json; default artifacts/judge_prompt_evolution_focused/<run-id>/summary.json",
    )
    p.add_argument(
        "--iteration-path",
        type=Path,
        default=None,
        help="Score a specific iteration file's prompt_after instead of the run's final_prompt "
        "(e.g. a pre-summarize round). Overrides --summary-path.",
    )
    p.add_argument(
        "--backend",
        choices=["openrouter", "bedrock", "gemini"],
        default="openrouter",
        help="'openrouter' uses OPENROUTER_API_KEY; 'bedrock' uses --aws-profile/--aws-region "
        "credentials via boto3's Converse API; 'gemini' calls the Google Gemini API directly "
        "using GEMINI_API_KEY (bypasses OpenRouter entirely).",
    )
    p.add_argument("--model", default=None,
                    help=f"OpenRouter model id (default {DEFAULT_MODEL}), Bedrock model id "
                    "(e.g. minimax.minimax-m2.5), or Gemini model id (e.g. gemini-3.1-flash-lite), "
                    "depending on --backend.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter only")
    p.add_argument("--aws-profile", default="tf_provisioner", help="Bedrock only")
    p.add_argument("--aws-region", default="us-east-1", help="Bedrock only")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=None,
                    help=f"Output token cap. Default {DEFAULT_MAX_TOKENS} for openrouter; "
                    "3000 for bedrock (MiniMax M2.5 and similar reasoning models can exhaust a "
                    "small budget before emitting the final JSON).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--max-failures", type=int, default=5)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(args.env_file)

    if args.iteration_path:
        system_prompt, summary = load_prompt_from_iteration(args.iteration_path)
        prompt_source_desc = str(args.iteration_path)
    else:
        summary_path = args.summary_path or (
            Path("artifacts/judge_prompt_evolution_focused") / args.run_id / "summary.json"
        )
        system_prompt, summary = load_evolved_prompt(summary_path)
        prompt_source_desc = str(summary_path)
    print(f"loaded evolved prompt from {prompt_source_desc} ({len(system_prompt)} chars)")
    print(f"optimizer model(s): {summary.get('optimizer_model')}")
    if summary.get("leakage_warning"):
        print(f"\n*** {summary['leakage_warning']} ***\n")

    if args.backend == "openrouter":
        model = args.model or DEFAULT_MODEL
        max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            print(
                "error: no OPENROUTER_API_KEY (or OPENAI_API_KEY) in the environment. "
                "Pass --env-file /path/to/.env.",
                file=sys.stderr,
            )
            return 2

        def make_call_fn(system: str, user: str):
            return make_openrouter_call_fn(
                system=system,
                user=user,
                model=model,
                temperature=args.temperature,
                api_key=api_key,
                base_url=args.base_url,
                max_tokens=max_tokens,
            )
    elif args.backend == "bedrock":
        if not args.model:
            print("error: --model is required with --backend bedrock", file=sys.stderr)
            return 2
        model = args.model
        max_tokens = args.max_tokens or 3000
        from baselines.llm_judge.bedrock_backend import make_client

        try:
            client = make_client(profile=args.aws_profile, region=args.aws_region)
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not create a Bedrock client: {exc}", file=sys.stderr)
            return 2

        bedrock_call = _make_bedrock_reasoning_safe_call_fn(
            client=client, model_id=model, temperature=args.temperature, max_tokens=max_tokens,
        )

        def make_call_fn(system: str, user: str):
            return lambda: bedrock_call(system, user)
    else:  # gemini
        model = args.model or "gemini-3.1-flash-lite"
        max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS
        api_key = os.getenv("GEMINI_API_KEY") or ""
        if not api_key:
            print("error: no GEMINI_API_KEY in the environment. Pass --env-file /path/to/.env.",
                  file=sys.stderr)
            return 2

        gemini_call = _make_gemini_call_fn(
            api_key=api_key, model=model, temperature=args.temperature, max_tokens=max_tokens,
        )

        def make_call_fn(system: str, user: str):
            return lambda: gemini_call(system, user)

    artifacts_dir = args.artifacts_dir or (
        Path("artifacts/judge_prompt_evolution_focused") / args.run_id / "eval"
        / (f"{args.backend}_{model.replace('/', '_').replace('.', '_')}"
           + ("_iter" + args.iteration_path.stem if args.iteration_path else ""))
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        system_prompt=system_prompt,
        model=model,
        temperature=args.temperature,
        workers=args.workers,
        max_attempts=args.max_attempts,
        max_failures=args.max_failures,
        artifacts_dir=artifacts_dir,
        make_call_fn=make_call_fn,
    )
    metrics["backend"] = args.backend
    metrics["seed_reference"] = SEED_REFERENCE
    metrics["evolution_run_id"] = args.run_id
    metrics["prompt_source"] = prompt_source_desc
    print_report(metrics)

    out_path = artifacts_dir / "metrics_all.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
