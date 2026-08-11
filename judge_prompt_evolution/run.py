"""Orchestrate the N-iteration judge-prompt-evolution loop.

Per iteration: sample a fresh contrastive batch of real-pair examples (never
touching the frozen holdout), hand the current prompt + batch to the
optimizer LLM, record what came back, and carry the revised prompt into the
next round. Every iteration is saved locally *and* pushed to LangSmith Hub as
its own commit (best-effort — see ``hub.py``).

Deliberately does **no** accuracy/AUC evaluation anywhere in this loop — that
is a separate, explicitly-confirmed step per the user's instruction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from judge_prompt_evolution.config import RunConfig
from judge_prompt_evolution.hub import (
    push_iteration_prompt,
    push_meta_prompt,
    push_seed_prompt,
    push_summarizer_prompt,
)
from judge_prompt_evolution.optimizer import run_one_iteration, run_summarization_step
from judge_prompt_evolution.sampling import ExampleBank
from judge_prompt_evolution.seed_prompt import SEED_JUDGE_PROMPT


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_seed_prompt(cfg: RunConfig) -> tuple[str, str]:
    """Return (prompt_text, description) for the configured seed source."""
    if cfg.seed_source == "naive":
        return (
            SEED_JUDGE_PROMPT,
            "seed: unmodified naive judge prompt (scored 0.6177 pair AUC on all-200 real pairs)",
        )
    if cfg.seed_source == "structured_cot":
        from baselines.llm_judge.prompt import SYSTEM_PROMPTS

        return (
            SYSTEM_PROMPTS["structured_cot"],
            "seed: unmodified structured_cot judge prompt (scored 0.6100 pair AUC on all-200 "
            "real pairs — the closest of any prior attempt to beating naive's 0.6177), "
            "imported read-only from baselines.llm_judge.prompt",
        )
    raise ValueError(f"unknown seed_source {cfg.seed_source!r}")


def _run_and_save_summary(
    *, cfg: RunConfig, current_prompt: str, after_iteration: int,
    iterations_dir: Path, history: list[dict[str, Any]],
) -> str:
    t0 = time.time()
    srecord = run_summarization_step(cfg=cfg, current_prompt=current_prompt, after_iteration=after_iteration)
    dt = time.time() - t0
    delta = len(srecord["prompt_after"]) - len(current_prompt)
    print(f"  summarize@{after_iteration:02d}  ({dt:.1f}s)  "
          f"{len(current_prompt)} -> {len(srecord['prompt_after'])} chars ({delta:+d})")
    _write_json(iterations_dir / f"{after_iteration:02d}s.json", srecord)
    if cfg.push_to_hub:
        push_iteration_prompt(
            text=srecord["prompt_after"], run_id=cfg.run_id,
            iteration=after_iteration, hub_owner=cfg.hub_owner, repo=cfg.hub_repo, kind="summarize",
        )
    history.append(srecord)
    return srecord["prompt_after"]


def run(cfg: RunConfig, *, resume: bool = False) -> dict[str, Any]:
    run_dir = cfg.run_dir
    iterations_dir = run_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id      {cfg.run_id}")
    print(f"iterations  {cfg.n_iterations}")
    print(f"examples/it {cfg.examples_per_iteration} "
          f"({cfg.n_positive_examples} pos / {cfg.n_hard_negative_examples} hard-neg / "
          f"{cfg.n_easy_negative_examples} easy-neg)")
    print(f"optimizer   {cfg.optimizer_model}")
    if cfg.split == "train":
        print(f"split       {cfg.split} (holdout never touched)")
    else:
        print(f"split       {cfg.split}  *** LEAKAGE WARNING: any AUC check after this run is "
              f"contaminated — the optimizer can see holdout pairs as examples, so no population "
              f"is left unseen. Exploratory only, not comparable to a train-sampled run. ***")
    print(f"summarize   every {cfg.summarize_every} iterations ({cfg.summarizer_variant})"
          if cfg.summarize_every else "summarize   off")

    seed_prompt, seed_description = resolve_seed_prompt(cfg)
    print(f"seed source {cfg.seed_source}")

    bank = ExampleBank(cfg)
    current_prompt = seed_prompt
    history: list[dict[str, Any]] = []
    start_iteration = 1

    if resume:
        # Chronological order: for a given index, the "optimize" record
        # (bare "{i:02d}.json") always precedes its "summarize" record
        # ("{i:02d}s.json", only present when summarize_every divides i).
        found: list[tuple[int, int, Path]] = []
        for p in iterations_dir.glob("*.json"):
            if p.stem.isdigit():
                found.append((int(p.stem), 0, p))
            elif p.stem.endswith("s") and p.stem[:-1].isdigit():
                found.append((int(p.stem[:-1]), 1, p))
        found.sort()

        last_optimize_idx = 0
        for idx, kind, p in found:
            record = json.loads(p.read_text(encoding="utf-8"))
            history.append(record)
            current_prompt = record["prompt_after"]
            if kind == 0:
                # only "optimize" steps consumed a sample batch
                bank.draw_batch(cfg)
                last_optimize_idx = idx

        if found:
            start_iteration = last_optimize_idx + 1
            print(f"resuming    {len(found)} record(s) on disk, "
                  f"last optimize iteration {last_optimize_idx}, starting at {start_iteration}")
            # a crash between an optimize step and its due summarize step
            # would otherwise silently skip that summarize call
            due_summary_missing = (
                cfg.summarize_every
                and last_optimize_idx % cfg.summarize_every == 0
                and not (iterations_dir / f"{last_optimize_idx:02d}s.json").exists()
            )
            if due_summary_missing:
                current_prompt = _run_and_save_summary(
                    cfg=cfg, current_prompt=current_prompt, after_iteration=last_optimize_idx,
                    iterations_dir=iterations_dir, history=history,
                )

    if cfg.push_to_hub and not resume:
        push_meta_prompt(hub_owner=cfg.hub_owner, repo=cfg.hub_repo)
        if cfg.summarize_every:
            push_summarizer_prompt(hub_owner=cfg.hub_owner, repo=cfg.hub_repo, variant=cfg.summarizer_variant)
        push_seed_prompt(
            hub_owner=cfg.hub_owner, repo=cfg.hub_repo,
            text=seed_prompt, description=seed_description, run_id=cfg.run_id,
        )

    if not resume:
        _write_json(
            run_dir / "iterations" / "00_seed.json",
            {
                "iteration": 0,
                "prompt_before": None,
                "prompt_after": seed_prompt,
                "rationale": seed_description,
                "contract_problems": [],
                "examples": [],
                "optimizer_model": None,
            },
        )

    for i in range(start_iteration, cfg.n_iterations + 1):
        t0 = time.time()
        examples = bank.draw_batch(cfg)
        record = run_one_iteration(
            cfg=cfg, current_prompt=current_prompt, examples=examples, iteration=i
        )
        dt = time.time() - t0

        if record["contract_problems"]:
            print(f"  iter {i:02d}/{cfg.n_iterations}  ({dt:.1f}s)  "
                  f"CONTRACT AUTO-REPAIRED: {record['contract_problems']}")
        else:
            print(f"  iter {i:02d}/{cfg.n_iterations}  ({dt:.1f}s)  ok — "
                  f"{len(record['prompt_after'])} chars")

        _write_json(iterations_dir / f"{i:02d}.json", record)

        if cfg.push_to_hub:
            push_iteration_prompt(
                text=record["prompt_after"],
                run_id=cfg.run_id,
                iteration=i,
                hub_owner=cfg.hub_owner,
                repo=cfg.hub_repo,
            )

        history.append(record)
        current_prompt = record["prompt_after"]

        if cfg.summarize_every and i % cfg.summarize_every == 0:
            current_prompt = _run_and_save_summary(
                cfg=cfg, current_prompt=current_prompt, after_iteration=i,
                iterations_dir=iterations_dir, history=history,
            )

    summary = {
        "run_id": cfg.run_id,
        "n_iterations": cfg.n_iterations,
        "optimizer_model": cfg.optimizer_model,
        "seed_source": cfg.seed_source,
        "example_split": cfg.split,
        "leakage_warning": None if cfg.split == "train" else (
            "Examples were sampled from split="
            + repr(cfg.split)
            + ", which includes the 69-pair holdout. Any AUC check on any population is "
            "contaminated by this run's own examples — not a generalization test, exploratory only."
        ),
        "summarize_every": cfg.summarize_every,
        "seed_prompt": seed_prompt,
        "final_prompt": current_prompt,
        "seed_prompt_auc_reference": {
            "note": "AUC of the *seed* prompt, measured previously — not re-measured by this run",
            "all_200_pair_auc": 0.6177 if cfg.seed_source == "naive" else 0.6100,
            "holdout_69_pair_auc": 0.6358 if cfg.seed_source == "naive" else 0.6336,
            "source": "docs/llm-judge-experiment.md",
        },
        "contract_warnings": [
            {"iteration": r["iteration"], "problems": r["contract_problems"]}
            for r in history
            if r["contract_problems"]
        ],
        "hub_repo": f"{cfg.hub_owner}/{cfg.hub_repo}" if cfg.hub_owner else cfg.hub_repo,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "full_history.json", history)

    print(f"\n--- done ---\nrun dir: {run_dir}")
    n_warn = len(summary["contract_warnings"])
    if n_warn:
        print(f"NOTE: {n_warn}/{cfg.n_iterations} iterations flagged contract warnings — inspect before use")
    print("No AUC was computed. Confirm before running any accuracy check on the result.")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-iterations", type=int, default=20)
    p.add_argument("--optimizer-model", default=None)
    p.add_argument("--optimizer-backend", choices=["openrouter", "gemini"], default=None,
                    help="'openrouter' (default) or 'gemini' (direct GEMINI_API_KEY, bypasses OpenRouter)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-source", choices=["naive", "structured_cot"], default=None)
    p.add_argument("--split", choices=["train", "all", "holdout"], default=None,
                    help="which real pairs the optimizer samples examples from. 'train' (default) "
                    "keeps the 69-pair holdout unseen, so a later all-200 AUC check has a clean "
                    "generalization component. 'all' shows every real pair as a possible example, "
                    "including holdout — any subsequent AUC check on any population is then "
                    "contaminated by design; use only for exploratory runs, never for a fair "
                    "comparison to a train-sampled run.")
    p.add_argument("--summarize-every", type=int, default=None,
                    help="insert a distillation-only step every N optimize iterations (0/unset = off)")
    p.add_argument("--summarizer-variant", choices=["aggressive", "gentle"], default=None,
                    help="'aggressive' (default) pushes toward shortness; 'gentle' explicitly says "
                    "length is not the objective, only merge genuinely repetitive wording")
    p.add_argument("--n-positive-examples", type=int, default=None)
    p.add_argument("--n-hard-negative-examples", type=int, default=None)
    p.add_argument("--n-easy-negative-examples", type=int, default=None)
    p.add_argument("--no-hub", action="store_true", help="skip LangSmith Hub pushes (local only)")
    p.add_argument("--resume", action="store_true",
                    help="continue an existing --run-id from its last saved iteration")
    args = p.parse_args(argv)

    kwargs: dict[str, Any] = {"n_iterations": args.n_iterations, "seed": args.seed}
    if args.run_id:
        kwargs["run_id"] = args.run_id
    if args.optimizer_model:
        kwargs["optimizer_model"] = args.optimizer_model
    if args.optimizer_backend:
        kwargs["optimizer_backend"] = args.optimizer_backend
    if args.seed_source:
        kwargs["seed_source"] = args.seed_source
    if args.split:
        kwargs["split"] = args.split
    if args.summarize_every is not None:
        kwargs["summarize_every"] = args.summarize_every
    if args.summarizer_variant:
        kwargs["summarizer_variant"] = args.summarizer_variant
    if args.no_hub:
        kwargs["push_to_hub"] = False
    if args.n_positive_examples is not None:
        kwargs["n_positive_examples"] = args.n_positive_examples
    if args.n_hard_negative_examples is not None:
        kwargs["n_hard_negative_examples"] = args.n_hard_negative_examples
    if args.n_easy_negative_examples is not None:
        kwargs["n_easy_negative_examples"] = args.n_easy_negative_examples

    cfg = RunConfig(**kwargs)
    run(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
