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
)
from judge_prompt_evolution.optimizer import run_one_iteration
from judge_prompt_evolution.sampling import ExampleBank
from judge_prompt_evolution.seed_prompt import SEED_JUDGE_PROMPT


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
    print(f"split       {cfg.split} (holdout never touched)")

    bank = ExampleBank(cfg)
    current_prompt = SEED_JUDGE_PROMPT
    history: list[dict[str, Any]] = []
    start_iteration = 1

    if resume:
        done = sorted(
            int(p.stem) for p in iterations_dir.glob("*.json") if p.stem.isdigit()
        )
        for n in done:
            record = json.loads((iterations_dir / f"{n:02d}.json").read_text(encoding="utf-8"))
            history.append(record)
            current_prompt = record["prompt_after"]
            # keep the sampler's draw sequence advancing past already-used
            # examples, so a resumed run doesn't repeat iteration 1..N's batches
            bank.draw_batch(cfg)
        if done:
            start_iteration = max(done) + 1
            print(f"resuming    {len(done)} iteration(s) already on disk, starting at {start_iteration}")

    if cfg.push_to_hub and not resume:
        push_meta_prompt(hub_owner=cfg.hub_owner, repo=cfg.hub_repo)
        push_seed_prompt(hub_owner=cfg.hub_owner, repo=cfg.hub_repo)

    if not resume:
        _write_json(
            run_dir / "iterations" / "00_seed.json",
            {
                "iteration": 0,
                "prompt_before": None,
                "prompt_after": SEED_JUDGE_PROMPT,
                "rationale": "seed: unmodified naive judge prompt (scored 0.6177 pair AUC on all-200 real pairs)",
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
                  f"CONTRACT WARNING: {record['contract_problems']}")
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

    summary = {
        "run_id": cfg.run_id,
        "n_iterations": cfg.n_iterations,
        "optimizer_model": cfg.optimizer_model,
        "seed_prompt": SEED_JUDGE_PROMPT,
        "final_prompt": current_prompt,
        "seed_prompt_auc_reference": {
            "note": "AUC of the *seed* prompt (unmodified naive judge), measured previously — "
                    "not re-measured by this run",
            "all_200_pair_auc": 0.6177,
            "holdout_69_pair_auc": 0.6358,
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-hub", action="store_true", help="skip LangSmith Hub pushes (local only)")
    p.add_argument("--resume", action="store_true",
                    help="continue an existing --run-id from its last saved iteration")
    args = p.parse_args(argv)

    kwargs: dict[str, Any] = {"n_iterations": args.n_iterations, "seed": args.seed}
    if args.run_id:
        kwargs["run_id"] = args.run_id
    if args.optimizer_model:
        kwargs["optimizer_model"] = args.optimizer_model
    if args.no_hub:
        kwargs["push_to_hub"] = False

    cfg = RunConfig(**kwargs)
    run(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
