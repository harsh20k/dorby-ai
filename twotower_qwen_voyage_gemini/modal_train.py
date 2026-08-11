"""Modal GPU entrypoint: twotower_qwen_bigbatch's winning micro-batch-6
recipe, retrained on the pairing_voyage_gemini rows instead of rrf_003.

Own app and own checkpoint volume — never touches twotower_qwen_bigbatch's or
twotower_voyage_gemini_ctrl's volumes. Shares only the HF model-download cache
(read/write-safe: keyed by model name), which already holds
Qwen3-Embedding-8B from the prior Qwen runs.

IMPORTANT — run this under the second Modal account (see repo .env's "# Modal
2" block) if a sibling experiment is launching GPU training concurrently under
the default account, to avoid both jobs contending for the same account's GPU
quota:

  export MODAL_TOKEN_ID=<Modal 2 token id> MODAL_TOKEN_SECRET=<Modal 2 secret> && \\
  modal run --detach twotower_qwen_voyage_gemini/modal_train.py \\
      --run-id qwen_voyage_gemini_001

  modal app logs <app-id>   # fetch mode, NOT -f; local_entrypoint() prints
                             # (final summary, holdout metrics, pull command)
                             # do NOT show up here — redirect the launching
                             # shell's stdout/stderr to a log file instead and
                             # tail that.
  modal volume get dorby-twotower-qwen-voyage-gemini-checkpoints <run_id> \\
      ./artifacts/twotower_qwen_voyage_gemini/<run_id>
  # NOTE: pulling a whole run directory in one call errors "Is a directory" on
  # this Modal CLI version — pull run_meta.json/run_result.json/
  # metrics_holdout.json/loss_history.json and adapter/ subfiles individually.

Keep the launching shell alive (`wait`), or `--detach` will still be cancelled
when the client process is terminated.

Preemption resilience (added after two B200 attempts were killed mid-training
by what looked like container preemption — see
docs/twotower-qwen-voyage-gemini-experiment.md): `train_remote` sets
`retries=modal.Retries(...)` so Modal auto-restarts "on the same input"
without needing a human to notice a dead container and manually relaunch, and
`train.py`'s `resolve_resume_checkpoint()` makes each restart resume from the
last epoch-boundary checkpoint (committed to the volume immediately after each
save, not just once at the end — see `_CommitOnSaveCallback`) instead of
crashing on the old unconditional "non-empty output_dir is an error" guard.
A restart before the *first* checkpoint (mid-epoch-1, before any
`save_strategy="epoch"` save has landed) still has nothing to resume from and
will still raise `FileExistsError` — that gap is inherent to epoch-granularity
checkpointing, not a bug in the resume logic itself.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-qwen-voyage-gemini"
CHECKPOINT_VOLUME = "dorby-twotower-qwen-voyage-gemini-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache

app = modal.App(APP_NAME)

# Same GPU twotower_qwen_bigbatch measured its micro-6 winner on: 8B params in
# bf16 (~16GB resident) plus activations for a real micro-batch comfortably
# fits an 80GB H100 (probe there found micro-batch 8 only at 27.46/79.18 GB).
GPU = "B200"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "peft>=0.14,<1",
        "accelerate>=0.30",
        "datasets>=2.19",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "einops",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source(
        "twotower_qwen_voyage_gemini", "twotower", "baselines", "synth_pipeline"
    )
    .add_local_dir("data", remote_path="/root/data")
    # Reuses twotower_voyage_gemini_ctrl's row file read-only — never copied or
    # regenerated here. Also carries that package's own run output, harmless
    # (~19MB) since the mount is read-only and unused by training.
    .add_local_dir("artifacts/twotower_voyage_gemini_ctrl", remote_path="/root/rows")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=8 * 60 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
    # B200 preemption resilience (https://modal.com/docs/guide/preemption):
    # Modal auto-restarts a preempted function "on the same input", so with
    # train.py's resolve_resume_checkpoint() fix (see that module for the full
    # story — two attempts were killed by preemption before this existed) each
    # restart resumes from the last epoch-boundary checkpoint instead of
    # crashing against its own partial state. single_use_containers=True is
    # the pattern from Modal's own long-training example
    # (https://modal.com/docs/examples/long-training): a fresh container per
    # attempt avoids any stale CUDA/driver state from a half-killed process
    # bleeding into the retry. Both confirmed available in modal==1.5.2
    # (this repo's pinned client) via `inspect.signature(modal.App.function)`
    # before use, rather than assumed from the docs alone.
    # initial_delay=10s + default exponential backoff (coefficient 2.0, capped
    # at 60s) rather than an immediate 0-delay retry, so 8 retries don't
    # hammer straight back into whatever caused the preemption.
    retries=modal.Retries(max_retries=8, initial_delay=10.0),
    single_use_containers=True,
)
def train_remote(
    run_id: str,
    rows_filename: str = "voyage_gemini_smoke002_multineg_k1.json",
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 6,
    eval_batch_size: int = 6,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    run_holdout: bool = True,
) -> dict:
    from pathlib import Path

    from twotower_qwen_voyage_gemini.config import build_config
    from twotower_qwen_voyage_gemini.train import run_training

    rows_path = Path("/root/rows") / rows_filename
    cfg = build_config(
        "qwen3-8b",
        run_id=run_id,
        rows_path=rows_path,
        negatives_per_anchor=negatives_per_anchor,
        epochs=epochs,
        seed=seed,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        output_dir=Path("/checkpoints") / run_id,
    )
    result = run_training(
        cfg,
        rows_path,
        negatives_per_anchor=negatives_per_anchor,
        dry_run=dry_run,
        run_holdout=run_holdout,
        real_holdout_data_dir=Path("/root/data"),
        real_holdout_split_path=Path("/root/data/synthetic/seed_split.json"),
        resume_from_checkpoint=resume_from_checkpoint,
        # Durably persist each epoch-boundary checkpoint the moment it's
        # written, not just once at the very end — see train.py's
        # _CommitOnSaveCallback docstring. Without this, a preemption between
        # a mid-training checkpoint save and the (previously single, final)
        # commit() below would lose that checkpoint despite it having been
        # "written", defeating the whole resume fix.
        checkpoint_commit_hook=checkpoints.commit,
    )
    checkpoints.commit()
    hf_cache.commit()
    return {
        "run_id": run_id,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "adapter_dir": result.get("adapter_dir"),
        "best_checkpoint": result.get("best_checkpoint"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "qwen_voyage_gemini_001",
    rows_filename: str = "voyage_gemini_smoke002_multineg_k1.json",
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 6,
    eval_batch_size: int = 6,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str = "",
    run_holdout: bool = True,
) -> None:
    """CLI: see module docstring for the Modal-account note and pull command."""
    eff = train_batch_size * gradient_accumulation_steps
    if eff != 12:
        print(
            f"WARNING: effective batch {train_batch_size}x{gradient_accumulation_steps}={eff} "
            "!= 12 — not comparable to twotower_qwen_bigbatch's qwen_micro6_r1."
        )
    result = train_remote.remote(
        run_id=run_id,
        rows_filename=rows_filename,
        negatives_per_anchor=negatives_per_anchor,
        epochs=epochs,
        seed=seed,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint or None,
        run_holdout=run_holdout,
    )
    print("=== Modal qwen-voyage-gemini train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} "
        f"./artifacts/twotower_qwen_voyage_gemini/{run_id}"
    )
