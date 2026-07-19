# Training on Modal (instead of local MPS)

Why: the two-tower LoRA fine-tune (`docs/two-tower-fine-tune-plan.md`) needs a
backward pass, not just inference — Apple's MPS backend has much weaker
support for autograd ops than CUDA, and `voyage-4-nano` loads via custom
remote code (`Qwen3BidirectionalModel`), so there's a real chance some
backward op silently falls back to CPU and craters training speed on a
MacBook. Modal gives a CUDA GPU on demand, billed per-second, callable from a
plain Python decorator — no notebook UI, no idle GPU cost.

At this scale (340M-param model, ~680 training examples, LoRA rank 8–16), a
single T4 should be enough and the whole fine-tune should be **minutes, not
hours** — cheap enough to just try rather than estimate precisely.

## 1. Account + CLI setup

```bash
pip install modal
modal setup   # opens a browser, links this machine to your Modal account
```

New accounts get **$30/month in free compute credit** — likely covers this
entire experiment (and several retries) without paying anything.

## 2. Push secrets (only if needed)

`voyage-4-nano` is public/Apache-2.0, so no gated-repo token is required to
download it. You'd only need a secret if training also calls out to
OpenRouter/Voyage/W&B from inside the Modal container:

```bash
modal secret create dorby-synth OPENROUTER_API_KEY=sk-... WANDB_API_KEY=...
```

Reference it in the function decorator with
`secrets=[modal.Secret.from_name("dorby-synth")]`; values show up as
`os.environ[...]` inside the container.

## 3. Define the training function

Put this alongside `twotower/train.py` (e.g. `twotower/modal_train.py`) once
that module exists. Skeleton:

```python
import modal

app = modal.App("dorby-twotower-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",  # same pin as requirements.txt for voyage-4-nano
        "sentence-transformers",
        "peft",
        "accelerate",
    )
)

# Persists LoRA checkpoints across runs/containers.
checkpoints = modal.Volume.from_name("dorby-twotower-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",                       # bump to "A10" or "L4" only if T4 turns out to be a bottleneck
    timeout=60 * 60,                # 1 hour ceiling; actual run should be far shorter
    volumes={"/checkpoints": checkpoints},
)
def train(epochs: int = 5, lora_rank: int = 8):
    from twotower.train import run_training  # your actual training entrypoint

    run_training(
        output_dir="/checkpoints/run_001",
        epochs=epochs,
        lora_rank=lora_rank,
    )
    checkpoints.commit()  # make the container's writes visible outside it


@app.local_entrypoint()
def main(epochs: int = 5, lora_rank: int = 8):
    train.remote(epochs=epochs, lora_rank=lora_rank)
```

`@app.local_entrypoint()` auto-exposes function args as CLI flags — e.g.
`--epochs 3` below — so you can sweep hyperparameters without editing code.

## 4. Run it

```bash
modal run twotower/modal_train.py
modal run twotower/modal_train.py --epochs 3 --lora-rank 16
```

This uploads your local code, builds/caches the image (first run only),
provisions a GPU, runs `train()` remotely, and streams logs back to your
terminal — no notebook, no manual instance management.

## 5. Pull the trained adapter back down

```bash
modal volume ls dorby-twotower-checkpoints
modal volume get dorby-twotower-checkpoints run_001 ./artifacts/twotower/run_001
```

Files under 16 MB can also be grabbed from the Modal dashboard directly.

## GPU choice & cost

| GPU | ~$/hr | When to use |
|-----|-------|-------------|
| T4  | ~$0.59 | Default — plenty for a 340M-param LoRA fine-tune on 680 examples |
| L4 / A10 | ~$1.10+ | Only if T4 profiling shows it's the bottleneck (unlikely at this scale) |

Billing is per-second with no idle charges, so a run that finishes in 10
minutes costs roughly 1/6th the hourly rate — realistically under $0.20 for
the whole experiment, well inside the free credit.

## Notes / gotchas

- **First run is slower** — Modal builds and caches the Docker image; later
  runs reuse the cached image and skip that step.
- **Model weights**: either bake `voyageai/voyage-4-nano` into the image at
  build time (faster cold start, image rebuild needed on model change) or
  download it at runtime into a second `modal.Volume` acting as a model
  cache (avoids re-downloading on every run, no rebuild needed) — same
  pattern as the "model cache" volume in Modal's own fine-tuning examples.
- **Debugging**: iterate locally with `dry_run`-style stubs before spending
  GPU time, same discipline as `synth_pipeline --dry-run`.
- Keep the eval protocol identical to the baseline comparison
  (`baselines/metrics.py`) regardless of where training happens — the
  holdout-based decision gate in the two-tower plan only means something if
  the metric definitions match exactly.

## References

- [GPU acceleration — Modal Docs](https://modal.com/docs/guide/gpu)
- [Volumes — Modal Docs](https://modal.com/docs/guide/volumes)
- [Secrets — Modal Docs](https://modal.com/docs/guide/secrets)
- [Efficient LLM Finetuning with Unsloth — Modal Docs](https://modal.com/docs/examples/unsloth_finetune)
- [Modal pricing](https://modal.com/pricing)
