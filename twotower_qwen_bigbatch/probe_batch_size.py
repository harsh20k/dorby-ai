"""Measure Qwen3-8B's real micro-batch ceiling before committing GPU hours.

The prior Qwen fine-tune ran at ``train_batch_size=1``. That number was never
probed: the OOM it cites was fp32 on a 40GB A100, and the same preset already
fixes that with bf16 weights (16GB) plus gradient checkpointing. This probe asks
what actually fits on an 80GB card *with those two settings on*, using one real
forward+backward at each size — same model, same LoRA config, same loss, real
text from the rrf_003 k=1 rows rather than short placeholders.

Both settings are deliberately reported per size, because a probe that silently
ran without gradient checkpointing would give a uselessly low ceiling.

    modal run twotower_qwen_bigbatch/probe_batch_size.py
    modal run twotower_qwen_bigbatch/probe_batch_size.py --batch-sizes 1,2,4,6,8,12
    modal run twotower_qwen_bigbatch/probe_batch_size.py --gpu H200
"""

from __future__ import annotations

import modal

from twotower_qwen_bigbatch.modal_train import GPU, hf_cache, image

app = modal.App("dorby-twotower-qwen-bigbatch-probe")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={"/cache/huggingface": hf_cache},
)
def probe_remote(
    batch_sizes: list[int], rows_filename: str = "rrf_003_multineg_k1.json"
) -> list[dict]:
    import json
    from pathlib import Path

    import torch

    from twotower.train import add_lora_adapter
    from twotower_qwen_bigbatch.config import build_config
    from twotower_qwen_bigbatch.model import build_model_with_dtype

    try:
        from sentence_transformers.sentence_transformer.losses import (
            MultipleNegativesRankingLoss,
        )
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    rows_path = Path("/root/rows") / rows_filename
    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    anchors = [r["anchor"] for r in rows]
    positives = [r["positive"] for r in rows]
    neg1 = [r["negatives"][0] for r in rows]

    cfg = build_config(
        "qwen3-8b",
        run_id="probe",
        rows_path=rows_path,
        negatives_per_anchor=1,
        output_dir=Path("/tmp/probe_unused"),
    )
    device = "cuda"
    model = build_model_with_dtype(
        cfg, device, torch_dtype=torch.bfloat16, gradient_checkpointing=True
    )
    add_lora_adapter(model, cfg)
    model.train()
    loss_fn = MultipleNegativesRankingLoss(model=model)

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    weights_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"GPU total {total_gb:.2f} GB | weights resident {weights_gb:.2f} GB")

    def make_batch(bs: int) -> dict[str, list[str]]:
        def take(src: list[str]) -> list[str]:
            return [src[i % len(src)] for i in range(bs)]

        return {
            "anchor": take(anchors),
            "positive": take(positives),
            "negative_1": take(neg1),
        }

    results: list[dict] = []
    for bs in sorted(batch_sizes):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = make_batch(bs)
            preprocess = getattr(model, "preprocess", None) or model.tokenize
            features = [
                {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in dict(preprocess(batch[col])).items()
                }
                for col in ("anchor", "positive", "negative_1")
            ]
            model.zero_grad(set_to_none=True)
            loss = loss_fn(features, None)
            loss.backward()
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            results.append(
                {
                    "batch_size": bs,
                    "status": "ok",
                    "peak_mem_gb": round(peak_gb, 2),
                    "total_mem_gb": round(total_gb, 2),
                }
            )
            print(f"  bs={bs:3d}  OK   peak {peak_gb:.2f} / {total_gb:.2f} GB")
        except torch.cuda.OutOfMemoryError as exc:  # noqa: BLE001
            results.append({"batch_size": bs, "status": "oom", "error": str(exc)[:200]})
            print(f"  bs={bs:3d}  OOM")
            torch.cuda.empty_cache()
            break
        except Exception as exc:  # noqa: BLE001
            results.append({"batch_size": bs, "status": "error", "error": str(exc)[:300]})
            print(f"  bs={bs:3d}  ERROR {str(exc)[:160]}")
            break
    return results


@app.local_entrypoint()
def main(batch_sizes: str = "1,2,4,6,8,12") -> None:
    sizes = [int(x) for x in batch_sizes.split(",")]
    results = probe_remote.remote(sizes)
    print("\n=== Qwen3-8B batch-size probe (bf16 + gradient checkpointing) ===")
    ok_sizes = []
    for r in results:
        print(r)
        if r["status"] == "ok":
            ok_sizes.append(r["batch_size"])
    if not ok_sizes:
        print("\nno batch size succeeded — even the smallest tested size OOM'd")
        return
    ceiling = max(ok_sizes)
    safe = max(1, int(ceiling * 0.75))
    candidates = [s for s in ok_sizes if s <= safe] or [ok_sizes[0]]
    print(f"\nceiling (largest OK tested): {ceiling}")
    print(f"recommended train_batch_size (~75% of ceiling): {max(candidates)}")
    print(
        "\nNote: effective batch must stay 12, so pick a micro-batch that divides "
        "12 (1, 2, 3, 4, 6, 12) to keep optimizer-step count matched with the "
        "control arm."
    )
