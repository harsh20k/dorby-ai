"""Empirical batch-size probe on the target GPU before committing to a real
training run.

rrf_triplet_voyage_nano_001 ran at train_batch_size=2 on an 80GB A100 with no
memory figures ever logged, so there is no existing ceiling to read off — this
runs one real forward+backward step (same model, same LoRA config, same loss,
realistic-length text pulled from the actual rrf_003 multi-negative rows, not
short placeholder strings) at increasing batch sizes and reports the largest
that doesn't OOM. The training run then uses ~75-80% of that ceiling as a
safety margin (headroom for the eval pass and any batch-to-batch variance in
sequence length).

Usage:
  modal run twotower_rrf_triplet_bigbatch/probe_batch_size.py
  modal run twotower_rrf_triplet_bigbatch/probe_batch_size.py --batch-sizes 8,16,32,64,96,128
"""

from __future__ import annotations

import modal

from twotower_rrf_triplet_bigbatch.modal_train import GPU, hf_cache, image

app = modal.App("dorby-twotower-rrf-triplet-bigbatch-probe")


@app.function(
    image=image,
    gpu=GPU,
    timeout=30 * 60,
    volumes={"/cache/huggingface": hf_cache},
)
def probe_remote(batch_sizes: list[int], rows_filename: str = "rrf_003_multineg_k2.json") -> list[dict]:
    import json
    from pathlib import Path

    import torch

    from twotower.train import add_lora_adapter, build_model
    from twotower_rrf_triplet_bigbatch.config import build_config

    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    rows_path = Path("/root/rows") / rows_filename
    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    anchors = [r["anchor"] for r in rows]
    positives = [r["positive"] for r in rows]
    neg1 = [r["negatives"][0] for r in rows]
    neg2 = [r["negatives"][1] for r in rows]

    cfg = build_config(
        "voyage-4-nano",
        run_id="probe",
        rows_path=rows_path,
        negatives_per_anchor=2,
        output_dir=Path("/tmp/probe_unused"),
    )
    device = "cuda"
    model = build_model(cfg, device)
    add_lora_adapter(model, cfg)
    model.train()
    loss_fn = MultipleNegativesRankingLoss(model=model)

    def make_batch(bs: int) -> dict[str, list[str]]:
        def take(src: list[str]) -> list[str]:
            return [src[i % len(src)] for i in range(bs)]

        return {
            "anchor": take(anchors),
            "positive": take(positives),
            "negative_1": take(neg1),
            "negative_2": take(neg2),
        }

    results = []
    for bs in sorted(batch_sizes):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = make_batch(bs)
            preprocess = getattr(model, "preprocess", None) or model.tokenize
            features = [
                {k: v.to(device) if hasattr(v, "to") else v for k, v in dict(preprocess(batch[col])).items()}
                for col in ("anchor", "positive", "negative_1", "negative_2")
            ]
            model.zero_grad(set_to_none=True)
            loss = loss_fn(features, None)
            loss.backward()
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            results.append({"batch_size": bs, "status": "ok", "peak_mem_gb": round(peak_gb, 2)})
        except torch.cuda.OutOfMemoryError as exc:  # noqa: BLE001
            results.append({"batch_size": bs, "status": "oom", "error": str(exc)[:200]})
            torch.cuda.empty_cache()
            break
        except Exception as exc:  # noqa: BLE001
            results.append({"batch_size": bs, "status": "error", "error": str(exc)[:300]})
            break
    return results


@app.local_entrypoint()
def main(batch_sizes: str = "8,16,32,64,96,128") -> None:
    sizes = [int(x) for x in batch_sizes.split(",")]
    results = probe_remote.remote(sizes)
    print("=== Batch-size probe results ===")
    ok_sizes = []
    for r in results:
        print(r)
        if r["status"] == "ok":
            ok_sizes.append(r["batch_size"])
    if ok_sizes:
        ceiling = max(ok_sizes)
        safe = max(1, int(ceiling * 0.75))
        # snap down to the largest tested size <= safe
        candidates = [s for s in ok_sizes if s <= safe] or [ok_sizes[0]]
        recommended = max(candidates)
        print(f"\nceiling (largest OK tested): {ceiling}")
        print(f"recommended train_batch_size (~75% of ceiling, snapped to tested value): {recommended}")
    else:
        print("\nno batch size succeeded — even the smallest tested size OOM'd or errored")
