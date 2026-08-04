"""Custom training loop: one shared tower (LoRA adapter), seeker side
embedded as three separate pieces (query, lookingFor, positioning) and
combined by a small learned gate (gate.py) before the contrastive loss.

Same reason as twotower_split/train.py for not using SentenceTransformerTrainer:
the seeker side needs a non-standard forward pass (three encodes + a combine
step) the trainer has no way to express. Reuses twotower.train's generic
helpers (build_model, add_lora_adapter) read-only; the loss is the same
from-scratch MultipleNegativesRankingLoss-equivalent used in twotower_split/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from baselines.metrics import retrieval_metrics
from twotower.config import TrainConfig
from twotower.train import add_lora_adapter, build_model, collect_env_metadata

from twotower_field_gate.config import PIECE_KEYS
from twotower_field_gate.data import FieldRow, carve_dev, load_field_rows
from twotower_field_gate.gate import FieldGate

LOSS_SCALE = 20.0


def _encode(model, texts: list[str], prompt: str, device: torch.device, truncate_dim: int) -> torch.Tensor:
    """Raw forward + truncate + normalize — matches SentenceTransformer.encode()'s
    own order exactly (sentence_transformers.util.truncate_embeddings slices
    the already-normalized pooled output, then encode() re-normalizes only if
    normalize_embeddings=True; verified numerically identical, max abs diff
    0.0, before wiring this in). ``model(feats)["sentence_embedding"]`` alone
    returns the *untruncated* native width (2048 for voyage-4-nano, not 1024
    — nano's LoRA adapter is fine-tuned on 2048-dim embeddings and only
    truncated to 1024 at inference; skipping the truncation step here would
    train on a different representation than every other number in this
    project was computed on)."""
    preprocess = getattr(model, "preprocess", None) or model.tokenize
    prefixed = [prompt + t for t in texts]
    feats = dict(preprocess(prefixed))
    feats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
    raw = model(feats)["sentence_embedding"]
    return raw[:, :truncate_dim]


def _encode_seeker(model, gate: FieldGate, rows: list[FieldRow], cfg: TrainConfig, device) -> torch.Tensor:
    piece_embs = []
    for key in PIECE_KEYS:
        texts = [r.pieces[key] for r in rows]
        emb = F.normalize(_encode(model, texts, cfg.query_prompt, device, cfg.truncate_dim), dim=-1)
        piece_embs.append(emb)
    return gate(piece_embs)


@torch.no_grad()
def _encode_eval(model, gate, rows: list[FieldRow], cfg: TrainConfig, device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        emb = _encode_seeker(model, gate, rows[i : i + batch_size], cfg, device)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


@torch.no_grad()
def _encode_cand_eval(model, texts: list[str], prompt: str, device, batch_size: int, truncate_dim: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        emb = F.normalize(_encode(model, texts[i : i + batch_size], prompt, device, truncate_dim), dim=-1)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def _build_dev_corpus(rows: list[FieldRow]) -> tuple[list[str], list[str]]:
    ids, texts, seen = [], [], set()
    for row in rows:
        if row.positive_id not in seen:
            seen.add(row.positive_id)
            ids.append(row.positive_id)
            texts.append(row.positive)
    for row in rows:
        for cid, text in zip(row.negative_ids, row.negatives):
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            texts.append(text)
    return ids, texts


def _dev_recall1(model, gate, dev_rows: list[FieldRow], cfg: TrainConfig, device) -> dict[str, float]:
    seeker_emb = _encode_eval(model, gate, dev_rows, cfg, device, cfg.eval_batch_size)
    corpus_ids, corpus_texts = _build_dev_corpus(dev_rows)
    corpus_emb = _encode_cand_eval(model, corpus_texts, cfg.document_prompt, device, cfg.eval_batch_size, cfg.truncate_dim)
    return retrieval_metrics(
        query_embs=seeker_emb,
        target_ids=[r.positive_id for r in dev_rows],
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )


def run_training(cfg: TrainConfig, rows_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, extraction_summary = load_field_rows(rows_path)
    train_rows, dev_rows = carve_dev(all_rows, seed=cfg.seed)
    print(f"rows: train={len(train_rows)} dev={len(dev_rows)} (source: {extraction_summary})")

    model = build_model(cfg, str(device))
    lora_info = add_lora_adapter(model, cfg)
    gate = FieldGate(n_pieces=len(PIECE_KEYS), dim=cfg.truncate_dim).to(device)
    print(f"lora: {lora_info}")
    print(f"gate params: {sum(p.numel() for p in gate.parameters())}")

    meta = collect_env_metadata(cfg, str(device))
    meta.update(
        {
            "experiment": "twotower_field_gate",
            "rows_path": str(rows_path),
            "piece_keys": list(PIECE_KEYS),
            "extraction_summary": extraction_summary,
            "n_train_rows": len(train_rows),
            "n_dev_rows": len(dev_rows),
            "lora": lora_info,
            "gate_params": sum(p.numel() for p in gate.parameters()),
            "loss_scale": LOSS_SCALE,
            "dry_run": dry_run,
        }
    )
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def step_loss(batch: list[FieldRow]) -> torch.Tensor:
        seeker = _encode_seeker(model, gate, batch, cfg, device)  # already normalized by gate
        pos = F.normalize(_encode(model, [r.positive for r in batch], cfg.document_prompt, device, cfg.truncate_dim), dim=-1)
        neg = F.normalize(_encode(model, [r.negatives[0] for r in batch], cfg.document_prompt, device, cfg.truncate_dim), dim=-1)
        candidates = torch.cat([pos, neg], dim=0)
        scores = seeker @ candidates.T * LOSS_SCALE
        labels = torch.arange(len(batch), device=device)
        return F.cross_entropy(scores, labels)

    if dry_run:
        batch = train_rows[: min(4, len(train_rows))]
        loss = step_loss(batch)
        loss.backward()
        print(f"dry_run smoke loss: {loss.item():.4f}")
        return meta

    params = [p for p in model.parameters() if p.requires_grad] + list(gate.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    rng = random.Random(cfg.seed)
    loss_history: list[dict[str, float]] = []
    dev_history: list[dict[str, Any]] = []
    best_recall1 = -1.0
    best_epoch = -1
    best_dir = output_dir / "best"

    model.train()
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        epoch_losses = []
        for i in range(0, len(order), cfg.train_batch_size):
            idx = order[i : i + cfg.train_batch_size]
            if len(idx) < 2:
                continue
            batch = [train_rows[j] for j in idx]
            loss = step_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            epoch_losses.append(loss.item())
            step += 1
            if step % cfg.logging_steps == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f}")
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else None
        loss_history.append({"epoch": epoch, "mean_loss": mean_loss, "steps": step})

        dev_metrics = _dev_recall1(model, gate, dev_rows, cfg, device)
        r1 = dev_metrics.get("recall@1", 0.0)
        dev_history.append({"epoch": epoch, **dev_metrics})
        print(f"epoch {epoch}: mean_loss={mean_loss} dev_recall@1={r1:.4f} dev_mrr={dev_metrics.get('mrr'):.4f}")

        if r1 > best_recall1:
            best_recall1 = r1
            best_epoch = epoch
            model.save_pretrained(str(best_dir / "adapter"))
            torch.save(gate.state_dict(), best_dir / "gate.pt")

        model.train()

    (output_dir / "loss_history.json").write_text(
        json.dumps({"train_loss": loss_history, "dev": dev_history}, indent=2) + "\n"
    )
    result = {
        **meta,
        "best_epoch": best_epoch,
        "best_dev_recall@1": best_recall1,
        "adapter_dir": str(best_dir / "adapter"),
        "gate_path": str(best_dir / "gate.pt"),
    }
    (output_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"best epoch: {best_epoch} (dev recall@1={best_recall1:.4f})")
    return result
