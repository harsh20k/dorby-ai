"""Custom training loop for two genuinely separate LoRA adapters — a query
tower and a candidate tower — instead of one shared model reading different
input text.

Why a custom loop, not ``SentenceTransformerTrainer``
-------------------------------------------------------
The trainer assumes one shared model encodes every dataset column. Here the
anchor column must go through a different set of weights than the
positive/negative columns, which the trainer has no way to express. This loop
reuses the codebase's generic, model-agnostic pieces read-only
(``twotower.train.build_model`` / ``add_lora_adapter``, ``baselines.metrics``)
and reimplements only what's genuinely new: routing two adapters and a
from-scratch MultipleNegativesRankingLoss-equivalent (in-batch + one explicit
hard negative, cross-entropy over similarity scaled by 20 — the same
formula the library loss uses, since the library class itself assumes one
model).
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

from twotower_split.data import MultiNegRow, carve_dev, load_multineg_rows

LOSS_SCALE = 20.0  # library default, matching top1_ctrl


def _encode(model, texts: list[str], prompt: str, device: torch.device) -> torch.Tensor:
    """Gradient-enabled encode: prepend the role prompt, tokenize, forward.
    Mirrors twotower.train.smoke_backward's tokenize->forward pattern."""
    preprocess = getattr(model, "preprocess", None) or model.tokenize
    prefixed = [prompt + t for t in texts]
    feats = dict(preprocess(prefixed))
    feats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
    return model(feats)["sentence_embedding"]


@torch.no_grad()
def _encode_eval(model, texts: list[str], prompt: str, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        emb = _encode(model, texts[i : i + batch_size], prompt, device)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def _build_dev_corpus(rows: list[MultiNegRow]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
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


def _dev_recall1(
    query_model, doc_model, dev_rows: list[MultiNegRow], cfg: TrainConfig, device: torch.device
) -> dict[str, float]:
    anchor_emb = _encode_eval(query_model, [r.anchor for r in dev_rows], cfg.query_prompt, device, cfg.eval_batch_size)
    corpus_ids, corpus_texts = _build_dev_corpus(dev_rows)
    corpus_emb = _encode_eval(doc_model, corpus_texts, cfg.document_prompt, device, cfg.eval_batch_size)
    metrics = retrieval_metrics(
        query_embs=anchor_emb,
        target_ids=[r.positive_id for r in dev_rows],
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )
    return metrics


def _mnrl_step(
    query_model, doc_model, batch: list[MultiNegRow], device: torch.device
) -> torch.Tensor:
    anchor = _encode(query_model, [r.anchor for r in batch], query_model._role_prompt, device)
    pos = _encode(doc_model, [r.positive for r in batch], doc_model._role_prompt, device)
    neg = _encode(doc_model, [r.negatives[0] for r in batch], doc_model._role_prompt, device)

    anchor = F.normalize(anchor, dim=-1)
    candidates = F.normalize(torch.cat([pos, neg], dim=0), dim=-1)
    scores = anchor @ candidates.T * LOSS_SCALE  # (B, 2B)
    labels = torch.arange(len(batch), device=device)
    return F.cross_entropy(scores, labels)


def run_training(cfg: TrainConfig, rows_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, extraction_summary = load_multineg_rows(rows_path)
    train_rows, dev_rows = carve_dev(all_rows, seed=cfg.seed)
    print(f"rows: train={len(train_rows)} dev={len(dev_rows)} (source: {extraction_summary})")

    query_model = build_model(cfg, str(device))
    query_lora = add_lora_adapter(query_model, cfg)
    query_model._role_prompt = cfg.query_prompt

    doc_model = build_model(cfg, str(device))
    doc_lora = add_lora_adapter(doc_model, cfg)
    doc_model._role_prompt = cfg.document_prompt

    print(f"query tower lora: {query_lora}")
    print(f"doc tower lora: {doc_lora}")

    meta = collect_env_metadata(cfg, str(device))
    meta.update(
        {
            "experiment": "twotower_split",
            "rows_path": str(rows_path),
            "extraction_summary": extraction_summary,
            "n_train_rows": len(train_rows),
            "n_dev_rows": len(dev_rows),
            "query_lora": query_lora,
            "doc_lora": doc_lora,
            "loss_scale": LOSS_SCALE,
            "dry_run": dry_run,
        }
    )
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if dry_run:
        # one smoke step, no optimizer update, to prove both adapters get gradients
        batch = train_rows[: min(4, len(train_rows))]
        loss = _mnrl_step(query_model, doc_model, batch, device)
        loss.backward()
        print(f"dry_run smoke loss: {loss.item():.4f}")
        return meta

    params = [p for p in query_model.parameters() if p.requires_grad] + [
        p for p in doc_model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    rng = random.Random(cfg.seed)
    loss_history: list[dict[str, float]] = []
    dev_history: list[dict[str, Any]] = []
    best_recall1 = -1.0
    best_epoch = -1
    best_dir = output_dir / "best"

    query_model.train()
    doc_model.train()
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        epoch_losses = []
        for i in range(0, len(order), cfg.train_batch_size):
            idx = order[i : i + cfg.train_batch_size]
            if len(idx) < 2:
                continue  # cross-entropy over a batch of 1 is degenerate
            batch = [train_rows[j] for j in idx]
            loss = _mnrl_step(query_model, doc_model, batch, device)
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

        dev_metrics = _dev_recall1(query_model, doc_model, dev_rows, cfg, device)
        r1 = dev_metrics.get("recall@1", 0.0)
        dev_history.append({"epoch": epoch, **dev_metrics})
        print(f"epoch {epoch}: mean_loss={mean_loss} dev_recall@1={r1:.4f} dev_mrr={dev_metrics.get('mrr'):.4f}")

        if r1 > best_recall1:
            best_recall1 = r1
            best_epoch = epoch
            query_model.save_pretrained(str(best_dir / "query_adapter"))
            doc_model.save_pretrained(str(best_dir / "doc_adapter"))

    (output_dir / "loss_history.json").write_text(
        json.dumps({"train_loss": loss_history, "dev": dev_history}, indent=2) + "\n"
    )
    result = {
        **meta,
        "best_epoch": best_epoch,
        "best_dev_recall@1": best_recall1,
        "query_adapter_dir": str(best_dir / "query_adapter"),
        "doc_adapter_dir": str(best_dir / "doc_adapter"),
    }
    (output_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"best epoch: {best_epoch} (dev recall@1={best_recall1:.4f})")
    return result
