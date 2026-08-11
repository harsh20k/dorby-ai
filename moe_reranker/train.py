"""Train and evaluate the MMoE re-ranker.

    PYTHONPATH=. .venv/bin/python -m moe_reranker.train --run-id moe_001
    PYTHONPATH=. .venv/bin/python -m moe_reranker.train --tau 0.2 --run-id moe_tau02
    PYTHONPATH=. .venv/bin/python -m moe_reranker.train --holdout   # final shot only

**The holdout is opt-in on purpose.** Without ``--holdout`` this reports
train-dev only. The real 69-pair holdout is a one-shot final check (the decision
gate in ``docs/two-tower-fine-tune-plan.md``); the standard error on any AUC
there is ±0.070, so repeatedly peeking at it while tuning is how you pick a
winner out of noise.

**Read the diagnostics before the score.** They are printed first, deliberately.
If the gate collapsed or routing tracks seeker identity, the accuracy number is
not evidence of anything regardless of what it says.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.metrics import pair_metrics
from moe_reranker import diagnostics
from moe_reranker.config import MoEConfig
from moe_reranker.data import load, within_seeker_triplet_count
from moe_reranker.model import MMoE, balance_loss, sharpen_loss, task_loss


def _gates_for(model: MMoE, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, g = model(torch.from_numpy(X))
    return g.numpy()


def _auc(model: MMoE, split: Any) -> float:
    scores = model.score(torch.from_numpy(split.X)).numpy()
    pos, neg = scores[split.y > 0.5], scores[split.y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(pair_metrics(pos, neg)["roc_auc"])


def train(cfg: MoEConfig, *, eval_holdout: bool = False) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("loading splits (leakage-safe via twotower.data.build_split_bundle)")
    splits = load(cfg)
    tr, dev, hold = splits["train"], splits["train_dev"], splits["holdout"]
    for name in ("train", "train_dev", "holdout"):
        s = splits[name]
        n_aux = int(s.aux_mask.sum())
        print(
            f"  {name:<10} {len(s.rows):>3} pairs "
            f"({int(s.y.sum())} pos / {int((1 - s.y).sum())} neg), "
            f"{n_aux} with a judge label"
        )

    n_trip, n_both, n_seek = within_seeker_triplet_count(tr)
    print(
        f"\n  within-seeker structure (train): {n_trip} triplets from {n_both} of "
        f"{n_seek} seekers carrying both classes"
    )
    if n_trip < 50:
        print(
            "  -> too few for within-seeker training on real pairs. Training pairwise;\n"
            "     the per-seeker base rate is NOT cancelled by construction here, so\n"
            "     Diagnostic 3 (routing vs seeker identity) is load-bearing, not optional."
        )

    n_features = tr.X.shape[1]
    print(f"\n  features: {n_features}  |  train rows: {len(tr.rows)}")
    model = MMoE(
        n_features=n_features,
        n_experts=cfg.n_experts,
        expert_hidden=cfg.expert_hidden,
        n_tasks=len(cfg.task_names),
        tau=cfg.tau,
        expert_dropout=cfg.expert_dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  parameters: {n_params}  ({n_params / max(1, len(tr.rows)):.1f} per training pair)"
    )
    if n_params > 4 * len(tr.rows):
        print("  WARNING: more than 4 parameters per training pair — expect overfitting")

    Xtr = torch.from_numpy(tr.X)
    ttr, mtr = tr.targets()
    Ttr, Mtr = torch.from_numpy(ttr), torch.from_numpy(mtr)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = {"dev_auc": -1.0, "epoch": -1, "state": None}
    history: list[dict[str, float]] = []
    since_best = 0

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        epoch_loss = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if len(idx) < 2:
                continue  # balance_loss needs a batch to average over
            opt.zero_grad()
            logits, gates = model(Xtr[idx])
            loss = (
                task_loss(logits, Ttr[idx], Mtr[idx], cfg.task_weights)
                + cfg.sharpen_weight * sharpen_loss(gates)
                + cfg.balance_weight * balance_loss(gates)
            )
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)

        dev_auc = _auc(model, dev)
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(1, len(perm)),
                "train_auc": _auc(model, tr),
                "dev_auc": dev_auc,
            }
        )
        if dev_auc > best["dev_auc"]:
            best = {
                "dev_auc": dev_auc,
                "epoch": epoch,
                "state": {k: v.clone() for k, v in model.state_dict().items()},
            }
            since_best = 0
        else:
            since_best += 1
            if since_best >= cfg.early_stop_patience:
                print(f"  early stop at epoch {epoch} (no dev gain in {since_best})")
                break

    # Selecting on train-dev, never on holdout. `docs/possible-bugs.md` #2 was
    # exactly this mistake in the twotower run: shipping the final epoch.
    assert best["state"] is not None
    model.load_state_dict(best["state"])
    print(f"\n  best epoch {best['epoch']} — train-dev AUC {best['dev_auc']:.4f}")

    # ---- diagnostics first, on train (where routing was actually learned) ----
    diag_tr = diagnostics.compute(_gates_for(model, tr.X), tr.seeker_ids)
    print("\n" + "=" * 72)
    print(diagnostics.render(diag_tr, cfg.task_names))
    print("=" * 72)

    result: dict[str, Any] = {
        "config": cfg.as_json(),
        "n_params": n_params,
        "n_features": n_features,
        "feature_count": n_features,
        "best_epoch": best["epoch"],
        "train_auc": _auc(model, tr),
        "train_dev_auc": best["dev_auc"],
        "within_seeker": {
            "triplets": n_trip,
            "seekers_with_both": n_both,
            "seekers": n_seek,
        },
        "diagnostics_train": diag_tr.as_json(),
        "history": history,
    }

    if eval_holdout:
        scores = model.score(torch.from_numpy(hold.X)).numpy()
        pos, neg = scores[hold.y > 0.5], scores[hold.y <= 0.5]
        pm = pair_metrics(pos, neg)
        diag_ho = diagnostics.compute(_gates_for(model, hold.X), hold.seeker_ids)
        result["holdout"] = {"pair": pm, "diagnostics": diag_ho.as_json()}
        print(f"\nHOLDOUT (one-shot) pair AUC {pm['roc_auc']:.4f}  AP {pm['average_precision']:.4f}")
        print("  SE on this AUC is about +/-0.070 (29 pos / 40 neg) — do not read")
        print("  small differences as real. Diagnostics on holdout routing:")
        print(diagnostics.render(diag_ho, cfg.task_names))
    else:
        print("\nholdout NOT evaluated (pass --holdout for the one-shot final check)")

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=float))
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"\nwrote {run_dir}/result.json and model.pt")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    d = MoEConfig()
    p.add_argument("--run-id", default=d.run_id)
    p.add_argument("--data-dir", type=Path, default=d.data_dir)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--tau", type=float, default=d.tau)
    p.add_argument("--n-experts", type=int, default=d.n_experts)
    p.add_argument("--expert-hidden", type=int, default=d.expert_hidden)
    p.add_argument("--expert-dropout", type=float, default=d.expert_dropout)
    p.add_argument("--sharpen-weight", type=float, default=d.sharpen_weight)
    p.add_argument("--balance-weight", type=float, default=d.balance_weight)
    p.add_argument("--emb-pca-dims", type=int, default=d.emb_pca_dims)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument(
        "--aux-weight",
        type=float,
        default=d.task_weights[1],
        help="Weight on the judge auxiliary task. 0 disables it (single-task MoE).",
    )
    p.add_argument(
        "--holdout",
        action="store_true",
        help="Evaluate the real 69-pair holdout. One-shot final check only.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    cfg = replace(
        MoEConfig(),
        run_id=a.run_id,
        data_dir=a.data_dir,
        epochs=a.epochs,
        lr=a.lr,
        tau=a.tau,
        n_experts=a.n_experts,
        expert_hidden=a.expert_hidden,
        expert_dropout=a.expert_dropout,
        sharpen_weight=a.sharpen_weight,
        balance_weight=a.balance_weight,
        emb_pca_dims=a.emb_pca_dims,
        seed=a.seed,
        task_weights=(1.0, a.aux_weight),
    )
    train(cfg, eval_holdout=a.holdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
