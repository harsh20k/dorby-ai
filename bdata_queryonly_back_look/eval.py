"""All-pairs B-data eval of queryonly_back_look_001 LoRA on voyage-4-nano.

Seeker = search query only; candidate = background + lookingFor.
No train/holdout split. Retrieval ranks each ACCEPT query against the
unique-candidate document pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from baselines.metrics import (
    neg_hardness_slice_metrics,
    pair_metrics,
    retrieval_metrics_from_ranks,
)

from bdata_queryonly_back_look.config import DEFAULT_CONFIG, ExperimentConfig
from bdata_queryonly_back_look.data import (
    BPair,
    load_pairs,
    retrieval_corpus,
    within_seeker_dual_label_groups,
    write_id_map,
)
from bdata_queryonly_back_look.encode import VoyageNanoEncoder, cosine_scores, pick_device
from bdata_queryonly_back_look.store import persist_from_eval_arrays


# Quoted from prior runs — different packing and/or population unless noted.
COMPARISON = {
    "queryonly_back_look_001_all200": {
        "artifact": "docs/twotower-queryonly-back-look-experiment.md",
        "population": "200 real seed pairs; same packing + same adapter",
        "pair_auc": 0.5983,
        "hard_neg_auc": 0.6564,
        "mrr": 0.4791,
        "recall_at_1": 0.30,
    },
    "bdata_voyage_nano_posbg_allpairs": {
        "artifact": "artifacts/bdata_voyage_nano_posbg/metrics.json",
        "population": "same B-data all-pairs; frozen nano; look+query vs pos+bg",
        "pair_auc": 0.5185,
        "hard_neg_auc": 0.4319,
        "mrr": 0.0320,
        "recall_at_1": 0.0048,
        "note": "same population, different packing, frozen not LoRA",
    },
    "bdata_voyage_nano_holdout_fullprofile": {
        "artifact": "artifacts/bdata_voyage_nano/metrics.json",
        "population": "B-data holdout; frozen nano; full-profile packing",
        "pair_auc": 0.4691,
        "hard_neg_auc": 0.3626,
        "note": "different population AND packing",
    },
    "bdata_tfidf_holdout": {
        "artifact": "artifacts/bdata_tfidf/metrics.json",
        "population": "B-data holdout; TF-IDF full-profile",
        "pair_auc": 0.5121,
        "hard_neg_auc": 0.4026,
    },
    "voyage_nano_frozen_full_200": {
        "artifact": "artifacts/voyage_nano/metrics.json",
        "population": "200 real seed pairs; frozen nano; full-profile",
        "pair_auc": 0.5614,
        "hard_neg_auc": 0.5064,
    },
}


def _split_by_label(pairs: list[BPair]) -> tuple[list[BPair], list[BPair]]:
    pos = [p for p in pairs if p.label == "ACCEPT"]
    neg = [p for p in pairs if p.label == "REJECT"]
    return pos, neg


def _within_seeker_auc(pairs: list[BPair], scores_by_id: dict[str, float]) -> dict[str, Any]:
    groups = within_seeker_dual_label_groups(pairs)
    per_seeker: list[float] = []
    for _sid, g in groups.items():
        y = [1] * len(g["ACCEPT"]) + [0] * len(g["REJECT"])
        s = [scores_by_id[p.pair_id] for p in g["ACCEPT"] + g["REJECT"]]
        if len(set(y)) < 2:
            continue
        try:
            per_seeker.append(float(roc_auc_score(y, s)))
        except ValueError:
            continue
    if not per_seeker:
        return {"n_seekers": 0, "mean_auc": None, "skipped": "no dual-label seekers"}
    return {
        "n_seekers": len(per_seeker),
        "mean_auc": float(np.mean(per_seeker)),
        "median_auc": float(np.median(per_seeker)),
        "min_auc": float(np.min(per_seeker)),
        "max_auc": float(np.max(per_seeker)),
    }


def encode_aligned(
    encoder: VoyageNanoEncoder,
    texts: Sequence[str],
    *,
    role: Literal["query", "document"],
    batch_size: int,
    cache_name: str,
) -> np.ndarray:
    texts_list = list(texts)
    if not texts_list:
        return encoder.encode([], role=role, batch_size=batch_size, cache_name=cache_name)

    unique: list[str] = []
    index: dict[str, int] = {}
    for t in texts_list:
        if t not in index:
            index[t] = len(unique)
            unique.append(t)

    unique_emb = encoder.encode(
        unique,
        role=role,
        batch_size=batch_size,
        cache_name=f"{cache_name}_u{len(unique)}",
        show_progress=True,
    )
    rows = [index[t] for t in texts_list]
    return unique_emb[np.asarray(rows, dtype=np.int64)]


def batched_retrieval_ranks(
    query_embs: np.ndarray,
    target_ids: list[str],
    candidate_ids: list[str],
    candidate_embs: np.ndarray,
    *,
    query_batch_size: int = 256,
) -> list[int]:
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    missing = [t for t in target_ids if t not in id_to_idx]
    if missing:
        raise KeyError(f"{len(missing)} target ids missing from candidate corpus")

    n_q = len(target_ids)
    ranks = np.empty(n_q, dtype=np.int32)
    target_idx = np.asarray([id_to_idx[t] for t in target_ids], dtype=np.int64)
    cand_t = np.ascontiguousarray(candidate_embs.T)

    try:
        import torch

        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False

    if use_cuda:
        import torch

        cand = torch.from_numpy(np.ascontiguousarray(candidate_embs)).to("cuda")
        for start in range(0, n_q, query_batch_size):
            end = min(start + query_batch_size, n_q)
            q = torch.from_numpy(np.ascontiguousarray(query_embs[start:end])).to("cuda")
            scores = q @ cand.T
            order = torch.argsort(scores, dim=1, descending=True, stable=True)
            tgt = torch.from_numpy(target_idx[start:end]).to("cuda").unsqueeze(1)
            pos = (order == tgt).to(torch.int64).argmax(dim=1) + 1
            ranks[start:end] = pos.detach().cpu().numpy()
            del q, scores, order, tgt, pos
        del cand
        torch.cuda.empty_cache()
        return ranks.tolist()

    for start in range(0, n_q, query_batch_size):
        end = min(start + query_batch_size, n_q)
        scores = query_embs[start:end] @ cand_t
        order = np.argsort(-scores, axis=1, kind="stable")
        tgt = target_idx[start:end][:, None]
        ranks[start:end] = (order == tgt).argmax(axis=1) + 1
    return ranks.tolist()


def run_eval(cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DEFAULT_CONFIG
    device = pick_device()
    if not cfg.adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Missing adapter at {cfg.adapter_dir}. Pull with:\n"
            "  modal volume get dorby-twotower-queryonly-back-look-checkpoints "
            "queryonly_back_look_001/adapter "
            "./artifacts/twotower_queryonly_back_look/queryonly_back_look_001/adapter"
        )
    print(f"device: {device}")
    print(
        f"model:  {cfg.model_name} + {cfg.adapter_dir} "
        f"(max_length={cfg.max_length}, truncate_dim={cfg.truncate_dim}, "
        f"batch_size={cfg.batch_size})"
    )

    pairs, id_map, contacts, meta = load_pairs(cfg)
    print(
        f"loaded {meta['n_resolved_pairs']} resolved pairs "
        f"({meta['n_accept']} ACCEPT / {meta['n_reject']} REJECT) "
        f"from {meta['n_unique_seekers']} seekers"
    )
    print(
        f"id map: {meta['id_map']['n_boardy']} boardy / "
        f"{meta['id_map']['n_minted']} minted / "
        f"{meta['id_map'].get('n_minted_collision', 0)} collision-minted / "
        f"{meta['id_map']['n_corpus']} retrieval-corpus people"
    )

    pos, neg = _split_by_label(pairs)
    print(f"all-pairs labels: {len(pos)} ACCEPT / {len(neg)} REJECT (no split)")

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_id_map(id_map, cfg.artifacts_dir / "id_map.json")

    encoder = VoyageNanoEncoder(
        model_name=cfg.model_name,
        device=device,
        max_length=cfg.max_length,
        truncate_dim=cfg.truncate_dim,
        cache_dir=cfg.artifacts_dir,
        adapter_dir=cfg.adapter_dir,
    )

    pos_s = encode_aligned(
        encoder, [p.seeker_text for p in pos], role="query",
        batch_size=cfg.batch_size, cache_name="all_pos_seeker",
    )
    pos_c = encode_aligned(
        encoder, [p.cand_text for p in pos], role="document",
        batch_size=cfg.batch_size, cache_name="all_pos_cand",
    )
    neg_s = encode_aligned(
        encoder, [p.seeker_text for p in neg], role="query",
        batch_size=cfg.batch_size, cache_name="all_neg_seeker",
    )
    neg_c = encode_aligned(
        encoder, [p.cand_text for p in neg], role="document",
        batch_size=cfg.batch_size, cache_name="all_neg_cand",
    )

    pos_scores = cosine_scores(pos_s, pos_c)
    neg_scores = cosine_scores(neg_s, neg_c)
    pair = pair_metrics(pos_scores, neg_scores)

    hardness = neg_hardness_slice_metrics(
        neg_scores=neg_scores,
        neg_seeker_texts=[p.hardness_seeker_text for p in neg],
        neg_cand_texts=[p.hardness_cand_text for p in neg],
        pos_scores=pos_scores,
    )

    scores_by_id = {
        **{p.pair_id: float(s) for p, s in zip(pos, pos_scores)},
        **{p.pair_id: float(s) for p, s in zip(neg, neg_scores)},
    }
    within = _within_seeker_auc(pairs, scores_by_id)
    if within["n_seekers"] < cfg.min_within_seeker_n:
        within = {
            **within,
            "reported": False,
            "note": (
                f"n_seekers={within['n_seekers']} < min_within_seeker_n="
                f"{cfg.min_within_seeker_n}; metric recorded but not headline"
            ),
        }
    else:
        within = {**within, "reported": True}

    corpus_ids, corpus_texts = retrieval_corpus(contacts, id_map)
    print(
        f"retrieval corpus: {len(corpus_ids)} unique match-role people "
        f"(query batch={cfg.retrieval_query_batch_size})"
    )
    corpus_emb = encode_aligned(
        encoder, corpus_texts, role="document",
        batch_size=cfg.batch_size, cache_name="corpus_cand",
    )
    corpus_set = set(corpus_ids)
    keep = [i for i, p in enumerate(pos) if p.match_contact_id in corpus_set]
    skipped = len(pos) - len(keep)
    if skipped:
        print(f"retrieval: skipping {skipped} ACCEPT pairs whose candidate is not in corpus")
    query_embs = pos_s[np.asarray(keep, dtype=np.int64)] if keep else pos_s[:0]
    target_ids = [pos[i].match_contact_id for i in keep]
    if keep:
        ranks = batched_retrieval_ranks(
            query_embs, target_ids, corpus_ids, corpus_emb,
            query_batch_size=cfg.retrieval_query_batch_size,
        )
        retrieval = retrieval_metrics_from_ranks(ranks, ks=cfg.retrieval_ks)
    else:
        retrieval = retrieval_metrics_from_ranks([], ks=cfg.retrieval_ks)
    retrieval = {
        **retrieval,
        "n_corpus": len(corpus_ids),
        "n_queries": len(keep),
        "n_skipped_not_in_corpus": skipped,
    }

    vector_store = persist_from_eval_arrays(
        cfg=cfg,
        pos=pos,
        neg=neg,
        pos_s=pos_s,
        pos_c=pos_c,
        neg_s=neg_s,
        neg_c=neg_c,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        corpus_emb=corpus_emb,
    )

    return {
        "model_name": cfg.model_name,
        "adapter_dir": str(cfg.adapter_dir),
        "device": str(device),
        "max_length": cfg.max_length,
        "truncate_dim": cfg.truncate_dim,
        "batch_size": cfg.batch_size,
        "source": meta,
        "pair": pair,
        "slices": {"neg_hardness": hardness},
        "within_seeker": within,
        "retrieval": retrieval,
        "vector_store": vector_store,
        "comparison": COMPARISON,
        "notes": {
            "label": "ACCEPT=positive, REJECT=negative; PENDING dropped",
            "encoder": "voyage-4-nano + queryonly_back_look_001 LoRA; ALL resolved pairs",
            "text_packing": "seeker=query only; candidate=background+lookingFor",
            "hardness": "full-profile+query text (same convention as queryonly_back_look eval)",
            "retrieval": "unique people with role candidate|both; minted/real ids; exact NumPy, not Chroma ANN",
            "vector_db": (
                "isolated Chroma under artifacts/bdata_queryonly_back_look/chroma/; "
                ".npy in vectors/ is the source of truth"
            ),
        },
    }


def print_bdata_metrics(metrics: dict[str, Any]) -> None:
    pair = metrics["pair"]
    print("\n=== Pair metrics (B-data ALL resolved, queryonly_back_look packing) ===")
    print(f"ROC-AUC:              {pair['roc_auc']:.4f}")
    print(f"Average Precision:    {pair['average_precision']:.4f}")
    print(
        f"Best-F1:              {pair['best_f1']:.4f} "
        f"@ threshold={pair['best_f1_threshold']:.4f} "
        f"(acc={pair['best_f1_accuracy']:.4f})"
    )
    print(f"Accuracy @ 0.5:       {pair['accuracy_at_0.5']:.4f}")
    print(
        f"Mean cosine (pos/neg/gap): "
        f"{pair['mean_cosine_positive']:.4f} / "
        f"{pair['mean_cosine_negative']:.4f} / "
        f"{pair['mean_cosine_gap']:.4f}"
    )
    print(f"n pos/neg:            {pair['num_positive']} / {pair['num_negative']}")

    hard = (metrics.get("slices") or {}).get("neg_hardness") or {}
    easy = hard.get("easy") or {}
    hard_b = hard.get("hard") or {}
    if easy or hard_b:
        print("\n=== Neg-hardness slices ===")
        if easy.get("pair_auc") is not None:
            print(
                f"easy-neg AUC:         {easy['pair_auc']:.4f} "
                f"(n_neg={easy.get('n_negatives')})"
            )
        if hard_b.get("pair_auc") is not None:
            print(
                f"hard-neg AUC:         {hard_b['pair_auc']:.4f} "
                f"(n_neg={hard_b.get('n_negatives')})"
            )

    within = metrics.get("within_seeker") or {}
    print("\n=== Within-seeker ranking ===")
    if within.get("mean_auc") is None:
        print(f"skipped: {within.get('skipped') or within.get('note')}")
    else:
        flag = "headline" if within.get("reported") else "informational only"
        print(
            f"mean AUC:             {within['mean_auc']:.4f} "
            f"(n_seekers={within['n_seekers']}, {flag})"
        )

    ret = metrics.get("retrieval") or {}
    print("\n=== Retrieval (ACCEPT queries vs unique-candidate pool) ===")
    if ret.get("n_queries"):
        print(f"MRR:                  {ret['mrr']:.4f}")
        print(f"mean / median rank:   {ret['mean_rank']:.1f} / {ret['median_rank']:.1f}")
        for k in (1, 5, 10, 50, 100):
            rk = ret.get(f"recall@{k}")
            if rk is not None:
                print(f"R@{k:<3}                 {rk:.4f}")
        print(
            f"n queries / corpus:   {int(ret['n_queries'])} / {int(ret.get('n_corpus') or 0)}"
        )
    else:
        print("skipped: no ACCEPT queries in corpus")

    print("\n=== Comparison (quoted) ===")
    for key, ref in (metrics.get("comparison") or {}).items():
        print(
            f"{key}: pair_auc={ref.get('pair_auc')} "
            f"hard_neg_auc={ref.get('hard_neg_auc')} "
            f"mrr={ref.get('mrr')}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B-data eval of queryonly_back_look_001 (isolated, all pairs)"
    )
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--unique-contacts", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--adapter-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--truncate-dim", type=int, default=None)
    p.add_argument("--retrieval-query-batch-size", type=int, default=None)
    return p.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    base = DEFAULT_CONFIG
    return ExperimentConfig(
        source_path=args.source or base.source_path,
        unique_contacts_path=args.unique_contacts or base.unique_contacts_path,
        artifacts_dir=args.artifacts_dir or base.artifacts_dir,
        adapter_dir=args.adapter_dir or base.adapter_dir,
        model_name=args.model or base.model_name,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        max_length=args.max_length if args.max_length is not None else base.max_length,
        truncate_dim=(
            args.truncate_dim if args.truncate_dim is not None else base.truncate_dim
        ),
        min_within_seeker_n=base.min_within_seeker_n,
        retrieval_query_batch_size=(
            args.retrieval_query_batch_size
            if args.retrieval_query_batch_size is not None
            else base.retrieval_query_batch_size
        ),
        retrieval_ks=base.retrieval_ks,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _cfg_from_args(args)
    metrics = run_eval(cfg)
    print_bdata_metrics(metrics)
    out_path = cfg.artifacts_dir / "metrics.json"
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
