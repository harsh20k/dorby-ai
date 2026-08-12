"""Run all 105 field/query combinations against the fine-tuned voyage_gemini_ctrl checkpoint.

Same sweep design as every prior field/query sweep in this project — this is
the identical 105-combo grid run against a third fine-tuned encoder: the
``voyage_gemini_ctrl_001`` LoRA adapter
(``baselines/twotower_voyage_gemini_ctrl_ckpt1130_field_sweep/encode.py``), `top1_ctrl`'s
exact recipe retrained on a bigger, newer synthetic batch (see this
package's ``__init__.py``). Encoding dedup is unchanged: candidate-side
embeddings depend only on ``candidate_fields`` (7 unique groups); seeker-side
embeddings depend only on (``seeker_fields``, ``use_query``) (15 unique
groups, including the query-only seeker). 22 unique encode groups feeding
all 105 combos' metrics, computed from cached embeddings.

Every combo gets the full metric suite this project tracks everywhere else
(``baselines/metrics.py``, unmodified): pair ROC-AUC, best-F1, accuracy@0.5,
hard/easy-negative-slice pair AUC, and retrieval MRR / mean & median rank /
Recall@1,5,10 / NDCG@1,5,10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baselines.llm_judge.real_pairs import load_real_pairs
from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
from baselines.twotower_voyage_gemini_ctrl_ckpt1130_field_sweep.encode import (
    VOYAGE_GEMINI_CTRL_CKPT1130_ADAPTER_DIR,
    VOYAGE_GEMINI_CTRL_CKPT1130_BASE_MODEL,
    VOYAGE_GEMINI_CTRL_CKPT1130_MAX_SEQ_LENGTH,
    VOYAGE_GEMINI_CTRL_CKPT1130_TRUNCATE_DIM,
    VoyageGeminiCtrlCkpt1130Encoder,
    cosine_scores,
)
from baselines.twotower_voyage_gemini_ctrl_ckpt1130_field_sweep.fields import Combo, all_combos, subset_id
from baselines.twotower_voyage_gemini_ctrl_ckpt1130_field_sweep.text import candidate_to_text, seeker_to_text
from baselines.voyage_nano.encode import pick_device


def _build_candidate_corpus(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    candidate_fields: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for record in positives + negatives:
        mid = record["matchContactId"]
        if mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)
        texts.append(candidate_to_text(record["matchContactFile"], candidate_fields))
    return ids, texts


def run_sweep(
    *,
    data_dir: Path,
    adapter_dir: Path = VOYAGE_GEMINI_CTRL_CKPT1130_ADAPTER_DIR,
    model_name: str = VOYAGE_GEMINI_CTRL_CKPT1130_BASE_MODEL,
    batch_size: int,
    max_length: int = VOYAGE_GEMINI_CTRL_CKPT1130_MAX_SEQ_LENGTH,
    truncate_dim: int = VOYAGE_GEMINI_CTRL_CKPT1130_TRUNCATE_DIM,
    artifacts_dir: Path,
    split_path: Path | None = None,
    combos: list[Combo] | None = None,
    on_progress: Any = None,
) -> list[dict[str, Any]]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  {model_name} + adapter {adapter_dir}")

    positives, negatives = load_real_pairs(data_dir, split="all", split_path=split_path)
    print(f"loaded {len(positives)} real positives, {len(negatives)} real negatives (split=all)")

    encoder = VoyageGeminiCtrlCkpt1130Encoder(
        adapter_dir=adapter_dir,
        model_name=model_name,
        device=device,
        max_seq_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )

    combos = combos if combos is not None else all_combos()

    # --- candidate-side: unique per candidate_fields (7 groups) ---
    cand_field_sets = sorted({c.candidate_fields for c in combos}, key=subset_id)
    candidate_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for i, candidate_fields in enumerate(cand_field_sets, 1):
        cid = subset_id(candidate_fields)
        print(f"[candidate {i}/{len(cand_field_sets)}] fields={cid}")
        pos_cand_texts = [candidate_to_text(r["matchContactFile"], candidate_fields) for r in positives]
        neg_cand_texts = [candidate_to_text(r["matchContactFile"], candidate_fields) for r in negatives]
        corpus_ids, corpus_texts = _build_candidate_corpus(positives, negatives, candidate_fields)

        pos_cand_emb = encoder.encode(
            pos_cand_texts, role="document", batch_size=batch_size, cache_name=f"pos_cand_{cid}"
        )
        neg_cand_emb = encoder.encode(
            neg_cand_texts, role="document", batch_size=batch_size, cache_name=f"neg_cand_{cid}"
        )
        corpus_emb = encoder.encode(
            corpus_texts, role="document", batch_size=batch_size, cache_name=f"corpus_{cid}"
        )
        candidate_cache[candidate_fields] = {
            "pos_texts": pos_cand_texts,
            "neg_texts": neg_cand_texts,
            "pos_emb": pos_cand_emb,
            "neg_emb": neg_cand_emb,
            "corpus_ids": corpus_ids,
            "corpus_emb": corpus_emb,
        }

    # --- seeker-side: unique per (seeker_fields, use_query) (15 groups) ---
    seeker_configs = sorted(
        {(c.seeker_fields, c.use_query) for c in combos},
        key=lambda sc: (subset_id(sc[0]), sc[1]),
    )
    seeker_cache: dict[tuple[tuple[str, ...], bool], dict[str, Any]] = {}
    for i, (seeker_fields, use_query) in enumerate(seeker_configs, 1):
        sid = f"{subset_id(seeker_fields)}_{'q' if use_query else 'noq'}"
        print(f"[seeker {i}/{len(seeker_configs)}] fields={sid}")
        pos_seeker_texts = [
            seeker_to_text(r["userContactFile"], seeker_fields, r["searchQuery"], use_query)
            for r in positives
        ]
        neg_seeker_texts = [
            seeker_to_text(r["userContactFile"], seeker_fields, r["searchQuery"], use_query)
            for r in negatives
        ]
        pos_seeker_emb = encoder.encode(
            pos_seeker_texts, role="query", batch_size=batch_size, cache_name=f"pos_seeker_{sid}"
        )
        neg_seeker_emb = encoder.encode(
            neg_seeker_texts, role="query", batch_size=batch_size, cache_name=f"neg_seeker_{sid}"
        )
        seeker_cache[(seeker_fields, use_query)] = {
            "pos_texts": pos_seeker_texts,
            "neg_texts": neg_seeker_texts,
            "pos_emb": pos_seeker_emb,
            "neg_emb": neg_seeker_emb,
        }

    # --- score every combo from cached embeddings (cheap) ---
    results: list[dict[str, Any]] = []
    for i, combo in enumerate(combos, 1):
        cand = candidate_cache[combo.candidate_fields]
        seek = seeker_cache[(combo.seeker_fields, combo.use_query)]

        pos_scores = cosine_scores(seek["pos_emb"], cand["pos_emb"])
        neg_scores = cosine_scores(seek["neg_emb"], cand["neg_emb"])
        pair = pair_metrics(pos_scores, neg_scores)

        pos_target_ids = [r["matchContactId"] for r in positives]
        retrieval = retrieval_metrics(
            query_embs=seek["pos_emb"],
            target_ids=pos_target_ids,
            candidate_ids=cand["corpus_ids"],
            candidate_embs=cand["corpus_emb"],
        )
        slices = slice_metrics(
            positives=positives,
            negatives=negatives,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            neg_seeker_texts=seek["neg_texts"],
            neg_cand_texts=cand["neg_texts"],
            query_embs=seek["pos_emb"],
            target_ids=pos_target_ids,
            candidate_ids=cand["corpus_ids"],
            candidate_embs=cand["corpus_emb"],
        )

        results.append(
            {
                "combo_id": combo.combo_id,
                "seeker_fields": list(combo.seeker_fields),
                "candidate_fields": list(combo.candidate_fields),
                "use_query": combo.use_query,
                "pair": pair,
                "retrieval": retrieval,
                "slices": slices,
            }
        )
        if on_progress:
            on_progress(i, len(combos), combo)
        elif i % 10 == 0 or i == len(combos):
            print(f"  scored {i}/{len(combos)} combos")

    return results


def summarize(results: list[dict[str, Any]]) -> None:
    ranked = sorted(results, key=lambda r: -r["pair"]["roc_auc"])
    print("\n=== Top 10 by pair AUC ===")
    for r in ranked[:10]:
        h = r["slices"]["neg_hardness"]
        hard = h["hard"].get("pair_auc")
        easy = h["easy"].get("pair_auc")
        mrr = r["retrieval"]["mrr"]
        print(
            f"  {r['pair']['roc_auc']:.4f}  hard={hard if hard is None else round(hard, 4)}  "
            f"easy={easy if easy is None else round(easy, 4)}  mrr={mrr:.4f}  {r['combo_id']}"
        )
