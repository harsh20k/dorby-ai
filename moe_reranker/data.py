"""Leakage-safe data assembly for the MoE re-ranker.

Splits come from ``twotower.data.build_split_bundle``, the canonical
leakage-safe loader — this module does **not** roll its own split. Holdout is the
frozen ``eval_pair_ids``; train-dev is a further user-disjoint carve from train.
``assert_no_holdout_leak`` is called here as it is in ``twotower/train.py``.

Two things worth knowing before touching this:

**Cached embeddings are aligned by position.** ``artifacts/voyage_nano/`` holds
whole-profile embeddings for the 200 real seed pairs, written when
``dataset_positive.json``/``dataset_negative.json`` still held exactly those 200.
Promotion appends, so the real seed pairs are still the first 100 of each file —
verified here at load time by ``_load_nano``, which refuses to proceed if the
first 100 rows are not all real or the corpus size no longer matches. Without
that check a future promotion could silently shift every feature by one row.

**The auxiliary task is free.** The naive LLM judge's verdicts are already cached
for all 200 real pairs (``artifacts/llm_judge/.../verdicts.json``), keyed by
``label:userId:matchId|prompt_hash``. That gives a second, related target on the
same rows at zero cost — which is the whole reason the multi-gate architecture is
worth using here rather than plain MoE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from moe_reranker.config import MoEConfig
from moe_reranker.features import FeatureBuilder
from twotower.data import assert_no_holdout_leak, build_split_bundle

N_REAL_SEED = 100


@dataclass
class Split:
    name: str
    rows: list[dict[str, Any]]
    y: np.ndarray  # (N,) accept/decline
    aux: np.ndarray  # (N,) judge yes/no
    aux_mask: np.ndarray  # (N,) which rows have an auxiliary label
    seeker_ids: list[str]
    X: np.ndarray | None = None  # filled by build_features
    raw: np.ndarray | None = None
    seeker_emb: np.ndarray | None = None
    cand_emb: np.ndarray | None = None

    def targets(self) -> tuple[np.ndarray, np.ndarray]:
        """(N, 2) targets and mask, task 0 = accept, task 1 = judge."""
        t = np.stack([self.y, self.aux], axis=1).astype(np.float32)
        m = np.stack([np.ones(len(self.y), bool), self.aux_mask], axis=1)
        return t, m


def _load_nano(cfg: MoEConfig) -> dict[str, Any]:
    """Cached whole-profile nano embeddings + the positional-alignment guard."""
    pos = json.loads((cfg.data_dir / "dataset_positive.json").read_text())
    neg = json.loads((cfg.data_dir / "dataset_negative.json").read_text())

    def is_synth(r: dict[str, Any]) -> bool:
        return str(r["matchContactId"]).startswith("cmsynth") or str(
            r["userContactId"]
        ).startswith("cmsynth")

    real_pos, real_neg = pos[:N_REAL_SEED], neg[:N_REAL_SEED]
    if any(is_synth(r) for r in real_pos + real_neg):
        raise RuntimeError(
            "artifacts/voyage_nano embeddings are aligned to the first 100 rows of "
            "each dataset file being the real seed pairs, and that no longer holds. "
            "Re-encode instead of trusting the cache."
        )

    emb = {
        k: np.load(cfg.nano_artifacts / f"emb_{k}.npy")
        for k in ("pos_seeker", "pos_cand", "neg_seeker", "neg_cand", "corpus")
    }
    for k in ("pos_seeker", "pos_cand", "neg_seeker", "neg_cand"):
        if emb[k].shape[0] != N_REAL_SEED:
            raise RuntimeError(f"emb_{k} has {emb[k].shape[0]} rows, expected {N_REAL_SEED}")

    corpus_ids: list[str] = []
    seen: set[str] = set()
    for r in real_pos + real_neg:
        mid = r["matchContactId"]
        if mid not in seen:
            seen.add(mid)
            corpus_ids.append(mid)
    if len(corpus_ids) != emb["corpus"].shape[0]:
        raise RuntimeError(
            f"corpus embedding has {emb['corpus'].shape[0]} rows but the first "
            f"{N_REAL_SEED}+{N_REAL_SEED} real pairs contain {len(corpus_ids)} unique "
            "candidates — the cache no longer matches the data"
        )

    # (userContactId, matchContactId) -> (side, index) for positional lookup.
    index: dict[tuple[str, str], tuple[str, int]] = {}
    for i, r in enumerate(real_pos):
        index[(r["userContactId"], r["matchContactId"])] = ("pos", i)
    for i, r in enumerate(real_neg):
        index.setdefault((r["userContactId"], r["matchContactId"]), ("neg", i))

    return {
        "emb": emb,
        "index": index,
        "corpus_ids": corpus_ids,
        "corpus_idx": {c: i for i, c in enumerate(corpus_ids)},
    }


def _load_judge(path: Path) -> dict[tuple[str, str], float]:
    """(userId, matchId) -> 1.0 if the judge said yes. Prompt hash is stripped."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[tuple[str, str], float] = {}
    for key, verdict in raw.items():
        stem = key.split("|", 1)[0]
        parts = stem.split(":")
        if len(parts) != 3:
            continue
        _, user_id, match_id = parts
        match = str(verdict.get("match", "")).strip().lower()
        if match in ("yes", "no"):
            out[(user_id, match_id)] = 1.0 if match == "yes" else 0.0
    return out


def _tfidf_channel(
    all_rows: Sequence[dict[str, Any]], train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TF-IDF cosine + rank percentile, with the vocabulary fit on train rows only.

    Returns (cos, rank_pct, corpus_matrix) where rank_pct is this candidate's
    percentile position among all candidates for this seeker — the cheap stand-in
    for the stage-1 retrieval rank the MoE would see in production.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    seeker_texts = [seeker_to_text(r["userContactFile"], r.get("searchQuery")) for r in all_rows]
    cand_texts = [candidate_to_text(r["matchContactFile"]) for r in all_rows]

    # Fit on train rows only so holdout vocabulary/IDF never informs the model.
    fit_corpus = [t for t, m in zip(seeker_texts, train_mask) if m] + [
        t for t, m in zip(cand_texts, train_mask) if m
    ]
    vec = TfidfVectorizer(min_df=1, sublinear_tf=True).fit(fit_corpus)

    S = vec.transform(seeker_texts)
    C = vec.transform(cand_texts)
    Sn = S.multiply(1.0 / (np.sqrt(S.multiply(S).sum(axis=1)) + 1e-12))
    Cn = C.multiply(1.0 / (np.sqrt(C.multiply(C).sum(axis=1)) + 1e-12))
    Sn, Cn = Sn.tocsr(), Cn.tocsr()

    cos = np.asarray(Sn.multiply(Cn).sum(axis=1)).ravel()
    full = np.asarray((Sn @ Cn.T).todense())  # seeker x every candidate in play
    rank_pct = np.array(
        [(full[i] < full[i, i]).mean() for i in range(full.shape[0])], dtype=np.float64
    )
    return cos, rank_pct, full


def load(cfg: MoEConfig) -> dict[str, Split]:
    """Build train / train_dev / holdout splits with features attached."""
    bundle = build_split_bundle(
        cfg.data_dir, cfg.split_path, include_synth=cfg.include_synth
    )
    assert_no_holdout_leak(bundle, split_path=cfg.split_path)

    nano = _load_nano(cfg)
    judge = _load_judge(cfg.judge_verdicts)

    def to_split(name: str, pairs: Sequence[Any]) -> Split:
        rows, y, aux, aux_mask, seekers = [], [], [], [], []
        skipped = 0
        for lp in pairs:
            r = lp.pair
            key = (r["userContactId"], r["matchContactId"])
            if key not in nano["index"]:
                # A promoted synthetic pair has no cached nano embedding.
                skipped += 1
                continue
            rows.append(r)
            y.append(1.0 if lp.label == "pos" else 0.0)
            seekers.append(r["userContactId"])
            if key in judge:
                aux.append(judge[key])
                aux_mask.append(True)
            else:
                aux.append(0.0)
                aux_mask.append(False)
        if skipped:
            print(f"  {name}: skipped {skipped} pairs with no cached nano embedding")
        return Split(
            name=name,
            rows=rows,
            y=np.array(y, dtype=np.float32),
            aux=np.array(aux, dtype=np.float32),
            aux_mask=np.array(aux_mask, dtype=bool),
            seeker_ids=seekers,
        )

    splits = {
        "train": to_split("train", bundle.train),
        "train_dev": to_split("train_dev", bundle.train_dev),
        "holdout": to_split("holdout", bundle.holdout),
    }
    build_features(cfg, splits, nano)
    return splits


def build_features(
    cfg: MoEConfig, splits: dict[str, Split], nano: dict[str, Any]
) -> FeatureBuilder:
    """Attach standardized feature matrices, fitting only on the train split."""
    order = ["train", "train_dev", "holdout"]
    all_rows: list[dict[str, Any]] = []
    bounds: dict[str, tuple[int, int]] = {}
    for name in order:
        start = len(all_rows)
        all_rows.extend(splits[name].rows)
        bounds[name] = (start, len(all_rows))

    train_mask = np.zeros(len(all_rows), bool)
    train_mask[slice(*bounds["train"])] = True

    tfidf_cos, tfidf_rank, _ = _tfidf_channel(all_rows, train_mask)

    # Nano cosine + rank percentile from the cached embeddings.
    emb, index = nano["emb"], nano["index"]
    corpus, corpus_idx = emb["corpus"], nano["corpus_idx"]
    nano_cos = np.zeros(len(all_rows))
    nano_rank = np.zeros(len(all_rows))
    seeker_vecs = np.zeros((len(all_rows), corpus.shape[1]), dtype=np.float32)
    cand_vecs = np.zeros((len(all_rows), corpus.shape[1]), dtype=np.float32)

    for i, r in enumerate(all_rows):
        side, j = index[(r["userContactId"], r["matchContactId"])]
        s = emb[f"{side}_seeker"][j]
        c = emb[f"{side}_cand"][j]
        seeker_vecs[i], cand_vecs[i] = s, c
        nano_cos[i] = float(s @ c)
        against = corpus @ s
        nano_rank[i] = float((against < against[corpus_idx[r["matchContactId"]]]).mean())

    builder = FeatureBuilder(emb_pca_dims=cfg.emb_pca_dims)
    raw_all = builder.raw(
        all_rows,
        nano_cos=nano_cos,
        nano_rank_pct=nano_rank,
        tfidf_cos=tfidf_cos,
        tfidf_rank_pct=tfidf_rank,
    )

    ts, te = bounds["train"]
    builder.fit(
        raw_all[ts:te],
        seeker_emb=seeker_vecs[ts:te],
        cand_emb=cand_vecs[ts:te],
    )

    for name in order:
        a, b = bounds[name]
        sp = splits[name]
        sp.raw = raw_all[a:b]
        sp.seeker_emb = seeker_vecs[a:b]
        sp.cand_emb = cand_vecs[a:b]
        sp.X = builder.transform(
            raw_all[a:b], seeker_emb=seeker_vecs[a:b], cand_emb=cand_vecs[a:b]
        )
    return builder


def within_seeker_triplet_count(split: Split) -> tuple[int, int, int]:
    """(n_triplets, n_seekers_with_both, n_seekers) for a split.

    Reported by ``train.py`` because it is the honest limit on within-seeker
    training: the real pairs have almost no seeker carrying both classes, so the
    per-seeker base rate *cannot* be cancelled by construction here the way it
    can on ``rrf_002``. Diagnostic 3 is the fallback.
    """
    from collections import defaultdict

    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for sid, label in zip(split.seeker_ids, split.y):
        by[sid][0 if label > 0.5 else 1] += 1
    both = [s for s, (p, n) in by.items() if p and n]
    return sum(by[s][0] * by[s][1] for s in both), len(both), len(by)
