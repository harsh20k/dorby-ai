"""Local Chroma cache for this experiment, built from persisted ``.npy`` files.

``.npy`` arrays are the source of truth (exact retrieval metrics stay on
NumPy). Chroma is a queryable layer over the same vectors, isolated under
``artifacts/bdata_queryonly_back_look_frozen/chroma/``. Rebuildable without a GPU
after ``modal volume get``.

Do not import or edit ``synth_pipeline/pairing_rrf/store.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bdata_queryonly_back_look_frozen.config import DEFAULT_CONFIG, ExperimentConfig
from bdata_queryonly_back_look_frozen.data import load_pairs, retrieval_corpus

CANDIDATE_COLLECTION = "candidates"
QUERY_COLLECTION = "queries"
DOC_TRUNCATE = 500
DEFAULT_ADD_CHUNK = 4000


def unique_first_seen(texts: Sequence[str]) -> tuple[list[str], list[int]]:
    """Same first-seen collapse ``eval.encode_aligned`` uses."""
    unique: list[str] = []
    index: dict[str, int] = {}
    rows: list[int] = []
    for text in texts:
        if text not in index:
            index[text] = len(unique)
            unique.append(text)
        rows.append(index[text])
    return unique, rows


def expand_unique(unique_mat: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    if unique_mat.size == 0:
        return unique_mat
    return unique_mat[np.asarray(rows, dtype=np.int64)]


def chroma_dir_for(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "chroma"


def vectors_dir_for(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "vectors"


def unique_cache_path(artifacts_dir: Path, cache_name: str, n_unique: int) -> Path:
    return Path(artifacts_dir) / f"emb_{cache_name}_u{n_unique}.npy"


def _truncate(text: str) -> str:
    text = text or ""
    if len(text) <= DOC_TRUNCATE:
        return text
    return text[:DOC_TRUNCATE]


def _clean_meta(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _add_chunked(
    collection: Any,
    *,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
    max_batch: int,
) -> None:
    for start in range(0, len(ids), max_batch):
        end = start + max_batch
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def persist_vector_store(
    *,
    vectors_dir: Path,
    chroma_dir: Path,
    corpus_ids: Sequence[str],
    corpus_emb: np.ndarray,
    corpus_texts: Sequence[str],
    query_ids: Sequence[str],
    query_emb: np.ndarray,
    query_meta: Sequence[dict[str, Any]],
    query_texts: Sequence[str],
    model_name: str,
) -> dict[str, Any]:
    """Write aligned ``.npy`` catalog + isolated Chroma collections."""
    if len(corpus_ids) != len(corpus_emb):
        raise ValueError("corpus ids/embeddings length mismatch")
    if len(query_ids) != len(query_emb):
        raise ValueError("query ids/embeddings length mismatch")
    if len(corpus_ids) != len(corpus_texts):
        raise ValueError("corpus ids/texts length mismatch")
    if len(query_ids) != len(query_texts) or len(query_ids) != len(query_meta):
        raise ValueError("query ids/texts/meta length mismatch")

    vectors_dir = Path(vectors_dir)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_dir / "corpus.npy", np.asarray(corpus_emb, dtype=np.float32))
    np.save(vectors_dir / "queries.npy", np.asarray(query_emb, dtype=np.float32))
    _write_json(vectors_dir / "corpus_ids.json", list(corpus_ids))
    _write_json(
        vectors_dir / "queries.json",
        [
            {"id": qid, **{k: v for k, v in meta.items() if v is not None}}
            for qid, meta in zip(query_ids, query_meta)
        ],
    )
    manifest = {
        "model_name": model_name,
        "frozen": True,
        "packing": "seeker=query_only; candidate=background+lookingFor",
        "n_candidates": len(corpus_ids),
        "n_queries": len(query_ids),
        "dim": int(corpus_emb.shape[1]) if len(corpus_emb) else None,
        "source_of_truth": "vectors/*.npy (Chroma is a queryable cache)",
        "chroma_dir": str(chroma_dir),
    }
    _write_json(vectors_dir / "manifest.json", manifest)

    try:
        import chromadb
    except ImportError:
        print("chromadb not installed — wrote vectors/*.npy only")
        return {**manifest, "backend": "npy-only"}

    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    for name in (CANDIDATE_COLLECTION, QUERY_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    space = {"hnsw:space": "cosine"}
    candidates = client.create_collection(name=CANDIDATE_COLLECTION, metadata=space)
    queries = client.create_collection(name=QUERY_COLLECTION, metadata=space)
    max_batch = int(getattr(client, "get_max_batch_size", lambda: DEFAULT_ADD_CHUNK)())
    max_batch = max(1, min(max_batch, DEFAULT_ADD_CHUNK))

    _add_chunked(
        candidates,
        ids=list(corpus_ids),
        embeddings=np.asarray(corpus_emb, dtype=np.float32).tolist(),
        documents=[_truncate(t) for t in corpus_texts],
        metadatas=[
            _clean_meta({"role": "document", "packing": "background_lookingfor"})
            for _ in corpus_ids
        ],
        max_batch=max_batch,
    )
    _add_chunked(
        queries,
        ids=list(query_ids),
        embeddings=np.asarray(query_emb, dtype=np.float32).tolist(),
        documents=[_truncate(t) for t in query_texts],
        metadatas=[_clean_meta({"role": "query", **row}) for row in query_meta],
        max_batch=max_batch,
    )
    print(
        f"chroma at {chroma_dir}: {candidates.count()} candidates, "
        f"{queries.count()} queries"
    )
    return {
        **manifest,
        "backend": "chroma",
        "n_candidates": int(candidates.count()),
        "n_queries": int(queries.count()),
    }


def persist_vector_store_from_unique_cache(
    *,
    artifacts_dir: Path,
    corpus_ids: Sequence[str],
    corpus_texts: Sequence[str],
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    query_meta: Sequence[dict[str, Any]],
    model_name: str,
    corpus_cache_name: str = "corpus_cand",
    query_cache_name: str = "all_pos_seeker",
) -> dict[str, Any]:
    """Expand unique encode caches to id-aligned arrays, then write Chroma."""
    artifacts_dir = Path(artifacts_dir)
    c_unique, c_rows = unique_first_seen(corpus_texts)
    q_unique, q_rows = unique_first_seen(query_texts)
    c_path = unique_cache_path(artifacts_dir, corpus_cache_name, len(c_unique))
    q_path = unique_cache_path(artifacts_dir, query_cache_name, len(q_unique))
    if not c_path.is_file():
        raise FileNotFoundError(
            f"Missing corpus cache {c_path}. Pull Modal volume first:\n"
            "  modal volume get dorby-bdata-queryonly-back-look-frozen allpairs "
            "./artifacts/bdata_queryonly_back_look_frozen --force"
        )
    if not q_path.is_file():
        raise FileNotFoundError(f"Missing query cache {q_path}")
    corpus_emb = expand_unique(np.load(c_path), c_rows)
    query_emb = expand_unique(np.load(q_path), q_rows)
    return persist_vector_store(
        vectors_dir=vectors_dir_for(artifacts_dir),
        chroma_dir=chroma_dir_for(artifacts_dir),
        corpus_ids=corpus_ids,
        corpus_emb=corpus_emb,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_emb=query_emb,
        query_meta=query_meta,
        query_texts=query_texts,
        model_name=model_name,
    )


def _save_pair_arrays(
    vectors_dir: Path,
    *,
    pos: Sequence[Any],
    neg: Sequence[Any],
    pos_s: np.ndarray,
    pos_c: np.ndarray,
    neg_s: np.ndarray,
    neg_c: np.ndarray,
) -> None:
    vectors_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_dir / "pos_seeker.npy", np.asarray(pos_s, dtype=np.float32))
    np.save(vectors_dir / "pos_cand.npy", np.asarray(pos_c, dtype=np.float32))
    np.save(vectors_dir / "neg_seeker.npy", np.asarray(neg_s, dtype=np.float32))
    np.save(vectors_dir / "neg_cand.npy", np.asarray(neg_c, dtype=np.float32))
    _write_json(vectors_dir / "pos_ids.json", [p.pair_id for p in pos])
    _write_json(vectors_dir / "neg_ids.json", [p.pair_id for p in neg])


def persist_from_eval_arrays(
    *,
    cfg: ExperimentConfig,
    pos: Sequence[Any],
    neg: Sequence[Any],
    pos_s: np.ndarray,
    pos_c: np.ndarray,
    neg_s: np.ndarray,
    neg_c: np.ndarray,
    corpus_ids: Sequence[str],
    corpus_texts: Sequence[str],
    corpus_emb: np.ndarray,
) -> dict[str, Any] | None:
    """Called from eval after encode. Metrics stay on NumPy, not Chroma."""
    keep = [p for p in pos if p.match_contact_id in set(corpus_ids)]
    if not keep:
        print("chroma: no ACCEPT queries in corpus — skipping")
        return None
    keep_idx = [i for i, p in enumerate(pos) if p.match_contact_id in set(corpus_ids)]
    query_emb = pos_s[np.asarray(keep_idx, dtype=np.int64)]
    try:
        _save_pair_arrays(
            vectors_dir_for(cfg.artifacts_dir),
            pos=pos, neg=neg,
            pos_s=pos_s, pos_c=pos_c, neg_s=neg_s, neg_c=neg_c,
        )
        return persist_vector_store(
            vectors_dir=vectors_dir_for(cfg.artifacts_dir),
            chroma_dir=chroma_dir_for(cfg.artifacts_dir),
            corpus_ids=corpus_ids,
            corpus_emb=corpus_emb,
            corpus_texts=corpus_texts,
            query_ids=[p.pair_id for p in keep],
            query_emb=query_emb,
            query_meta=[
                {
                    "seeker_id": p.seeker_id,
                    "label": p.label,
                    "match_contact_id": p.match_contact_id,
                    "match_type": p.match_type,
                }
                for p in keep
            ],
            query_texts=[p.seeker_text for p in keep],
            model_name=cfg.model_name,
        )
    except Exception as exc:  # noqa: BLE001 — eval must still write metrics.json
        print(f"chroma persist failed ({exc!r}); vectors/metrics still valid")
        return {"backend": "failed", "error": repr(exc)}


def ingest_from_disk(cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    """Rebuild Chroma from Modal-pulled unique ``.npy`` caches. No GPU."""
    cfg = cfg or DEFAULT_CONFIG
    pairs, id_map, contacts, _meta = load_pairs(cfg)
    pos = [p for p in pairs if p.label == "ACCEPT"]
    neg = [p for p in pairs if p.label == "REJECT"]
    corpus_ids, corpus_texts = retrieval_corpus(contacts, id_map)
    corpus_set = set(corpus_ids)
    keep = [p for p in pos if p.match_contact_id in corpus_set]
    # Unique caches were encoded on ALL ACCEPT seekers, not the retrieval subset.
    pos_s = _expand_named_cache(
        cfg.artifacts_dir, "all_pos_seeker", [p.seeker_text for p in pos]
    )
    keep_idx = [i for i, p in enumerate(pos) if p.match_contact_id in corpus_set]
    query_emb = pos_s[np.asarray(keep_idx, dtype=np.int64)]
    corpus_unique, corpus_rows = unique_first_seen(corpus_texts)
    corpus_emb = expand_unique(
        np.load(unique_cache_path(cfg.artifacts_dir, "corpus_cand", len(corpus_unique))),
        corpus_rows,
    )
    info = persist_vector_store(
        vectors_dir=vectors_dir_for(cfg.artifacts_dir),
        chroma_dir=chroma_dir_for(cfg.artifacts_dir),
        corpus_ids=corpus_ids,
        corpus_emb=corpus_emb,
        corpus_texts=corpus_texts,
        query_ids=[p.pair_id for p in keep],
        query_emb=query_emb,
        query_meta=[
            {
                "seeker_id": p.seeker_id,
                "label": p.label,
                "match_contact_id": p.match_contact_id,
                "match_type": p.match_type,
            }
            for p in keep
        ],
        query_texts=[p.seeker_text for p in keep],
        model_name=cfg.model_name,
    )
    try:
        pos_s = _expand_named_cache(cfg.artifacts_dir, "all_pos_seeker", [p.seeker_text for p in pos])
        pos_c = _expand_named_cache(cfg.artifacts_dir, "all_pos_cand", [p.cand_text for p in pos])
        neg_s = _expand_named_cache(cfg.artifacts_dir, "all_neg_seeker", [p.seeker_text for p in neg])
        neg_c = _expand_named_cache(cfg.artifacts_dir, "all_neg_cand", [p.cand_text for p in neg])
        _save_pair_arrays(
            vectors_dir_for(cfg.artifacts_dir),
            pos=pos, neg=neg,
            pos_s=pos_s, pos_c=pos_c, neg_s=neg_s, neg_c=neg_c,
        )
        info["pair_arrays"] = True
    except FileNotFoundError as exc:
        print(f"pair-array caches missing ({exc}); chroma catalog still written")
        info["pair_arrays"] = False
    return info


def _expand_named_cache(artifacts_dir: Path, cache_name: str, texts: Sequence[str]) -> np.ndarray:
    unique, rows = unique_first_seen(texts)
    path = unique_cache_path(artifacts_dir, cache_name, len(unique))
    if not path.is_file():
        raise FileNotFoundError(path)
    return expand_unique(np.load(path), rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build isolated local Chroma DB from this experiment's .npy caches"
    )
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--unique-contacts", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = DEFAULT_CONFIG
    cfg = ExperimentConfig(
        source_path=args.source or base.source_path,
        unique_contacts_path=args.unique_contacts or base.unique_contacts_path,
        artifacts_dir=args.artifacts_dir or base.artifacts_dir,
        model_name=base.model_name,
        batch_size=base.batch_size,
        max_length=base.max_length,
        truncate_dim=base.truncate_dim,
        min_within_seeker_n=base.min_within_seeker_n,
        retrieval_query_batch_size=base.retrieval_query_batch_size,
        retrieval_ks=base.retrieval_ks,
    )
    info = ingest_from_disk(cfg)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
