#!/usr/bin/env python3
"""Load the field-isolation embeddings (fetched from Modal) into a local Chroma DB.

Consumes artifacts/voyage_nano_field_isolation/{embeddings.npy,meta.json},
written by baselines/voyage_nano_field_isolation/modal_embed_space.py and
pulled down with `modal volume get`. Upserts every row (whole-profile,
field-alone, section-alone) into a persistent Chroma collection on disk --
open-source, no server, just a local directory -- so downstream steps
(the 3D viz, or any future nearest-neighbor query over this experiment)
read from the vector DB rather than re-touching the raw .npy.

Usage:
  python scripts/load_field_isolation_to_chroma.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "artifacts" / "voyage_nano_field_isolation"
DEFAULT_DB_DIR = DEFAULT_RUN_DIR / "chroma"
COLLECTION_NAME = "holdout_field_isolation_voyage_nano"


def _row_id(i: int, row: dict) -> str:
    parts = [str(i), row["contactId"], row["kind"]]
    if row.get("field"):
        parts.append(row["field"])
    if row.get("sectionIndex") is not None:
        parts.append(str(row["sectionIndex"]))
    return ":".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    ap.add_argument("--collection", default=COLLECTION_NAME)
    ap.add_argument("--reset", action="store_true", help="drop and rebuild the collection")
    args = ap.parse_args(argv)

    emb = np.load(args.run_dir / "embeddings.npy")
    meta = json.loads((args.run_dir / "meta.json").read_text())
    rows = meta["rows"]
    if len(rows) != emb.shape[0]:
        raise ValueError(f"meta rows ({len(rows)}) != embeddings ({emb.shape[0]})")

    contacts_meta = {c["id"]: c for c in meta["contacts"]}

    args.db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.db_dir))

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"model": meta["model_name"], "dim": meta["truncate_dim"]},
    )

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, row in enumerate(rows):
        cid = row["contactId"]
        cmeta = contacts_meta[cid]
        ids.append(_row_id(i, row))
        embeddings.append(emb[i].tolist())
        documents.append(row.get("text") or "")
        metadatas.append(
            {
                "contactId": cid,
                "kind": row["kind"],
                "field": row.get("field") or "",
                "sectionIndex": row["sectionIndex"] if row.get("sectionIndex") is not None else -1,
                "roles": ",".join(cmeta["roles"]),
                "pairCount": cmeta["pairCount"],
            }
        )

    # Chroma's HNSW add has a batch-size ceiling well above this row count,
    # but chunk anyway so this scales if the experiment grows.
    CHUNK = 500
    for start in range(0, len(ids), CHUNK):
        end = start + CHUNK
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"collection '{args.collection}' at {args.db_dir}: {collection.count()} vectors")
    kinds = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    for kind, n in sorted(kinds.items()):
        print(f"  {kind:14s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
