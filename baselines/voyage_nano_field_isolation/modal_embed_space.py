"""Modal GPU job: embed every holdout contact whole-profile, each profile field
*alone*, and each lookingFor section *alone*.

Sibling experiment to baselines/voyage_nano_sectioned/modal_embed_space.py.
That earlier run swapped lookingFor down to one section but kept every other
field in the text, so the "rest of the profile" always dominated the vector
and points barely moved. This run isolates instead of swaps: each row is
built from exactly one field's text and nothing else, so there is no shared
context left to dominate. For each unique contact in the frozen 69-pair real
holdout it emits:

  * one "whole" embedding    -- profile_to_text(profile), searchQuery excluded
  * one "field" embedding per non-empty profile field -- just "field: value",
    e.g. "positioning: ..." alone, no other field present
  * one "section_alone" embedding per lookingFor paragraph -- just
    "lookingFor: <that one paragraph>" alone (only emitted when lookingFor
    has more than one paragraph; the single-paragraph case is already
    covered by the "field" row for lookingFor)

Everything is embedded in one encoder pass so all vectors share a space, then
written to a Modal volume as float32 .npy + a meta.json describing each row.
Loading into a local Chroma vector DB and PCA/rendering happen locally in
scripts/load_field_isolation_to_chroma.py and
scripts/build_field_isolation_embedding_space_3d.py.

Usage:
  modal run baselines/voyage_nano_field_isolation/modal_embed_space.py
  modal volume get dorby-sectioning-eval embed_space_fields_holdout \
      ./artifacts/voyage_nano_field_isolation/embed_space_fields_holdout
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-field-isolation-embed-space"
RESULTS_VOLUME = "dorby-sectioning-eval"  # shared namespace with the sibling experiment, keyed by run_id
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("baselines", "synth_pipeline")
    .add_local_file("data/dataset_positive.json", remote_path="/root/data/dataset_positive.json")
    .add_local_file("data/dataset_negative.json", remote_path="/root/data/dataset_negative.json")
    .add_local_file(
        "data/synthetic/seed_split.json",
        remote_path="/root/data/synthetic/seed_split.json",
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def embed_remote(
    run_id: str = "embed_space_fields_holdout",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 8192,
    truncate_dim: int = 1024,
) -> dict:
    import json
    from pathlib import Path

    import numpy as np

    from baselines.bert_frozen.text import PROFILE_FIELDS, _nonempty, profile_to_text
    from baselines.holdout import filter_to_holdout
    from baselines.voyage_nano.encode import VoyageNanoEncoder, pick_device
    from baselines.voyage_nano_sectioned.text import split_looking_for_sections

    data_dir = Path("/root/data")
    positives = json.loads((data_dir / "dataset_positive.json").read_text())
    negatives = json.loads((data_dir / "dataset_negative.json").read_text())
    positives, negatives = filter_to_holdout(
        positives, negatives, data_dir / "synthetic" / "seed_split.json"
    )
    print(f"holdout: {len(positives)} positives, {len(negatives)} negatives")

    contacts: dict[str, dict] = {}

    def touch(cid: str, profile: dict, role: str) -> None:
        entry = contacts.setdefault(
            cid, {"id": cid, "profile": profile, "roles": set(), "pair_count": 0}
        )
        entry["roles"].add(role)
        entry["pair_count"] += 1

    edges: list[dict] = []
    for label, pairs in (("pos", positives), ("neg", negatives)):
        for p in pairs:
            touch(p["userContactId"], p["userContactFile"], "seeker")
            touch(p["matchContactId"], p["matchContactFile"], "candidate")
            edges.append(
                {
                    "source": p["userContactId"],
                    "target": p["matchContactId"],
                    "label": label,
                    "searchQuery": p.get("searchQuery"),
                }
            )
    print(f"unique contacts: {len(contacts)}")

    # Build the flat text list: whole-profile row, then one row per isolated
    # field, then one row per isolated lookingFor section, per contact.
    # searchQuery is deliberately never included -- this is a profile-space map.
    texts: list[str] = []
    rows: list[dict] = []
    for cid, entry in contacts.items():
        profile = entry["profile"]
        rows.append(
            {"contactId": cid, "kind": "whole", "field": None, "sectionIndex": None, "text": None}
        )
        texts.append(profile_to_text(profile))

        for field in PROFILE_FIELDS:
            value = profile.get(field)
            if _nonempty(value):
                rows.append(
                    {
                        "contactId": cid,
                        "kind": "field",
                        "field": field,
                        "sectionIndex": None,
                        "text": value.strip(),
                    }
                )
                texts.append(f"{field}: {value.strip()}")

        looking_for = (profile.get("lookingFor") or "").strip()
        if looking_for:
            sections = split_looking_for_sections(looking_for)
            if len(sections) > 1:
                for i, section in enumerate(sections):
                    rows.append(
                        {
                            "contactId": cid,
                            "kind": "section_alone",
                            "field": "lookingFor",
                            "sectionIndex": i,
                            "text": section,
                        }
                    )
                    texts.append(f"lookingFor: {section}")

    n_field = sum(1 for r in rows if r["kind"] == "field")
    n_section = sum(1 for r in rows if r["kind"] == "section_alone")
    print(
        f"embedding {len(texts)} texts "
        f"({len(contacts)} whole + {n_field} field-alone + {n_section} section-alone)"
    )

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    encoder = VoyageNanoEncoder(
        model_name=model,
        device=pick_device(),
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )
    # role="document": profiles/fields/sections are being placed as points in a
    # space, not used as retrieval queries, so every row gets the same
    # asymmetric prefix.
    emb = encoder.encode(texts, role="document", batch_size=batch_size, cache_name=None)
    emb = np.asarray(emb, dtype=np.float32)
    print(f"embeddings: {emb.shape}")

    np.save(artifacts_dir / "embeddings.npy", emb)
    meta = {
        "run_id": run_id,
        "model_name": model,
        "truncate_dim": truncate_dim,
        "max_length": max_length,
        "n_pairs": len(positives) + len(negatives),
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "n_contacts": len(contacts),
        "n_fields": n_field,
        "n_sections": n_section,
        "rows": rows,
        "contacts": [
            {
                "id": cid,
                "roles": sorted(e["roles"]),
                "pairCount": e["pair_count"],
                "profile": {k: e["profile"].get(k) for k in PROFILE_FIELDS},
            }
            for cid, e in contacts.items()
        ],
        "edges": edges,
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    results.commit()

    return {
        "run_id": run_id,
        "n_contacts": len(contacts),
        "n_fields": n_field,
        "n_sections": n_section,
        "shape": list(emb.shape),
    }


@app.local_entrypoint()
def main(
    run_id: str = "embed_space_fields_holdout",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 8192,
    truncate_dim: int = 1024,
    gpu: str = "L4",
) -> None:
    call = embed_remote if gpu == "L4" else embed_remote.with_options(gpu=gpu)
    result = call.remote(
        run_id=run_id,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    print("=== field-isolation embed-space run finished ===")
    print(result)
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/voyage_nano_field_isolation/{result['run_id']}"
    )
