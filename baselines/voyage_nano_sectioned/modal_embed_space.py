"""Modal GPU job: embed every holdout contact whole-profile *and* per-lookingFor-section.

Visualization-only experiment (no metrics, no scoring). For each unique contact
appearing in the frozen 69-pair real holdout it emits:

  * one "whole" embedding  -- profile_to_text(profile), searchQuery excluded
  * one "section" embedding per lookingFor paragraph -- the same full profile
    with lookingFor swapped to that single section (the exact text variant
    baselines/voyage_nano_sectioned/text.py already builds for scoring)

Everything is embedded in one encoder pass so all vectors share a space, then
written to a Modal volume as float32 .npy + a meta.json describing each row.
PCA / rendering happens locally in scripts/build_holdout_embedding_space_3d.py.

Usage:
  modal run baselines/voyage_nano_sectioned/modal_embed_space.py
  modal volume get dorby-sectioning-eval embed_space_holdout \
      ./artifacts/voyage_nano_sectioned_modal/embed_space_holdout
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-lookingfor-embed-space"
RESULTS_VOLUME = "dorby-sectioning-eval"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # reuse the existing HF model cache

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
    # Mount the three data files individually rather than add_local_dir("data"):
    # in a git worktree, data/ contains a `data` symlink back to the real repo
    # data dir, which add_local_dir would follow into infinite recursion.
    .add_local_file("data/dataset_positive.json", remote_path="/root/data/dataset_positive.json")
    .add_local_file("data/dataset_negative.json", remote_path="/root/data/dataset_negative.json")
    .add_local_file(
        "data/synthetic/seed_split.json",
        remote_path="/root/data/synthetic/seed_split.json",
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

PROFILE_FIELDS = [
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
]


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def embed_remote(
    run_id: str = "embed_space_holdout",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 8192,
    truncate_dim: int = 1024,
) -> dict:
    import json
    from pathlib import Path

    import numpy as np

    from baselines.bert_frozen.text import profile_to_text
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

    # Unique contacts across both sides of every holdout pair, with the roles
    # they play (a contact can be a seeker in one pair and a candidate in another).
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

    # Build the flat text list: whole-profile row then section rows, per contact.
    # searchQuery is deliberately never included -- this is a profile-space map.
    texts: list[str] = []
    rows: list[dict] = []
    for cid, entry in contacts.items():
        profile = entry["profile"]
        rows.append({"contactId": cid, "kind": "whole", "sectionIndex": None, "text": None})
        texts.append(profile_to_text(profile))

        looking_for = (profile.get("lookingFor") or "").strip()
        sections = split_looking_for_sections(looking_for) if looking_for else []
        for i, section in enumerate(sections):
            variant = dict(profile)
            variant["lookingFor"] = section
            rows.append(
                {
                    "contactId": cid,
                    "kind": "section",
                    "sectionIndex": i,
                    "text": section,
                }
            )
            texts.append(profile_to_text(variant))

    n_sections = sum(1 for r in rows if r["kind"] == "section")
    print(f"embedding {len(texts)} texts ({len(contacts)} whole + {n_sections} sections)")

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    encoder = VoyageNanoEncoder(
        model_name=model,
        device=pick_device(),
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )
    # role="document": profiles are being placed as points in a space, not used
    # as retrieval queries, so every row gets the same asymmetric prefix.
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
        "n_sections": n_sections,
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
        "n_sections": n_sections,
        "shape": list(emb.shape),
    }


@app.local_entrypoint()
def main(
    run_id: str = "embed_space_holdout",
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
    print("=== embed-space run finished ===")
    print(result)
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/voyage_nano_sectioned_modal/{result['run_id']}"
    )
