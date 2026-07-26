#!/usr/bin/env python3
"""Embed each isolated profile field (and each ``lookingFor`` ask) for an rrf_* batch.

Companion to ``synth_pipeline/pairing_rrf/run.py``'s main embedding step, which
embeds whole profiles + section-swapped seeker variants. This script embeds a
third kind of row -- one field, or one ask, completely alone -- in the same
Qwen3-Embedding-8B space, so ``scripts/build_rrf_browser.py`` can plot isolated
fields next to the existing whole-profile/section anchors.

Reads the batch's ``run_summary.json`` for which profile-generation run and
embed model produced it, and the batch's ``embeddings/manifest.json`` for which
contact ids are seekers vs candidates (only those touched by the batch's pairs
are embedded -- no point isolating fields for profiles the batch never used).

Usage:
  python scripts/embed_rrf_field_isolation.py --batch-id rrf_002
  python scripts/embed_rrf_field_isolation.py --batch-id rrf_002 --backend local  # smoke test, small/free model only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth_pipeline.pairing.profiles import load_profile_run  # noqa: E402
from synth_pipeline.pairing_rrf import field_isolation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--backend", choices=("local", "modal"), default="modal",
                     help="modal runs on an A100 -- required for Qwen3-Embedding-8B")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args(argv)

    batch_dir = args.artifacts_dir / "pairing_rrf" / args.batch_id
    run_summary = json.loads((batch_dir / "run_summary.json").read_text(encoding="utf-8"))
    cfg = run_summary["config"]
    model_name = cfg["embed_model"]

    manifest_path = batch_dir / "embeddings" / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeker_ids = {r["contact_id"] for r in existing_manifest["seeker"]}
    candidate_ids = {r["contact_id"] for r in existing_manifest["candidate"]}

    profiles = load_profile_run(Path(cfg["profile_run"]))
    by_id = {p.contact_id: p.profile for p in profiles}
    missing = (seeker_ids | candidate_ids) - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} contact ids from {batch_dir} not found in "
                          f"{cfg['profile_run']} -- profile-run mismatch?")

    seekers = {cid: by_id[cid] for cid in seeker_ids}
    candidates = {cid: by_id[cid] for cid in candidate_ids}

    rows = field_isolation.build_rows(seekers, candidates)
    texts = [r.text for r in rows]
    n_field = sum(1 for r in rows if r.kind == "field")
    n_section = sum(1 for r in rows if r.kind == "section_alone")
    print(f"{len(seekers)} seekers + {len(candidates)} candidates -> "
          f"{len(texts)} isolated rows ({n_field} field-alone, {n_section} ask-alone)")

    if args.backend == "modal":
        from synth_pipeline.pairing_rrf.modal_embed import embed_isolated_via_modal
        vecs = embed_isolated_via_modal(
            texts, model_name=model_name, batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        from baselines.hf_embedding.encode import get_encoder_class
        encoder_cls = get_encoder_class(model_name)
        encoder = encoder_cls(model_name, device=None, max_length=args.max_length, truncate_dim=None)
        vecs = encoder.encode(texts, role="document", batch_size=args.batch_size)

    out_dir = batch_dir / "field_isolation"
    field_isolation.persist(out_dir, rows, vecs, model_name=model_name)
    print(f"wrote {out_dir} ({vecs.shape[0]} vectors, dim {vecs.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
