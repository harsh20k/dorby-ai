#!/usr/bin/env python3
"""Worker: embed one half of a text list on Modal, save to a .npz file.

Invoked as a subprocess by embed_two_accounts.py with its own MODAL_TOKEN_ID /
MODAL_TOKEN_SECRET env vars set, so two halves can run concurrently on two
separate Modal accounts. Not part of any experiment package — this is a
one-off operational speedup for a single run, not a reusable pipeline stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="JSON: {seeker_texts, candidate_texts}")
    p.add_argument("--output", type=Path, required=True, help="where to write the .npz result")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=4096)
    args = p.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    seeker_texts = payload["seeker_texts"]
    candidate_texts = payload["candidate_texts"]
    print(f"[worker {args.output.stem}] embedding {len(seeker_texts)} seeker + "
          f"{len(candidate_texts)} candidate texts", flush=True)

    from synth_pipeline.pairing_rrf_qwen_judge.modal_embed import embed_via_modal

    seeker_mat, cand_mat = embed_via_modal(
        seeker_texts,
        candidate_texts,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    import numpy as np

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, seeker=seeker_mat, candidate=cand_mat)
    print(f"[worker {args.output.stem}] done: seeker {seeker_mat.shape} candidate {cand_mat.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
