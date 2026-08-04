"""Isolated copy of ``synth_pipeline.pairing_rrf`` with a Qwen3-32B-on-Bedrock judge.

Per this repo's experiment-isolation rule, this is a duplicate-then-edit of
``synth_pipeline/pairing_rrf/`` rather than an in-place change — swapping the
judge model changes what any batch produced here means, and the original
package's batches (rrf_002, etc.) must stay reproducible from their own
unmodified code.

Retrieval (embedding + BM25 + RRF fusion) is identical to ``pairing_rrf``. Only
``judge.py`` differs: verdicts come from ``qwen.qwen3-32b-v1:0`` via AWS Bedrock
instead of ``google/gemini-3.1-flash-lite`` via OpenRouter — ~3x cheaper per
call (docs/llm-judge-experiment.md: ~$0.03/200 pairs vs ~$0.10/200), at the
cost of pair AUC 0.5802 vs 0.6358 (hard-negative AUC barely moves: 0.6224 vs
0.6466).

Labels remain a model's opinion, not real accept/decline outcomes. Batches stay
in their own ``artifacts/pairing_rrf_qwen_judge/<batch_id>/`` namespace and
nothing is promoted into ``data/dataset_*.json``.
"""

from __future__ import annotations

__all__ = [
    "embed",
    "fuse",
    "judge",
    "label",
    "prompt_hub",
    "query_gen",
    "recall",
    "run",
    "sections",
    "store",
]
