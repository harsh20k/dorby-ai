"""Qwen3-Embedding-8B fine-tune at a real micro-batch, not micro-batch 1.

The question
-----------
``twotower_rrf_triplet_ablation`` established, on voyage-4-nano at fixed
effective batch, that **micro-batch size is the lever that moves retrieval**:
raising it 2 → 6 gained +0.052 MRR and +0.069 recall@1, while the second
negative per row hurt. The mechanism is ``MultipleNegativesRankingLoss``'s
in-batch negatives, which scale with the *true* micro-batch, not with the
gradient-accumulated effective batch.

Qwen3-Embedding-8B has the highest pair AUC of any model measured in this
project (frozen 0.6595, above Voyage-4-large's 0.6086). Its single fine-tune,
``twotower_rrf_triplet/rrf_triplet_qwen3_8b_h100_002``, ran at
``train_batch_size=1`` — exactly the starved corner the ablation later showed to
be worst — and gained almost nothing: 0.6595 → 0.6672 pair AUC, inside the
measured ±0.013 noise band, with hard-negative AUC and recall@10 both going
*down*.

This experiment asks whether that flat result was the model or the batch size.

Why micro-batch 1 was never a measured ceiling
----------------------------------------------
The prior preset justified ``train_batch_size=1`` with an OOM observed at
**fp32 on a 40GB A100**. That is a weights-memory problem, and the same preset
already solves it twice over — ``torch_dtype=bfloat16`` (16GB resident instead
of 32GB) and ``gradient_checkpointing_override=True``. The ceiling was never
re-probed after those landed, and never on an 80GB card, where the binding
constraint is activations. ``probe_batch_size.py`` measures it rather than
inheriting the assumption.

Design
------
Two arms, both on the *identical* rows Arm A trained on
(``artifacts/twotower_rrf_triplet_ablation/rrf_003_multineg_k1.json``, mounted
read-only), both at effective batch 12 and therefore the same 245 optimizer
steps, both at Qwen's own established ``lr=1e-4``:

    qwen_micro1   micro-batch 1,  accum 12   control (prior run's setting)
    qwen_microN   micro-batch N,  accum 12/N treatment (N from the probe)

Only micro-batch differs, so the comparison reads exactly like Arm C vs Arm A.

Isolation
---------
Nothing under ``twotower/``, ``twotower_rrf_triplet/``,
``twotower_rrf_triplet_bigbatch/``, or ``twotower_rrf_triplet_ablation/`` is
modified. ``data.py`` and ``eval_dev.py`` are byte-identical copies of the
ablation's (modulo the package rename) and are pinned as such by
``tests/test_qwen_bigbatch_copies.py``. ``model.py`` and ``checkpoint.py`` are
copies of ``twotower_rrf_triplet``'s bf16 loading path, which nano never needed.
Generic helpers from ``twotower.train``/``twotower.eval`` and all scoring in
``baselines.metrics`` are imported unchanged, so results stay comparable with
every prior run.

Caveat carried forward
----------------------
``rrf_003``'s labels are an LLM judge's opinion on synthetic profiles, not real
accept/decline outcomes. Judge accuracy on the hard slice is 0.5942. Any model
trained here is scored on the **real** 69-pair holdout, never on held-out
synthetic pairs.
"""
