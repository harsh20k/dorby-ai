"""Qwen3-Embedding-8B fine-tuned on the pairing_voyage_gemini batch, at
twotower_qwen_bigbatch's winning micro-batch-6 recipe.

The question
------------
``twotower_qwen_bigbatch`` found, on ``rrf_003`` (583 synthetic rows), that
raising Qwen3-Embedding-8B's true micro-batch from 1 to 6 (effective batch held
at 12 in both arms) produced real gains that *generalised* to all 200 real
pairs — pair AUC +0.053, hard-negative AUC +0.104, MRR +0.100, recall@1
+0.100, recall@10 +0.080 against its own frozen 4096-dim baseline — unlike the
same micro-batch lever applied to voyage-4-nano, whose fine-tuning gains on
that batch did not generalise (+0.0001 pair AUC, +0.0000 recall@1). Micro-6 was
the clear winner between the package's two arms on every population measured
(all-200 pair AUC 0.5947 vs micro-1's 0.5604, MRR 0.3031 vs 0.2734, hard-neg
0.5608 vs 0.4828).

Separately, ``twotower_voyage_gemini_ctrl`` retrained voyage-4-nano's own
winning recipe (``top1_ctrl_001``) on a new, much bigger and measurably leakier
synthetic source — ``artifacts/pairing_voyage_gemini/smoke_test_002``
(3,008 multi-negative rows / 1,921 seekers, vs. rrf_003's 583/91) — and found
nano's gains again did not clearly generalise the way Qwen's did on rrf_003.

This experiment asks the natural next question: does Qwen's fine-tuning
advantage — real, generalizing gains, unlike nano's — carry over from
``rrf_003`` to this newer, bigger, leakier batch too? Nothing about the
question or the recipe changes; only the training rows do.

Design
------
Qwen3-Embedding-8B, LoRA rank 8 / alpha 16 on q/k/v/o_proj, bf16 weights +
gradient checkpointing (load-bearing for any micro-batch above 1 on an 8B
model), micro-batch 6 / accum 2 (effective batch 12), lr 1e-4 (Qwen's own
established rate, not nano's 2e-4), 5 epochs, H100 — copied verbatim from
``twotower_qwen_bigbatch/config.py``'s ``qwen3-8b`` preset and pinned against
it by ``tests/test_qwen_voyage_gemini.py``. The only variable is the training
rows: ``artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_smoke002_multineg_k1.json``
(3,008 rows / 1,921 seekers / 0% padding), mounted read-only from
``twotower_voyage_gemini_ctrl/``'s output rather than copied or regenerated —
that package already built it from ``artifacts/pairing_voyage_gemini/
smoke_test_002`` via ``scripts/build_rrf_multineg_triplets.py``, and rebuilding
it here would risk a second, silently-divergent copy of the same rows.

**Leakage caveat carried forward from `twotower_voyage_gemini_ctrl`'s own
findings on this batch** (measured before any training happened): candidate-
profile-only AUC 0.758, seeker-identity-only AUC 0.780 — both worse than
``rrf_003``'s. Lexical circularity is clean (TF-IDF query-candidate cosine AUC
0.481, near chance) — this is a base-rate/candidate-identity shortcut, not a
keyword-overlap one, and the query-level triplet format partially (not fully)
guards against it. Labels are an LLM judge's opinion on unreviewed, retrieved
synthetic profiles (``smoke_test_002`` is still a pipeline pilot batch), not
real accept/decline outcomes.

Isolation
---------
New top-level package. Nothing under ``twotower_qwen_bigbatch/``,
``twotower_voyage_gemini_ctrl/``, ``twotower/``, or ``baselines/`` is modified.
``data.py``, ``eval_dev.py``, ``model.py``, and ``checkpoint.py`` are
byte-identical copies of ``twotower_qwen_bigbatch``'s (modulo the package
rename), pinned as such by ``tests/test_qwen_voyage_gemini.py`` — same
discipline ``tests/test_qwen_bigbatch_copies.py`` already established one
package upstream. ``train.py`` is a near-copy of
``twotower_qwen_bigbatch/train.py``'s micro-6 path, with its CLI defaults
fixed to the single winning recipe (the upstream file's own CLI defaulted
to the *baseline* corner, requiring flag overrides for micro-6, and its
``main()`` had a latent bug building a ``"voyage-4-nano"`` preset that would
KeyError against this file's ``qwen3-8b``-only preset table — not hit there
because that arm was always launched through Modal, which builds its config
directly; fixed here rather than reproduced). Generic helpers from
``twotower.train``/``twotower.eval`` and all scoring in ``baselines.metrics``
are imported unchanged, so results stay comparable with every prior run in
this project. ``eval.py`` is self-contained: it reuses
``twotower.eval.run_eval_cli`` (model loading + ``baselines.metrics``)
directly for a real 69-pair holdout sanity check only — no all-200 eval here;
that is scored separately via ``eval_real_full/``, whose ``guard.py`` and
``modal_eval.py`` this package does not touch.

GPU cost note
-------------
An 8B backbone at micro-batch 6 on 3,008 rows (5.2x ``rrf_003``'s row count)
costs meaningfully more GPU time than either the nano runs in this project or
Qwen's own ``rrf_003`` run — budget accordingly before launching.
"""
