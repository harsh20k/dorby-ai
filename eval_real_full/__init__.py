"""Full-real-dataset evaluation: frozen voyage-4-nano vs. the Arm A LoRA adapter.

Why this is a separate top-level package rather than a flag on ``twotower.eval``
=============================================================================
Every published two-tower number in this repo is measured on the frozen 69-pair
holdout. That population has 29 positive queries, which forced the ablation to
report a recall@1 noise floor of exactly 1/29 = 0.0345 — one query. Conclusions
at that resolution are fragile.

This experiment asks a different question on a different population: **how do
frozen nano and the fine-tuned adapter compare across all 200 real pairs?**

That evaluation is legitimate here for a reason specific to Arm A, and it must
not be generalised to other runs:

    Arm A trained on ``exports/rrf_datasets/rrf_003`` — 583 rows built entirely
    from synthetic profiles. It saw **zero real pairs and zero real profiles**.

Verified before building this package:
  * ``rrf_003``'s profiles were generated with the v3+ profile-gen prompt, from
    which the ``{ref_example_1}`` real-profile seeding was removed in v2 (see
    ``docs/rrf-pairing-pipeline.md``). No real profile text conditioned
    generation.
  * None of the 297 real contact ids appears anywhere in the ``rrf_003``
    manifest.

So all 200 real pairs are out-of-sample for Arm A, and the 131 "train" pairs are
unseen by it in exactly the same sense as the 69 holdout pairs. **This does not
hold for ``twotower`` runs that trained on real pairs** (``run_001``,
``arm_a_real_only``); evaluating those here would leak. ``guard.py`` refuses
adapters not on the allowlist for that reason.

Isolation
---------
Nothing under ``twotower/``, ``twotower_rrf_triplet/``,
``twotower_rrf_triplet_bigbatch/``, ``twotower_rrf_triplet_ablation/``, or
``baselines/`` is modified. Those are imported read-only, and scoring goes
through ``baselines.metrics`` via ``twotower.eval.evaluate_pairs`` unchanged, so
numbers stay comparable with every prior run.

Input-data provenance
---------------------
``data/`` holds real contact profiles and is gitignored, so this package does
**not** copy that content into its own namespace the way the isolation rule
normally prescribes — doing so would commit real profile data to git. Instead
``freeze.py`` records pair ids plus per-pair SHA-256 digests, and ``--verify``
proves the source has not shifted since import. Provenance without exposure.

Read the corpus-size caveat in ``eval.py`` before comparing any retrieval metric
across subsets.
"""
