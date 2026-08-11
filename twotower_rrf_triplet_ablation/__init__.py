"""Ablation of the bigbatch experiment: which lever moved retrieval —
micro-batch size, or negatives per row?

`rrf_triplet_voyage_nano_bigbatch_001` changed both at once (plus effective
batch: gradient_accumulation_steps went 4->2 while train_batch_size went 2->6),
so its recall@1 gain (0.276 -> 0.345) could not be attributed. This package
runs the missing corners of a 2x2 grid with **effective batch held fixed at
12** in every arm, so micro-batch (which controls in-batch negatives for
MultipleNegativesRankingLoss) is separated from effective batch (which controls
update smoothness):

                    k=1                     k=2
    micro 2   abl_c_baseline          abl_b_negs_only
    micro 6   abl_a_batch_only        (bigbatch_001, already done)

All four corners run the identical 245 optimizer steps on the same 643
(anchor, positive) pairs, lr=2e-4, 5 epochs, voyage-4-nano, A100-80GB.

Isolation: this package imports nothing from twotower_rrf_triplet/ or
twotower_rrf_triplet_bigbatch/ (each is a prior experiment's frozen code) and
writes only under artifacts/twotower_rrf_triplet_ablation/. It reuses only
twotower/'s generic, unmodified helpers, exactly as both prior packages do.

Deliberate deviation from the one-package-per-run pattern: all three arms share
this single package rather than getting a copy each, because the arms must run
byte-identical code — separate copies could drift and confound the very
comparison this ablation exists to make. Per-arm isolation is preserved via
distinct run_ids, artifact directories, and run_meta.json configs.

See docs/twotower-rrf-triplet-ablation-experiment.md.
"""
