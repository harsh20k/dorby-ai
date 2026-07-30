"""Targeting recall@1 directly, even at the cost of pair AUC.

The problem
-----------
No fine-tune in this project has improved top-1 retrieval on real data. Measured
on all 200 real pairs (`docs/eval-real-full-experiment.md`):

    frozen voyage-4-nano   recall@1 = 0.1800  (18 of 100 queries)
    Arm A, fine-tuned      recall@1 = 0.1800  (+0.0000)
    Qwen3-8B micro-6       recall@1 = 0.1400
    Voyage-4-large         recall@1 = 0.1300

The best recall@1 anywhere in the project belongs to an **untrained** model, and
fine-tuning nano moved it by exactly zero. Meanwhile Arm A *did* improve the rest
of the ranking substantially — mean rank 17.9 → 12.1, median 7 → 5, recall@10
59 → 64 — so the model pulls the right person up the list without converting that
into first place.

Two causes, both addressed here
-------------------------------
**1. The loss never concentrated on the top competitor.** Every prior run called
`MultipleNegativesRankingLoss(model=model)` with pure defaults: `scale=20.0`,
`hardness_mode=None`, `hardness_strength=0.0`. The softmax temperature was never
tuned and the library's hard-negative weighting was never switched on, so
gradient spread evenly over all in-batch negatives — which pushes the positive
into the top-N, not to rank 1.

**2. Checkpoint selection was structurally blind to recall@1.** The ablation's
dev evaluator reports `beat_all_accuracy`: did the positive out-score *its own*
1-2 negatives. That is a triplet test over 2-3 candidates, while the real metric
ranks against ~178. The gap is not academic — it is why correcting checkpoint
selection in the ablation made holdout numbers *worse*: the selector was
optimising a different task, so its choice was near-arbitrary.

What this package changes
-------------------------
* `config.py` — `scale=50.0` (up from 20.0), `hardness_mode="hard_negatives"`,
  `hardness_strength=1.0`, and `primary_metric="recall@1"`.
* `eval_dev.py` — `CorpusRecallDevEvaluator` ranks every dev anchor against the
  full dev corpus and delegates scoring to `baselines.metrics.retrieval_metrics`,
  the same function the real holdout uses. Dev and holdout now compute recall@1
  identically by construction.

Everything else is held identical to Arm A (`abl_a_batch_only_v2`), the current
best nano configuration: same `rrf_003_multineg_k1.json` rows, micro-batch 6,
accum 2 (effective batch 12 → 245 optimizer steps), lr 2e-4, 5 epochs,
`save_total_limit=5`. Any difference is attributable to the two changes above.

Stated in advance, so it is not rationalised afterwards
-------------------------------------------------------
A peakier softmax is worse-calibrated, so **pair AUC is expected to drop**. That
is an accepted cost: this arm exists to find out whether recall@1 can be moved at
all. Judged on recall@1 against Arm A v2's 0.1800 (all 200 real pairs) and
0.3793 (holdout).

Isolation
---------
Nothing under `twotower/`, `twotower_rrf_triplet/`,
`twotower_rrf_triplet_bigbatch/`, `twotower_rrf_triplet_ablation/`, or
`twotower_qwen_bigbatch/` is modified. `data.py` is a byte-identical copy of the
ablation's (modulo the package rename) and is pinned by
`tests/test_top1_optimised.py`. `twotower.train`/`twotower.eval` helpers and all
of `baselines.metrics` are imported unchanged, so results stay comparable with
every prior run.

Caveat carried forward: `rrf_003`'s labels are an LLM judge's opinion on
synthetic profiles, not real accept/decline outcomes (judge accuracy on the hard
slice is 0.5942). Scoring is on the real pairs only, never on held-out synthetic.
"""
