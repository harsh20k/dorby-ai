"""voyage_gemini_ctrl's exact recipe, plus the KL-leash-to-frozen-base term
from `twotower_kl_reg/`, retrained on the bigger `pairing_voyage_gemini` batch.

`twotower_kl_reg` tested the KL-divergence-against-frozen-base idea
(`losses.py::KLRegularizedMNRL`, `kl_weight=0.5`) on `top1_ctrl`'s recipe and
`rrf_003` (643 rows, 583 train / 60 dev) — a small, cleaner synthetic batch —
and found a small net negative on all 200 real pairs: pair AUC 0.5504 vs
`top1_ctrl`'s 0.5683, every metric slightly worse (see
`docs/twotower-kl-reg-experiment.md`). This package asks the same question
again on `voyage_gemini_ctrl_001`'s recipe and data instead: does the KL leash
still hurt (or help) once the training population is 4.7x bigger and drawn
from a differently-shaped, measurably leakier pipeline
(`pairing_voyage_gemini/smoke_test_002`)? The prior negative result was
measured on one small batch; this repeats it on a much larger one to see
whether that conclusion generalizes or was specific to the small-data regime.

Two things are combined, each copied from its own source rather than
reimplemented:

1. **Recipe + data**: everything from `twotower_voyage_gemini_ctrl/` —
   LoRA rank 8 / alpha 16 / dropout 0.05 on q/k/v/o_proj (983,040 trainable
   params), micro-batch 6 / accum 2 (effective batch 12), lr 2e-4, 5 epochs,
   `voyage-4-nano` at native 1024-dim truncation, `CorpusRecallDevEvaluator`
   recall@1 checkpoint selection, full-profile text on both sides. Training
   rows: `artifacts/twotower_voyage_gemini_ctrl/
   voyage_gemini_smoke002_multineg_k1.json` (3,008 rows / 1,921 seekers / 0%
   padding), mounted read-only from that package's artifacts dir — never
   copied or regenerated, matching how `twotower_kl_reg` itself mounted
   `rrf_003_multineg_k1.json` read-only from `twotower_rrf_triplet_ablation`.
2. **Loss mechanism**: `twotower_kl_reg/losses.py::KLRegularizedMNRL` and its
   `train.py`'s custom training loop (needed because the KL term requires a
   second forward pass through the same PEFT model with its adapter toggled
   off via `disable_adapters()`/`enable_adapters()`, not `SentenceTransformer
   Trainer`'s stock loss interface) — copied verbatim, `kl_weight=0.5`
   unchanged, matching the prior experiment exactly so this is a true retest
   of the same idea, not a new one.

Isolated package: `data.py` and `eval_dev.py` are copies of
`twotower_voyage_gemini_ctrl`'s (pinned by
`tests/test_voyage_gemini_kl.py`), `losses.py` is a copy of
`twotower_kl_reg`'s (same pin file) — not imports, per the isolation rule.
Nothing under `twotower_voyage_gemini_ctrl/`, `twotower_kl_reg/`, or
`twotower/` is modified. Own Modal app (`dorby-twotower-voyage-gemini-kl`)
and own checkpoint volume
(`dorby-twotower-voyage-gemini-kl-checkpoints`) — shares only the read-only
row file and the project's common HF model-download cache volume.

**Leakage caveat carried over from `voyage_gemini_ctrl_001`, applies
identically here** (same source batch): `pairing_voyage_gemini/smoke_test_002`
is an unreviewed pilot batch that measured leakier than `rrf_003` before any
training happened — candidate-profile-only AUC 0.758, seeker-identity-only
AUC 0.780, 43.5% of seekers all-positive/all-negative across every query they
asked, vs. lexical circularity that stayed clean (TF-IDF query-candidate
cosine AUC 0.481, near chance). See
`scripts/leakage_check_pairing_voyage_gemini.py` and
`docs/twotower-voyage-gemini-ctrl-experiment.md`'s own caveats section.

`eval.py` in this package computes a **holdout-only** (69-pair) sanity check.
All-200 scoring (the number this project's standing rule says actually
counts) is deliberately out of scope here — `eval_real_full/guard.py` and
`eval_real_full/modal_eval.py` are registration points shared with a parallel
experiment and are handled separately, once both land, to avoid two agents
racing to edit the same files. See `docs/twotower-voyage-gemini-kl-
experiment.md` for the full writeup.
"""
