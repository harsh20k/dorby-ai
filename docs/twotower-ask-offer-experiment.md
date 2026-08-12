# Ask Tower, Offer Tower — training two towers jointly on the reciprocal score

**Status: run complete (`ask_offer_001`, voyage-4-nano, Modal A100-80GB).**
Mixed result, and a genuinely different one from the zero-training version of
this idea. **On the more trustworthy all-200 population, the reciprocal term
does real, meaningful work** (pair AUC 0.5126 → 0.5714, +0.059) — but the
trained forward-only tower it's rescuing is weaker than the existing
single-tower champion, and weaker than frozen Voyage-4-nano itself on
all-200. **On the 69-pair holdout, the reciprocal term does nothing**
(0.6138 → 0.6121, essentially flat/slightly negative). Net: this design does
not currently beat `voyage_gemini_ctrl`'s combined score (0.6587 all-200 /
0.7345 holdout, `docs/reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md`)
or its own forward-only number, despite training two dedicated towers instead
of reusing one frozen model. Treat this as a real, informative negative
result, not a bug to explain away — see "Reading the result" below for what
it does and doesn't tell us.

Design doc (written before this ran, kept as the audit trail):
[`docs/html/reciprocal-two-tower-training-plan.html`](https://claude.ai/code/artifact/373941ab-0755-4948-af61-9eba1fd34bc8).
Builds on `baselines/reciprocal_static/` (zero-training version of this
score, `docs/reciprocal-static-experiment.md`).

A post-hoc, no-fitting λ sweep on top of these same two towers
(`reciprocal_lambda_grid_ask_offer/`, same mechanics as the other three
`reciprocal_lambda_grid*` sweeps) is plotted alongside them in
[`docs/html/reciprocal-lambda-grid.html`](https://claude.ai/code/artifact/c522e5a2-a1fb-4a77-8f96-ed8e9c403431):
holdout curve peaks at λ=0.15 (0.6138→0.6293, a small genuine bump near the
trained λ=1.75's flat result); all-200 curve rises to the grid boundary
(λ=1.90, 0.5126→0.5723) without a real interior peak, same "not a peak, a
shelf" caveat as `voyage_gemini_ctrl`'s sweep.

## What was built

New isolated package **`twotower_ask_offer/`** — nothing under `twotower/`,
`baselines/reciprocal_static/`, or `twotower_voyage_gemini_ctrl/` touched.

| File | Role |
|---|---|
| `import_rows.py` | Freezes the exact 3008 rows `voyage_gemini_ctrl_001` trained on, resolved back to raw profile dicts (source is `pairing_voyage_gemini/smoke_test_002`'s staged pairs, read-only, never read live at train time) — same population, so this experiment differs from `voyage_gemini_ctrl` in exactly one place: two towers + the reciprocal loss, not the data too. |
| `config.py` | `AskOfferConfig`: wraps one shared `TrainConfig` (LoRA rank 8, batch 6, grad-accum 2, lr 2e-4 — identical to `voyage_gemini_ctrl`'s recipe) plus `lam` (fixed, default 1.75) and training knobs. |
| `data.py` | Loads frozen rows, seeker-disjoint train/dev carve (same convention as every other `twotower_*` package). |
| `model.py` | Builds two independent LoRA adapters from the same frozen base (`build_two_towers`); a differentiable encode helper with manual prompt-prepending, since prompt handling outside `SentenceTransformerTrainer` isn't automatic. |
| `loss.py` | The one genuinely new piece of math: `combine_and_cross_entropy` builds an N×2N score matrix `S = s_fwd + λ·s_rev` (positives then explicit negatives, same in-batch-negative shape as today's `MultipleNegativesRankingLoss` recipe) and applies cross-entropy on the diagonal. Pinned by 5 tensor-only unit tests, no model needed. |
| `eval_dev.py` | Corpus-recall dev evaluator — ranks each dev seeker against every unique dev candidate using the *combined* score, not plain cosine, so checkpoint selection tracks what the loss actually optimizes. |
| `train.py` | Hand-rolled training loop (not `SentenceTransformerTrainer` — two independently-weighted models optimized jointly against one shared-output loss doesn't fit that abstraction). Per-epoch checkpoint + dev eval + best-checkpoint-by-recall@1 selection, same discipline as every other package here (never silently ship the final epoch — `docs/possible-bugs.md` #2). |
| `eval.py` | Real holdout/all-200 scoring, same metric shape as `baselines/reciprocal_static/eval.py`, reusing `baselines.metrics` unchanged. |
| `modal_train.py` / `modal_eval.py` | Modal A100-80GB entrypoints (two towers + optimizer state concurrently needs more memory than a single-tower run; local MPS smoke-tested the mechanism only, then OOM'd on the real loop as expected — confirming this needs Modal, not proving a bug). |

12 unit tests pass (`tests/test_twotower_ask_offer.py`): loss-matrix shape and
label correctness, a hand-computed-formula match, a perfect-pool sanity check
(near-zero loss when the true pair is a trivial match), shape-mismatch and
too-small-pool error paths, a text-reuse pin against
`baselines.reciprocal_static.text` (imported, never duplicated), and config
sanity.

**Known scope simplification:** no mixed precision (fp32 throughout) — noted
rather than silently claimed to match the Trainer-based packages' bf16/fp16
handling. LoRA on `voyage-4-nano` over ~2700 rows / 5 epochs was cheap enough
on an A100-80GB that this didn't block the run.

## What each tower saw

Same split as `reciprocal_static` and the design doc, reused read-only
(`baselines.reciprocal_static.text`):
- **Ask tower**: `lookingFor` only. Seeker side also gets `searchQuery`.
- **Offer tower**: `positioning` + `background` only.
- Every other field (notes, preferences, scheduling) — unused, same
  narrowing this repo's field-pair experiments already used.

## Training

2710 train rows / 298 dev rows (2710 rows / 452 steps per epoch, batch 6,
grad-accum 2 — effective batch 12, matching `voyage_gemini_ctrl_001`). Train
loss fell from ~1.0–1.5 (epoch 1) to ~0.05–0.4 (epoch 5), smoothly, no
divergence. Dev recall@1 by epoch (evaluated against the combined score over
a 462-candidate dev corpus):

| epoch | dev recall@1 | dev MRR |
|---|---|---|
| 2 | 0.359 | 0.563 |
| 3 | **0.369** | 0.557 |
| 4 | 0.359 | 0.538 |

Best-checkpoint selection picked **epoch 3** — the same early-peak-then-decay
pattern this project has hit before (e.g. the `voyage_gemini_ctrl`
checkpoint-1130 sweep), and exactly why checkpoint selection exists rather
than shipping the final epoch by default.

## Results

`λ = 1.75` (fixed at training time, never touched real labels — see "λ:
fixed, not learned" in the design doc).

| population | n (pos/neg) | corpus | pair AUC fwd-only | pair AUC combined | Δ | hard-neg AUC (combined) | easy-neg AUC (combined) | MRR (fwd-only retrieval) | R@1 |
|---|---|---|---|---|---|---|---|---|---|
| holdout | 69 (29/40) | 65 | 0.6138 | 0.6121 | **−0.0017** | 0.6379 | 0.6828 | 0.5324 | 0.3103 |
| all-200 | 200 (100/100) | 178 | 0.5126 | **0.5714** | **+0.0588** | 0.5236 | 0.6952 | 0.4317 | 0.2700 |

## Reading the result

**The all-200 gain is real and in the direction the design predicted; the
holdout result is not.** This project's own established finding
(`docs/baseline-results-real200.md`, cited in `CLAUDE.md`) is that the
69-pair holdout carries no reliable information among strong models — Spearman
correlation with the all-200 ranking is −0.029 across the top 6 models tested
so far. This result adds a data point consistent with that: holdout says
"the reciprocal term did nothing," all-200 says "it added 0.059 AUC," and
per this project's own prior, all-200 is the one to trust.

**But the more important number is the forward-only tower itself, and it's
weak.** 0.5126 all-200 AUC is close to chance, and *below* frozen
Voyage-4-nano's own forward score in `reciprocal_static` (0.5638) — training
two dedicated towers on narrower text (one field each) did not produce a
better forward signal than the frozen, zero-training, full-field split. The
combined score's 0.5714 is doing real work rescuing a weak base, not adding
polish to a strong one. The likely cause isn't "two towers" per se — this
project's field-pair experiments already found that narrowing seeker/candidate
text to fewer fields costs general ranking ability even when it helps
hard-negative resistance (`docs/twotower-field-pairs-experiment.md`) — and
this run narrows further, to a single field per tower.

**Against the reigning champion:** `voyage_gemini_ctrl` (one shared tower,
full-profile text, the *existing* fine-tuning recipe) reaches 0.6108 all-200 /
0.7164 holdout forward-only alone, and 0.6587 / 0.7345 combined with a
post-hoc (zero-training) reciprocal sweep on top
(`docs/reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md`). This
experiment's combined score (0.5714 / 0.6121) does not beat that on either
population. **The untied, jointly-trained ask/offer design does not currently
outperform the cheaper alternative — one tower, full-profile text, reciprocal
term applied post-hoc for free.**

**hard-neg < easy-neg on all-200** (0.5236 vs 0.6952) — consistent with, not
contradicting, `reciprocal_static`'s own all-200 slice (0.5478 vs 0.7208).
Both zero-training and trained versions of this score are better at ranking
lexically-distant negatives than lexically-close ones; that pattern didn't
change with training.

## What this actually resolves, and what it doesn't

Confirms the mechanism is buildable and trains stably (loss falls smoothly,
grads reach both towers, checkpoint selection works, real holdout/all-200
eval runs cleanly) — the plumbing risk flagged in the design doc's odds
estimate ("~1-in-4 chance it beats the current cold result and holds up
rather than being another mirage") didn't materialize as a plumbing failure;
it materialized as the predicted risk itself: more capacity on the same small
data, without a clear win. Does **not** resolve whether untying the encoders
specifically was the wrong move, versus the narrower single-field text being
the wrong move — those are confounded here, same as the design doc warned
untying would be a third variable if capacity kept growing without isolating
it. A cleaner next step would test one field-widening or one architecture
change at a time, not both together as this run did relative to
`reciprocal_static`.

## Repro

```bash
python -m twotower_ask_offer.import_rows            # freeze rows (idempotent, ~1s)
python -m twotower_ask_offer.import_rows --verify    # confirm frozen copy matches source

modal run --detach twotower_ask_offer/modal_train.py --run-id ask_offer_001
modal volume get dorby-twotower-ask-offer-checkpoints ask_offer_001 \
    ./artifacts/twotower_ask_offer/ask_offer_001

modal run twotower_ask_offer/modal_eval.py --run-id ask_offer_001   # holdout + all-200

pytest tests/test_twotower_ask_offer.py -q
```

## Next steps (not done here)

- **Isolate the confound.** Either widen each tower back toward multi-field
  text (closer to `voyage_gemini_ctrl`'s full-profile input, still split by
  role) while keeping two towers, or keep the single-field split but tie the
  two towers back to one shared model — either change alone would tell us
  which variable actually cost the forward-only signal.
- **Try the calibration eq. 101** (`w1·s_fwd + w2·s_recip + b`, fit on
  train-dev) instead of the single fixed `λ` — cheap, reuses this run's saved
  embeddings, no retraining needed.
- **A larger or less-leaky training batch.** `pairing_voyage_gemini`'s
  `smoke_test_002` is the same batch `voyage_gemini_ctrl` used, already
  flagged as measurably leakier than `rrf_003`
  (`docs/twotower-voyage-gemini-ctrl-experiment.md`) — worth knowing whether
  that's suppressing or inflating either arm's numbers before trusting either
  too far.
