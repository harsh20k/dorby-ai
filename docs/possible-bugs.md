# Possible bugs

Running log of suspected pipeline/data issues found during review, not yet
root-caused or fixed. Add an entry when something looks wrong but hasn't
been confirmed/patched; move it out (or mark fixed) once resolved.

## 1. Long seeker `lookingFor` gets truncated inconsistently with `searchQuery` in negative generation

**Where:** `synth_pipeline/nodes/generate.py` (negative-generation path),
prompt `synth_pipeline/prompts/generate_neg.md`.

**Found in:** `artifacts/synth/batch_500_001/staged/neg_cmsynthtcfm6yqmmq5rxb5lif.json`
(seeker `cmsynth2gb2fna51xgpcxuu1w`, seeded from real pair
`neg:cmohrqn4800kimu02s837gmav:...` in `data/dataset_negative.json`).

**What's wrong:** The prompt instructs the model to "keep the seeker's
`userContactFile` and `searchQuery` essentially the same as the seed." The
`searchQuery` was copied verbatim from the seed ("hands-on science
researchers founders or CSOs building biotech and deep-tech companies...").
But the seed's `lookingFor` field is a sprawling ~15-section field
(accumulated over many real update timestamps) covering real estate/car
wash funding, growth roles, *and* a "Science-First Biotech Founders and
CSOs" section that the query is drawn from almost word-for-word.

The generated pair's `userContactFile.lookingFor` kept only 4 of the ~15
sections (Commercial Real Estate, Growth Roles, Car Wash Funding, Real
Estate Capital) and dropped the biotech section — the one section that
actually justifies the `searchQuery`. Result: a seeker profile with no
stated interest in biotech, paired with a biotech-seeking query.

**Why it matters:** The judge still labeled the pair a valid negative, but
for the wrong reason — its own `judge_raw.reason` says "zero alignment
between the user's search query... and the user's actual professional
focus," failing role/side/stage/prefs axes. The pair was tagged
`failure_mode: wrong_stage` (Eleanor's Series B stage vs. Max's early-stage
ask), but the actual defect is seeker-query inconsistency introduced during
generation, not a genuine stage mismatch on the match side. Training on
this teaches "reject when query doesn't match profile" (an artifact of
generation) rather than the intended failure-mode semantics.

**Suspected root cause:** very long/multi-section `lookingFor` fields
likely push the generation LLM toward summarizing/compressing rather than
copying verbatim, since the instruction is qualitative ("essentially the
same") with no explicit length/fidelity constraint.

**Suggested next step:** grep `data/dataset_negative.json` /
`dataset_positive.json` for seed users with unusually long `lookingFor`
strings, cross-reference which batch_500_001 pairs were generated from
those seeds, and spot-check whether the same section-dropping pattern shows
up elsewhere. If confirmed widespread, consider passing `lookingFor`
through unmodified (string substitution rather than LLM regeneration) for
the seeker side, since the prompt already says it should stay unchanged.

**Status:** unconfirmed at scale — one instance found via manual review
(2% sample). Human-approved as staged despite the defect (2026-07-19); flag
for exclusion or fix before scaling generation further.

## 2. `select_best_checkpoint` silently falls back to the final epoch instead of the best one

**Where:** `twotower/train.py::select_best_checkpoint()`.

**Found in:** `artifacts/twotower/run_001/run_result.json` — full 5-epoch
run, see `docs/twotower-run-001-results.md`.

**What's wrong:** `run_training()` is supposed to reload whichever epoch
checkpoint scored best on `train_dev_{primary_metric}` (default
`pair_auc`), by reading `train_dev_metrics_epoch*.json` files that
`PairMetricsEvaluator.__call__` is meant to write under `output_path` on
each epoch-end eval. In `run_001`, epoch 3 scored highest (`pair_auc:
0.989`) but `select_best_checkpoint` returned `{"source":
"final_in_memory", "reason": "no_metric_files"}` — it found zero matching
files and silently used whatever was in memory at the end of training
(epoch 5, `pair_auc: 0.986`), instead of raising or warning loudly.

**Why it matters:** the "pick best checkpoint" mechanism exists
specifically to avoid shipping an overfit/regressed final epoch — see the
epoch-3-vs-epoch-5 divergence between pair metrics (still climbing) and
retrieval metrics (declining since epoch 2) in
`docs/twotower-run-001-results.md`. Right now it's a no-op that happens to
land close to the true best (0.986 vs 0.989 AUC) by luck, not by design.
On a run where epochs diverge further, this would silently ship a
materially worse adapter with a `run_result.json` that looks like the
selection logic ran successfully.

**Suspected root cause:** likely one of — (a) `SentenceTransformerTrainer`
doesn't pass `output_path` to the evaluator on training-time epoch-end
calls (only on an explicit standalone `evaluate()`), so
`PairMetricsEvaluator` never writes the file in the first place; or (b) HF
Trainer passes `epoch` as a float (e.g. `1.0`) to the evaluator, producing
filenames like `train_dev_metrics_epoch1.0.json` that don't parse cleanly
against `select_best_checkpoint`'s `int(stem.rsplit("epoch", 1)[-1]...)`
parsing, even if the glob itself still matches.

**Suggested next step:** add a print/log inside
`PairMetricsEvaluator.__call__` confirming whether `output_path` is
non-`None` during training (not just standalone eval calls), and check
`artifacts/twotower/run_001/checkpoints/` for any
`train_dev_metrics_epoch*` files that did get written but weren't found by
the glob (naming/parsing mismatch) vs. none being written at all (missing
`output_path`). Fix before running holdout eval, since the two-tower plan's
decision gate treats holdout as a one-time check — it should run against a
provably-best checkpoint, not the last one by accident.

**Status:** confirmed in `run_001` (2026-07-19); low impact this run (0.3
AUC points), but the safety mechanism itself is broken and unverified at
larger epoch counts or noisier runs.

## 3. Baseline vs. twotower comparison is currently invalid (population, split, and length mismatches)

**Where:** `baselines/*/eval.py::load_pairs()` vs.
`twotower/data.py::build_split_bundle()`.

**What's wrong:** the cached baseline metrics
(`artifacts/{bert_frozen,voyage_nano,voyage_large}/metrics.json`) were
computed by reading `data/dataset_positive.json`/`dataset_negative.json`
directly, with no train/holdout split applied — at the time, those files
held only the original 200 real seed pairs (100/100). Since then, both
files grew to 660 pairs via `promote.py` (320/340, including 460 promoted
synthetic pairs). A naive re-run of the baseline scripts today would now
silently score against a totally different, synth-polluted population
without any code change or warning. Separately, the twotower training run
used `max_seq_length=4096` while the cached baselines used `8192`, and the
first twotower numbers available are on a 61-pair train-dev slice (which
is ~80% synthetic) rather than the 69-pair frozen real holdout — three
independent axes of mismatch stacked into what looks like a single
apples-to-apples table.

**Why it matters:** the raw comparison in
`docs/twotower-run-001-results.md` shows twotower pair AUC (0.986) nearly
2x the production Voyage-4-large baseline (0.573) — a genuinely alarming-
looking number if read at face value. In reality it mostly reflects the
fine-tune scoring well against data statistically close to its own
training distribution, not evidence it beats production Voyage-4-large on
real matching quality.

**Suggested next step:** build a holdout-only baseline rerun (filter to
`twotower/data.py::build_split_bundle().holdout`, the frozen 69-pair set)
at matched `max_seq_length`, then extend
`scripts/export_baseline_results.py`'s hardcoded `BASELINES` tuple to also
read twotower's `metrics_holdout.json` (different path convention:
`artifacts/twotower/<run_id>/metrics_holdout.json` vs. the other
baselines' flat `artifacts/<name>/metrics.json`).

**Status:** partially resolved 2026-07-19 — ran `twotower/eval.py --split
holdout` directly against the real 69-pair frozen holdout (matched
population on the twotower side; baseline rows still pending a matched
rerun). Result: twotower holdout pair AUC 0.578 vs. Voyage-4-large 0.573 —
essentially tied, not the ~2x gap the train-dev table implied. See
`docs/twotower-run-001-results.md` for the full table. This confirms the
comparison-invalidity concern was real: the raw table was misleading.
Baseline rows in the corrected table are still the old 200-pair/8192
cached numbers, not yet rerun on the matched 69-pair/4096 population —
close this out once that rerun happens.

## 4. `run_001` LoRA adapter likely overfit to synthetic-generation artifacts, not real matching semantics

**Where:** training data composition (`data/dataset_negative.json` /
`dataset_positive.json`, synthetic portion) + `twotower/train.py`.

**Found in:** `docs/twotower-run-001-results.md` — real holdout eval vs.
train-dev eval for the same `run_001` adapter, no retraining between the
two.

**What's wrong:** train-dev pair AUC (0.986, 61 pairs, 50 synthetic) vs.
real holdout pair AUC (0.578, 69 real pairs) is an enormous, unexplained-
by-noise gap for the same model weights. Worse, the real-holdout
hard-negative slice AUC is 0.4845 — below chance — meaning on real
negatives that are lexically close to the query (exactly what the
synthetic hard negatives were designed to simulate), the fine-tuned model
performs worse than a coin flip. Train and train-dev pull 410/530 and
50/61 pairs respectively from the same LLM-generated distribution
(`generate_pos.md`/`generate_neg.md` via `synth_pipeline`); the model
appears to have learned to distinguish which generation prompt produced a
given profile rather than learning role/stage/side/geo/prefs matching
semantics.

**Supporting evidence:** grepped all 240 synthetic negatives in
`data/dataset_negative.json` for explicit generator meta-commentary
phrases ("critical distinction," "mistaken for," "has never," "is a
mismatch," etc.) — 9/240 (3.75%) contain one outright (e.g. the
Alex/ScaleWorks Capital pair, `docs/possible-bugs.md` context from manual
review: "*frequently mistaken for an operator... has never held an
operating role... critical distinction for intro matching*" — the
generator wrote the answer directly into the profile text). This is very
likely the visible tip of a larger, subtler pattern of structural/stylistic
differences between the two generation prompts that a small LoRA adapter
(983K trainable params, 5 epochs, contrastive loss driving training loss to
~0.004) can pick up as a shortcut well before learning the intended
semantic distinctions.

**Why it matters:** this is the central finding of the `run_001`
experiment. It means the ~660-pair promoted dataset, as currently
generated, does not straightforwardly produce a better matcher than the
frozen production baseline — more epochs or more of the same synthetic
data is unlikely to fix it and could make the shortcut-learning worse, per
`data/synthetic/strategy.md`'s own stated risk ("teaching old-model blind
spots" / mode collapse), which predicted a version of this failure mode
before it was observed.

**Suggested next step:** (a) strip/rewrite overt meta-commentary from
synthetic profiles — the 9 confirmed cases are an easy first pass; (b)
audit for subtler structural tells (section-header conventions, phrasing
templates, sentence-length distributions) that differ systematically
between `generate_pos.md` and `generate_neg.md` outputs, e.g. by training a
trivial classifier on synthetic-only text to predict pos/neg label without
seeing the seeker/candidate relationship at all — if that classifier scores
well above chance, it quantifies exactly how "gameable" the current
synthetic data is; (c) consider generating positives and negatives from a
single shared prompt/pass (one call decides both, or a diff-based negative
mutation of a real positive) rather than two structurally different
prompts, to remove the shortcut signal at the source.

**Status:** confirmed 2026-07-19, root cause not yet fixed. Per the
decision gate in `docs/two-tower-fine-tune-plan.md` ("if lift shows on
train but not holdout, stop and diagnose — don't scale data yet"), do not
generate a larger synthetic batch or launch `run_002` until this is
addressed.
