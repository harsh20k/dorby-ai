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

**Status:** resolved 2026-07-19. Added `--holdout-only` to all three
baseline `eval.py` scripts (filters via new `baselines/holdout.py::
filter_to_holdout`, reusing `synth_pipeline.split.load_split`) and reran
all three restricted to the same 69-pair holdout at `max_length=4096`
(BERT stays at its native 512-token architectural cap). Extended
`scripts/export_baseline_results.py` with a second output
(`docs/baseline-results-holdout.md`/`.json`) that reads the `*_holdout`
artifact dirs plus twotower's `metrics_holdout.json` directly (different
path convention, handled via `add_twotower_holdout_run()`).

**Corrected finding — the gap was previously understated, not overstated:**
matching the population changed the baseline numbers more than the
twotower number. Voyage-4-large's *true* holdout retrieval quality is much
better than the stale full-200-pair cached number suggested (MRR 0.529 vs.
the old 0.310 — a *smaller*, holdout-only candidate corpus of 65 unique
candidates makes retrieval easier than the full ~168-candidate pool the
original number was computed against). On the properly matched table:

| | pair AUC | MRR | NDCG@10 | Recall@10 | Top-1 |
|---|---|---|---|---|---|
| Frozen BERT | 0.460 | 0.137 | 0.156 | 0.310 | 0.069 |
| Voyage-4-nano | 0.579 | 0.461 | 0.523 | 0.759 | 0.276 |
| Voyage-4-large (prod) | 0.609 | 0.529 | 0.604 | 0.862 | 0.345 |
| **twotower run_001** | 0.578 | **0.283** | **0.359** | 0.655 | **0.069** |

twotower is roughly tied with Voyage-nano on pair AUC (binary classify-this-
one-pair) but clearly **worse than both Voyage baselines on every retrieval
metric** — Top-1 accuracy ties the near-random BERT baseline (0.069). Full
table: `docs/baseline-results-holdout.md`. This sharpens (not softens) the
`run_001` verdict in `docs/twotower-run-001-results.md`: it isn't just
failing to beat Voyage-large, it's underperforming Voyage-nano on ranking
quality specifically — consistent with #4's diagnosis (the model learned a
binary classify-this-one-candidate shortcut, not something that transfers
to ranking many candidates against each other).

**Added 2026-07-20: `baselines/tfidf/`, a plain TF-IDF-cosine lexical
baseline** (no neural model), as a lexical floor. Result: TF-IDF pair AUC
is **0.592 — higher than both `run_001` (0.578) and `arm_a_real_only`
(0.579)**. Simple keyword-overlap cosine similarity beats the fine-tuned
LoRA adapter on the exact metric it was trained to optimize. TF-IDF is
weakest of all baselines on retrieval (MRR 0.248, only ahead of BERT),
consistent with keyword overlap being useful for coarse binary separation
but not for ranking the single best candidate among several
keyword-similar ones. Its neg-hardness split (easy AUC 0.755, hard AUC
0.502 — exactly chance) is a clean sanity check that the hardness slice
metric works as intended by construction. See
`docs/twotower-run-001-results.md` for the full six-way table.

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

**Quantified via `scripts/check_synth_cheatability.py`** (2026-07-19): a
trivial TF-IDF + logistic regression classifier, given **only the
candidate's own profile text** — no seeker, no query, no relationship
between the two — predicts pos/neg label with:

| Population | n pos / n neg | 5-fold CV ROC-AUC |
|---|---|---|
| Real seed pairs only | 100 / 100 | **0.487** (pure chance) |
| Synthetic (`batch_500_001`) only | 220 / 240 | **0.992** (near-perfect) |
| Combined | 320 / 340 | 0.911 |

Gap: **+0.505**. Real candidate profiles carry essentially zero label
signal in isolation — correct, since whether a real person is a good
match depends entirely on who's asking. Synthetic candidate profiles are
almost perfectly separable by surface text alone, confirming the shortcut
is pervasive across the whole synthetic set, not confined to the 9
profiles with blatant meta-commentary — this is a stronger and more
complete piece of evidence than the phrase-grep, and fully explains
`run_001`'s inflated train-dev AUC (0.986) without needing any other
mechanism.

**Base experiment — real-data control, in detail (2026-07-19):** ran the
same classifier on real-only data with per-fold breakdown and a feature
inspection, as a formal control to contrast against the synthetic result
above (not just a summary number).

Per-fold AUC (5-fold CV, seed=42): `[0.497, 0.335, 0.542, 0.516, 0.518]`,
mean 0.482, **std 0.075**. Compare to the synthetic-only run: std 0.0054 —
14x tighter. This is the more important signal than the mean AUC alone: a
classifier that's found a real, generalizable pattern scores consistently
across folds; one that's fitting to sampling noise in a small (200-row)
dataset bounces around near 0.5 fold-to-fold, which is exactly what the
real-data run does. The synthetic run's tight, stable ~0.99 across every
fold is strong independent evidence that it reflects a real systematic
artifact, not an overfit statistical fluke — the real-data control is a
methodological sanity check that the diagnostic isn't just prone to
finding spurious signal in any small dataset.

Top TF-IDF coefficients on the real-data control (full fit): negative-
leaning — `and`, `munich`, `engineer`, `startup`, `trade`, `technical`,
`the`, `europe`, `singapore`, `tax`, `nyc`; positive-leaning — `health`,
`sports`, `robotics`, `women`, `chris`, `ventures`, `robert`, `05 27`
(a date fragment), `longevity`, `lp`, `ecommerce`. These are stopwords and
person/company/location proper nouns scattered across unrelated topics —
the model is keying on incidental identity details that happen to
correlate with label by chance in a small sample, not on anything about
*how the profile was written*. Contrast with the synthetic-batch feature
list (`generate_neg.md` fix validation, above): narrative/explanatory
language ("critical distinction," "has never," "mistaken for") before the
fix, then post-fix topic-clustering words ("campus," "festival," "health,"
"climate") — a qualitatively different, structural failure mode, not
noise. The real-data control confirms the diagnostic distinguishes real
structural artifacts from small-sample noise correctly.

**Suggested next step:** (a) strip/rewrite overt meta-commentary from
synthetic profiles — the 9 confirmed cases are an easy first pass; (b) fix
`synth_pipeline/llm.py::truncate_pair_for_prompt`'s 400-character seed
truncation, which likely also contributes structural distortion (see #1);
(c) close the stylistic gap between `generate_pos.md` (has a "Match
Boardy CRM tone" instruction) and `generate_neg.md` (lacks it); (d)
consider generating positives and negatives from a single shared
prompt/pass (one call decides both, or a diff-based negative mutation of a
real positive) rather than two structurally different prompts, to remove
the shortcut signal at the source; (e) re-run
`scripts/check_synth_cheatability.py` after each fix to confirm the
synthetic-only AUC drops toward the real-only ~0.487 floor.

**Fixes applied (2026-07-19):** (a) `synth_pipeline/llm.py::
truncate_pair_for_prompt` no longer truncates the seed pair (was 400 chars;
real `lookingFor` fields run up to 23,672 chars, p95 4,874 — the truncation
was hiding most of the field from the generator, root cause of #1); (b)
`generate_neg.md` now explicitly bans meta-commentary give-away phrases;
(c) `generate_neg.md` now has the same "Match Boardy CRM tone" instruction
`generate_pos.md` already had, closing a structural gap. Pushed as
`synth-generate-neg:v2` on LangSmith Hub.

**Validation (pilot batch, 2026-07-19):** generated `batch_pilot_fix_001`
(20/20 attempts, 36 staged, 90% yield — consistent with `batch_500_001`'s
93%, no regression in generation quality) with the fixed prompt, then
re-ran `scripts/check_synth_cheatability.py --batch-dir
artifacts/synth/batch_pilot_fix_001`:

| Population | n pos / n neg | CV ROC-AUC |
|---|---|---|
| Real seed pairs (unchanged control) | 100 / 100 | 0.487–0.532 (CV-fold dependent) |
| `batch_500_001` (unfixed, old) | 220 / 240 | 0.992 |
| **`batch_pilot_fix_001` (fixed prompt)** | 19 / 17 | **0.863** |

Material drop (0.992 → 0.863), clearing the plan's decision-gate bar. Not
fully closed to the real-data floor — inspected the classifier's top
TF-IDF coefficients on the pilot batch and found **no meta-commentary
words** (previously would expect "mistaken," "distinction," etc.); the
top features are now pure topic/domain vocabulary ("campus," "festival,"
"health," "climate"), consistent with small-sample topic clustering (only
36 examples across ~15 distinct seed queries) rather than a persistent
structural pos/neg tell. Caveat: this is a noisy read at n=36 — a larger
validation batch would give a cleaner signal, and the real arbiter is the
Arm C real-holdout eval in Phase 4, not this proxy metric in isolation.

**Confirmed with a training experiment, not just the cheatability proxy
(2026-07-19):** trained `arm_a_real_only` — identical recipe to `run_001`
but 111 real pairs only, zero synthetic. On the same real holdout, Arm A
beat `run_001` (real + 410 unfixed synth) on every metric except pair AUC
(a tie): MRR 0.388 vs 0.283, NDCG@10 0.471 vs 0.359, Recall@10 0.793 vs
0.655, hard-negative AUC 0.500 (chance) vs 0.4845 (below chance). Using
~1/5th the training data and none of it synthetic produced a *better*
model. This is decisive: the unfixed synthetic data wasn't just failing to
help, it was **actively degrading** the model relative to not using it at
all. Full table: `docs/twotower-run-001-results.md`.

**Status:** root cause fixed and validated at small scale 2026-07-19, and
independently confirmed harmful-not-neutral via the Arm A real-only
control. Decision: Arm C (full-scale regeneration with fixed prompts +
train real+fixed-synth, bar to clear = Arm A's numbers, not just
Voyage-large's) is the remaining step, scale/timing not yet scheduled —
deliberate pause per user decision, not a technical blocker. Re-run
`scripts/check_synth_cheatability.py` against the full-scale regenerated
batch once available to confirm the drop holds at scale before promoting
it.

## 5. `twotower.eval.evaluate_pairs` always includes the search query, even for a model trained without it

**Where:** `twotower/data.py::LabeledPair.seeker_text` (hardcoded to
`seeker_to_text(profile, searchQuery)`), consumed unconditionally by
`twotower/eval.py::evaluate_pairs`, and therefore by every caller of that
function — `eval_real_full/eval.py::run_eval`,
`twotower_top1_optimised/train.py`'s in-training holdout eval, etc.

**Found in:** `twotower_no_query/` — the first all-200 eval of
`no_query_001` (an adapter trained on `profile_to_text` seeker text, no
query anywhere in training) went through `eval_real_full.eval.run_eval` like
every other adapter in this project, which built seeker text via
`LabeledPair.seeker_text` — profile *plus* query, unconditionally. User
caught this by inspection ("i think that is also using search query") before
the result was written up as final.

**What's wrong:** for every other adapter in this project (`run_001`,
`arm_a_real_only`, the RRF triplet series, `top1_ctrl`), `seeker_text`
correctly matches training — they all trained on profile+query concatenated,
so evaluating on the same text is right. `no_query_001` is the first adapter
whose training text diverges from `seeker_text`, and nothing in
`evaluate_pairs` or its callers parameterizes seeker-text construction, so it
silently fed the model out-of-distribution input (query tokens never seen in
training) rather than raising or requiring an explicit override.

**Impact measured:** re-scored `no_query_001` on seeker text that matches
its training (`profile_to_text`, via `twotower_query_weighted.eval`'s
already-published `profile_only` path, read-only reuse) instead of
`evaluate_pairs`'s query-included text. All-200 real pairs:

| Eval text | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| Mismatched (`evaluate_pairs`, profile+query — the bug) | 0.5718 | 0.5550 | 0.3371 | 0.18 | 0.67 |
| Matched (profile only, correct) | 0.5574 | 0.5374 | 0.2827 | 0.13 | 0.62 |

The mismatched number is *higher* on every metric than the matched one — the
model does slightly better on text it never trained on than on its own
training distribution, most likely because the extra (unfamiliar) query
tokens still add some lexical signal even unattended-to. Either way, the
original write-up's headline number (R@1 0.18, "matches the untrained
baseline") was not measuring what it claimed to measure. See
`docs/twotower-no-query-experiment.md` for the corrected comparison and full
discussion.

**Suggested next step:** `twotower.eval.evaluate_pairs`/`LabeledPair` have no
way to express "score this pair's seeker side without the query" — any future
adapter trained on non-standard seeker text needs its own eval path (as
`twotower_no_query/modal_eval_matched.py` now does by reusing
`twotower_query_weighted.eval`) rather than the shared `evaluate_pairs`,
until/unless a `seeker_text_variant` parameter is added there. Not fixing the
shared function itself — every other published number in this project depends
on its current unconditional query-inclusion being correct.

**Status:** confirmed and worked around 2026-08-03 via a matched-distribution
eval in `twotower_no_query/`; the shared `evaluate_pairs` function is
unchanged (fixing it would need to become opt-in, not default, to avoid
retroactively changing every prior published number).

## 6. Custom training loops that bypass `.encode()` train on the untruncated embedding, not the 1024-dim one everything else uses

**Where:** `twotower_split/train.py`'s `_encode` (and the same pattern
originally in `twotower_field_gate/train.py`, fixed before publishing).

**Found in:** building `twotower_field_gate/`, while adding a second custom
training loop (needed because `SentenceTransformerTrainer` can't route
non-standard forward passes) and checking why a `FieldGate` linear layer
sized for 1024-dim inputs raised a shape-mismatch error on real data.

**What's wrong:** `voyage-4-nano`'s native pooled embedding is
**2048-dimensional**, not 1024 — `truncate_dim=1024` (used everywhere else
in this project) is a real truncation, applied only inside
`SentenceTransformer.encode()`'s own post-processing
(`sentence_transformers.util.truncate_embeddings`, a plain
`embeddings[..., :truncate_dim]` slice of the already-normalized pooled
output, then renormalized if `normalize_embeddings=True`). A raw
`model(features)["sentence_embedding"]` forward pass — the pattern both
`twotower_split/train.py` and the first draft of `twotower_field_gate/
train.py` used, needed because `.encode()` disables gradients — **skips this
entirely** and returns the full 2048-dim vector. Verified numerically:
manually slicing to `[:1024]` and renormalizing exactly reproduces
`.encode(..., truncate_dim=1024, normalize_embeddings=True)` (max abs diff
0.0), confirming the mechanism and the fix.

**Why it matters:** every other embedding in this project — every baseline,
every prior fine-tune, every eval script — is computed at 1024 dims. A
custom loop that trains on the untruncated 2048-dim space is optimizing a
LoRA adapter for a representation that gets thrown away and re-sliced before
comparison at eval time; the adapter has no reason to put the most useful
signal specifically in the first 1024 of its 2048 output dimensions.

**Impact:** `twotower_field_gate/train.py` was fixed before any GPU spend
(caught by a shape-mismatch crash, not silently). `twotower_split/train.py`
was **not** retroactively fixed or rerun — its published result
(`docs/twotower-split-experiment.md`) trained on the native 2048-dim space
and was only truncated to 1024 at eval time. This doesn't necessarily
invalidate that result (the negative finding — split towers underperform —
might well hold regardless), but it is a real methodological gap between how
that adapter trained and how it was scored, on record as a caveat rather
than silently left unstated.

**Suggested next step:** any future custom training loop that bypasses
`.encode()` should truncate-then-renormalize explicitly, as
`twotower_field_gate/train.py`'s fixed `_encode` now does. Re-running
`twotower_split` with the fix applied, to see whether it changes the
already-negative result, is a candidate follow-up but not yet scheduled.

**Status:** confirmed and fixed in `twotower_field_gate/` 2026-08-04;
`twotower_split/`'s already-published number is left as-is with this caveat
recorded, not retroactively edited (per the isolation rule — its code is
frozen, matching what actually produced the published result).
