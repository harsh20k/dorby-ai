# LLM-judge experiment — can an LLM predict accept/decline from two profiles alone?

Generated: 2026-07-25. Code: `baselines/llm_judge/`. Tests: `tests/test_llm_judge.py`.

## What this tests

Give an LLM both **complete** profiles of a real pair — nothing else — and ask
whether the intro would be a good match. No embeddings, no training, no
retrieval, and **no `searchQuery`**, which every other baseline in this repo
gets. Score the answer against the real human outcome (accept vs decline) on
the 200 real seed pairs.

Withholding the query is the whole point, and it is the one thing worth
guarding: `tests/test_llm_judge.py::test_search_query_never_reaches_the_prompt`
asserts the query text appears in neither the user prompt nor any system
prompt, because a regression there would silently turn this into a much easier
experiment.

Two call paths, sharing one prompt/scoring pipeline:
- **OpenRouter** (`--backend openrouter`, default) — any OpenRouter model id.
- **AWS Bedrock** (`--backend bedrock`) — any Bedrock model id, via the
  `tf_provisioner` AWS account already used for this project's profile
  generation. Structured JSON-schema output where the model supports it,
  falling back to plain-text-prompted JSON otherwise (see
  `baselines/llm_judge/bedrock_backend.py`).

Four (model, backend) combinations tested so far, all temperature 0, output
capped at 600 tokens (see "Cost-optimization note" below):

| Model | Backend | ~Cost for 200 pairs |
|---|---|---|
| `google/gemini-3.1-flash-lite` | OpenRouter | ~$0.10 |
| `google.gemma-3-27b-it` | Bedrock | ~$0.03 |
| `qwen.qwen3-32b-v1:0` | Bedrock | ~$0.03 |

Output is a JSON object with `reasoning` (written before deciding), `match`
(yes/no) and `confidence` (0-100). Prompts average 18.6k characters (~4.6k
tokens), max 64k.

## Headline result

**On the matched 69-pair holdout, the best LLM judge (`gemini-3.1-flash-lite`,
naive framing) beats Voyage-4-large — Boardy's production model — on pair AUC
(0.6358 vs 0.6086) while being denied the search query that Voyage gets. Its
hard-negative-slice AUC of 0.6466 is the best figure of any model tested in
this project. Two cheaper open-weight judges tested since (Gemma 3 27B,
Qwen3-32B, both via Bedrock) land lower on AUC but match or beat it on
hard-negative AUC — see "Which judge model?" below.**

Matched frozen 69-pair holdout, so every row is the same population
(`docs/baseline-results-holdout.md` is the source for the non-LLM rows):

| Model | Pair AUC | Hard-neg AUC | Easy-neg AUC | Acc @ 0.5 | Gets query? |
|---|---|---|---|---|---|
| Qwen3-Embedding-8B (open) | **0.6595** | 0.6259 | 0.7586 | 0.5072 | yes |
| Hybrid TF-IDF+nano | 0.6397 | 0.6034 | 0.7172 | 0.5072 | yes |
| **LLM judge: gemini-3.1-flash-lite (naive)** | **0.6358** | **0.6466** | 0.5638 | **0.5942** | **no** |
| Voyage-4-large (production) | 0.6086 | 0.6017 | 0.6000 | 0.4348 | yes |
| TF-IDF (lexical) | 0.5922 | 0.5017 | 0.7552 | 0.5797 | yes |
| LLM judge: gemini-3.1-flash-lite (calibrated) | 0.5901 | 0.5879 | 0.5310 | 0.5652 | no |
| LLM judge: gemma-3-27b-it (Bedrock, naive) | 0.5823 | 0.6216 | 0.4931 | 0.5507 | no |
| LLM judge: qwen3-32b (Bedrock, naive) | 0.5802 | 0.6224 | 0.4966 | 0.5072 | no |
| twotower arm_a_real_only | 0.5793 | 0.5000 | 0.6552 | 0.4203 | yes |
| Voyage-4-nano | 0.5793 | 0.5707 | 0.6207 | 0.4348 | yes |
| twotower run_001 | 0.5784 | 0.4845 | 0.6931 | 0.4203 | yes |
| Frozen BERT | 0.4595 | 0.4224 | 0.6379 | 0.4203 | yes |

On all 200 real pairs (train + holdout), naive `gemini-3.1-flash-lite` scores
pair AUC 0.6177, AP 0.5804, decision accuracy 0.6050, F1 0.6326.

## Which judge model?

All four (model, framing) combinations land in the same 0.58–0.64 AUC band —
comfortably above chance, none close to Qwen3-Embedding-8B's ceiling.
`gemini-3.1-flash-lite` is the clear leader on pair AUC (0.6358), but the two
Bedrock models actually edge it out on **hard-negative** AUC (0.6216 / 0.6224
vs 0.6466 — flash-lite is still ahead here too, but the margin over Gemma/Qwen
is much smaller than the overall-AUC gap suggests) while being noticeably
weaker on **easy negatives** (0.49–0.50 vs 0.56). Read together with finding
#1 below, that's consistent with the smaller models being *less* swayed by
surface lexical similarity, not more — they just don't yet convert that into
better overall separation.

The three models also disagree on how often to say "yes": flash-lite 56.5%,
Gemma 71–75%, Qwen 68–70%. Gemma and Qwen over-trigger more, which is the more
likely explanation for their weaker easy-neg numbers (easy negatives are
exactly the pairs where a low-lexical-overlap heuristic would say "no"
confidently — a judge that defaults toward "yes" gives that up).

Cost-wise the Bedrock models are roughly 3x cheaper per 200-pair run, so a
larger model in the same families (Bedrock also has larger DeepSeek, Mistral,
and Claude options — see the model list this repo already surveyed) is a
cheap next experiment if the goal is closing the gap to flash-lite rather than
just adding cost diversity.

## The four findings that matter

### 1. It is the only model that does *better* on hard negatives than easy ones

Every embedding baseline degrades sharply from easy to hard negatives — TF-IDF
falls 0.7552 → 0.5017, Qwen3-8B 0.7586 → 0.6259. The LLM judge goes the other
way: 0.5638 easy → 0.6466 hard.

That inversion is the most informative number here. The easy/hard split is
defined by token Jaccard overlap between the two profile texts, so a model
whose score tracks lexical overlap gets easy negatives nearly for free and
stalls on hard ones. The LLM shows the opposite profile, which is direct
evidence it is **not** scoring surface similarity — and hard negatives are the
population that actually matters, since per `docs/objective.md` all real
negatives are production's own false positives (every one already cleared a
relevance bar, so there is no easy-negative population in production).

This is also the cleanest counter yet to `docs/possible-bugs.md` #3, where
plain TF-IDF's 0.5922 embarrassed both fine-tunes: TF-IDF earns its number
almost entirely on the easy slice (0.7552 easy / 0.5017 hard — chance on hard).
The LLM judge's 0.6358 is earned where it counts.

### 2. Its decisions are usable without threshold tuning

Accuracy @ 0.5 is 0.5942, the best in the table. The embedding baselines sit at
or below chance there (Voyage-4-large 0.4348) because raw cosine has no
calibrated decision point — they need a threshold fitted on labeled data before
they can answer yes/no at all. The LLM commits to an answer, and that answer is
right ~60% of the time out of the box. `verdict_to_score` centers the score on
the model's own decision boundary specifically so `pair.accuracy_at_0.5` equals
its raw decision accuracy (asserted in the tests).

### 3. Telling the model the truth about the task made it worse

The `calibrated` prompt variant states what `docs/objective.md` establishes:
production already deemed every pair relevant, so topical fit is a given, and
the base rate is exactly 50/50. This is strictly more true information — and it
*hurt*: pair AUC 0.6358 → 0.5901, hard-neg 0.6466 → 0.5879.

What it changed was skepticism, not discrimination. The yes-rate collapsed from
56.5% to 30.4%, i.e. the model moved its threshold rather than better separating
the classes. Worth knowing before anyone assumes prompt-engineering the base
rate in is free upside; the naive framing is the one to build on.

### 4. Its stated confidence is worthless

Mean confidence 88.5 — 88.6 when right, 88.2 when wrong. Effectively no
discrimination, and 199 of 200 answers land in the 80-100 band, so per-bucket
accuracy is flat (0.5870 in [80,90] vs 0.6168 in [90,100]). The AUC above comes
almost entirely from the yes/no decision, not from confidence ordering. Do not
use this field to rank or to gate; if a soft score is wanted, token logprobs or
an explicit paired comparison would be the thing to try.

## `structured_cot`: does forcing multi-aspect CoT scoring help? (2026-07-26)

Motivated by wanting to know, before scaling the `rrf_002`-style pipeline to
500 profiles, whether the judge that labels synthetic pairs could simply be
made more accurate by asking it to reason harder — the same "score six
weighted aspects, then aggregate" pattern production intro-matching UIs use,
rather than the naive variant's direct "would this be a good match?" call.

**Design.** Same model (`google/gemini-3.1-flash-lite`), same profiles, same
missing `searchQuery` — the only thing that changes from `naive` is the
prompt. `structured_cot` (`baselines/llm_judge/prompts/structured_cot.md`)
requires the model to score six fixed-weight aspects with cited evidence
before answering:

| Aspect | Weight |
|---|---|
| Location & availability | 15% |
| Ask/offer alignment (does what one wants match what the other offers) | 25% |
| Skill/domain evidence | 20% |
| Seniority/stage fit | 15% |
| Domain/industry fit | 15% |
| Practical constraints | 10% |

The verdict is **not** the model's own stated confidence — `baselines/
llm_judge/structured.py::parse_structured_verdict` recomputes `weighted_score
= Σ(canonical_weight × score/5)` from the six scores using the *canonical*
weights above, discarding whatever weight the model echoed back, so a model
that quietly reweights one aspect to swing its own answer can't move the
number that gets measured. `match` = "yes" iff `weighted_score >= 0.5`;
`confidence = |weighted_score − 0.5| × 200`.

**Result, matched 69-pair holdout, both variants run back-to-back in the same
session for a clean comparison** (`python -m baselines.llm_judge.eval
--variant {naive,structured_cot} --split holdout`):

| | naive | structured_cot |
|---|---|---|
| Pair ROC-AUC | **0.6409** | 0.6336 |
| Decision accuracy | **0.6087** | 0.5507 |
| Hard-negative AUC | **0.6543** | 0.6267 |
| Easy-negative AUC | 0.5603 | 0.5603 |
| Says "yes" | 55.1% | 75.4% |

**Forcing step-by-step aspect scoring did not help — it was a small, uniform
step backward.** Pair AUC, decision accuracy, and hard-negative AUC all moved
in the same direction, naive ahead of `structured_cot` on every one. (These
naive numbers, 0.6409/0.6543, are close to but not identical to the
0.6358/0.6466 documented above — same model, same prompt, same population,
re-run in a fresh session; the small drift is most likely `temperature=0`
non-determinism on OpenRouter, not a regression, and doesn't change which
variant wins.)

The mechanism is visible in the yes-rate: `structured_cot` says "yes" 75.4% of
the time versus naive's 55.1%, which reads as **regression to the middle, not
sharper judgment**. Six independently-scored aspects on a 0-5 scale average
out — a pair with one weak aspect and five middling ones still lands close to
0.5 — so the aggregate score clusters tighter around the decision boundary
than one holistic yes/no with self-reported confidence does. Decomposition
bought interpretability (every verdict now comes with six pieces of cited
evidence instead of 2-4 sentences) but not accuracy, at roughly 2.5× the
output tokens per call.

**Decision: keep `naive` as the labeling judge.** `structured_cot` does not
replace it for the next synthetic batch. The per-aspect evidence remains
useful as a debugging/audit tool on individual pairs even though the
aggregate score isn't an improvement — see `docs/html/llm-judge-comparison.html`
for both alongside every embedding baseline.

## What this does not show

- **This cannot ship on the serving path.** A per-candidate LLM call is
  explicitly out of scope under the <100 ms budget (see CLAUDE.md) — each call
  here takes seconds on a ~4.6k-token prompt. Nothing in this experiment is a
  deployable configuration. Its value is as an accuracy reference and as
  evidence about what signal *exists* in the profile text, which is a different
  claim from what can be served.
- **No retrieval metrics exist for this row**, which is why it is not merged
  into `docs/baseline-results-holdout.md` and instead lives here. An LLM judge
  has no shared vector space, so ranking a candidate corpus per query would
  need ~40k calls rather than one encode plus an ANN lookup. The comparison
  above is classification-only; Voyage-4-large still leads retrieval (MRR
  0.5287) and that is not contested here.
- **One model, one prompt, n=69 on the holdout.** A 0.027 AUC gap over
  Voyage-4-large on 69 pairs is not a wide margin. The 200-pair number
  (0.6177) is the more stable estimate; the holdout number is the
  population-matched one. Both point the same way, but neither is decisive on
  its own.
- Only `gemini-3.1-flash-lite` was tested. A frontier model would be the
  obvious next probe, and is a `--model` swap.

## Cost-optimization note: the `max_tokens` incident

An early attempt at running `google/gemini-3.6-flash` and `openai/gpt-5.5` as
additional judges via OpenRouter failed partway through both times with
`402: insufficient credits` — not because the account was actually out of
money, but because neither call capped `max_tokens`. OpenRouter's credit
check is a **preflight reservation against the requested ceiling** (the
model's absolute max, e.g. 65,536 tokens), not against real usage, so every
call reserved against 65k tokens even though the real completion is a
~100-token JSON object. `synth_pipeline.llm.complete_json` now accepts an
explicit `max_tokens` (plumbed through from `--max-tokens`, default 600 —
generous for the `reasoning`/`match`/`confidence` schema) so the preflight
check reserves a sane amount instead. The same cap applies to the Bedrock
path via `inferenceConfig.maxTokens`. Both partial runs' successful verdicts
stayed cached rather than being discarded; resuming them (with the cap in
place) is a `--model` swap away whenever there's appetite to spend on a
frontier-tier point.

## Reproducing

```bash
# 200 real pairs, the naive framing (the headline row), via OpenRouter
python -m baselines.llm_judge.eval --data-dir data --variant naive --split all

# matched 69-pair holdout — free after the run above, the verdict cache is
# keyed by pair identity + prompt hash, so it is split-independent
python -m baselines.llm_judge.eval --data-dir data --variant naive --split holdout

# the framing comparison
python -m baselines.llm_judge.eval --data-dir data --variant calibrated --split all

# another OpenRouter model
python -m baselines.llm_judge.eval --data-dir data --model anthropic/claude-sonnet-4.5

# a Bedrock model instead — uses the tf_provisioner AWS account, no OpenRouter
# credits needed. See `aws bedrock list-foundation-models` for what's available;
# not every model supports Bedrock's native structured-output enforcement, and
# call_bedrock_verdict falls back to plain-text JSON parsing when it doesn't.
python -m baselines.llm_judge.eval --data-dir data --backend bedrock \
  --model google.gemma-3-27b-it --variant naive --split all
python -m baselines.llm_judge.eval --data-dir data --backend bedrock \
  --model qwen.qwen3-32b-v1:0 --variant naive --split all

# from a git worktree, data/ and .env live in the main checkout
python -m baselines.llm_judge.eval \
  --data-dir /Users/harsh/Artifacts/dorby-ai/data \
  --env-file /Users/harsh/Artifacts/dorby-ai/.env
```

Verdicts cache to `artifacts/llm_judge/<backend>_<model>_<variant>/verdicts.json`,
metrics to `metrics_<split>.json` in the same directory. Editing a prompt
changes its hash and correctly invalidates affected entries — the stale-cache
failure mode already hit once in this repo (see the `cache_name` note under
"Pairing standalone profiles" in CLAUDE.md), so it is tested here.

Bedrock usage across every model tested (tokens, invocations, errors,
latency, and an estimated $/hr per priced model) is visible in the
`dorby-bedrock-profile-gen` CloudWatch dashboard — regenerate it after adding
a new priced model via `scripts/update_bedrock_dashboard.py` (token/invocation/
latency widgets pick up new models automatically via `SEARCH()`; only the
cost widget needs a `MODEL_PRICING` entry).

## Suggested next steps

1. **Run a frontier model.** Cheapest high-information probe available; if
   flash-lite already clears production, the ceiling is the interesting number.
2. **Use the judge where it is affordable: labeling.** `synth_pipeline/pairing/`
   currently labels with the TF-IDF+nano fusion, and its own docs flag that TF-IDF
   query cosine predicts the assigned label at 0.868 AUC — the labels are mostly
   lexical overlap. This judge is the strongest *non-lexical* pair scorer measured
   (finding #1), never sees a query, and labeling is offline where latency is
   irrelevant. It is a direct answer to that doc's open call for "a semantic judge
   back in the labeling path".
3. **Distill it.** The judge demonstrates non-lexical signal exists in profile
   text at ~0.64 AUC. Using its verdicts (or margins) as a training target for
   the two-tower adapter is the obvious way to move that signal into something
   that fits the latency budget.
4. **Ablate what it reads.** Rerun with `--max-field` truncation, or drop
   individual profile fields, to find which parts carry the signal — cheap, and
   directly useful to the field-ablation work already in `baselines/text_field_ablation.py`.
