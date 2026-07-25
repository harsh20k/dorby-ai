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

Model: `google/gemini-3.1-flash-lite` via OpenRouter, temperature 0. Output is
a JSON object with `reasoning` (written before deciding), `match` (yes/no) and
`confidence` (0-100). Cost was ~$0.10 for all 200 pairs; prompts average 18.6k
characters (~4.6k tokens), max 64k.

## Headline result

**On the matched 69-pair holdout, the LLM judge beats Voyage-4-large —
Boardy's production model — on pair AUC (0.6358 vs 0.6086) while being denied
the search query that Voyage gets. Its hard-negative-slice AUC of 0.6466 is
the best figure of any model tested in this project.**

Matched frozen 69-pair holdout, so every row is the same population
(`docs/baseline-results-holdout.md` is the source for the non-LLM rows):

| Model | Pair AUC | Hard-neg AUC | Easy-neg AUC | Acc @ 0.5 | Gets query? |
|---|---|---|---|---|---|
| Qwen3-Embedding-8B (open) | **0.6595** | 0.6259 | 0.7586 | 0.5072 | yes |
| Hybrid TF-IDF+nano | 0.6397 | 0.6034 | 0.7172 | 0.5072 | yes |
| **LLM judge (naive, no query)** | **0.6358** | **0.6466** | 0.5638 | **0.5942** | **no** |
| Voyage-4-large (production) | 0.6086 | 0.6017 | 0.6000 | 0.4348 | yes |
| TF-IDF (lexical) | 0.5922 | 0.5017 | 0.7552 | 0.5797 | yes |
| LLM judge (calibrated, no query) | 0.5901 | 0.5879 | 0.5310 | 0.5652 | no |
| twotower arm_a_real_only | 0.5793 | 0.5000 | 0.6552 | 0.4203 | yes |
| Voyage-4-nano | 0.5793 | 0.5707 | 0.6207 | 0.4348 | yes |
| twotower run_001 | 0.5784 | 0.4845 | 0.6931 | 0.4203 | yes |
| Frozen BERT | 0.4595 | 0.4224 | 0.6379 | 0.4203 | yes |

On all 200 real pairs (train + holdout), the naive judge scores pair AUC
0.6177, AP 0.5804, decision accuracy 0.6050, F1 0.6326.

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

## Reproducing

```bash
# 200 real pairs, the naive framing (the headline row)
python -m baselines.llm_judge.eval --data-dir data --variant naive --split all

# matched 69-pair holdout — free after the run above, the verdict cache is
# keyed by pair identity + prompt hash, so it is split-independent
python -m baselines.llm_judge.eval --data-dir data --variant naive --split holdout

# the framing comparison
python -m baselines.llm_judge.eval --data-dir data --variant calibrated --split all

# another model
python -m baselines.llm_judge.eval --data-dir data --model anthropic/claude-sonnet-4.5

# from a git worktree, data/ and .env live in the main checkout
python -m baselines.llm_judge.eval \
  --data-dir /Users/harsh/Artifacts/dorby-ai/data \
  --env-file /Users/harsh/Artifacts/dorby-ai/.env
```

Verdicts cache to `artifacts/llm_judge/<model>_<variant>/verdicts.json`, metrics
to `metrics_<split>.json` in the same directory. Editing a prompt changes its
hash and correctly invalidates affected entries — the stale-cache failure mode
already hit once in this repo (see the `cache_name` note under "Pairing
standalone profiles" in CLAUDE.md), so it is tested here.

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
