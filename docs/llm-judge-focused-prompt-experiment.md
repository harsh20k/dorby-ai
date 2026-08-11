# LLM-judge focused-prompt experiment — searchQuery + trimmed fields

Generated: 2026-08-08. Code: `baselines/llm_judge_with_pos_look_pos_back_look/`.
Isolated variation of `baselines/llm_judge` (see `docs/llm-judge-experiment.md`
for the original) — new package, not an edit, per the "ML / data-science
experiments" isolation rule in CLAUDE.md. Everything in the original
`llm_judge` package (its prompts, cached verdicts, results) is untouched.

## What changed vs. the original ("naive") LLM judge

The original experiment's whole point was withholding `searchQuery` and
showing complete profiles. This variant asks a different question — with
more of what Voyage gets, does the LLM do even better? — by making two
deliberate changes:

1. **`searchQuery` is given to the model.** Every other LLM-judge run in
   this project withholds it.
2. **Only a subset of profile fields is shown**, not the complete profile:
   - seeker (Person A): `positioning` + `lookingFor`
   - candidate (Person B): `positioning` + `background` + `lookingFor`
3. The system prompt tells the model explicitly what to look for:
   complementary need/supply (A's `lookingFor` mapping to something concrete
   in B's `positioning`/`background`), two-way fit (B's own `lookingFor`
   also compatible with what A offers), and to weigh concrete stated
   preferences over shared keywords/vibes.

Model: `google/gemini-3.1-flash-lite`, temperature 0, `max_tokens=600`. Run
twice, through two backends, to check whether OpenRouter passes the request
through to Google faithfully:

- **OpenRouter** (`--backend openrouter`, default) — same proxy every other
  LLM-judge run in this project uses.
- **Google Generative Language API directly** (`--backend google`,
  `GEMINI_API_KEY`) — plain REST to `generateContent`, forcing JSON output
  via `responseMimeType: application/json`. New: `baselines/llm_judge_with_pos_look_pos_back_look/google_backend.py`.

They are **not** the same result — see "OpenRouter vs. direct Google API"
below.

Prompt pushed to LangSmith Prompt Hub as
[`llm_judge_with_pos_look_pos_back_look`](https://smith.langchain.com/prompts/llm_judge_with_pos_look_pos_back_look/eae1dd7c)
via `scripts/push_llm_judge_focused_prompt.py` — the hub copy is a record of
what was run, not pulled at eval time; `baselines/llm_judge_with_pos_look_pos_back_look/prompt.py`
is the source of truth actually executed, matching how the original
`llm_judge` package keeps its prompt inline too.

## Results

Matched 69-pair holdout (same population as `docs/baseline-results-holdout.md`
and `docs/llm-judge-experiment.md`):

| Model | Pair AUC | Hard-neg AUC | Easy-neg AUC | Acc @ 0.5 | Gets query? |
|---|---|---|---|---|---|
| Qwen3-Embedding-8B (open) | 0.6595 | 0.6259 | 0.7586 | 0.5072 | yes |
| **LLM judge: gemini-3.1-flash-lite (focused, w/ query, direct Google API)** | **0.6530** | **0.6784** | 0.5603 | 0.6087 | **yes** |
| Hybrid TF-IDF+nano | 0.6397 | 0.6034 | 0.7172 | 0.5072 | yes |
| LLM judge: gemini-3.1-flash-lite (naive, no query, OpenRouter) | 0.6358 | 0.6466 | 0.5638 | 0.5942 | no |
| LLM judge: gemini-3.1-flash-lite (focused, w/ query, OpenRouter) | 0.6203 | 0.6517 | 0.5414 | **0.6232** | yes |
| Voyage-4-large (production) | 0.6086 | 0.6017 | 0.6000 | 0.4348 | yes |
| TF-IDF (lexical) | 0.5922 | 0.5017 | 0.7552 | 0.5797 | yes |
| LLM judge: gemini-3.1-flash-lite (calibrated) | 0.5901 | 0.5879 | 0.5310 | 0.5652 | no |
| LLM judge: gemma-3-27b-it (Bedrock, naive) | 0.5823 | 0.6216 | 0.4931 | 0.5507 | no |
| LLM judge: qwen3-32b (Bedrock, naive) | 0.5802 | 0.6224 | 0.4966 | 0.5072 | no |

On all 200 real pairs, direct Google API: pair AUC 0.6451 (decision accuracy
0.5950, F1 0.6197). OpenRouter, same model/prompt: pair AUC 0.6177 (decision
accuracy 0.5900, F1 0.5980) — matching the naive judge's 200-pair AUC
(also 0.6177) almost exactly, likely coincidence given the two prompts
differ substantially.

## OpenRouter vs. direct Google API — not the same result

The whole reason for adding a `--backend google` option was to sanity-check
that OpenRouter passes the exact same request through to Google faithfully.
It does not produce the same result: calling Google's Generative Language
API directly with the identical system/user prompt, model id, temperature,
and `max_tokens` scores **meaningfully higher** on every pair metric —
holdout pair AUC 0.6530 vs. 0.6203 (+0.033), hard-neg AUC 0.6784 vs. 0.6517,
200-pair AUC 0.6451 vs. 0.6177. Decision accuracy at 0.5 is the one metric
that goes the other way (0.6087 direct vs. 0.6232 via OpenRouter), so this
isn't simply "direct is strictly better," but pair AUC — the metric this
whole project ranks by — is not close.

**Two known differences between the paths, either of which could explain
this:**

- The direct-API call sets `responseMimeType: application/json` (Gemini's
  native structured-output constraint), while the OpenRouter path goes
  through `synth_pipeline.llm.complete_json`'s JSON-mode handling — not
  necessarily the same enforcement mechanism, and a different constraint
  mechanism can change token-level sampling even at temperature 0.
- OpenRouter is a proxy: it can silently substitute quantized variants, cap
  context differently, or route through infrastructure with its own
  sampling quirks, none of which is visible from the client side. This
  project has no way to confirm OpenRouter's `google/gemini-3.1-flash-lite`
  and Google's own `gemini-3.1-flash-lite` are bit-identical model weights.

**Implication for every other OpenRouter-only number in this project
(the naive/calibrated judge, `docs/llm-judge-experiment.md`'s headline
0.6358):** those have only ever been measured through OpenRouter. This one
apples-to-apples check suggests the true direct-API number for those prompts
could plausibly be higher too — worth a follow-up run before treating
OpenRouter's numbers as the ceiling for any Gemini-family judge.

## Reading the result

**Via the direct Google API, this is the second-best pair AUC of any
config tested in this project** (0.6530, behind only Qwen3-Embedding-8B's
0.6595) and the **best hard-negative AUC of anything tested** (0.6784,
beating the original naive judge's 0.6466). Via OpenRouter, the same prompt
lands lower (0.6203) — between the naive no-query judge (0.6358) and
Voyage-4-large (0.6086) — which is the OpenRouter-vs-Google backend gap
documented above, not a property of the prompt itself.

So on the question this experiment actually set out to answer — does giving
the model the query plus a more targeted (if narrower) view of the profile,
with explicit matching criteria, help? — **the honest answer depends on
which transport measured it**, and the more trustworthy of the two (calling
Google directly, no proxy in between) says yes, clearly: 0.6530 vs. the
original naive judge's 0.6358.

**On hard negatives specifically, this is now the strongest result of any
kind in the project, embeddings included.** Combined with the original naive
judge's inversion finding (hard-neg AUC beats easy-neg AUC for LLM judges
generally), this is further evidence LLM judges resist the lexical-overlap
shortcut that dominates every embedding baseline — and doing so *while*
seeing the query and a prompt naming concrete matching criteria, not despite
it.

**What it does not show:** this is one prompt variant, one model, two runs
(n=69 on the holdout) — the "not a wide margin" caveat from the original
experiment applies here too, more so with two backends to compare across.
It also isn't a clean ablation: the query and the field trimming changed
together relative to `naive`, so this result alone can't say which change
(or their combination) drives the difference. A follow-up ablation (query
alone, trimmed fields alone) would be needed to attribute it, and the
OpenRouter/Google backend gap is itself unresolved — see the section above.

## Reproducing

```bash
# 200 real pairs, via OpenRouter (default)
python -m baselines.llm_judge_with_pos_look_pos_back_look.eval \
  --data-dir data --env-file .env --split all

# 200 real pairs, via the direct Google API (GEMINI_API_KEY)
python -m baselines.llm_judge_with_pos_look_pos_back_look.eval \
  --data-dir data --env-file .env --backend google --split all

# matched 69-pair holdout (free after the run above — same cache-by-prompt-hash
# convention as baselines/llm_judge); add --backend google for that path
python -m baselines.llm_judge_with_pos_look_pos_back_look.eval \
  --data-dir data --env-file .env --split holdout

# from a worktree: --data-dir /Users/harsh/Artifacts/dorby-ai/data --env-file /Users/harsh/Artifacts/dorby-ai/.env
```

## Related: the same fields, on embeddings instead of an LLM

`docs/voyage-field-selected-experiment.md` re-runs Voyage-4-nano and
Voyage-4-large with this experiment's exact field selection + searchQuery,
to separate "does this field selection help" from "is an LLM judge better
than an embedding model" — field selection helps both Voyage models, but
only on easy negatives; the LLM judge's hard-negative advantage holds even
when both are given identical inputs.

`docs/llm-judge-seeker-background-experiment.md` tests one more field-set
variant on this exact prompt: adding the seeker's own `background`. It made
results worse (holdout pair AUC 0.6530→0.6302) — the focused prompt without
seeker `background` remains the best config found.

Verdicts cache to
`artifacts/llm_judge_with_pos_look_pos_back_look/<backend>_<model>/verdicts.json`,
same key/hash convention as `baselines/llm_judge` — its own namespace, so it
cannot collide with or invalidate the original experiment's cache.
