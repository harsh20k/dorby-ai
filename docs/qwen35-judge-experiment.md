# Qwen3.5-4B judge experiment — self-hosted, open-weight, with searchQuery

Generated: 2026-08-04. Code: `baselines/qwen35_judge/`.

## What this tests

Same shape as `docs/llm-judge-experiment.md` (prompt an LLM with both complete
profiles, ask for a yes/no match + confidence, score against real accept/
decline on the 200 real seed pairs) but two deliberate differences:

- **Self-hosted, not an API call.** `Qwen/Qwen3.5-4B` (Alibaba, released
  ~Feb/Mar 2026, open-weight, small enough to fine-tune later under the
  latency budget) runs on a Modal A10G from Hugging Face weights, not through
  OpenRouter/Bedrock. Chosen as a fine-tuning candidate: a small model that
  can eventually merge a LoRA adapter with no serving-side latency add-on
  (see CLAUDE.md's "hard latency budget" note), unlike a per-candidate frontier
  API call.
- **`searchQuery` is included**, the one deliberate prompt difference from
  `llm_judge`'s `naive` variant, which withholds it. This run is a first
  attempt at whether adding the query helps a judge — `llm_judge`'s own
  query-ablation work only tested this for embedding baselines, not an LLM
  judge (see CLAUDE.md "Query ablation").

Prompt is `naive` (system prompt copied verbatim from `baselines/llm_judge/
prompt.py`) plus a `PERSON A'S SEARCH QUERY` block prepended to the user
message. Source of truth is `baselines/qwen35_judge/prompts/naive_query.md`,
pushed to LangSmith Hub as `-/qwen35-judge-naive-query:v1` so the exact prompt
behind these numbers is a named, inspectable commit — the eval script pulls it
from the Hub at run time rather than trusting a local copy.

Not fine-tuned. This is the frozen-weights baseline number to fine-tune against.

## Headline result

**Pair ROC-AUC 0.5888, decision accuracy 0.5700, all 200 real pairs, 200/200
scored (no failures).** Middling: edges out Voyage-4-large (0.5726) and
several open-weight embedding baselines by a small margin, but sits well
below the strongest models in `docs/baseline-results-real200.md`, and well
below `gemini-3.1-flash-lite` naive (0.6177 on the same all-200 population,
`docs/llm-judge-experiment.md`) despite that judge not getting the query at
all.

| Model | Pair AUC | Decision Acc | Hard-neg AUC | Easy-neg AUC | Gets query? |
|---|---|---|---|---|---|
| gemini-3.1-flash-lite (naive, no query) | 0.6177 | 0.6050 | — | — | no |
| Voyage-4-large (production) | 0.5726 | — | 0.5422 | 0.6540 | yes |
| **Qwen3.5-4B (naive + query, this run)** | **0.5888** | **0.5700** | **0.6271** | **0.5624** | **yes** |
| Qwen3-Embedding-8B (open) | 0.5529 | — | 0.4680 | 0.7208 | yes |

(Top two rows from `docs/llm-judge-experiment.md` / `docs/baseline-results-real200.md`
for reference — not re-run in this session, cited as published.)

## What's actually interesting here

**Hard negatives beat easy negatives (0.6271 vs 0.5624)** — same inversion
`gemini-3.1-flash-lite` showed and no embedding baseline does. That's evidence
this model, too, is reasoning about the match rather than measuring lexical
overlap between the two profile texts, even at 4B parameters and even with
the query included. This is the one number worth carrying into a fine-tune
decision: there is real, non-lexical signal for a 4B model to sharpen.

**Its stated confidence is worthless** — mean confidence 90.2, 90.4 when
correct vs 89.9 when wrong. Same failure mode `llm-judge-experiment.md`
documented for `gemini-3.1-flash-lite`. Don't use the confidence field for
ranking; only the yes/no.

**Adding `searchQuery` did not clearly help.** 0.5888 is worse than
`gemini-3.1-flash-lite`'s 0.6177 without the query — but this is two
different models, so this run alone cannot isolate the query's effect. A
same-model with/without-query ablation (swap `naive_query.md` for the
query-free `naive` prompt, same Qwen3.5-4B, same run) is the next thing to
run before concluding anything about the query itself.

## Debugging notes (useful before extending this script)

Three bugs hit building the Modal runner, in case they recur:

1. **`add_local_python_source` only mounts `.py` files.** The local prompt
   `.md` (only used as an import-time fallback; the real prompt is pulled
   from LangSmith Hub) wasn't shipped to the container until the `prompts/`
   dir was mounted explicitly via `add_local_dir`.
2. **`transformers` didn't yet recognize the `qwen3_5` architecture** — the
   model is new enough (~Feb/Mar 2026) that no pinned release supports it.
   Fixed by installing `transformers` from the GitHub `main` branch, which
   in turn needed `git` added to the (otherwise git-less) `debian_slim`
   Modal image via `apt_install("git")`.
3. **Qwen3.5 defaults to "thinking" mode** — with `max_new_tokens=300` every
   single pair failed identically (`no JSON object in model output`) because
   the model spent its whole budget on paragraphs of chain-of-thought before
   ever reaching the JSON verdict. Fixed with `enable_thinking=False` in
   `tokenizer.apply_chat_template(...)`, plus a bump to `max_new_tokens=500`
   as margin.

## What this does not show

- **Not deployable as-is** — same caveat as `llm_judge`: a per-candidate LLM
  call is out of scope under the <100 ms serving budget. Fine-tuning this
  model is the path toward something deployable, not this frozen-weights run.
- **One model, one prompt, n=200, no retrieval metrics** — same shape
  limitation as `llm_judge`, and for the same reason (no shared vector space
  to rank a candidate corpus with).
- **Query effect not isolated** — see above; this run alone doesn't prove the
  query helped or hurt.

## Reproducing

```bash
# push the prompt (only needed once, or after editing naive_query.md)
python -m baselines.qwen35_judge.push_prompt --tag v1

# run on all 200 real pairs (Modal A10G, ~25-30 min sequential, no batching)
modal run baselines/qwen35_judge/modal_eval.py --split all --run-id qwen35_4b_naive_query

# pull the full metrics.json down locally
modal volume get dorby-qwen35-judge-eval qwen35_4b_naive_query/metrics_all.json \
  artifacts/qwen35_judge/metrics_all.json
```

## Suggested next steps

1. **Same-model query ablation** — rerun with the query-free `naive` prompt on
   this same Qwen3.5-4B to isolate whether the query helped or hurt, before
   drawing any conclusion from the comparison above.
2. **Fine-tune it.** This is the frozen-weights baseline (0.5888) to fine-tune
   against — LoRA SFT on real accept/decline pairs, scored on the same 200-pair
   population, is the natural next step per this session's discussion.
3. **Batch/parallelize generation** if this script is run again at any scale —
   `judge_one` currently runs strictly sequentially with no concurrency,
   unlike `llm_judge`'s 8-way threaded API calls, which is why this 200-pair
   run took ~25-30 minutes against Qwen3.5-4B's fast token-generation on an A10G.
