# Judge prompt evolution — can automatic prompt optimization beat the naive judge?

Generated: 2026-07-31. Code: `judge_prompt_evolution/` (isolated package, no files
under `baselines/llm_judge/` or `data/` were edited). Two runs so far, both
against the same seed prompt and process, differing only in the
meta-optimizer instructions (see "Run 2" below). Published trace (updated to
show both runs):
[Judge Prompt Evolution — evo_001 / evo_002](https://claude.ai/code/artifact/1e089702-6f90-4676-98e4-6c7e69813119).

## What this tests

The LLM-judge experiment (`docs/llm-judge-experiment.md`) found that a plain,
unengineered ("naive") system prompt fed to `google/gemini-3.1-flash-lite`
scores pair ROC-AUC **0.6177** on all 200 real pairs — beating two more
sophisticated prompt variants (`calibrated`, `structured_cot`) that were tried
by hand. This experiment asks a different question: can an *automatic*
prompt-optimization loop, with no human editing the prompt directly, do better
than a human's best manual attempt?

**Design — pure prompt evolution, no accuracy feedback inside the loop.**
Each round: an "optimizer" LLM is shown the current judge prompt plus a fresh,
deliberately contrastive batch of 4 real labeled example pairs (2 accepted, 1
hard-declined, 1 easy-declined — hardness = token-Jaccard overlap between the
two profiles, median split), and asked to propose a revised prompt. The
revised prompt becomes the input to the next round, with a *different* sample
of examples. Batches are drawn without replacement from the **train split
only** (`data/synthetic/seed_split.json`) — the 69-pair holdout was never
shown to the optimizer, and no AUC/accuracy check ran at any point during the
20 rounds, by deliberate design (per instruction: inspect the evolved prompt
qualitatively first, run the accuracy check only afterward, as an explicitly
separate, confirmed step).

20 rounds total, split across two optimizer models mid-run: **Sonnet 4.5**
(rounds 1-9) then **Deepseek-v4-pro** (rounds 10-20), switched deliberately —
see "Two bugs found running this" below. All 20 iterations, plus the fixed
meta-optimizer instructions, are versioned on LangSmith Hub
(`judge-prompt-evolution` / `judge-prompt-evolution-meta` repos) in addition
to the local JSON trace under `artifacts/judge_prompt_evolution/evo_001/`.

## Headline result: run 1 (evo_001) lost on every metric; run 2 (evo_002) closed about half the gap but still lost

Final-round prompt scored against **all 200 real pairs**, same judge model
(`gemini-3.1-flash-lite`) and calling conventions as the reference run
(temperature 0, no `searchQuery`, complete profiles):

| Metric | evo_001 (v1 meta-prompt) | evo_002 (v2 meta-prompt) | Seed (naive, reference) |
|---|---|---|---|
| Pair ROC-AUC | 0.5734 | **0.5918** | **0.6177** |
| Decision accuracy | 0.5650 | 0.5650 | 0.6050 |
| Decision F1 | 0.5584 | 0.6167 | 0.6326 |
| Average precision | 0.5409 | 0.5610 | 0.5804 |
| Hard-neg AUC (all-200 split) | 0.6300 | 0.6157 | — (0.6466 on holdout pop.) |
| Easy-neg AUC (all-200 split) | 0.5454 | 0.5920 | — |

Both still clearly above chance, both strictly worse than the ~1KB unmodified
prompt either started from. Full metrics: `artifacts/judge_prompt_evolution/evo_00{1,2}/eval/metrics_all.json`.

## What actually happened across the 20 rounds

Prompt length by round (chars): **1,011 (seed) → 24,905 (round 10, peak,
Sonnet's last round, 32 numbered rules) → 6,420 (round 11, Deepseek's first
round) → 18,463 (round 20, final)**.

1. **Sonnet's 9 rounds were purely additive.** Every round appended 2-3 new
   numbered "rule" entries targeting whatever specific pair it had just been
   shown, and never once removed or merged anything. By round 9 there were 26
   such rules; round 10 pushed it to 32 at 24.9KB.
2. **The additions were example-specific, not general principles** — the
   clearest tell, quoted verbatim from round 6's rule 16: *"An investor who
   writes **$200k-$1M checks** as a follower at seed to Series A and prefers
   post-revenue companies with **$500k+ ARR** is not a fit for a pre-revenue
   founder raising **$500k** at pre-seed... Pay special attention to whether
   their sector focus (**sports, health, media**) actually overlaps..."* — the
   literal dollar figures and sector names from one training pair, baked in
   as if they were the rule itself, rather than abstracted into something like
   "verify the investor's check size and revenue bar actually match the
   founder's stage."
3. **Deepseek's first move, unprompted, was a 74% rewrite** — its round-11
   rationale states verbatim: *"Condensed the 32 specific rules into broader,
   principle-based categories to improve generalization and reduce cognitive
   load on the judge."* Nothing in the meta-optimizer instructions asked for
   consolidation; Deepseek did it on its own initiative on its very first
   turn. It then resumed the same additive pattern for its remaining 10
   rounds (6.4KB → 18.5KB), ending shorter than Sonnet's peak but still 18×
   the seed.
4. **The final AUC check shows the consolidation didn't fix the underlying
   problem.** Even Deepseek's more organized version — fewer, broader
   categories instead of 32 narrow ones — still lost to the 4-paragraph seed
   prompt. Structure and length were not the bottleneck; the loop as a whole
   was optimizing against a small, non-held-out sample of examples and
   overfit to it regardless of which model was doing the writing.

This is the same failure shape `docs/llm-judge-experiment.md` already found
twice by hand: `calibrated` (more true context) and `structured_cot` (more
structured reasoning) both scored below `naive`. A third, independent attempt
at "more sophisticated" — this time fully automated, with real contrastive
examples driving each edit — lost by an even larger margin.

## Run 2 (`evo_002`): rewriting the meta-prompt to push generalization directly

Manual inspection of `evo_001` pointed at the meta-optimizer instructions
themselves as a likely cause: the original text said *"treat this as
incremental sharpening, not a one-shot rewrite"* (encouraging pure addition)
and explicitly told the optimizer which examples were "hard" vs. "easy"
rejects (handing it a shortcut — label the pattern, don't find it). Rewrote
the meta-prompt by hand, collaboratively, in several passes:

1. Dropped the hard/easy-negative framing entirely — the optimizer now only
   ever sees "accepted" / "declined", never a hardness label. (`sampling.py`
   still stratifies the *sampling* by hardness internally, for batch
   diversity; it just no longer discloses that label in what the optimizer
   reads.)
2. Replaced "incremental sharpening" with an explicit instruction to **revise
   the rubric, not append to it** — reorganize, merge, or cut, and generalize
   any example-specific detail (a number, a name) into a principle instead of
   copying it in.
3. Cut everything not load-bearing: the 200/100/100 population framing, the
   "every round" language, and restated filler — down from ~280 words to
   ~180.

Full before/after text is in the published browser's "Meta-prompt" panel.
This is now pushed to LangSmith Hub as `judge-prompt-evolution-meta`, commit
tag `v2` (the original is tag-less, i.e. the repo's initial commit).

**Result: meaningfully better, still not a win.** `evo_002` ran all 20 rounds
on Deepseek-v4-pro only (no model switch this time, staying off Anthropic
models per standing cost guidance) and produced a much healthier growth
curve — **1,011 → 13,142 chars, smooth and monotonic**, no 25KB spike, no
sudden 74% rewrite (nothing to rewrite away). Manual spot-checks of the
rationales show genuine rubric edits ("broadened the network-value rule to
treat capital-seeking as a strongly implicit need") rather than new numbered
rules citing specific dollar figures. Pair AUC recovered from 0.5734 to
0.5918 (roughly half the gap to the seed's 0.6177), and F1 recovered nearly
all the way (0.6167 vs. seed's 0.6326). But it still lost on every headline
metric — see the table above.

**Four attempts, four losses — and the closest one is structured_cot, not
either evolution run.** `docs/llm-judge-experiment.md` only had
`structured_cot`'s holdout number (0.6336 vs. naive's 0.6409, a −0.0073 gap)
on record; confirming it on **all 200 real pairs** for direct comparability
with the evolution runs gives **pair AUC 0.6100** (decision accuracy 0.5700,
F1 0.6560) — a −0.0077 gap from naive's 0.6177, closely matching the holdout
finding and comfortably closer than either `evo_001` (−0.044) or `evo_002`
(−0.026):

| Attempt | Pair AUC (all 200) | Delta from naive |
|---|---|---|
| Seed (naive) | **0.6177** | — |
| structured_cot | 0.6100 | −0.0077 |
| evo_002 | 0.5918 | −0.0259 |
| calibrated | — (holdout only: 0.5901, −0.0508) | |
| evo_001 | 0.5734 | −0.0443 |

Two hand-designed variants and two independently-run automatic optimizations
— one of the latter explicitly engineered (based on inspecting the first
failure) to avoid the specific overfitting pattern found the first time —
and all four still land below the ~4-paragraph naive prompt on pair AUC.
Fixing the obvious overfitting mechanism in `evo_002` helped substantially
but did not flip the result, and it still didn't get as close as
`structured_cot`'s six-aspect scored rubric does. That's the motivation for
the next run: seed the evolution loop from `structured_cot` instead of
`naive` — start from the closest-performing alternative structure instead of
the winner, and see whether the loop can hold or improve on a 0.61-AUC
starting point rather than degrade a 0.6177 one.

## Two bugs found running this (worth knowing before reusing `judge_prompt_evolution/`)

1. **`optimizer.py` initially never passed an API key to `complete_json`** —
   the very first launch failed immediately with a `RuntimeError: Missing API
   key`, even though `OPENROUTER_API_KEY` was set in `.env`. Fixed by
   explicitly passing `api_key=os.getenv("OPENROUTER_API_KEY")` and
   `base_url=DEFAULT_OPENROUTER_BASE_URL` (`synth_pipeline.llm.complete_json`
   otherwise defaults to an empty key unless a `cfg` object is threaded
   through, which this call site didn't do).
2. **`max_tokens=3000` was too small once the prompt started growing.** The
   revised prompt naturally gets longer each round, so a fixed output cap
   sized for round 1 becomes a real failure mode by round 7: the completion
   got cut off mid-JSON-string and crashed the whole loop with a
   `JSONDecodeError`. Fixed by raising `optimizer_max_tokens` to 8000 and
   adding a 3-attempt retry with backoff around the optimizer call
   (`run_one_iteration` in `optimizer.py`).
3. **LangSmith tags can only point at one commit each** — reusing the bare
   `run_id` as a Hub commit tag across all 20 iterations 409-conflicted from
   the second push onward (`"Nothing to commit"` / `"Tag already exists"`).
   Fixed by giving every iteration a tag unique to itself
   (`f"{run_id}--iter-{NN}"`) in `hub.py::push_iteration_prompt`.
4. **`json.loads` strict mode rejects a raw newline inside a JSON string
   value** — found in `evo_002`, twice, at the exact same error location
   (`"Unterminated string starting at: line 2 column 21"`) across two
   separate runs, both times on the model's most complex response of the
   run. The location is structural (right after `"updated_prompt": "`
   opens), not content-dependent — Deepseek was emitting an unescaped
   literal newline inside the multi-paragraph prompt string it was
   returning, which Python's `json` module refuses to parse in strict mode
   regardless of retries, since the same input keeps producing the same
   failure. Fixed by adding a local lenient parser
   (`json.loads(text, strict=False)`, which permits control characters
   inside strings — the standard fix for this exact failure mode) in
   `optimizer.py`, calling OpenRouter directly via `langchain_openai`
   instead of through `synth_pipeline.llm.complete_json` (which is strict
   and shared elsewhere in the repo — fixed locally rather than touching
   that shared file, per the isolation rule).

All crashes happened mid-run with already-completed iterations safely on
disk; `run.py --resume` was added to continue from the last saved iteration
instead of re-paying for earlier rounds — used three times across the two
runs (`evo_001` at rounds 7 and 10, the latter also the point the optimizer
model was switched from Sonnet to Deepseek per standing cost guidance;
`evo_002` at round 11, then again at round 20 after the strict-JSON fix).

## What this does not show

- **One run, one seed, one example-sampling strategy.** 20 rounds, 4 examples
  each, hardness-balanced but not otherwise tuned. A different batch
  composition (e.g., more hard negatives per round, or explicitly instructing
  the optimizer to consolidate every N rounds rather than only append) might
  behave differently — untested here.
- **The loop never got a held-out accuracy signal during optimization.** That
  was deliberate (per the experiment's design goal — see "Design" above), but
  it is also very likely *why* it overfit: nothing in the loop could tell it
  a rule was over-specific until this final, single AUC check at the very
  end. A loop with an intermediate eval-and-revert step is a natural next
  variant, not attempted here.
- **Two optimizer models, not isolated from each other.** Because the run
  switched from Sonnet to Deepseek mid-stream (for cost reasons, not a
  designed comparison), this cannot cleanly attribute the final result to
  either model alone. The 74% consolidation is clearly attributable to
  Deepseek's first round specifically, but the overall AUC loss reflects the
  full 20-round chain, not one model in isolation.

## Reproducing

```bash
# evo_001 (v1 meta-prompt: hard/easy framing, "incremental sharpening")
python scripts/run_judge_prompt_evolution.py --run-id evo_001
python scripts/run_judge_prompt_evolution.py --run-id evo_001 --resume --optimizer-model deepseek/deepseek-v4-pro

# evo_002 (v2 meta-prompt: accepted/declined only, "revise the rubric, not append")
python scripts/run_judge_prompt_evolution.py --run-id evo_002 --optimizer-model deepseek/deepseek-v4-pro
python scripts/run_judge_prompt_evolution.py --run-id evo_002 --resume --optimizer-model deepseek/deepseek-v4-pro

# the AUC check against all 200 real pairs (isolated eval script, reads
# baselines/llm_judge/ read-only, writes to artifacts/judge_prompt_evolution/evo_00N/eval/)
python -m judge_prompt_evolution.eval_evolved --run-id evo_001
python -m judge_prompt_evolution.eval_evolved --run-id evo_002

# structured_cot confirmed on all 200 real pairs (existing baselines/llm_judge
# code, unmodified — was previously only measured on the 69-pair holdout)
python -m baselines.llm_judge.eval --data-dir data --variant structured_cot --split all
```

Every iteration's exact prompt text, rationale, and which examples produced
it is in `artifacts/judge_prompt_evolution/evo_00{1,2}/iterations/*.json` and
mirrored as LangSmith Hub commits. The published browser
(`artifacts/judge_prompt_evolution/evo_001/browser.html`) shows both runs
side by side, including full seed-vs-final prompt text and the meta-prompt
diff between v1 and v2.

## Bottom line

**Naive stays the judge, for now.** `docs/llm-judge-experiment.md`'s naive
prompt (pair AUC 0.6177 on all 200 real pairs) is still the best judge
prompt found anywhere in this project, having beaten two hand-designed
variants and two independent automatic-optimization runs. `structured_cot`
(0.6100) is the closest challenger by a wide margin over either evolution
run — which is why the next step is seeding the evolution loop from
`structured_cot` rather than `naive` (see `evo_003`, tracked separately once
run). Do not promote any evolved prompt to a labeling path
(`synth_pipeline/pairing_rrf/` etc. should keep using the naive framing
already in place) until one actually beats 0.6177.
