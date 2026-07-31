# Judge prompt evolution — can automatic prompt optimization beat the naive judge?

Generated: 2026-07-31. Code: `judge_prompt_evolution/` (isolated package, no files
under `baselines/llm_judge/` or `data/` were edited). Four runs so far, each
isolating one variable — meta-prompt version, seed prompt, and a periodic
summarization step. Published trace (all four runs, selectable):
[Judge Prompt Evolution — evo_001 → evo_004](https://claude.ai/code/artifact/1e089702-6f90-4676-98e4-6c7e69813119).

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

`evo_001` ran 20 rounds split across two optimizer models mid-run: **Sonnet
4.5** (rounds 1-9) then **Deepseek-v4-pro** (rounds 10-20), switched
deliberately per standing cost guidance. `evo_002`, `evo_003`, and `evo_004`
all use Deepseek-v4-pro throughout, 20 rounds each. Every iteration across
all four runs, plus the meta-optimizer instructions (v1 and v2) and the
summarizer prompt, are versioned on LangSmith Hub (`judge-prompt-evolution` /
`judge-prompt-evolution-meta` repos) in addition to the local JSON trace
under `artifacts/judge_prompt_evolution/evo_00{1,2,3,4}/`.

## Headline result: five attempts to beat the naive judge prompt, five losses

Final prompt from each run scored against **all 200 real pairs**, same judge
model (`gemini-3.1-flash-lite`) and calling conventions as the reference run
(temperature 0, no `searchQuery`, complete profiles):

| Attempt | Pair ROC-AUC | Decision acc. | F1 | Δ vs. naive |
|---|---|---|---|---|
| **Seed (naive)** | **0.6177** | 0.6050 | 0.6326 | — |
| structured_cot (hand-designed) | 0.6100 | 0.5700 | 0.6560 | −0.0077 |
| evo_002 (v2 meta, naive seed) | 0.5918 | 0.5650 | 0.6167 | −0.0259 |
| evo_003 (v2 meta, structured_cot seed) | 0.5918 | 0.5600 | 0.6071 | −0.0259 |
| calibrated (hand-designed, holdout only) | 0.5901 | — | — | −0.0276 (holdout) |
| evo_001 (v1 meta, naive seed) | 0.5734 | 0.5650 | 0.5584 | −0.0443 |
| evo_004 (v2 meta, naive seed, +summarize/5) | 0.5700 | 0.5200 | 0.6571 | −0.0477 |

Every evolved prompt still clearly above chance, every one strictly worse
than whatever it started from. Full metrics:
`artifacts/judge_prompt_evolution/evo_00{1,2,3,4}/eval/metrics_all.json`.

## Run 1 (`evo_001`): what actually happened across the 20 rounds

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

**`structured_cot` confirmed on all 200 real pairs** (it only had a holdout
number before, 0.6336 vs. naive's 0.6409): **pair AUC 0.6100**, a −0.0077 gap
from naive — closely matching the holdout finding, and comfortably closer
than either evolution run at this point.

## Run 3 (`evo_003`): seeding from structured_cot instead of naive

If naive's own rubric-editing loop converges to ~0.59, does starting from a
different, higher-scoring structure (structured_cot's six weighted aspects,
0.6100 AUC) hold up better, or converge to the same place? Same v2
meta-prompt, same process, Deepseek-v4-pro throughout — the only change is
the seed (`--seed-source structured_cot`, importing
`baselines.llm_judge.prompt.SYSTEM_PROMPTS["structured_cot"]` read-only).

The v2 meta-prompt's hard constraint (plain `reasoning`/`match`/`confidence`
JSON, no aspect-weighting) forced an immediate restructure: **round 1
collapsed the 3,605-char six-aspect seed down to 1,960 chars**, discarding
the weighted-aspect format entirely rather than working within it, before
growing back up over the remaining 19 rounds to 8,669 chars.

**Result: pair AUC 0.5918 — identical to `evo_002` to four decimal places**,
despite starting 77 points of AUC higher (0.6100 vs. 0.6177) and from a
structurally different prompt. Two runs, two different starting prompts, two
different structures, the same final score. That is the most informative
single data point in this experiment so far: it suggests the ~0.59 result
isn't a property of the *seed* being flawed or the *starting structure* being
wrong — it's an attractor of the optimization process itself, regardless of
where it starts. Whatever this loop does over 20 rounds of small-batch
rubric editing pulls a judge prompt down to roughly the same place.

## Run 4 (`evo_004`): does forcing periodic consolidation help?

The v2 meta-prompt already asks the optimizer to "revise, not append" every
round, and `evo_002`/`evo_003` show it mostly complies. Does *forcing* an
explicit consolidation pass — not just relying on that instruction — do
better? Added `summarize_every=5`: after every 5th optimize round, a
separate, concise distillation-only prompt (`prompts/summarizer.md`, no
example batch, just "merge redundant rules into general principles, cut
restated points") rewrites the current prompt before the next round
continues. Same v2 meta-prompt, same naive seed as `evo_002` — isolating
summarization as the only new variable.

**The summarization step worked exactly as designed, three times.** Rounds
5, 10, and 15 each triggered a real distillation, genuinely merging
redundant paragraphs rather than truncating — e.g. round 10's step folded
"multiple paragraphs about two-way benefit, implied needs, and constraint
checking into concise principles" and cut the prompt from 3,552 to 1,496
chars (58% reduction).

**The fourth (final, round 20) summarization pass broke the output contract
— three times in a row.** Despite `prompts/summarizer.md`'s explicit hard
constraint to preserve the `reasoning`/`match`/`confidence` JSON format, all
three attempts at the round-20 summarize step dropped that entire paragraph,
compressing it away as if it were removable boilerplate:

| Attempt | Chars (in → out) | Kept the JSON contract? |
|---|---|---|
| 1st | 4,017 → 1,125 | No |
| 2nd (retry) | 4,017 → 1,125 | No |
| 3rd (retry) | 4,017 → **2,071** | **Yes** |

Same input every time (`optimize` round 20's output, unchanged), materially
different compression ratios and outcomes across three calls at
temperature 0.4 — not deterministic, but a real, repeatable failure mode:
**the more aggressively a prompt gets compressed, the more likely a
structural/format instruction gets treated as cuttable content alongside the
substantive rubric.** No other step in this whole experiment (60+ optimize
calls, 4 summarize calls across two runs) ever dropped the contract except
the two most-compressed calls here.

**Result, using the third (contract-valid) attempt's output: pair AUC
0.5700** — the *worst* AUC of any variant tried except the original,
already-diagnosed-as-overfit `evo_001` (0.5734). Despite ending with the
shortest, cleanest-looking final prompt of any run (2,071 chars) and a
summarization mechanism that worked correctly most of the time, forcing
periodic compression did not help — it cut real discriminating detail along
with the redundancy it was meant to remove, landing below both
non-summarized v2 runs (0.5918 each).

**Five attempts, five losses**, and the ranking so far: naive (0.6177) >
structured_cot (0.6100) > evo_002 ≈ evo_003 (0.5918) > evo_001 (0.5734) >
evo_004 (0.5700). Neither of the two specific fixes tried — generalizing
away from example-specific rules (`evo_002`/`evo_003`) or forcing periodic
consolidation (`evo_004`) — closed the gap to naive; the second one, in
fact, opened it slightly wider than the original overfitting fix already
had.

## Six bugs/failure modes found running this (worth knowing before reusing `judge_prompt_evolution/`)

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

5. **A genuinely empty model response** — `evo_004` round 20's first attempt
   returned zero-length content, which fails JSON parsing with a different,
   less informative error (`"Expecting value: line 1 column 1"`) than bug #4
   above. Unlike bug #4, this did not repeat on retry — a single `--resume`
   fixed it, consistent with ordinary API-level flakiness rather than a
   structural parsing issue.
6. **The summarizer can drop the required output contract under aggressive
   compression** — not a code bug (nothing crashed), but a real content
   failure mode: `evo_004`'s final summarize step compressed the 4,017-char
   round-20 prompt down to ~1,125 chars twice in a row, both times silently
   deleting the `reasoning`/`match`/`confidence` JSON-output paragraph
   despite `prompts/summarizer.md` explicitly listing it as a hard
   constraint to preserve. A third retry at a less extreme compression ratio
   (2,071 chars) kept the contract. See "Run 4" above — `validate_contract`
   catches this (it's why the problem was visible at all) but nothing
   currently retries automatically on a contract violation; each occurrence
   here was caught and re-run by hand.

All crashes happened mid-run with already-completed iterations safely on
disk; `run.py --resume` was added to continue from the last saved iteration
instead of re-paying for earlier rounds — used across all four runs
(`evo_001` at rounds 7 and 10, the latter also the point the optimizer model
was switched from Sonnet to Deepseek per standing cost guidance; `evo_002`
at round 11, then again at round 20 after the strict-JSON fix; `evo_004` at
round 20 for the empty-response retry, then twice more to re-run just the
round-20 summarize step until it kept the contract).

## What this does not show

- **Four variants, one example-sampling strategy.** 20 rounds, 4 examples
  each, hardness-balanced (internally) but not otherwise tuned, across all
  four runs. A different batch size or composition is untested.
- **The loop never got a held-out accuracy signal during optimization**, in
  any of the four runs. That was deliberate (see "Design" above), but is
  also very likely central to why every variant converges to roughly the
  same ~0.57-0.59 band regardless of seed or process tweak: nothing inside
  any of these loops could tell it a rule was over-specific, or that a
  compression pass had gone too far, until the single AUC check at the very
  end. A loop with an intermediate eval-and-revert step (scored on a
  further-held-out slice of train, never the real holdout) is the most
  promising untried variant given what four attempts now show.
- **`evo_001`'s two optimizer models are not isolated from each other** —
  since it switched from Sonnet to Deepseek mid-stream for cost reasons, not
  as a designed comparison, its result cannot cleanly attribute to either
  model alone. `evo_002`, `evo_003`, and `evo_004` all use Deepseek-v4-pro
  throughout, so they don't have this confound.
- **`evo_004`'s summarization result is from one seed (round 20) with a
  demonstrated 2-in-3 failure rate** at that specific compression ratio (see
  bug #6). A less aggressive summarizer instruction, or one that validates
  its own output and retries automatically, might land differently — this
  run used whichever attempt happened to keep the contract, not necessarily
  the best possible distillation.

## Reproducing

```bash
# evo_001 (v1 meta-prompt: hard/easy framing, "incremental sharpening")
python scripts/run_judge_prompt_evolution.py --run-id evo_001
python scripts/run_judge_prompt_evolution.py --run-id evo_001 --resume --optimizer-model deepseek/deepseek-v4-pro

# evo_002 (v2 meta-prompt: accepted/declined only, "revise the rubric, not append")
python scripts/run_judge_prompt_evolution.py --run-id evo_002 --optimizer-model deepseek/deepseek-v4-pro

# evo_003 (v2 meta-prompt, seeded from structured_cot instead of naive)
python scripts/run_judge_prompt_evolution.py --run-id evo_003 --seed-source structured_cot --optimizer-model deepseek/deepseek-v4-pro

# evo_004 (v2 meta-prompt, naive seed, forced distillation every 5 rounds)
python scripts/run_judge_prompt_evolution.py --run-id evo_004 --seed-source naive --optimizer-model deepseek/deepseek-v4-pro --summarize-every 5

# the AUC check against all 200 real pairs (isolated eval script, reads
# baselines/llm_judge/ read-only, writes to artifacts/judge_prompt_evolution/evo_00N/eval/)
python -m judge_prompt_evolution.eval_evolved --run-id evo_001   # ...evo_002, evo_003, evo_004

# structured_cot confirmed on all 200 real pairs (existing baselines/llm_judge
# code, unmodified — was previously only measured on the 69-pair holdout)
python -m baselines.llm_judge.eval --data-dir data --variant structured_cot --split all
```

Every iteration's exact prompt text, rationale, and which examples produced
it is in `artifacts/judge_prompt_evolution/evo_00{1,2,3,4}/iterations/*.json`
(evo_004 additionally has `NNs.json` files for its summarize steps) and
mirrored as LangSmith Hub commits. The published browser
(`artifacts/judge_prompt_evolution/evo_001/browser.html`) shows all four runs
via a toggle, including full seed-vs-final prompt text, the meta-prompt v1→v2
diff, and the summarizer prompt.

## Bottom line

**Naive stays the judge.** `docs/llm-judge-experiment.md`'s naive prompt
(pair AUC 0.6177 on all 200 real pairs) is still the best judge prompt found
anywhere in this project, having now beaten two hand-designed variants and
three independent automatic-optimization variants (five attempts total).
`structured_cot` (0.6100) remains the closest challenger by a wide margin
over any evolution run. Two specific fixes were tried against the
overfitting diagnosed in `evo_001` — generalizing the meta-prompt away from
example-specific rules, and forcing periodic consolidation — and neither
closed the gap; the loop appears to have a ~0.57-0.59 attractor regardless of
seed, structure, or these particular process changes. Do not promote any
evolved prompt to a labeling path (`synth_pipeline/pairing_rrf/` etc. should
keep using the naive framing already in place) until one actually beats
0.6177.
