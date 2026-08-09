# Judge prompt evolution, focused seed — run 10 (`evo_focused_001`)

Generated: 2026-08-08. Code: `judge_prompt_evolution_focused/` (isolated
package — nothing under `judge_prompt_evolution/`, `baselines/`, or `data/`
was edited; verified with `git diff`, empty). This is the tenth prompt-evolution
run in the project and the first to evolve the **focused** judge prompt rather
than the naive one. The previous nine are in
`docs/judge-prompt-evolution-experiment.md`.

## What changed vs. the previous nine runs

The loop itself is unchanged: 20 rounds, an optimizer LLM sees the current
judge prompt plus a small batch of real labeled pairs and rewrites the prompt,
no AUC feedback anywhere inside the loop. Four deliberate differences:

1. **Seed = the focused prompt**, not naive. The focused prompt gives the
   judge the `searchQuery` and trims the profile fields (seeker:
   `positioning` + `lookingFor`; candidate: `positioning` + `background` +
   `lookingFor`). It scores **0.6451** pair AUC on all 200 real pairs — a
   higher bar than naive's 0.6177. Source:
   `docs/llm-judge-focused-prompt-experiment.md`.
2. **The optimizer sees the focused field set, not complete profiles.** This
   is the substantive code change (`sampling.py::Example.render`). In the
   previous nine runs the optimizer was shown both people's full profiles and
   never the query, while the judge it was writing for saw the same. Here both
   sides had to move together — otherwise the optimizer writes rules about
   fields the judge is never shown.
3. **Optimizer = `gemini-3.1-flash-lite` via the direct Google API**, from
   round 1, same model and backend as the final scoring run.
4. **Examples sampled from all 200 real pairs.** Chosen deliberately: this
   prompt's intended downstream job is labeling *new* synthetic pairs
   (`synth_pipeline/pairing_rrf/`), not scoring these 200, so a richer example
   pool costs nothing that matters for that use. The consequence is recorded
   plainly rather than dressed up — the end-of-run AUC is measured on pairs the
   optimizer saw, so it answers "did the loop improve the prompt", not "does
   this generalize to unseen pairs".

Everything else matches **evo_006**, the best clean run of the previous nine:
4 examples per round (2 accepted / 1 hard-declined / 1 easy-declined, hardness
never disclosed to the optimizer), gentle summarizer every 5 rounds, v2-style
meta-prompt, contract auto-repair on.

Two prompt files needed adapting to the focused framing, since both previously
hardcoded "two complete profiles": `prompts/meta_optimizer.md` (now describes
the exact six fields plus the query, and forbids instructing the judge to use a
field it will not be shown) and `prompts/summarizer_gentle.md` (one line).

## Headline result: the focused seed also survives, by the widest margin yet

All 200 real pairs, `gemini-3.1-flash-lite`, direct Google API, temperature 0,
200/200 scored, no failures.

| Attempt | Pair ROC-AUC | Decision acc. | F1 | Δ vs. seed |
|---|---|---|---|---|
| **Focused seed (unevolved), re-scored here as a control** | **0.6474** | 0.5950 | 0.6197 | — |
| Focused seed, previously published number | 0.6451 | 0.5950 | 0.6197 | — |
| evo_focused_001 (evolved, 20 rounds) | 0.5885 | 0.5900 | 0.5543 | **−0.0589** |

The seed was re-scored through this package's own code path rather than
trusting the published 0.6451: it came back **0.6474**, a 0.0023 difference
attributable to model nondeterminism, which confirms the eval path is
equivalent and the gap below is real rather than a plumbing artifact.

**Ten runs, ten losses.** No automatic prompt-evolution run in this project
has ever beaten its own seed. This is also the largest gap of the ten — the
previous worst was evo_004 at −0.0477, and the best was evo_006 at −0.0072.
Starting from a stronger prompt did not help; it gave the loop more to destroy.

Full metrics:
`artifacts/judge_prompt_evolution_focused/evo_focused_001/eval/gemini_gemini-3_1-flash-lite/metrics_all.json`.

## The interesting part: the whole loss is the confidence channel going dead

Decision accuracy fell 0.5950 → 0.5900, five thousandths — essentially
unchanged. Pair AUC fell 0.6474 → 0.5885. Those two metrics read different
parts of the verdict: accuracy uses only `match` (yes/no), while pair AUC uses
the **confidence-signed score** (`verdict_to_score` — a "yes" at 90 outranks a
"yes" at 60, and both outrank any "no").

Scoring each verdict set both ways isolates it exactly:

| | yes/no alone | confidence-signed | what confidence contributes |
|---|---|---|---|
| Seed | 0.5950 | **0.6474** | **+0.0524** |
| Evolved | 0.5900 | 0.5885 | **−0.0015** |

The evolved prompt's yes/no calls are as good as the seed's (0.5900 vs
0.5950). Its **confidence numbers have stopped carrying any information** —
layering them on top moves AUC by −0.0015, i.e. nothing. In the seed, that
same channel is worth +0.0524, which is most of what makes the focused prompt
good. The loop did not make the judge worse at deciding; it destroyed the
judge's ability to say how sure it was.

The cause is visible in the final prompt. The seed spells out what
`confidence` means:

> "confidence" is confidence in the answer you gave, not the probability of
> "yes": answering "no" with confidence 90 means you are 90% sure it is not a
> good match. Use the full 0-100 range - say 55 when it is close to a
> coin-flip and 95 only when it is clear-cut.

The evolved prompt compressed the entire output contract into one trailing
sentence:

> Respond with a single JSON object containing: 'reasoning' (2-4 sentences
> explaining your assessment), 'match' ('yes' or 'no'), and 'confidence'
> (0-100).

All three keys survive, so the judge still answers and `validate_contract`
still passes — but the definition of confidence is gone.

Worth noting what did **not** happen, since it was the first guess: the
confidence values did not collapse into a single number. Their spread actually
widened slightly (sd 4.34 evolved vs 3.88 seed; 4 distinct values vs 5, both
clustered in 85-100). The model still emits varied confidences — they simply no
longer track whether it is right. Without the "confidence in the answer you
gave, not the probability of yes" clause, a plausible reading is that
confidence drifts toward meaning something closer to "how good is this match",
which is redundant with `match` rather than orthogonal to it. That is a
hypothesis this run does not test.

The yes-rate also fell, 56.5% → 42.0%, consistent with the more
rejection-oriented rubric described below.

**This is a bug in `validate_contract`, not just an outcome.** It checks that
the three key names and the word JSON appear; it does not check that the
confidence semantics survive. Logged as failure mode #9 below.

## Negative-hardness slices

| Slice | n | Seed | Evolved | Δ |
|---|---|---|---|---|
| hard-neg | 50 | 0.6590 | 0.6018 | −0.0572 |
| easy-neg | 25 | 0.6570 | 0.6014 | −0.0556 |

The degradation is uniform across both slices — this is not a case of losing
one kind of negative. Note that on this all-200 population the seed's hard and
easy slices are already level (0.659 vs 0.657); the hard-above-easy inversion
reported for the focused prompt in `docs/llm-judge-focused-prompt-experiment.md`
(0.6784 vs 0.5603) is a 69-pair-holdout result on a different population, not a
claim this run contradicts. Hard negatives are the only population that exists
in production, so the −0.057 there is the number that matters most.

## What the 20 rounds actually did

Prompt length: **1,388 (seed) → 1,772 chars (final)**, a well-behaved sawtooth
— growth to 2,400-3,500 chars, then a gentle summarize pass cutting it back,
four times. No runaway 25KB spike like evo_001, no aggressive over-compression
like evo_004.

**One contract repair fired, at round 9** (`contract_repaired: true`), where
the optimizer dropped `reasoning`/`confidence` and any mention of JSON
entirely. The auto-repair added the block back and the run self-healed — the
same mechanism that took evo_008 from evo_007's 13-of-20 failures down to
1-of-20, working as designed.

Reading the rationales, the loop spent its 20 rounds oscillating over the same
few ideas rather than converging. "Principle of Asymmetric Value" was
introduced in round 8, re-introduced in round 10, and refined in round 12;
"Principle of Direct Utility" appears in rounds 2, 10, and 15; the
peer-to-peer / "both parties seeking the same outcome" trap is added in rounds
3, 12, and 16. Each round sees four new examples, patches the rubric toward
them, and the next round patches back. The three summarize passes merged
these into broader principles each time, and the next five rounds re-split
them.

The final rubric is coherent and reads well — three principles (Functional
Utility and Strategic Fit, Mutual Value Exchange, Proven Capacity). It is also
noticeably more rejection-oriented than the seed ("a match must be rejected
if", "Do not approve matches based on generalist potential"), consistent with
the recall drop (0.5100, TP=51 FN=49): it says no more often and more firmly.
That mirrors the `calibrated` finding from the original LLM-judge experiment —
making the judge more skeptical does not make it more discriminating.

## Failure mode #9 (new): a contract can degrade while staying valid

Adding to the eight in `docs/judge-prompt-evolution-experiment.md`:

`optimizer.py::validate_contract` treats the output contract as present if the
strings `reasoning`, `match`, `confidence` and a mention of JSON all appear. A
one-line restatement of the contract passes that check while silently dropping
the confidence-scale definition — which is load-bearing for every
confidence-signed metric downstream. The prompt stayed valid and the *scoring
signal* degraded.

If this loop is run again, `validate_contract` should assert the confidence
*semantics* survive (the "not the probability of yes" clause and the
full-range instruction), not just the three key names — or, more simply, treat
the contract as immutable and re-append it verbatim every round rather than
letting the optimizer restate it.

## Reproducing

```bash
source .venv/bin/activate

# 20 rounds (defaults are the recipe above; ~$0.05 in Gemini calls)
python -m judge_prompt_evolution_focused.run --run-id evo_focused_001

# score the evolved prompt on all 200 real pairs
python -m judge_prompt_evolution_focused.eval_evolved \
  --run-id evo_focused_001 --backend gemini --model gemini-3.1-flash-lite

# seed control through the identical code path
python -m judge_prompt_evolution_focused.eval_evolved \
  --iteration-path artifacts/judge_prompt_evolution_focused/evo_focused_001/iterations/00_seed.json \
  --backend gemini --model gemini-3.1-flash-lite \
  --artifacts-dir artifacts/judge_prompt_evolution_focused/evo_focused_001/eval_seed_control

python -m pytest tests/test_judge_prompt_evolution_focused.py -q
```

Needs `GEMINI_API_KEY` in `.env`. Verdicts cache per prompt hash, so re-runs
are free.

## What this does and does not show

- It **does** show that the focused prompt, like naive and `structured_cot`
  before it, is not improved by this optimization loop — now across three
  different seeds and ten runs.
- It **does** localize this particular loss to one channel: the yes/no
  survived intact and the confidence signal died, with the dropped
  confidence-definition paragraph as the proximate suspect. That is a concrete,
  fixable mechanism rather than "evolution just doesn't work here" — though the
  fix is untested, so treat the causal link as strongly indicated, not proven.
- It **does not** measure generalization. Examples came from all 200 pairs by
  design; the AUC is on pairs the optimizer saw.
- It **does not** rule out that a loop with accuracy feedback inside it would
  do better. Every run so far has been blind by construction — the optimizer
  has never once been told whether its previous edit helped.
