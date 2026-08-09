# LLM-judge candidate field-permutation experiment

Generated: 2026-08-08. Code: `baselines/llm_judge_candidate_pos_back/`,
`baselines/llm_judge_candidate_pos_look/`, `baselines/llm_judge_candidate_back_look/`.
Isolated variations of `baselines/llm_judge_with_pos_look_pos_back_look` (the
focused-prompt experiment, see `docs/llm-judge-focused-prompt-experiment.md`)
— new packages, not edits, per the "ML / data-science experiments" isolation
rule in CLAUDE.md.

## What this tests

The focused prompt fixes the seeker at `positioning` + `lookingFor` and
gives the candidate all three of `positioning` + `background` +
`lookingFor` (pair AUC 0.6451 on the 200 real pairs). This experiment holds
the seeker fixed and the searchQuery included, and drops the candidate to
each of the three possible **two-field** combinations, to see which
candidate field matters most and how each pairing trades off hard-neg vs.
easy-neg AUC. Direct Google API only (`gemini-3.1-flash-lite`), matching the
other focused-prompt experiments. Scored on all 200 real pairs only (no
holdout run for this sweep).

## Results

**200 real pairs (all):**

| Candidate fields | Pair AUC | Hard-neg AUC | Easy-neg AUC |
|---|---|---|---|
| positioning + background + lookingFor (full, focused prompt) | **0.6451** | **0.6711** | 0.6132 |
| background + lookingFor (drop positioning) | 0.6356 | 0.6449 | 0.6390 |
| positioning + background (drop lookingFor) | 0.6274 | 0.6419 | 0.6112 |
| positioning + lookingFor (drop background) | 0.6274 | 0.5792 | 0.7082 |

## Reading the result

**No two-field candidate beats the full three-field candidate** — every
permutation drops pair AUC below 0.6451, confirming each of the three
candidate fields is contributing something, not just adding noise.

**Dropping `positioning` hurts least.** `background + lookingFor` is the
best of the three two-field configs (0.6356, closest to the full 0.6451)
and keeps a hard-neg AUC (0.6449) close to the full config's (0.6711) —
`positioning` (a static "who I am" statement) is the least load-bearing
candidate field here, echoing the seeker-side finding
(`docs/llm-judge-seeker-field-isolation-experiment.md`) that `positioning`
alone is the weakest seeker field too.

**Dropping `background` hurts hard-negative discrimination the most.**
`positioning + lookingFor` has both the lowest pair AUC (tied at 0.6274)
and by far the worst hard-neg AUC (0.5792 — the only one of the three
two-field configs where hard-neg AUC drops *below* easy-neg AUC, 0.7082).
That's the same pattern every embedding baseline shows and the opposite of
what makes the LLM judge distinctive elsewhere in this project — losing
`background` seems to push the judge back toward surface/lexical matching
on the harder cases specifically. `background` is what lets the model tell
a lexically-similar-but-wrong candidate apart from a real one.

**`positioning + background` (drop `lookingFor`) sits in between** — same
pair AUC as `positioning + lookingFor` (0.6274) but a much better hard-neg
AUC (0.6419 vs 0.5792), suggesting `background` is doing more of the
hard-negative work than `lookingFor` on the candidate side (candidate
`lookingFor` is two-way-fit signal, not the primary discriminator — see the
system prompt's "two-way fit" clause).

**Net effect: keep all three candidate fields if hard-neg AUC matters, but
if forced to drop one, drop `positioning` first, `background` last.** The
ranking on both pair AUC and hard-neg AUC is consistent:
`background+lookingFor` > `positioning+background` ≈ `positioning+lookingFor`
(same pair AUC, but positioning+lookingFor's hard-neg collapse makes it the
worst practical choice of the three).

## Reproducing

```bash
python -m baselines.llm_judge_candidate_pos_back.eval \
  --data-dir data --env-file .env --backend google --split all
python -m baselines.llm_judge_candidate_pos_look.eval \
  --data-dir data --env-file .env --backend google --split all
python -m baselines.llm_judge_candidate_back_look.eval \
  --data-dir data --env-file .env --backend google --split all
```
