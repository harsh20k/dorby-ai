# LLM-judge seeker field-isolation experiment

Generated: 2026-08-08. Code: `baselines/llm_judge_lookingfor_only/`,
`baselines/llm_judge_positioning_only/`. Isolated variations of
`baselines/llm_judge_with_pos_look_pos_back_look` (the focused-prompt
experiment, see `docs/llm-judge-focused-prompt-experiment.md`) — new
packages, not edits, per the "ML / data-science experiments" isolation rule
in CLAUDE.md.

## What this tests

The focused prompt gives the seeker `positioning` + `lookingFor` (candidate:
`positioning` + `background` + `lookingFor`) and scores 0.6451 pair AUC on
the 200 real pairs / 0.6530 on the holdout — the best LLM-judge config found
so far (`docs/llm-judge-seeker-background-experiment.md` already showed
*adding* seeker `background` makes it worse). This experiment goes the other
direction: drop to **one** seeker field at a time, to isolate how much of the
focused prompt's result comes from `lookingFor` alone versus `positioning`
alone. Candidate fields, the searchQuery, and the system prompt are all held
identical to the focused experiment — only the seeker field set changes.
Direct Google API only (`gemini-3.1-flash-lite`), same backend the focused
experiment found more trustworthy than OpenRouter.

## Results

| Config | Holdout (n=69) Pair AUC | Holdout Hard-neg AUC | 200-pair Pair AUC |
|---|---|---|---|
| Focused (positioning + lookingFor) | 0.6530 | 0.6784 | 0.6451 |
| **Seeker: lookingFor only** | **0.6698** | **0.7164** | 0.6327 |
| Seeker: positioning only | 0.5914 | 0.5621 | 0.6087 |

## Reading the result

**`lookingFor` is carrying the focused prompt, and on the holdout it does
slightly *better* alone than combined with `positioning`.** Dropping the
seeker to `lookingFor` only *raised* holdout pair AUC (0.6530→0.6698) and
hard-neg AUC (0.6784→0.7164) — the single best hard-neg AUC of any LLM-judge
config tested in this project so far. On the 200-pair population it's close
behind the two-field focused version (0.6451→0.6327), not a clean win there,
but the holdout — the population the project's headline numbers are judged
on — favors the single-field version.

**`positioning` alone is much weaker and roughly matches chance-plus on the
holdout.** 0.5914 holdout pair AUC, hard-neg AUC 0.5621 — both well below
both other configs, and closer to the embedding baselines' typical hard-neg
degradation than to the LLM judge's usual pattern of hard-neg ≥ easy-neg.
`positioning` (a fairly static "who I am" statement) evidently isn't where
the model finds its matching signal; `lookingFor` (what the seeker is
explicitly asking for) is.

**Consistent with the seeker-background finding.** `docs/llm-judge-seeker-
background-experiment.md` found adding a third field (`background`) to the
seeker hurt performance; this experiment shows the two fields already in the
seeker set aren't equal contributors either — one (`lookingFor`) is doing
essentially all the work, and the other (`positioning`) is close to inert or
mildly diluting when combined with it (200-pair AUC dips slightly with
`positioning` added: 0.6327→0.6451, a ~0.02 gain — small enough that the
combined two-field version and lookingFor-alone are close on this
population, while the holdout favors dropping `positioning` entirely).

**Net effect: `lookingFor` alone is a legitimate simpler alternative to the
two-field focused prompt, particularly if hard-neg AUC is the metric that
matters most.** It's the best hard-neg AUC found in the project to date. The
two-field focused version remains marginally ahead on the 200-pair AUC, so
this isn't an unambiguous replacement — but it establishes `lookingFor` as
the field driving the focused prompt's advantage over full-profile
baselines, not `positioning`.

## Reproducing

```bash
python -m baselines.llm_judge_lookingfor_only.eval \
  --data-dir data --env-file .env --backend google --split all
python -m baselines.llm_judge_lookingfor_only.eval \
  --data-dir data --env-file .env --backend google --split holdout

python -m baselines.llm_judge_positioning_only.eval \
  --data-dir data --env-file .env --backend google --split all
python -m baselines.llm_judge_positioning_only.eval \
  --data-dir data --env-file .env --backend google --split holdout
```
