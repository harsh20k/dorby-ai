# LLM-judge seeker-background experiment

Generated: 2026-08-08. Code: `baselines/llm_judge_with_pos_look_back_pos_back_look/`.
Isolated variation of `baselines/llm_judge_with_pos_look_pos_back_look`
(the focused-prompt experiment, see `docs/llm-judge-focused-prompt-experiment.md`)
— new package, not an edit.

## What changed

Exactly one thing versus the focused prompt: the seeker gets `background`
added to its field set (`positioning` + `lookingFor` + `background`, was
`positioning` + `lookingFor`). Candidate fields (`positioning` +
`background` + `lookingFor`), the searchQuery, and — deliberately — the
system prompt text are all unchanged, so this isolates the effect of the one
field addition rather than conflating it with a prompt rewrite. Direct
Google API only (`gemini-3.1-flash-lite`), the backend the focused
experiment found more trustworthy than OpenRouter.

Prompt pushed to LangSmith Hub as
[`llm_judge_with_pos_look_back_pos_back_look`](https://smith.langchain.com/prompts/llm_judge_with_pos_look_back_pos_back_look/eae1dd7c).

## Results

| Config | Holdout (n=69) Pair AUC | Holdout Hard-neg AUC | 200-pair Pair AUC |
|---|---|---|---|
| Focused (no seeker background) | 0.6530 | 0.6784 | 0.6451 |
| **Focused + seeker background** | **0.6302** | **0.6690** | **0.6220** |

## Reading the result

**Adding seeker `background` made it worse, consistently, on both
populations.** Pair AUC dropped 0.6530→0.6302 on the holdout and
0.6451→0.6220 on all 200 pairs — small (~0.02–0.03) but the same direction
both times, not noise flipping sign. Hard-neg AUC dropped too (0.6784→0.6690).

One plausible read: `background` is more free-text/narrative than
`positioning`/`lookingFor`, so adding it to the seeker side likely increases
surface lexical overlap with the candidate's own `background` field (which
was already in the candidate set) — nudging the judge back toward the kind
of surface-similarity signal the focused prompt's smaller field set seemed
to help it avoid. Not confirmed here (would need the same
Jaccard-overlap-vs-AUC breakdown the embedding baselines get), but consistent
with `docs/voyage-field-selected-experiment.md`'s finding that this project's
easy/hard-negative split tracks lexical overlap directly.

**Net effect: don't add seeker `background` to this prompt.** The focused
variant without it remains the best LLM-judge configuration found so far.

## Reproducing

```bash
python -m baselines.llm_judge_with_pos_look_back_pos_back_look.eval \
  --data-dir data --env-file .env --split all
python -m baselines.llm_judge_with_pos_look_back_pos_back_look.eval \
  --data-dir data --env-file .env --split holdout
```
