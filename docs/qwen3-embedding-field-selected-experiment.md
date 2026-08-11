# Qwen3-Embedding-8B field-selected experiment

Generated: 2026-08-08. Code: `baselines/hf_embedding_field_selected/`.
Isolated variation of `baselines/hf_embedding` (new package, not an edit) —
same encoder machinery (`MODEL_REGISTRY`, `get_encoder_class`), only the
text packing changes, matching `baselines/voyage_large_field_selected/` and
`baselines/voyage_nano_field_selected/`.

## What this tests

The strongest embedding baseline in this project (`Qwen/Qwen3-Embedding-8B`,
`docs/hf-embedding-baseline-findings.md` — the first model of any kind here
to beat Boardy's own production model, Voyage-4-large, on full profiles)
re-run with the focused LLM judge's exact field selection: seeker =
`positioning` + `lookingFor` + searchQuery, candidate = `positioning` +
`background` + `lookingFor`, instead of the complete profile. Ran on Modal,
A100-40GB (A10G OOMs on this 8B model, same finding as
`baselines/voyage_nano_field_selected/modal_eval.py`).

## Results

| Config | Holdout (n=69) Pair AUC | Hard-neg AUC | Easy-neg AUC | 200-pair Pair AUC |
|---|---|---|---|---|
| Qwen3-Embedding-8B, full profile | 0.6595 | 0.6259 | 0.7586 | — |
| **Qwen3-Embedding-8B, field-selected** | **0.6862** | 0.5690 | 0.8552 | **0.5840** |

**0.6862 is the best pair AUC of any configuration tested in this entire
project** — ahead of Qwen3-Embedding-8B's own full-profile score (0.6595),
the best LLM judge (0.6530), and field-selected Voyage-4-large (0.6431).

## Reading the result

**Field selection helps Qwen3-Embedding-8B on the holdout, the same
direction as both Voyage models** (`docs/voyage-field-selected-experiment.md`)
— trimming to three fields plus the query improves ranking over the complete
profile, at least on this population.

**But the gain is entirely on easy negatives, and hard negatives get
worse.** Hard-neg AUC drops 0.6259→0.5690, easy-neg AUC rises sharply
0.7586→0.8552 — the identical pattern seen in both Voyage field-selected
runs. This is the same story repeating a third time: shorter, topic-dense
text makes lexical/topical overlap a stronger signal for cosine similarity,
which is free money on easy negatives (defined by low lexical overlap) but
actively hurts on hard negatives (defined by high lexical overlap that
doesn't reflect a real match) — the population that matters, since every
real negative in production is already a lexically-plausible false positive.

**So "best AUC" and "best on the population that matters" are two different
answers now.** Qwen3-Embedding-8B field-selected has the best *overall* pair
AUC in the project (0.6862); the LLM judge (focused, direct Google API) still
has the best *hard-negative* AUC (0.6784 vs. 0.5690) by a wide margin. If the
goal is the strongest offline classification number, this is the new leader.
If the goal is deployable signal on production's actual hard-negative
population, the LLM judge (not deployable under the <100ms budget as-is, but
informative as a reference/distillation target) is still ahead — and this
embedding model, being a frozen bi-encoder, *is* servable under that budget,
which the LLM judge is not.

**200-pair number is notably lower than the holdout number** (0.5840 vs.
0.6862) — a larger gap between splits than most other configs in this
project show. Worth treating the holdout number as the population-matched
one to trust (same caveat as everywhere else in this project) rather than
averaging the two.

## Reproducing

```bash
modal run baselines/hf_embedding_field_selected/modal_eval.py \
  --model Qwen/Qwen3-Embedding-8B --holdout-only --run-id qwen_qwen3-embedding-8b_holdout
modal run baselines/hf_embedding_field_selected/modal_eval.py \
  --model Qwen/Qwen3-Embedding-8B --no-holdout-only --run-id qwen_qwen3-embedding-8b_all

modal volume get dorby-hf-embedding-field-selected-eval qwen_qwen3-embedding-8b_holdout/metrics.json \
  ./artifacts/hf_embedding_field_selected_modal/qwen_qwen3-embedding-8b_holdout/metrics.json
modal volume get dorby-hf-embedding-field-selected-eval qwen_qwen3-embedding-8b_all/metrics.json \
  ./artifacts/hf_embedding_field_selected_modal/qwen_qwen3-embedding-8b_all/metrics.json
```
