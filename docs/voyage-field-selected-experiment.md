# Voyage field-selected experiment — same fields + query as the focused LLM judge

Generated: 2026-08-08. Code: `baselines/voyage_large_field_selected/`,
`baselines/voyage_nano_field_selected/`. Isolated variation of
`baselines/voyage_large` / `baselines/voyage_nano` (new packages, not edits —
see the "ML / data-science experiments" isolation rule in CLAUDE.md).

## What this tests

`docs/llm-judge-focused-prompt-experiment.md`'s focused LLM judge trims each
profile to a specific field subset and adds the searchQuery: seeker =
`positioning` + `lookingFor` + query, candidate = `positioning` +
`background` + `lookingFor`. Every other Voyage baseline in this project
embeds the **complete** profile instead. This experiment re-runs Voyage-4-nano
and Voyage-4-large with the LLM judge's exact field selection, so the two
approaches (LLM judge vs. embedding cosine) can be compared on **identical
inputs**, not just identical pairs — isolating "does the field selection
itself help/hurt an embedding model" from "is an LLM judge better at this
task than an embedding model."

Text packing: `baselines/voyage_large_field_selected/text.py` /
`baselines/voyage_nano_field_selected/text.py` (deliberately duplicated
between the two packages and from the LLM-judge prompt module — see each
file's docstring for why). Both packages reuse the frozen encoders
(`VoyageLargeEncoder`, `VoyageNanoEncoder`) and `baselines/metrics.py`
unmodified — only the text packing changes.

Voyage-4-nano ran on Modal (A100-40GB; the L4 default OOM'd on this
workload, same failure mode already documented for 7-8B HF models in
`docs/hf-embedding-baseline-findings.md` — see
`baselines/voyage_nano_field_selected/modal_eval.py`). Voyage-4-large ran
locally via the API (cheap, disk-cached).

## Results

**200 real pairs (all)** — the population this experiment was run for:

| Model | Pair AUC | Hard-neg AUC | Easy-neg AUC | Acc @ 0.5 |
|---|---|---|---|---|
| Voyage-4-large, full profile (production config) | 0.5726 | 0.5422 | 0.6540 | 0.505 |
| **Voyage-4-large, field-selected** | **0.5961** | 0.5250 | 0.6612 | 0.495 |
| Voyage-4-nano, full profile | 0.5614 | — | — | — |
| **Voyage-4-nano, field-selected** | **0.5658** | 0.4550 | 0.6768 | 0.500 |
| LLM judge: focused, direct Google API | **0.6451** | — | — | 0.595 |
| LLM judge: focused, OpenRouter | 0.6177 | — | — | 0.590 |
| LLM judge: naive (no query, full profile) | 0.6177 | — | — | — |

**69-pair matched holdout** (for comparison against the rest of the project's
headline numbers):

| Model | Pair AUC | Hard-neg AUC | Easy-neg AUC |
|---|---|---|---|
| Voyage-4-large, full profile (production) | 0.6086 | 0.6017 | 0.6000 |
| **Voyage-4-large, field-selected** | **0.6431** | 0.5586 | 0.7483 |
| Voyage-4-nano, full profile | 0.5793 | 0.5707 | 0.6207 |
| **Voyage-4-nano, field-selected** | **0.6060** | 0.5034 | 0.7931 |
| LLM judge: focused, direct Google API | 0.6530 | 0.6784 | 0.5603 |
| LLM judge: focused, OpenRouter | 0.6203 | 0.6517 | 0.5414 |

## Reading the result

**Field selection helps both Voyage models, on both populations.** Trimming
to `positioning`/`lookingFor`/`background` and keeping the query raised pair
AUC for Voyage-4-large (0.5726→0.5961 on 200 pairs, 0.6086→0.6431 on
holdout) and Voyage-4-nano (0.5614→0.5658 on 200 pairs, 0.5793→0.6060 on
holdout) in every comparison run. This is the opposite of what
`docs/field-ablation-voyage-nano.md`-style single-field-removal experiments
might predict — here, *removing* `notes`, `introPreferences`,
`personalPreferences`, and `meetingAndSchedulingPreferences` entirely (not
just one of them) net improves ranking, suggesting some of that text is
diluting the cosine signal rather than adding to it, at least for these
three retained fields.

**But field selection helps embeddings on easy negatives, not hard ones —
the opposite of what happened for the LLM judge.** Both Voyage models'
hard-neg AUC *drops* under field selection (large: 0.6017→0.5586 holdout;
nano: 0.5707→0.5034 holdout) while easy-neg AUC rises sharply (large:
0.6000→0.7483; nano: 0.6207→0.7931). That's consistent with every other
embedding-baseline finding in this project (`docs/possible-bugs.md` #3,
the LLM-judge-experiment findings): cosine similarity leans on lexical/topical
overlap, and a shorter, more topic-dense text (three fields instead of eight)
makes that overlap *stronger*, not weaker — it doesn't teach the model to
resist surface similarity the way the LLM judge's prompt does.

**Even with identical inputs, the LLM judge still leads on the metric that
matters most.** On the 69-pair holdout, the focused judge (direct API)
reaches hard-neg AUC 0.6784 vs. field-selected Voyage-4-large's 0.5586 —
a wide gap, on the same text both models were given. This is the cleanest
evidence yet in this project that the LLM judge's advantage isn't about what
information it sees, but about what it does with it: it doesn't just have
better inputs, it makes better use of the same ones.

**On overall pair AUC, field-selected Voyage-4-large (holdout: 0.6431) now
sits above the naive no-query LLM judge (0.6358)** and close to the focused
judge via OpenRouter (0.6203) — worth noting since it means Voyage-4-large,
given the right text and the query, is more competitive with LLM judges than
the project's earlier headline comparisons (full-profile Voyage vs. no-query
LLM judge) suggested. It still trails the focused judge via the direct
Google API (0.6530) and Qwen3-Embedding-8B (0.6595).

**Update:** `docs/qwen3-embedding-field-selected-experiment.md` runs this
same field selection on Qwen3-Embedding-8B and finds an even bigger jump
(0.6595→0.6862 holdout) — now the best overall pair AUC in the project,
though with the same hard-neg/easy-neg tradeoff documented here.

## Reproducing

```bash
# Voyage-4-large, field-selected, 200 real pairs (local, API, cached)
python -m baselines.voyage_large_field_selected.eval --data-dir data

# Voyage-4-large, field-selected, 69-pair holdout (free — same cache dir)
python -m baselines.voyage_large_field_selected.eval --data-dir data --holdout-only \
  --artifacts-dir artifacts/voyage_large_field_selected_holdout

# Voyage-4-nano, field-selected — Modal GPU only (local MPS not used for this
# experiment; L4 OOMs on this workload, use A100-40GB)
modal run baselines/voyage_nano_field_selected/modal_eval.py --gpu A100-40GB --batch-size 4 --max-length 4096
modal run baselines/voyage_nano_field_selected/modal_eval.py --holdout-only --gpu A100-40GB --batch-size 4 --max-length 4096
modal volume get dorby-voyage-nano-field-selected-eval real_all/metrics.json ./artifacts/voyage_nano_field_selected_modal/real_all/metrics.json
modal volume get dorby-voyage-nano-field-selected-eval real_holdout/metrics.json ./artifacts/voyage_nano_field_selected_modal/real_holdout/metrics.json
```

Chart: both populations are plotted side by side in the published
[LLM judge vs. embedding baselines](https://claude.ai/code/artifact/12f8f93b-8fc4-41e5-bc13-b05ce8ab45fa)
artifact (`docs/html/llm-judge-comparison.html`,
`scripts/build_llm_judge_browser.py`) — the two field-selected Voyage rows
are tagged "field-selected" in the full comparison table.
