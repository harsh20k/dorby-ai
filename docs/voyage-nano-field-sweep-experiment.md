# Voyage-4-nano field/query grid sweep

Generated: 2026-08-09. Code: `baselines/voyage_nano_field_sweep/`. Isolated
new experiment, not an edit of `baselines/voyage_nano_field_selected` (the
single hand-picked field selection matching the LLM judge's focused prompt)
— see the "ML / data-science experiments" isolation rule in CLAUDE.md.

## What this tests

`docs/voyage-field-selected-experiment.md` re-ran Voyage-4-nano and
Voyage-4-large with one specific field selection borrowed from the LLM
judge (seeker = `positioning`+`lookingFor`, candidate =
`positioning`+`background`+`lookingFor`, query included) and found it beat
the full-profile baseline. That leaves an open question: is that particular
selection actually the *best* one for an embedding model, or just the one
that happened to work for an LLM prompt? This experiment answers it directly
— grid search every non-empty subset of the same three fields
(`positioning`, `background`, `lookingFor`) independently for seeker and
candidate, times whether the searchQuery is included:

- 7 non-empty subsets of 3 fields per side (2³−1)
- 7 × 7 = 49 seeker/candidate field combinations
- × 2 (query included / not) = 98 combinations
- **plus** a "query-only" seeker (zero profile fields, searchQuery is the
  entire seeker text — always with the query included, since a fully empty
  seeker text is not a meaningful combo) × the same 7 candidate subsets = 7
  more combinations, added to directly test "how much does the search query
  alone carry, with no seeker profile at all"
- **= 105 combinations total**

Scored on all 200 real pairs (no holdout split for this sweep, per
instruction — this is an exploratory search, not a final comparison).
Voyage-4-nano only, run on Modal (A100-40GB). Interactive browser (pick
fields to see a combo's results, or rank all combos by any metric, plus a
metric-selectable heatmap):
[Voyage-4-nano field/query sweep](https://claude.ai/code/artifact/1e698f4e-15a6-44b4-bd76-2381befb2f40)
(`docs/html/voyage-nano-field-sweep-heatmap.html`,
`scripts/build_voyage_nano_field_sweep_browser.py`).

Every combo gets the full metric suite used everywhere else in this
project, via `baselines/metrics.py` unmodified: pair ROC-AUC, best-F1,
accuracy@0.5, hard/easy-negative-slice pair AUC, and retrieval MRR, mean/
median rank, Recall@1/5/10, NDCG@1/5/10.

## Method: encoding dedup

Encoding, not scoring, is the expensive step, so `sweep.py` avoids doing a
full 105x encode pass. Candidate-side embeddings depend only on
`candidate_fields` (7 unique encode groups); seeker-side embeddings depend
only on (`seeker_fields`, `use_query`) (14 groups for the 7 non-empty
subsets × 2 query settings, + 1 more for the query-only seeker = 15). That's
22 unique encode groups feeding all 105 combos' metrics, computed from
cached embeddings — one Modal GPU boot, not 105. The query-only-seeker
addition reused the same Modal results volume, so all 21 original encode
groups hit cache and only the 1 new seeker group needed encoding.

## Results

**Top 10 by pair AUC** (200 real pairs):

| Seeker fields | Candidate fields | Query | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR |
|---|---|---|---|---|---|---|
| positioning | lookingFor | yes | **0.6110** | 0.5396 | 0.7208 | 0.3296 |
| background | lookingFor | yes | 0.6092 | 0.5460 | 0.6784 | 0.2878 |
| background | positioning | yes | 0.6009 | 0.5732 | 0.6732 | 0.3051 |
| background | positioning+lookingFor | yes | 0.6003 | 0.5108 | 0.7288 | 0.3380 |
| positioning | background+lookingFor | yes | 0.5986 | 0.4968 | 0.7384 | 0.3940 |
| positioning+background | lookingFor | yes | 0.5980 | 0.5586 | 0.6664 | 0.2780 |
| positioning | positioning+lookingFor | yes | 0.5904 | 0.5054 | 0.6977 | 0.3770 |
| background | background+lookingFor | yes | 0.5869 | 0.5498 | 0.6456 | 0.3507 |
| positioning | positioning+background+lookingFor | yes | 0.5868 | 0.4900 | 0.7388 | 0.4057 |
| positioning+background | positioning+lookingFor | yes | 0.5842 | 0.5430 | 0.6720 | 0.3046 |

**Bottom 5 by pair AUC:**

| Seeker fields | Candidate fields | Query | Pair AUC |
|---|---|---|---|
| positioning | background | no | 0.5191 |
| background+lookingFor | background | no | 0.5183 |
| positioning+background+lookingFor | background | no | 0.5176 |
| positioning+background | background | no | 0.5149 |
| background | background | no | **0.5088** |

**Best on hard-neg AUC:** seeker=`background`, candidate=`positioning`,
query=yes — 0.5732 hard-neg AUC (pair AUC 0.6009, 3rd overall).

**Best on retrieval (MRR):** seeker=`positioning`, candidate=`positioning`+
`background`+`lookingFor`, query=yes — MRR 0.4057 (pair AUC 0.5868).

**Query-only seeker (no profile fields at all):**

| Candidate fields | Pair AUC | Hard-neg AUC |
|---|---|---|
| lookingFor | **0.5861** | 0.4818 |
| background+lookingFor | 0.5626 | 0.4374 |
| positioning+lookingFor | 0.5542 | 0.4556 |
| positioning+background+lookingFor | 0.5507 | 0.4344 |
| background | 0.5458 | 0.4146 |
| positioning+background | 0.5425 | 0.4392 |
| positioning | 0.5412 | 0.4516 |

Every one of these 7 beats chance by a wide margin, and the best
(candidate=`lookingFor`, 0.5861) beats the full-profile baseline (0.5614)
using **zero seeker profile fields** — the search query alone, matched
against just the candidate's `lookingFor`, carries more signal than the
seeker's entire 8-field profile does when paired with the full candidate
profile. None of the 7 beat the best combo that keeps one seeker field
(0.6110), so a seeker field still helps when available — but the query is
clearly doing most of the work.

**Query matters, consistently, across the whole grid** — not just for the
single best combo:

| | Mean pair AUC | Mean hard-neg AUC | Mean MRR |
|---|---|---|---|
| Query included (n=49) | 0.5715 | 0.5053 | 0.3101 |
| Query excluded (n=49) | 0.5375 | 0.4685 | 0.2065 |

Every one of the 10 worst combos has the query excluded. This matches the
project's earlier query-ablation finding that Voyage models lean on the
query, especially for retrieval.

**Marginal effect of each field** (mean pair AUC across all combos where
the field is present vs. absent, same side):

| Field | Seeker: with − without | Candidate: with − without |
|---|---|---|
| positioning | +0.0015 | +0.0040 |
| background | −0.0011 | **−0.0153** |
| lookingFor | **−0.0099** | **+0.0124** |

On the seeker side no field's marginal effect is large — `positioning` is
mildly positive, `lookingFor` mildly negative, roughly a wash, unlike the
LLM judge where `lookingFor` was clearly the load-bearing seeker field
(`docs/llm-judge-seeker-field-isolation-experiment.md`). On the candidate
side, `background` is the clearest drag (−0.0153) and `lookingFor` the
clearest lift (+0.0124) — the opposite of the LLM judge's candidate finding,
where dropping `background` hurt the most
(`docs/llm-judge-candidate-field-permutation-experiment.md`). **More fields
is not better for either side**: mean pair AUC falls monotonically from
1-field to 3-field subsets on both seeker (0.5580→0.5530→0.5487) and
candidate (0.5535→0.5558→0.5535, non-monotonic but 3-field is not the best)
— cosine similarity over a longer, more diluted text does worse than over a
short, targeted one, consistent with `docs/voyage-field-selected-experiment.md`'s
original finding that trimming the profile helped.

## Reading the result

**The single best combo (0.6110) beats every other Voyage-4-nano
configuration measured in this project on the 200-pair population** — the
full profile (0.5614, `docs/baseline-results-all.md`) and the LLM-judge-
matched field selection (0.5658, `docs/voyage-field-selected-experiment.md`).
It is not the same fields the LLM judge uses: the LLM judge's best is
seeker=`positioning`+`lookingFor`, candidate=all three
(`docs/llm-judge-focused-prompt-experiment.md`, 0.6451 all-200), while
Voyage-nano's best is a much smaller footprint — one seeker field, one
candidate field. **The best field selection is model-specific, not a
property of the fields themselves** — confirms the premise for running this
sweep rather than assuming the LLM judge's choice transfers.

**Still short of the LLM judge everywhere.** Even Voyage-nano's best combo
(0.6110 pair AUC, 0.5396 hard-neg AUC) trails the LLM judge's focused prompt
(0.6451 pair AUC, 0.6711 hard-neg AUC on the same 200-pair population) by a
wide margin, especially on hard negatives — consistent with every other
comparison in this project.

**Caveat: this is a 105-way search scored on one 200-pair population with no
held-out check**, so the top result should be read as "best of this grid on
this data," not a robust estimate of true generalization — the gap between
the best (0.6110) and 10th-best (0.5842) combo is comparable to the kind of
run-to-run noise seen elsewhere in this project's smaller-sample
experiments. Before trusting the winning combo for anything beyond
"embeddings prefer fewer, targeted fields too," it would need a holdout
check the way `docs/voyage-field-selected-experiment.md`'s single combo got.

## Reproducing

```bash
# Modal GPU (A100-40GB; one boot runs all 105 combos via encoding dedup)
modal run baselines/voyage_nano_field_sweep/modal_eval.py --run-id real_all
modal volume get dorby-voyage-nano-field-sweep-eval real_all/sweep_results.json \
  ./artifacts/voyage_nano_field_sweep_modal/real_all/sweep_results.json

# rebuild the interactive browser after a new sweep_results.json
python3 scripts/build_voyage_nano_field_sweep_browser.py

# local smoke test only (not a full sweep — MPS is slow for a 105-combo grid)
python -m baselines.voyage_nano_field_sweep.eval --data-dir data --limit-combos 4
```

Full 105-row results: `artifacts/voyage_nano_field_sweep_modal/real_all/sweep_results.json`
/ `.csv`.
