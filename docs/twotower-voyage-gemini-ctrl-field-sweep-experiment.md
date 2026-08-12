# A field/query sweep against the newest fine-tune sets a new project record

## Question

`voyage_gemini_ctrl_001` (`twotower_voyage_gemini_ctrl`) is `top1_ctrl`'s
exact recipe retrained on a bigger, newer (but unreviewed, measurably
leakier) synthetic batch — `pairing_voyage_gemini/smoke_test_002`. On its
default full-profile text it already set the best all-200 pair AUC of any
fine-tune in the project (0.6081), though it trails `queryonly_back_look
_001` on hard-neg AUC/MRR/R@1/R@10
(`docs/twotower-voyage-gemini-ctrl-experiment.md`). That doc explicitly
flagged the field/query sweep methodology as the natural next step for this
checkpoint, not yet run. This experiment runs it: the same generalized
105-combo seeker/candidate field x search-query grid already run against
frozen Voyage-4-nano, `top1_ctrl`, and `queryonly_back_look_001`, this time
against `voyage_gemini_ctrl_001`.

## Method

New isolated package `baselines/twotower_voyage_gemini_ctrl_field_sweep/` —
`fields.py`/`text.py` copied unchanged from the prior sweep packages (same
105-combo grid design); only `encode.py` changes, pointing at the
`voyage_gemini_ctrl_001` LoRA adapter (pulled from the
`dorby-twotower-voyage-gemini-ctrl-checkpoints` Modal volume — confirmed
identical LoRA shape to every prior checkpoint: rank 8, alpha 16, dropout
0.05, q/k/v/o_proj, `voyage-4-nano` base, 1024-dim truncation, 4096 max
sequence length). Same encoding-dedup trick, same full metric suite
(`baselines/metrics.py`, unmodified), same population (all 200 real pairs,
178-candidate corpus).

```bash
modal run baselines/twotower_voyage_gemini_ctrl_field_sweep/modal_eval.py
modal volume get dorby-twotower-voyage-gemini-ctrl-field-sweep-eval real_all/sweep_results.json \
    ./artifacts/twotower_voyage_gemini_ctrl_field_sweep_modal/real_all/sweep_results.json
```

## Results — all 200 real pairs, full 105-combo grid

| | Pair AUC | Hard-neg AUC (grid-own split) | Easy-neg AUC | MRR | R@1 |
|---|---|---|---|---|---|
| **Grid best: seeker=`background`, candidate=`lookingFor`, query=yes** | **0.6855** | 0.6732 | 0.7084 | 0.3947 | 0.24 |
| Query-only-seeker best: seeker=none, candidate=`lookingFor`, query=yes | 0.6716 | 0.6164 | 0.8016 | 0.3941 | 0.23 |
| `voyage_gemini_ctrl_001` canonical full-profile baseline | 0.6081 | 0.6264 | 0.6544 | 0.4506 | 0.26 |
| This sweep's own full-profile+query row (sweep's own text builder) | 0.6027 | — | — | — | — |

**New project-wide record — the first embedding-based model of any kind
(frozen or fine-tuned) to beat the LLM judge's own best pair AUC.** The grid
best (0.6855) beats:
- `top1_ctrl_field_sweep`'s grid best (0.6395, +0.046)
- `queryonly_back_look_field_sweep`'s grid best (0.6349, +0.051)
- Frozen Voyage-4-nano's grid best (0.6110, +0.075)
- **The LLM judge's own best pair AUC (0.6451,
  `docs/llm-judge-focused-prompt-experiment.md`), by +0.040** — every prior
  field/query sweep in this project landed within or below that number;
  this is the first to clear it.
- This checkpoint's own canonical full-profile baseline (0.6081, +0.077) —
  field selection buys as much here as it did for every checkpoint tested
  so far, on top of a checkpoint that was already the project's best on raw
  pair AUC.

**The hard/easy-neg gap is the tightest of any top combo in the project.**
0.6732 hard vs 0.7084 easy — both numbers are high in absolute terms, and
the 0.035 gap is far smaller than the typical embedding-baseline pattern
(TF-IDF: 0.75 easy vs 0.50 hard) — not the full LLM-judge-style inversion
`queryonly_back_look_001`'s training combo showed, but closer to it than
any other fine-tune's grid-best row. (Standard caveat carried from the
`queryonly_back_look` sweep write-up: this sweep classifies hard/easy using
each combo's own trimmed text, not the canonical full-profile-pinned split,
so this number is relative-within-the-grid, not directly comparable to the
project's canonical hard-neg-AUC record of 0.6564.)

**A query-only seeker (zero profile fields) alone already beats the LLM
judge's best**, at 0.6716 pair AUC with just `candidate=lookingFor` — the
search query text alone, run through this checkpoint's encoder, out-predicts
every LLM-judge prompt variant tested in this project on pair AUC.

**Search query still matters everywhere** — mean pair AUC with the query
included (0.6351, n=56) beats without it (0.5895, n=49), the same direction
every model in this project shows; every one of the 5 worst combos in the
grid excludes the query.

**Seeker field-count effect is monotonically worse, like `top1_ctrl`'s
pattern, not `queryonly_back_look`'s plateau.** Mean pair AUC by seeker
size: 0 fields (query-only) 0.6317 → 1 field 0.6144 → 2 fields 0.6141 → 3
fields (full) 0.6024. Candidate side is the first sweep in this project
where mean AUC is monotonically **best at 1 field**, not 2 (1-field 0.6168,
2-field 0.6139, 3-field 0.6047) — though the single best candidate field
either way is `lookingFor` alone, consistent with every prior sweep.

**Caveat: 105-way search on one 200-pair population, no held-out check.**
The gap between the best (0.6855) and 10th-best (0.6636) combo is real but
not huge — read this as "field selection helps this checkpoint even more
than the others," not as a validated production config without a holdout
check. The underlying training data for `voyage_gemini_ctrl_001` is also an
unreviewed pilot batch measurably leakier than `rrf_003`
(`docs/twotower-voyage-gemini-ctrl-experiment.md`'s own caveat) — this
sweep does not re-litigate that; it answers "given this checkpoint, does
field selection help," not "is this checkpoint's training data trustworthy."

## What this means

Every field/query sweep run against a fine-tuned checkpoint so far has
found the same lever works: trimming both sides down to a short, targeted
text representation (usually seeker=one identity field, candidate=
`lookingFor` alone) beats that checkpoint's own full-profile training text
by a wide margin — and here, for the first time, that lever pushes an
embedding-only, sub-100ms-serving-cost model past the LLM judge's own
ceiling. Combined with `docs/twotower-top1-ctrl-field-sweep-experiment.md`
and `docs/twotower-queryonly-back-look-field-sweep-experiment.md`, the
pattern is now three-for-three: a frozen-model field/query search is a
reliable, free way to find how much headroom a checkpoint has left before
spending more GPU budget on it.

Published artifact: https://claude.ai/code/artifact/ed83c211-f865-4d8b-9e6a-56d20a027804
