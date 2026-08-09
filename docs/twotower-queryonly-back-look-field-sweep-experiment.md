# Does the new best fine-tune generalize past its own training text?

## Question

`queryonly_back_look_001` (`twotower_queryonly_back_look/`) is, as of this
experiment, the new best two-tower fine-tune in the project on every tracked
metric (`docs/twotower-queryonly-back-look-experiment.md`, all-200: pair AUC
0.5983, hard-neg AUC 0.6564, MRR 0.4791, R@1 0.30 — beating `top1_ctrl`'s
0.5683 / 0.5484 / 0.3550 / 0.19 on every metric). It was fine-tuned on one
specific text representation — seeker = search query only (zero profile
fields), candidate = `background` + `lookingFor` — found by brute-force
search against the *frozen* `top1_ctrl` checkpoint first
(`baselines/twotower_top1_ctrl_field_sweep`) and only then actually trained.

The same generalized 105-combo field/query sweep already run against frozen
Voyage-4-nano and frozen `top1_ctrl` was re-run here against
`queryonly_back_look_001`: does fine-tuning on one text representation help
or hurt the model's embeddings for every *other* representation, or is the
gain narrowly confined to the exact combo it was trained on?

## Method

New isolated package `baselines/twotower_queryonly_back_look_field_sweep/` —
`fields.py`/`text.py` copied unchanged from
`twotower_top1_ctrl_field_sweep` (same 105-combo grid design: every
non-empty subset of `positioning`/`background`/`lookingFor` per side x
query on/off, plus a query-only seeker x the same 7 candidate subsets); only
`encode.py` changes, pointing at the `queryonly_back_look_001` LoRA adapter
(pulled from the `dorby-twotower-queryonly-back-look-checkpoints` Modal
volume) instead of `top1_ctrl`'s. Same encoding-dedup trick (22 unique
encode groups feeding all 105 combos), same full metric suite
(`baselines/metrics.py`, unmodified), same population (all 200 real pairs,
178-candidate corpus).

```bash
modal run baselines/twotower_queryonly_back_look_field_sweep/modal_eval.py
modal volume get dorby-twotower-queryonly-back-look-field-sweep-eval real_all/sweep_results.json \
    ./artifacts/twotower_queryonly_back_look_field_sweep_modal/real_all/sweep_results.json
```

**Sanity check against the canonical eval passed exactly.** The sweep's own
row for the training combo (seeker=none, candidate=background+lookingFor,
query=yes) reproduces `docs/twotower-queryonly-back-look-experiment.md`'s
canonical numbers to 4 decimals: pair AUC 0.5983, MRR 0.4791, Recall@1 0.30
— all identical. That confirms the new encoder wrapper loads the right
checkpoint and encodes text the same way the canonical eval does.

**One number does *not* match, and it isn't a bug.** The canonical doc
reports hard-neg AUC 0.6564 / easy-neg AUC 0.5700 for this exact combo (the
finding that this model shows the LLM judge's inverted hard>easy signature).
This sweep's own row for the identical combo shows hard=0.5206 /
easy=0.7060 — the *opposite* order. The difference is methodology, not the
model: the canonical eval pins the hard/easy split to the full-profile+query
baseline text for every row (a fixed difficulty ruler, `docs/twotower-
queryonly-back-look-experiment.md`'s own stated design choice), while this
sweep — like `twotower_top1_ctrl_field_sweep` and `voyage_nano_field_sweep`
before it — classifies hard/easy using each combo's *own* trimmed text, so
the notion of "hard negative" itself shifts per cell. **Read every hard/easy
number in this sweep as relative-within-the-grid only; the
project's-headline hard-neg-AUC record (0.6564) is a canonical-eval number,
not reproduced or contradicted by anything here.**

## Results — all 200 real pairs, full 105-combo grid

| | Pair AUC | Notes |
|---|---|---|
| Grid best: seeker=`positioning`, candidate=`lookingFor`, query=yes | **0.6349** | hard 0.5748 / easy 0.6972 (grid-own split), MRR 0.3429, R@1 0.20 |
| Grid best, query-only seeker: seeker=none, candidate=`lookingFor`, query=yes | 0.6323 | MRR 0.4039, R@1 0.25, R@5 0.60 |
| Training combo (seeker=none, candidate=`background`+`lookingFor`) | 0.5983 | matches canonical eval exactly |
| This sweep's own full-profile+query row (both sides all 3 fields) | 0.5917 | sweep's own text builder, not the canonical `top1_ctrl` baseline builder |
| `top1_ctrl` canonical full-profile baseline (for reference) | 0.5683 | from `docs/twotower-queryonly-back-look-experiment.md` |

**The grid's overall best (0.6349) is slightly *below*
`twotower_top1_ctrl_field_sweep`'s own grid best (0.6395,
seeker=`background`/candidate=`lookingFor`).** Fine-tuning on one specific
text representation raised the ceiling for that exact combo and its close
neighbors (empty seeker, `positioning`-only seeker, all with `lookingFor`
candidates), but did not raise the ceiling of the whole 105-combo landscape
— `top1_ctrl`'s frozen-grid champion is still marginally ahead of
`queryonly_back_look_001`'s own best cell.

**Query-only-seeker combos got specifically better, though.** The best
query-only-seeker row in this grid (0.6323, candidate=`lookingFor`) beats
`top1_ctrl_field_sweep`'s equivalent best (0.6220) by +0.010 — the direction
you'd expect, since `queryonly_back_look_001` was fine-tuned toward exactly
this seeker shape (just not this exact candidate field set).

**Search query still matters everywhere** — mean pair AUC with the query
included (0.5983, n=56) beats without it (0.5700, n=49), the same direction
every model in this project shows.

**Seeker field-count effect flattens instead of continuing to fall.**
Mean pair AUC by seeker size: 0 fields (query-only) 0.6007 → 1 field 0.5836
→ 2 fields 0.5843 → 3 fields (full) 0.5842. `top1_ctrl_field_sweep` showed a
monotonic decline all the way down (0.5915→0.5813→0.5766→0.5692); here the
big drop is only from 0→1 field, then it plateaus rather than keeps falling
— consistent with a model that's been specifically nudged toward the
zero-profile-seeker shape rather than one that's simply "worse with more
text" in general. Candidate side still peaks at 2 fields, not 3 (1-field
0.5843, 2-field 0.5862, 3-field 0.5844), matching every other sweep in this
project.

## What this means

Fine-tuning on the field-sweep's own best-found text representation is a
**narrow win, not a general one**: it took the model from a full-profile
baseline of 0.5683 (`top1_ctrl`) to 0.5983 on that specific combo — genuinely
the new project-best two-tower fine-tune — but it did not push the
*ceiling* of the field/query search space past what a frozen model already
found, and other combos this checkpoint was never trained on land within
noise of where `top1_ctrl`'s frozen embeddings already put them. The
practical takeaway for future rounds carries over unchanged from
`docs/twotower-top1-ctrl-field-sweep-experiment.md`: search the frozen model
first to find a good text representation, then decide whether training on
it is worth the GPU spend — but expect the win to be local to that
representation, not a lift across the whole grid.

Published artifact: https://claude.ai/code/artifact/b9baf2d7-e7de-4962-918b-a3fe48b7ccf2
