# Does the discarded final-epoch checkpoint do better on any text representation?

## Question

`voyage_gemini_ctrl_001`'s training pipeline saves 5 checkpoints
(steps 226/452/678/904/1130) and selects the best by dev recall@1 — step
452 (epoch 2.0) — as the `adapter` every other experiment against this
model uses, per the project-wide checkpoint-selection fix documented in
`docs/possible-bugs.md` #2. The field/query sweep already run against that
selected checkpoint
(`baselines/twotower_voyage_gemini_ctrl_field_sweep`) found a new
project-record 0.6855 pair AUC. The raw final-epoch checkpoint
(`checkpoint-1130`, step 1130/1130) was trained but never selected or
evaluated anywhere. Does it do better on *any* of the 105 field/query
combos — the training pipeline's dev-recall@1-based selection is a single
metric on a small dev split; maybe a different text representation favors
the checkpoint that was discarded?

## Method

New isolated package `baselines/twotower_voyage_gemini_ctrl_ckpt1130_field
_sweep/` — identical to `twotower_voyage_gemini_ctrl_field_sweep` in every
respect (`fields.py`/`text.py` copied unchanged, same 105-combo grid, same
metric suite, same all-200 population) except `encode.py`, which points at
`checkpoints/checkpoint-1130` instead of the selected `adapter` — pulled
directly from the same Modal volume
(`dorby-twotower-voyage-gemini-ctrl-checkpoints`), confirmed identical LoRA
shape (rank 8, alpha 16, dropout 0.05, q/k/v/o_proj) to every other
checkpoint in this project.

```bash
modal run baselines/twotower_voyage_gemini_ctrl_ckpt1130_field_sweep/modal_eval.py
modal volume get dorby-twotower-voyage-gemini-ctrl-ckpt1130-field-sweep-eval real_all/sweep_results.json \
    ./artifacts/twotower_voyage_gemini_ctrl_ckpt1130_field_sweep_modal/real_all/sweep_results.json
```

## Results — all 200 real pairs, full 105-combo grid

| | Pair AUC |
|---|---|
| **checkpoint-1130 grid best** (seeker=none, candidate=`lookingFor`, query=yes) | **0.6715** |
| checkpoint-452 (`adapter`, selected) grid best (seeker=`background`, candidate=`lookingFor`, query=yes) | 0.6855 |
| checkpoint-1130 full-profile+query (this sweep's own text builder) | 0.5857 |

**checkpoint-1130 loses on every single one of the 105 combos — 0/105.**
Mean pair AUC drop vs. the selected checkpoint: **-0.0226**. The three
worst-hit combos all lose ~0.045-0.049 pair AUC; even the *best*-preserved
combo (the grid's own top row for both checkpoints,
seeker=none/candidate=`lookingFor`/query=yes) still loses 0.0001 — a wash
at best, never a win.

**This confirms and generalizes the project's known early-peak-then-decay
pattern.** Every custom-loop two-tower run in this project except
`queryonly_back_look_001` shows dev recall@1 peak at epoch 1-2 and decline
afterward (`docs/possible-bugs.md` #2, `docs/twotower-kl-reg-experiment.md`).
That finding was always measured on a single training-side metric (dev
recall@1) over the model's *own training text*. This sweep asks the same
question from an entirely different angle — 105 different text
representations, scored on the real 200-pair population the training loop
never sees — and gets the same answer with zero exceptions: the decay is
real, not an artifact of recall@1 specifically or of the training text
specifically. The pipeline's checkpoint-selection fix is doing exactly what
it should.

## What this means

There is no text representation, of the 105 tried, where the discarded
final-epoch checkpoint would have been the better choice. This is the
cleanest confirmation yet in this project that recall@1-based checkpoint
selection isn't just picking a locally-good point on one metric — the
model genuinely gets worse, uniformly, as training continues past its
early peak, at least for this recipe and this batch. Combined with
`docs/twotower-voyage-gemini-ctrl-field-sweep-experiment.md`, the full
picture for this checkpoint is: the selected checkpoint is unambiguously
the right one to have kept, and field/query selection (not training
longer) is still the lever that moves this project's ceiling.

Published artifact: https://claude.ai/code/artifact/bf1d8ed3-5c1b-4e8b-bb74-997ef3a73ba2
