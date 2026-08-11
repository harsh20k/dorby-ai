"""Two-tower fine-tune: BOTH sides trimmed to positioning + background only.

Motivation
----------
The field-isolation experiment found `positioning`/`background`/`lookingFor`
are the only profile fields that carry real person-identity signal alone;
the free eval-time sweep (`field_pairs_sweep/`, frozen voyage-4-nano, all
200 real pairs) then tested all three field-pair combinations on the seeker
side against the *full* candidate profile and found none beat
`concat_baseline` or `query_only` — `pos_background` was in fact the
weakest of the three (MRR 0.2060, R@1 0.10).

This experiment asks a different question: what if the candidate side is
trimmed to the same two fields too, and the model is actually fine-tuned
(not just scored frozen) on that narrower representation? Three field-pair
combinations are planned as three separate packages; this is the first,
`positioning + background` on both sides — the pair the free sweep scored
weakest, used here as the starting point per the user's own choice.

One shared tower, not two
--------------------------
Trimming the candidate side to match the seeker side does not change this:
`twotower_split/` already tested giving query and candidate fully
independent towers and lost to a single shared tower. The shared tower
already handles asymmetric content between the two sides via Voyage's own
query/document role prompts — every experiment in this project, including
this one, has asymmetric seeker/candidate text (even here, the seeker's
`positioning`+`background` describes a different person than the
candidate's). Two towers would only be justified if the two sides were a
genuinely different kind of content (e.g. text vs. image); both sides here
are still short profile-field text drawn from the same schema.

Recipe: `top1_ctrl`'s exact settings, one input change
-------------------------------------------------------
Every setting is held identical to `top1_ctrl_001`
(`twotower_top1_optimised/`, run with `--loss-scale 20.0
--hardness-mode ""` — the project's best two-tower fine-tune, plain
library-default `MultipleNegativesRankingLoss`, `primary_metric=
"recall@1"` checkpoint selection): LoRA rank 8 / alpha 16 / dropout 0.05 on
q/k/v/o_proj, micro-batch 6 / accum 2 (effective batch 12), lr 2e-4, 5
epochs, `voyage-4-nano` at its native 1024-dim truncation. The only thing
that changes is what text goes into `anchor`/`positive`/`negatives`:
`positioning`+`background` only, on both seeker and candidate, produced by
`scripts/build_rrf_multineg_triplets_pos_bg.py` (a new row-builder script,
not a modification of the one `top1_ctrl` used) from the same 643-row
`rrf_003` population `top1_ctrl` trained on — same query_keys, same
seekers, same seed, so the two runs are trained on exactly the same pairs
and differ only in field selection.

Isolation
---------
New top-level package. `data.py`/`eval_dev.py` are copies of
`twotower_top1_optimised`'s (not imports), pinned by
`tests/test_field_pos_bg.py`. `config.py`/`train.py` are near-copies with
the hardness-weighting knob removed (this arm never uses it — top1_ctrl's
control corner is plain `MultipleNegativesRankingLoss(scale=20.0)`) and
import paths updated. Generic helpers from `twotower.train`/`twotower.eval`
are imported read-only, exactly as every other package in this project
does. Evaluation reuses `eval_real_full.eval.run_eval` unmodified for
scoring on the real 200 pairs — but note the real pairs still carry their
*full* profiles; `eval_real_full` has no notion of a trimmed candidate
side, so this arm cannot be scored through that path unchanged. See
`eval.py` for the field-trimmed scoring path this experiment needs
instead.
"""
