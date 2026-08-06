"""Two-tower fine-tune: BOTH sides trimmed to background + lookingFor only.

Third of three field-pair packages (sibling to `twotower_field_pos_bg/`
and `twotower_field_pos_look/`) — same design, different field pair. See
`twotower_field_pos_bg/__init__.py` for the full rationale (field-isolation
motivation, why one shared tower rather than two, why `top1_ctrl`'s exact
recipe is held fixed); this docstring only covers what differs.

The free eval-time sweep (`field_pairs_sweep/`, frozen voyage-4-nano, seeker
side only vs. full candidate profile) placed `background_lookingfor` in the
middle of the three field pairs (MRR 0.2426, R@1 0.12, R@10 0.51 — behind
`pos_lookingfor`'s 0.2879/0.15/0.56, ahead of `pos_background`'s
0.2060/0.10/0.40). This package tests whether that ordering holds once the
candidate side is also trimmed to the same two fields and the model is
actually fine-tuned, using the identical recipe `field_pos_bg_001` and
`field_pos_look_001` used so all three results are directly comparable.

Row file: `scripts/build_rrf_multineg_triplets_bg_look.py`, drawing on the
same 643-row `rrf_003` population (`background` + `lookingFor` only, on
both seeker and candidate) — same query_keys, same seekers, same seed as
every other package in this family.
"""
