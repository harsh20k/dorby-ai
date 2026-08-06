"""Two-tower fine-tune: BOTH sides trimmed to positioning + lookingFor only.

Second of three field-pair packages (sibling to `twotower_field_pos_bg/`
and `twotower_field_bg_look/`) — same design, different field pair. See
`twotower_field_pos_bg/__init__.py` for the full rationale (field-isolation
motivation, why one shared tower rather than two, why `top1_ctrl`'s exact
recipe is held fixed); this docstring only covers what differs.

The free eval-time sweep (`field_pairs_sweep/`, frozen voyage-4-nano, seeker
side only vs. full candidate profile) found `pos_lookingfor` the *strongest*
of the three field pairs on the seeker side alone (MRR 0.2879, R@1 0.15,
R@10 0.56 — clearly ahead of `pos_background`'s 0.2060/0.10/0.40). This
package tests whether that edge holds up once the candidate side is also
trimmed to the same two fields and the model is actually fine-tuned, using
the identical recipe `field_pos_bg_001` used so the two results are directly
comparable.

Row file: `scripts/build_rrf_multineg_triplets_pos_look.py`, drawing on the
same 643-row `rrf_003` population (`positioning` + `lookingFor` only, on
both seeker and candidate) — same query_keys, same seekers, same seed as
every other package in this family.
"""
