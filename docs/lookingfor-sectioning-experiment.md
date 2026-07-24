# lookingFor field-sectioning experiment (single-seeker pilot)

## The idea being tested

Today, every baseline embeds a candidate's whole profile — including the
entire `lookingFor` field — as one string, producing one vector. When
`lookingFor` lists several distinct asks (e.g. "Fundraising" / "Scouting" /
"Founders", each its own paragraph, which most real profiles already do —
533 of 660 records have a multi-paragraph `lookingFor`), that one vector is
effectively an average over all of them. A seeker asking about only one ask
can score worse than they should, dragged down by averaging with unrelated
asks.

The fix under test: split `lookingFor` on the paragraph breaks that already
exist in the data (no header parsing, no inference), embed each section
separately alongside the rest of the unchanged profile, and score a
candidate as the **max** over its own sections rather than one blurred
average. Implementation: `baselines/voyage_nano_sectioned/`.

Two directions were tried, both against the same known-good/known-bad pairs
so the comparison is apples-to-apples:

1. **Candidate-sectioned** (`eval.py`): seeker text unchanged (whole profile
   + searchQuery, one embedding); candidate's `lookingFor` split into N
   section-texts (full profile each time, `lookingFor` swapped to one
   section); score = max cosine over the candidate's own sections.
2. **Seeker-sectioned** (`eval_seeker.py`, the mirror): candidate text
   unchanged (today's baseline, one embedding); seeker's own `lookingFor`
   split into N section-texts the same way; score = max cosine over the
   seeker's own sections against the one candidate embedding.

If a `lookingFor` field has 0 or 1 sections, both scripts fall back to the
identical whole-profile text used today — no artificial fragmentation.

## Pilot: single seeker, `cm5yrbzzj01uhui0ly2n8tqpi`

Chosen as a cheap first check before running the full 69-pair holdout: this
seeker has exactly 3 positive and 5 negative pairs, all investor-flavored
candidates (deep-tech, fintech, private-markets), and their own `lookingFor`
has 4 distinct sections (Fundraising / Scouting / Founders / Direct Capital
for Vidadeya) — each pair's `searchQuery` targets a different one of those
asks (foodtech/agritech round vs. aviation SPV vs. Abacus Lelle GP raise),
so it's a good stress test for whether splitting recovers per-ask signal.
Model: local `voyage-4-nano`, matching the existing baseline exactly except
for the sectioning change.

Per-candidate cosine scores, all three variants:

| Candidate | Label | Baseline (whole/whole) | Candidate-sectioned | Seeker-sectioned |
|---|---|---|---|---|
| cm6jy153... | POS | 0.5875 | 0.587 | 0.5877 |
| cmhxhht0... | POS | 0.5841 | 0.584 | **0.6102** |
| cmll05yrh... | POS | 0.5780 | 0.579 | 0.5798 |
| cmhm7k0s4... | NEG | 0.6295 | 0.630 | 0.6285 |
| cmblaqcb7... | NEG | 0.6424 | 0.640 | 0.6380 |
| cmn8fihbf... | NEG | 0.6799 | 0.677 | 0.6736 |
| cmpr3cumx... | NEG | 0.6255 | 0.628 | 0.6220 |
| cmj0x41pj... | NEG | 0.6036 | 0.606 | 0.6032 |

Pair AUC (n=8, all 5 negatives outscoring all 3 positives is AUC 0.0):

| Variant | Pair AUC |
|---|---|
| Baseline (whole profile both sides) | 0.0000 |
| Candidate-sectioned | 0.0000 |
| Seeker-sectioned | 0.0667 |

## Reading

All three variants get this seeker essentially backwards — every negative
outscores every positive, for the same underlying reason in every case: all
8 candidates are lexically/topically in the same investor space as the
query, and nano can't separate "same vertical" from "actually the right
fit" regardless of how the text is chunked. Splitting `lookingFor` didn't
rescue this seeker.

The one place sectioning moved a score meaningfully was seeker-sectioned on
`cmhxhht0...` (aviation-SPV pair): 0.5841 → 0.6102, a real jump that pushed
it above one negative it previously lost to. That's consistent with the
theory — the seeker's own `lookingFor` has an aviation-adjacent-sounding
section that, isolated from the other three asks, matches this candidate
better than the averaged whole field did. But it wasn't enough to flip the
overall ranking, and candidate-sectioning (the original hypothesis) showed
essentially no movement at all here (±0.003 on every score).

**This is one seeker, n=8 pairs — noisy, not conclusive.** It's a case where
the baseline was already fully broken (AUC 0), so there was limited room
for either sectioning direction to prove itself one way or the other. What
it does show: the failure mode here isn't simple field-blur — it looks more
like nano being unable to distinguish investors who fit the deal from
investors who merely share the deal's jargon (a "same-vertical, wrong-fit"
problem, similar in flavor to the neg-hardness slice already tracked in
`docs/baseline-metrics.md`). A real read on whether sectioning helps in
general needs the full 69-pair holdout, not one hard seeker — not yet run.

## Reproducing

```bash
# candidate-sectioned, one seeker
python -m baselines.voyage_nano_sectioned.eval \
  --data-dir data --user-id cm5yrbzzj01uhui0ly2n8tqpi \
  --artifacts-dir artifacts/voyage_nano_sectioned_user_test

# seeker-sectioned (mirror), same seeker
python -m baselines.voyage_nano_sectioned.eval_seeker \
  --data-dir data --user-id cm5yrbzzj01uhui0ly2n8tqpi \
  --artifacts-dir artifacts/voyage_nano_sectioned_seeker_user_test

# full 69-pair holdout, either direction (drop --user-id, add --holdout-only)
python -m baselines.voyage_nano_sectioned.eval --data-dir data --holdout-only
python -m baselines.voyage_nano_sectioned.eval_seeker --data-dir data --holdout-only
```
