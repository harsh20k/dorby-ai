# lookingFor field-sectioning experiment

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

## Full 69-pair holdout (Modal, `voyage-4-nano`, `max_length=4096` to match the baseline exactly)

Run via `baselines/voyage_nano_sectioned/modal_eval.py` on Modal L4 GPUs
(mirrors `twotower/modal_train.py`'s pattern). First attempt at the
default `batch_size=16` hit a CUDA OOM on the larger sectioned corpus;
retried at `batch_size=4` (same as the local default) successfully. Ran
once more at `max_length=8192` (this package's default) and once at
`max_length=4096` (the baseline's setting) — the two barely differ
(±0.001 AUC), confirming truncation isn't a factor; numbers below are the
`max_length=4096` runs, directly comparable to
`docs/baseline-results-holdout.md`.

| Metric | Baseline (voyage-4-nano) | Candidate-sectioned | Seeker-sectioned |
|---|---|---|---|
| Pair ROC-AUC | 0.5793 | 0.5681 | **0.5957** |
| Average precision | 0.5123 | 0.4991 | **0.5681** |
| MRR | 0.4610 | 0.4342 | **0.4934** |
| Mean rank | 6.7931 | 7.4828 | **6.6552** |
| Median rank | 2.0 | 3.0 | 2.0 |
| Recall@1 (top-1 acc) | 0.2759 | 0.2414 | **0.3448** |
| Recall@5 | 0.6552 | 0.6207 | 0.6207 |
| Recall@10 | **0.7586** | 0.7241 | 0.6897 |
| NDCG@1 | 0.2759 | 0.2414 | **0.3448** |
| NDCG@5 | 0.4908 | 0.4602 | **0.5029** |
| NDCG@10 | **0.5230** | 0.4948 | 0.5261 |

n=29 positive / 40 negative pairs, 29 retrieval queries — identical
population to every other row in `docs/baseline-results-holdout.md`.

### Reading — this is where the pilot's "noisy n=8" caveat resolves

**Candidate-sectioned (the original hypothesis) loses to the baseline on
every single metric.** Splitting the *candidate's* `lookingFor` and taking
the best-matching section does not recover signal — if anything it costs
a little (pair AUC -0.011, MRR -0.027, NDCG@10 -0.028). The theory that
candidates' blurred multi-ask embeddings are hurting match quality does
not hold up at n=69.

**Seeker-sectioned (the mirror direction) beats the baseline on most
metrics, most notably at the top of the ranking:** pair AUC +0.016
(0.596 vs 0.579), average precision +0.056, top-1 retrieval accuracy
34.5% vs 27.6% (a real jump — over a third more queries land their best
candidate in the #1 slot), NDCG@1/5/10 all improve. It loses a little at
the tail (R@10 0.690 vs 0.759, P@10 unreported here but similarly down) —
sectioning the seeker's own `lookingFor` sharpens the top of the list at
some cost to broader recall.

**Takeaway:** the pilot's single hard seeker was misleading about which
direction mattered. The averaging problem worth fixing isn't "a
candidate's several asks get blurred together" (candidate-sectioning) —
it's "a *seeker's* several asks get blurred into one query vector"
(seeker-sectioning). That's a sensible result in hindsight: a candidate's
`lookingFor` sections are typically all real, valid things they want, so
scoring by their best-matching one is a reasonable model of "does this
candidate have *anything* that fits" — but a seeker in this dataset often
has several unrelated fundraise/hiring/partnership threads going at once,
and the one active `searchQuery` for a given pair only cares about one of
them; splitting `lookingFor` and letting the seeker's *own* on-topic
section drive the score, rather than diluting it with the seeker's other
unrelated asks, is what actually helps here.

**Not yet done:** this used local `voyage-4-nano` only, matching the
pilot and the primary comparison table's `voyage-4-nano` column. Whether
seeker-sectioning also helps `voyage-4-large` (Boardy's production model,
currently the best baseline at pair AUC 0.6086 / MRR 0.5287) is the
natural next check before drawing conclusions about production impact.

## Follow-up: softer aggregation (PR #13)

Hard max keeps only a seeker's single best-matching section; tried
`topk_mean` (mean of top-2 sections) and `softmax` (temperature-weighted
average over all sections, T=0.05) as softer alternatives, same holdout,
same model — see `baselines/voyage_nano_sectioned/aggregate.py`.

| Aggregation | Pair AUC | MRR | Top-1 | Recall@10 |
|---|---|---|---|---|
| Baseline (no sectioning) | 0.5793 | 0.4610 | 27.6% | 0.7586 |
| max (hard) | 0.5957 | 0.4934 | 34.5% | 0.6897 |
| topk_mean (k=2) | 0.5940 | 0.5127 | 37.9% | 0.6897 |
| softmax (T=0.05) | 0.5983 | 0.5149 | 37.9% | 0.6897 |

All three sectioned variants land at the identical 0.6897 Recall@10 —
softening the aggregation improved MRR/top-1 a little further over hard
max, but did not recover any of the Recall@10 lost to sectioning in the
first place. Whatever seeker-sectioning trades away at the tail of the
ranking isn't about the max-vs-average choice.

## Follow-up: layering onto the hybrid TF-IDF+voyage baseline (PR #12)

The strongest frozen baseline before this work was `hybrid_tfidf_voyage`
(TF-IDF + voyage-4-nano, late-fused; pair AUC 0.6397, MRR 0.4043 — see
`docs/baseline-results-holdout.md`). Built
`baselines/hybrid_tfidf_voyage_seeker_sectioned/`: identical to that
baseline except the voyage channel's seeker/query side uses
seeker-sectioning (max over sections) instead of one whole-profile
embedding; TF-IDF and the candidate side are untouched.

| Variant | Pair AUC | MRR | Top-1 | Recall@10 |
|---|---|---|---|---|
| Hybrid TF-IDF+voyage (previous best) | 0.6397 | 0.4043 | 27.6% | 0.7931 |
| Seeker-sectioned voyage alone | 0.5957 | 0.4934 | 34.5% | 0.6897 |
| **Hybrid + seeker-sectioning** | **0.6483** | **0.4392** | **31.0%** | **0.7931** |

This is now the best frozen baseline measured on this holdout — better
AUC and MRR than plain hybrid, better top-1, and Recall@10 holds at the
hybrid's already-strong level (no tradeoff, unlike sectioning alone).
Average precision is roughly flat (0.520 → 0.516). The fit-set fusion
still weights TF-IDF heavily (alpha ≈ 0.95 on ~131 real fit pairs), so
most of the ranking is still lexical, but the sectioned voyage channel's
contribution clearly pulls through.

## Reproducing

```bash
# candidate-sectioned, one seeker (local)
python -m baselines.voyage_nano_sectioned.eval \
  --data-dir data --user-id cm5yrbzzj01uhui0ly2n8tqpi \
  --artifacts-dir artifacts/voyage_nano_sectioned_user_test

# seeker-sectioned (mirror), same seeker (local)
python -m baselines.voyage_nano_sectioned.eval_seeker \
  --data-dir data --user-id cm5yrbzzj01uhui0ly2n8tqpi \
  --artifacts-dir artifacts/voyage_nano_sectioned_seeker_user_test

# full 69-pair holdout, either direction, local (drop --user-id, add --holdout-only)
python -m baselines.voyage_nano_sectioned.eval --data-dir data --holdout-only --max-length 4096
python -m baselines.voyage_nano_sectioned.eval_seeker --data-dir data --holdout-only --max-length 4096

# same, on Modal (L4 GPU; batch_size=4 avoids the OOM the default batch_size=16 hits)
modal run baselines/voyage_nano_sectioned/modal_eval.py --holdout-only --direction candidate --batch-size 4 --max-length 4096
modal run baselines/voyage_nano_sectioned/modal_eval.py --holdout-only --direction seeker --batch-size 4 --max-length 4096
modal volume get dorby-sectioning-eval <run_id> ./artifacts/voyage_nano_sectioned_modal/<run_id>
```
