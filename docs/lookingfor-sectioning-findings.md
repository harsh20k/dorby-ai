# Findings: does splitting `lookingFor` into sections improve matching?

**Status: tested, mixed result — one direction helps, the other doesn't.**
Full experiment log, per-candidate scores, and repro commands:
[`lookingfor-sectioning-experiment.md`](lookingfor-sectioning-experiment.md).

## The question

Every baseline (`baselines/{tfidf,bert_frozen,voyage_nano,voyage_large}/`)
embeds a contact's `lookingFor` field as one string, even when it lists
several distinct, independently-labeled asks (e.g. "### Fundraising" /
"### Scouting" / "### Founders" — 533 of 660 dataset records already write
`lookingFor` this way). One embedding over several unrelated asks is
effectively an average of them. The question: does that averaging cost us
match quality, and if so, on which side of the seeker/candidate pair does
it matter?

## The test

New sibling baseline `baselines/voyage_nano_sectioned/`: split `lookingFor`
on the blank-line paragraph breaks that already exist in the data (no
header parsing, no LLM, no inference), embed each section separately
alongside the rest of the profile unchanged, and score by the **max**
cosine over sections rather than one blurred whole-field embedding. Two
directions, tested independently against the same 69-pair real holdout
(`data/synthetic/seed_split.json`) used by every other baseline in
`docs/baseline-results-holdout.md`:

- **Candidate-sectioned** — split the *candidate's* `lookingFor`; seeker
  side untouched.
- **Seeker-sectioned** — split the *seeker's own* `lookingFor`; candidate
  side untouched.

Model: local `voyage-4-nano`, same encoder/settings as the existing
`voyage_nano` baseline row, run on Modal L4 GPUs
(`baselines/voyage_nano_sectioned/modal_eval.py`).

## Results (69-pair holdout: 29 positive / 40 negative pairs, 29 retrieval queries)

| Metric | Baseline | Candidate-sectioned | Seeker-sectioned |
|---|---|---|---|
| Pair ROC-AUC | 0.579 | 0.568 (↓) | **0.596 (↑)** |
| Average precision | 0.512 | 0.499 (↓) | **0.568 (↑)** |
| MRR | 0.461 | 0.434 (↓) | **0.493 (↑)** |
| Top-1 retrieval accuracy | 27.6% | 24.1% (↓) | **34.5% (↑)** |
| Recall@10 | **0.759** | 0.724 (↓) | 0.690 (↓) |

## What this means

**Candidate-sectioning — the original hypothesis — doesn't hold up.**
Splitting a candidate's asks and scoring by their best-matching one loses
to the plain baseline on every metric. A candidate's several `lookingFor`
sections are typically all real, valid things they're open to, so scoring
by whichever one fits best isn't actually different in kind from scoring
the averaged whole — it doesn't add signal, and averages out slightly
worse here.

**Seeker-sectioning — the mirror direction — genuinely helps, especially
at the top of the ranking.** Pair AUC and average precision both improve
meaningfully, and top-1 retrieval accuracy jumps from 27.6% to 34.5% (over
a third more queries land their true best match in the #1 slot). The
tradeoff: recall further down the list (R@10) gets slightly worse — the
seeker's other, unrelated asks apparently still add a little useful
breadth for looser matches, even if they dilute precision at the top.

**Why the asymmetry:** a seeker's `lookingFor` often holds several live,
unrelated threads at once (e.g. this dataset's pilot seeker had
fundraising, hiring, and investor asks going in parallel), but any single
`searchQuery` for a given pair is only ever about one of them. Splitting
lets that one on-topic section drive the score instead of being diluted by
the seeker's other, irrelevant asks. Candidates don't have this same
per-pair mismatch — whichever ask they're read against, they meant all of
them — so there's nothing to un-blur on that side.

## Not yet done

- This only tested local `voyage-4-nano`. Boardy's production model,
  `voyage-4-large`, is currently the strongest baseline (pair AUC 0.609,
  MRR 0.529) — whether seeker-sectioning's gain holds there is the natural
  next check before this could matter for production.
- Only tried against real holdout pairs; not tested against synthetic data
  or the two-tower fine-tune.
