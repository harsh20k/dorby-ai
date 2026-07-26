# Synthetic pair batch `rrf_001`

Produced by `scripts/generate_rrf_dataset.py`. The exact settings are in
`preset_used.json` — re-running with that file reproduces this batch.

- Profile pool: `run_20260726_105943`
- Pairs shortlisted: 675
- Pairs labeled: 675
- Balance: 189 positive / 486 negative
- Density: 7.4176 edges per node (real data: 0.673)
- Judge cost: $0.6230 (measured, from OpenRouter usage)

## What these labels are

A model's opinion, not real accept/decline outcomes. The judge
(`google/gemini-3.1-flash-lite`, naive framing) measured 0.6358 pair AUC and
0.5942 decision accuracy on the real 69-pair holdout. There is no deadband:
every judged pair is labeled, so borderline pairs carry some wrong labels.

Negatives are the more interesting half — each was surfaced by both a dense
embedding channel and BM25 and still refused by the judge, which is closer to
the real data's "production recommended it, a human declined" than any
constructed mismatch.

**Not promoted.** Nothing here is merged into `data/dataset_*.json`.
