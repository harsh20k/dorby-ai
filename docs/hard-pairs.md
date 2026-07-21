# Hard pairs in the original dataset (Voyage-4-nano / -large)

Which of the **original 200 real pairs** (100 positive / 100 negative seed —
`matchContactId` without the `cmsynth` prefix) do the frozen Voyage baselines
get wrong, and how badly? Reconstructed entirely from the on-disk embedding
caches (`artifacts/voyage_large/emb/`, `artifacts/voyage_nano/emb_*.npy`) —
**zero API calls**.

Reproduce:

```bash
.venv/bin/python scripts/hard_pairs.py     # reads data/ + artifacts/, writes artifacts/hard_pairs/
```

Outputs (under `artifacts/hard_pairs/`, gitignored — regenerate locally):
`hard_pairs_large.{csv,json}`, `hard_pairs_nano.{csv,json}` (every pair, ranked
hardest-first), and `hard_pairs_consensus.json` (both-model view).

## Method

Scoring is identical to `baselines/voyage_*/eval.py`:
`score = cosine(seeker_emb, candidate_emb)` on L2-normalized embeddings, where
`seeker = seeker_to_text(userContactFile, searchQuery)` (input_type `query`) and
`candidate = candidate_to_text(matchContactFile)` (input_type `document`). The
decision boundary `t` is each model's own `pair.best_f1_threshold` from
`metrics.json`.

Signed margin to the boundary (positive = correct, negative = misclassified):
- positive pair: `margin = score - t`
- negative pair: `margin = t - score`

**Validation:** reconstructed pair AUC reproduces the reported `metrics.json`
value exactly — voyage-4-large 0.5726, voyage-4-nano 0.5614 — so these per-pair
scores are the eval, not an approximation.

## Headline finding

Both models barely separate positives from negatives on the original seed
(AUC ≈ 0.57 / 0.56, just above chance), and **the errors are almost entirely
false positives** — labeled-not-a-match pairs that still embed as highly similar
to the query:

| model | threshold | hard positives (false neg) | hard negatives (false pos) |
|---|---|---|---|
| voyage-4-large | 0.517 | 1 / 100 | 99 / 100 |
| voyage-4-nano  | 0.578 | 2 / 100 | 94 / 100 |

(The false-positive counts are inflated by the F1-optimal threshold sitting low
to keep recall — the robust signal is the *margin ranking*, not the raw count.
Negatives routinely score 0.65–0.77 cosine, overlapping the positive range.)

**94 negatives are misclassified by *both* models** (`hard_pairs_consensus.json`,
sorted by summed margin). These are the genuinely hard negatives: topically
on-query but violating one axis of Boardy's matching semantics — exactly the
"hard negative" regime `data/synthetic/strategy.md` targets, and consistent with
the near-chance hard-negative-slice AUC seen elsewhere.

Hard *positives* barely exist for either model — when a pair is truly a good
intro, both Voyage models rank it high. The whole difficulty is telling apart
the plausible-looking non-matches.

### Hardest negatives (wrong by both, top of consensus list)

The query themes that most reliably fool both models:

- adjacent **company-side healthcare / digital-health operators** with budget
- **enterprise B2B / industrial-manufacturing sales** leaders
- **early-stage VC / angel / family-office** investor intros
- **NYC Tech Week AI-native founder** networking
- **seed→Series A fundraise** intros

Full per-pair detail (ids, scores, margins) is in the CSV/JSON outputs.
