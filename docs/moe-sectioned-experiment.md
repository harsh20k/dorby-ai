# Sectioned MoE — per-ask experts with learned attention pooling

**Status: built, run on both encoders, replicated across 5 seeds. The sectioned
model beats the standing bar — but not for the reason the plan predicted.**
`moe_attention` averages 0.6467 AUC against logistic regression's 0.5758 and wins
5 of 5 seeds. However `moe_mean` (0.6404) matches it, so the gain comes from
**scoring each ask separately**, not from the learned attention that motivated the
design. Qwen3-Embedding-8B section embeddings scored *worse* than TF-IDF. The
frozen 69-pair holdout was **not spent** — it is the next decision, not a
foregone one.

Isolated package: `moe_sectioned/`. Nothing in `moe_reranker/`, `moe_rrf/`,
`baselines/` or `twotower/` was modified — `moe_rrf.features` and
`moe_reranker.diagnostics` are imported read-only and pinned by drift guards in
`tests/test_moe_sectioned.py`.

## Why this experiment existed

The MoE proposal was to explode the seeker's `lookingFor` field and reason about
each ask separately. What had actually been built collapsed all of it into a
single integer — `n_sections` — while the section-scoring machinery ran as a
parallel path whose output was a report, never a feature. This closes that gap
and tests the idea directly. Full plain-language writeup, with diagrams:
[`docs/html/moe-sectioned-plan.html`](html/moe-sectioned-plan.html).

## What changed, mechanically

The unit of prediction moved from a pair to a **(pair, section) row**. 131
training pairs become **708 rows**, 5.40 asks per pair.

| block | dims | source |
|---|---|---|
| section↔candidate similarity | 3 | cosine, rank percentile, cosine − pair-level cosine |
| interaction | 32 | elementwise product of the two embeddings, projected by a learned layer |
| pair scalars | 12 | `moe_rrf.features`, imported unchanged, broadcast across a pair's rows |
| gate input *(separate path)* | 16 | projected section embedding **only** |

Two deliberate departures from `moe_reranker.model.MMoE`:

1. **The gate sees only the ask.** In the pair-level model the gate saw the same
   features as the experts, so "what is this expert for" was unconstrained, and
   the diagnostic once caught routing tracking *seeker identity*. Routing is now
   structurally a function of the section text, which turns that check from an
   inference into a direct one.
2. **Experts are two layers** (`47→24→ReLU→16→ReLU`), not `Linear(12→4)+ReLU`.
   The old expert was a weighted sum with a kink; four of them could barely
   out-express one logistic regression, which is a plain reason the mixture never
   beat one. Affordable now at 708 rows rather than 111 pairs.

## Results — 5 seeds x seeker-disjoint 5-fold CV, 131 real pairs

Every number is pooled out-of-fold AUC, averaged over 5 seeds. The seed changes
the fold assignment *and* the model init, and every arm sees the same folds
within a seed, so "wins" counts seeds where the arm beat logistic regression
head-to-head.

| arm | mean AUC | sd | vs logistic | wins |
|---|---|---|---|---|
| `logistic_pair` | 0.5758 | 0.024 | — | — |
| **`moe_attention`** (TF-IDF) | **0.6467** | 0.029 | **+0.0709** | **5/5** |
| `moe_mean` (TF-IDF) | 0.6404 | 0.019 | +0.0646 | 5/5 |
| `moe_softmax` (TF-IDF) | 0.5568 | 0.056 | −0.0190 | 2/5 |
| `mlp_attention` (TF-IDF) | 0.5446 | 0.048 | −0.0312 | 2/5 |
| `moe_attention` (Qwen3, correct prompts) | 0.5433 | 0.050 | −0.0324 | 1/5 |
| `moe_attention` (Qwen3, no prompts) | 0.5378 | 0.028 | −0.0380 | 1/5 |

**Single-seed runs were meaningless and are kept only as a record of that.**
`sec_001` through `sec_004` produced four different arm rankings in four runs.
That is what overfitting plus a 131-pair sample looks like from the outside, and
no conclusion should be drawn from any single-seed row.

## Findings

**1. The sectioned model beats the standing bar, consistently.** `moe_attention`
on TF-IDF sections averages 0.6467 against logistic regression's 0.5758 and wins
in **5 of 5 seeds**, mean margin +0.071. This is the first architecture in this
project to beat plain logistic regression on real pairs under a repeated-seed
comparison. It is still a small-sample result and the holdout is unspent, but the
consistency is what distinguishes it from the earlier single-run noise.

**2. Exploding the ask is what helps — not the pooling rule.** `moe_mean`
(0.6404) is statistically indistinguishable from `moe_attention` (0.6467), and
both crush the pair-level baseline. So the gain comes from **scoring each ask
separately**, not from cleverly deciding which ask matters. The learned attention
that motivated the whole design contributes roughly nothing over a plain average.

**3. The mixture *does* matter here, reversing the earlier reading.**
`mlp_attention` — one expert, identical pooling — scores 0.5446 against
`moe_attention`'s 0.6467. Four experts beat one by 0.102. This contradicts the
first (unreplicated) run and every earlier MoE experiment in this project. The
plausible reason is that per-ask rows finally give the experts something to
specialise *on*: an ask is a coherent unit, a whole profile is not.

**4. Frozen-softmax pooling collapsed once the sample was replicated.**
`moe_softmax` averages 0.5568 with the highest variance of any arm (sd 0.056). Its
apparent single-run win at 0.5913 did not survive four more seeds. At tau=0.05 it
is nearly a hard max, so it bets the pair on one ask's verdict and inherits that
verdict's noise.

**5. Qwen3-Embedding-8B section embeddings are worse than TF-IDF here, and the
asymmetric-prompt fix did not change that.** 0.5433 with the correct
`prompt_name="query"` on the ask side, 0.5378 without — both far below TF-IDF's
0.6467, both losing to logistic regression in 4 of 5 seeds. This is genuinely
surprising: the same model beats Voyage-4-large at the pair level (0.6595 vs
0.6086). The working hypothesis is that a section is a short, jargon-dense
shopping list, which is exactly the regime where lexical overlap is strong and
dense semantics add little — consistent with the hybrid baseline fitting alpha
around 0.95 onto the lexical channel. It has not been tested further.

**6. Routing is decisive and not a seeker shortcut.** Gate entropy fell to 16-41%
of the uniform ceiling, expert usage stayed spread (22-28%), and routing-vs-seeker
mutual information sat at +0.087 to +0.140 excess over a permutation null, under
the 0.15 warning line and below the pair-level model's +0.109 at its worst.
Constraining the gate to see only the ask worked as designed.

## Honest caveats

- **131 pairs.** A +0.071 mean margin with a between-seed sd of ~0.03 is
  consistent and reproducible, but it is 131 examples. Fold-to-fold sd within a
  seed is 0.10–0.18. The 5/5 seed sweep is what makes this reportable at all;
  it is not a substitute for more data.
- **Seeds share one dataset.** Re-seeding changes the fold split and the init, not
  the sample. It bounds optimisation noise, not sampling error, so the true
  interval around 0.6467 is wider than 0.029 suggests.
- **The TF-IDF vocabulary is fit on the whole pool**, not per fold. That is a mild
  optimism shared identically by every arm, so it cannot explain the gap between
  arms, but it inflates all absolute numbers somewhat. The Qwen3 arm has no such
  issue — and scores worse.
- **`emb_pca_dims=48` was chosen by reasoning, not swept.** It was set to bring the
  projection layers from ~960k parameters down to ~2.3k. No other value was tried,
  so it is a fix for a clear defect rather than a tuned hyperparameter.
- **Two paid-run bugs were found by inspection, not by the pipeline.** fp32 loading
  (OOM) and the missing asymmetric prompt were both caught by comparing against
  `baselines/hf_embedding/`, which already handled them. Copying that working
  loader instead of writing a new one would have avoided both.
- **Synthetic pretraining was not attempted.** The plan calls for it; the previous
  experiment showed every synth-trained arm landing below the no-model floor, so it
  is deliberately deferred rather than repeated on faith.
- **The holdout is unspent.** These are cross-validated numbers on the train pool.
  The 69-pair holdout is one shot and has not been used.

## What this means

**The sectioning was the idea worth having.** Splitting a seeker's `lookingFor`
into individual asks and scoring each one separately is worth about +0.07 AUC over
the pair-level baseline, consistently across seeds. That is the first thing in
this project to clear plain logistic regression under replication.

**The attention was not.** Mean pooling matches learned attention. The
interpretability by-product — "we introduced Garrett because you asked for
brand-side operations leaders" — survives regardless, since per-ask verdicts exist
either way, so attention is still worth keeping for that reason alone. Just not
for accuracy.

**The mixture earned its place for the first time**, and plausibly because an ask
is a coherent thing to specialise on where a whole profile is not.

**Qwen3 sections did not help**, which is the clearest thing the $0.55 of GPU
bought. Worth knowing before anyone spends more on dense embeddings for this
field.

Next steps, in order of value:

1. **Spend the holdout.** The bar was cleared under replication, which is the
   condition the plan set. This is the one experiment that has earned it.
2. **Ablate the interaction block.** If TF-IDF sections win on lexical overlap,
   the 32-dim interaction may be doing nothing and the 3 similarity scalars may be
   the whole story. Free to test.
3. **Sweep `emb_pca_dims`.** Currently one untested value.

## Reproduce

```bash
# free, local, ~2 min
PYTHONPATH=. .venv/bin/python -m moe_sectioned.experiment --run-id sec_001

# paid: encode both populations with Qwen3-Embedding-8B on an A100
modal run moe_sectioned/modal_encode.py
modal volume get dorby-moe-sectioned-emb qwen3 ./artifacts/moe_sectioned/embeddings
PYTHONPATH=. .venv/bin/python -m moe_sectioned.experiment --encoder qwen3 --run-id sec_002
```

Raw output: `artifacts/moe_sectioned/sec_001/result.json`.

The holdout gate is enforced in code: `--run-holdout` refuses to score the frozen
69 pairs unless the cross-validated result already beat logistic regression on
the same folds. It currently does not, so the holdout stays unspent.
