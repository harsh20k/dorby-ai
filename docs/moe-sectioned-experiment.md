# Sectioned MoE — per-ask experts with learned attention pooling

**Status: built and run locally on the free encoder. The recommendation lost to
its own control arm.** The frozen-softmax pooling that the earlier aggregation
sweep already picked beats the learned attention it was supposed to generalise,
and a single expert matches four. Nothing beat plain logistic regression. The
frozen 69-pair holdout was **not spent**.

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

## Results — seeker-disjoint 5-fold CV on 131 real pairs, TF-IDF encoder

| arm | AUC (pooled OOF) | mean-of-folds | fold std | what it isolates |
|---|---|---|---|---|
| `logistic_pair` | 0.5484 | **0.6363** | 0.160 | the standing bar |
| `moe_attention` | 0.5516 | 0.5865 | 0.167 | **the recommendation** |
| **`moe_softmax`** | **0.5913** | 0.6151 | 0.131 | frozen pooling — the control |
| `mlp_attention` | 0.5796 | 0.5844 | 0.153 | 1 expert: is the mixture doing anything? |
| `moe_mean` | 0.5218 | 0.5510 | 0.169 | the null: does selectivity matter at all? |

**Two AUC columns, because this repo has used both conventions and they
disagree.** `AUC (pooled OOF)` concatenates every out-of-fold prediction and
scores once. `mean-of-folds` averages the per-fold AUCs, which is what
`moe_rrf/experiment.py` reports — and it reproduces that experiment's published
logistic figure (0.6363 here vs 0.6398 there), confirming the two are measuring
the same thing under different estimators. At ~26 pairs per fold, mean-of-folds
is the noisier and more optimistic of the two; **the pooled column is the one to
trust**, and this discrepancy is worth knowing before comparing any number in
this repo against any other.

## Findings

**1. The recommendation lost to its own control.** `moe_softmax` (0.5913) beats
`moe_attention` (0.5516). The argument for learned attention was that the winning
frozen-softmax pooling *is* attention with fixed weights, so learning those
weights should generalise it. It doesn't. Learning ~16 extra parameters on 131
labels costs more than the flexibility buys.

**2. The mixture is not carrying its weight — again.** `mlp_attention`, with a
single expert, scores 0.5796 against `moe_attention`'s 0.5516. One expert beats
four under identical pooling. This is the redundancy risk stated in the plan
before running, and it is now the **fourth** independent time in this project
that the MoE machinery has failed to pay for itself.

**3. Selective pooling does beat averaging.** `moe_mean` (0.5218) is worst by a
clear margin. So the *sectioned* framing is not worthless — treating asks
separately and then weighting them is better than treating the whole field as one
blob. It is the *learned* weighting and the *mixture* that add nothing.

**4. Nothing beat the bar.** On mean-of-folds, logistic regression's 0.6363 leads
every arm. On pooled OOF, `moe_softmax`'s 0.5913 leads logistic's 0.5484 — but
see the caveat below before reading anything into that.

**5. Routing is decisive and mostly not a seeker shortcut.** Gate entropy fell to
0.574 of a 1.386 ceiling (41%), so the gate is genuinely routing rather than
averaging, and expert usage stayed spread (9.3%–34.8%) rather than collapsing.
Routing-vs-seeker mutual information was 0.254 against a permutation null of
0.167 — an excess of **+0.087**, below the 0.15 warning threshold and lower than
the pair-level model's +0.109. Constraining the gate to see only the ask did
reduce the seeker shortcut, as designed. That mechanism works; it just isn't
worth anything on the metric.

## Honest caveats

- **Every difference here is inside the noise.** Fold-to-fold standard deviations
  are 0.131–0.169 on 5 folds over 131 pairs. `moe_softmax` leading
  `moe_attention` by 0.040 is not a result that would survive a different seed.
  What the table supports is "no arm clearly separated from the others," not a
  ranking.
- **The two AUC estimators disagree about which arm wins**, which is itself
  evidence the sample is too small to rank arms.
- **This ran on TF-IDF, not Qwen3.** The plan's step 1 — encode both populations
  with Qwen3-Embedding-8B, the only model measured to beat Voyage-4-large here —
  is a paid Modal run and has **not** been done. The lexical encoder is a floor,
  not the test. Section embeddings from a real semantic model could change the
  interaction block substantially; they could also change nothing.
- **The TF-IDF vocabulary is fit on the whole pool**, not per fold. That is a mild
  optimism shared identically by all five arms, so it cannot explain a difference
  between them, but it inflates every absolute number a little.
- **Synthetic pretraining was not attempted.** The plan calls for it; the previous
  experiment showed every synth-trained arm landing below the no-model floor, so
  it is deliberately deferred rather than repeated on faith.

## What this means

The sectioned framing is worth keeping — `moe_mean` losing to everything says
per-ask reasoning beats whole-field reasoning. The **mixture of experts** and the
**learned pooling** are the parts that keep failing to earn their place.

The cheapest informative next step is not more architecture. It is the Qwen3
encode, because it is the one component of the original proposal never actually
tested, and because it changes what the model can see rather than how it reasons
about what it sees.

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
