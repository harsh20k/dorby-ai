# Multi-gate mixture-of-experts re-ranker over `lookingFor`

**Status: built, cross-validated, and not testable at the current data size.** The
architecture works and the diagnostics are clean, but on 111 real training pairs
it is statistically indistinguishable from using no model at all. The bottleneck
is data, not architecture. The real 69-pair holdout was deliberately **not spent**.

Plain-language writeup with charts: [`docs/html/moe-reranker-review.html`](html/moe-reranker-review.html).

## The idea

A two-stage retrieve-then-rerank design. Stage 1 generates candidates with the
fine-tuned two-tower model plus BM25 (this already exists as
`synth_pipeline/pairing_rrf/`). Stage 2 re-ranks that shortlist with a mixture of
experts, where the seeker's `lookingFor` field is exploded into its separate asks.

The original sketch named five experts after the synthetic negative generator's
failure modes (`wrong_side`, `wrong_stage`, `wrong_role`, `geo_mismatch`,
`prefs_conflict`). That version is not what was built, for a reason recorded below.

## Isolation

Everything lives in **`moe_reranker/`**. No file in `baselines/`, `twotower/`, or
`synth_pipeline/` was modified.

| File | Role |
|---|---|
| `config.py` | `MoEConfig` — every knob, single source of truth |
| `features.py` | the bottom-network feature builder (train-only fitting) |
| `aggregation.py` | the 7 section-aggregation shapes, incl. the veto family |
| `section_scoring.py` | scoring loop; imports shared baseline helpers read-only |
| `model.py` | `MMoE` + the three loss terms |
| `diagnostics.py` | the three gate diagnostics |
| `data.py` | leakage-safe assembly via `twotower.data.build_split_bundle` |
| `train.py` | training + one-shot holdout CLI |
| `import_rrf.py` | read-only frozen copy of a `pairing_rrf` batch |
| `scripts/compare_section_aggregation.py` | experiment 1 |
| `scripts/moe_cv_compare.py` | experiment 2 |
| `tests/test_moe_aggregation.py`, `tests/test_moe_reranker.py` | 56 tests |

**On the deliberate duplication.** `aggregation.py` reimplements `max`,
`topk_mean`, and `softmax` rather than importing them from
`baselines/voyage_nano_sectioned/aggregate.py`. That is the isolation boundary:
adding veto-shaped modes would otherwise mean editing a module the
lookingFor-sectioning experiment owns. Duplication is only safe when pinned, so
`test_matches_shared_baseline_on_shared_modes` asserts the two implementations
agree numerically on all three shared modes — if the shared one changes, that
test fails and says so.

## Why the five named experts were dropped

The five axes are the synthetic negative generator's *construction recipe*, not an
observed property of real declines. Checked directly in the data:

- **0 of 100** real negatives carry a `failure_mode`.
- **339 of 340** negatives in `dataset_negative.json` have none — the labels only
  survive in `artifacts/synth/*/staged/` (~52–55 per mode) and `promote.py` drops
  them.

Real negatives are intros production recommended and humans declined, with no
reason recorded (see `docs/objective.md`). So there is no supervision for per-axis
experts, and no evidence real declines even decompose into those five buckets.

**The resolution** (user's call, and the right one): make each expert's objective a
**learned parameter**. Specialization then emerges from the final accept/decline
signal and needs no per-axis labels — which is also how MoE is done in practice.
The five axes were demoted from expert heads to **input features** (geo overlap,
stage gap, role overlap/complementarity), where they need no labels at all.

The cost of that resolution: unnamed experts don't explain themselves, so the
interpretability argument for the MoE weakens. `structured_cot` already showed
decomposition buys interpretability but not accuracy
(`docs/llm-judge-experiment.md`).

## Experiment 1 — how should expert opinions combine?

Eleven aggregation shapes over one cached embedding pass, matched 69-pair holdout.
Free to run; nothing re-encodes.

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_section_aggregation.py --holdout-only
```

Two families. **Relevance-shaped** asks "does any ask fit?" (`max`, `topk_mean`,
`softmax`). **Veto-shaped** asks "is any ask badly violated?" (`min`, `softmin`,
`noisy_or`) — the "one dealbreaker sinks the intro" theory. `mean` is the control,
i.e. the aggregation shape that lost as `structured_cot`.

Ordered most-veto → most-relevance:

| mode | family | pair AUC | hard-neg AUC | MRR | R@10 |
|---|---|---|---|---|---|
| `min` | veto | 0.5836 | 0.5724 | 0.5159 | 0.6552 |
| `softmin(τ=0.01)` | veto | 0.5871 | 0.5759 | 0.5128 | 0.6552 |
| `softmin(τ=0.05)` | veto | 0.5871 | 0.5828 | 0.5150 | 0.6897 |
| `softmin(τ=0.20)` | veto | 0.5931 | 0.5914 | 0.5146 | 0.6897 |
| `mean` / `noisy_or` | average / veto | 0.5940 | 0.5931 | 0.5147 | 0.6897 |
| `topk_mean(k=2)` | relevance | 0.5940 | 0.6052 | 0.5127 | 0.6897 |
| `max` | relevance | 0.5957 | **0.6155** | 0.4934 | 0.6897 |
| `softmax(τ=0.01)` | relevance | 0.5974 | 0.6138 | 0.4946 | **0.7241** |
| `softmax(τ=0.20)` | relevance | 0.5966 | 0.5966 | **0.5151** | 0.6897 |
| **`softmax(τ=0.05)`** | relevance | **0.5983** | 0.6052 | 0.5149 | 0.6897 |

**Verdict: relevance-shaped, with a tunable τ. The veto hypothesis is not
supported.** Pair AUC and hard-negative AUC both climb monotonically at every step
away from the veto.

**Read honestly.** No individual gap is significant. Paired bootstrap (20,000
resamples) on `softmax(τ=0.05)` − `min`: **+0.0146, 95% CI [−0.021, +0.051]**,
P(relevance better) ≈ 0.79. The Hanley–McNeil SE on any single AUC here is
**±0.070**, five times the entire spread of results. What carries weight is the
ordered ladder — six configurations, two metrics, same direction — not any row.

Two design consequences:

- `softmax(τ=0.05)` beats `τ→0` (hard argmax, 0.596), so **sharper is not
  automatically better**. Keep τ tunable.
- `noisy_or` landed on `mean`'s numbers to four decimals. That is expected, not
  broken: over the narrow cosine band this data occupies, `log p` is nearly linear,
  so the geometric mean induces the same ranking. The probabilistic veto has no
  teeth because cosine 0.26 maps to p=0.63, nowhere near "dealbreaker." Hard `min`
  is the honest test, and it came last.

The `softmax` mode is exactly the professor's Idea 1,
`g_m(x) = exp(a_m/τ) / Σ exp(a_j/τ)`, with the gate over *sections* rather than
experts — so that slide's mechanism already had a measured result in this repo.

### Bug: a fake 0.8500 AUC from four lines of normalization

The first `noisy_or` mapped cosines to probabilities by rescaling against the
min/max of the score matrix it was handed. But positives and negatives are
aggregated in **separate calls**, so positives were rescaled by 0.333–0.767 and
negatives by 0.265–0.908 — an identical cosine became a higher probability on the
positive side. It reported **pair AUC 0.8500 / AP 0.8598** against a field where
nothing exceeds 0.66.

The implausibility was the tell, not the code. Section count was ruled out as an
explanation first (0.5435 alone).

Fixed with a data-independent map, `p = (1 + cos) / 2`. Pinned by
`test_record_score_is_batch_independent`, which asserts for **all 7 modes** that a
record's score cannot change based on which other records share its batch —
verified to fail against the old implementation.

## Experiment 2 — does the MMoE earn its parameters?

Built exactly to spec: 3 experts, shared bottom over 14 features, two gates
(accept/decline + judge-score), τ tunable, both entropy terms, expert dropout,
all three diagnostics logged from epoch one.

```bash
PYTHONPATH=. .venv/bin/python -m moe_reranker.train --run-id moe_001
PYTHONPATH=. .venv/bin/python scripts/moe_cv_compare.py --folds 5
```

**Multi-gate means multi-task**, which is the actual claim of MMoE — related tasks
share experts while each learns its own mixing weights. The task set here:

| task | label source | volume |
|---|---|---|
| accept / decline (**the real objective**) | human outcome | **111 train pairs** |
| would this be a good intro? | LLM judge, already cached for all 200 real pairs | free |

A scarce main task beside an abundant related one is the regime where MMoE earns
its keep. With one task it degenerates into ordinary MoE.

### Finding 1 — it overfits before it learns

Train AUC climbs 0.545 → **0.861** while train-dev AUC *falls* 0.512 → **0.345**,
monotonically. Early stopping selects **epoch 1**. Not a tuning problem.

### Finding 2 — nothing is separable, including doing nothing

A single 20-pair train-dev split (14 pos / 6 neg) has AUC granularity 1/84 and
cannot adjudicate anything, so the real test is seeker-disjoint 5-fold CV over the
111-pair train pool, four models on identical features:

| model | mean AUC | fold std |
|---|---|---|
| nano cosine, no model at all | 0.5282 | 0.138 |
| logistic regression | 0.5251 | 0.144 |
| MoE, single task | 0.5434 | 0.065 |
| **MMoE, multi-task** | **0.5536** | 0.067 |

The winner's margin over *doing nothing* is 0.025 against a fold spread of 0.067.
Every model is within one standard deviation of every other.

**The MMoE is not broken and did not fail — the experiment has no power.** Two
things survived: multi-task beat single-task in 3 of 5 folds (+0.010 mean,
directionally what MMoE claims), and both MoE variants were **half as volatile**
across folds. The regularization buys stability, not accuracy.

### Finding 3 — the slide's bottom network assumes data this project lacks

Feeding two 1024-d embeddings into the shared bottom, as drawn, is ~30,000
parameters against 111 examples. The build uses **14 engineered scalars** instead
(cosine, TF-IDF, retrieval rank percentiles, geo/stage/role proxies) for a total of
**280 parameters**. Raw-embedding input exists behind `--emb-pca-dims`, off by
default.

### The holdout was not spent

CV says nothing is separable, so running the one-shot 69-pair check would burn the
project's one clean measurement to confirm a null. `--holdout` is opt-in and
remains unused.

## The three diagnostics

Adopted from the slides, plus one this dataset needs.

1. **Average expert usage** `ḡ_m` — barplot; one dominant expert means collapse.
   Observed: 40% / 38% / 22%, no collapse.
2. **Per-example gate entropy** `H(g(x))` — catches a gate that just averages.
   Observed: 0.086 of a 1.099 ceiling (8% of uniform), sharply decisive.
3. **Routing vs seeker identity** (added) — mutual information between the chosen
   expert and *which seeker* the pair belongs to. On `rrf_002`, seeker identity
   alone predicted the label at 0.687 AUC because 12 of 40 seekers rejected every
   candidate; a gate can learn that shortcut. The professor's setting (click-rate,
   many users) doesn't need this. This one does — especially since real data has
   only 19 within-seeker triplets, so the base rate **cannot** be cancelled by
   construction here.

### Bug: diagnostic 3 was saturated and its raw number meaningless

It first reported normalized MI **0.815**, which reads as a five-alarm fire. With
111 rows over **75 seekers**, most seekers appear once, so knowing the seeker
nearly determines the row and therefore its routing. **Uniformly random routing
scores 0.706** on this data (p95 0.759). The real excess is **+0.109** —
statistically detectable (p=0.000) but modest.

It now reports the observed MI, a permutation null mean, the **excess**, and a
p-value, warning only on excess ≥ 0.15 with p ≤ 0.05. Two tests pin it: random
routing must show near-zero excess despite high raw MI, and genuinely
seeker-determined routing must still be caught.

## A gap in the slides worth closing

**Sharpening alone collapses the gate.** `L_sharp = (1/N)·H(g(x))` pushes every
example toward one-hot, but nothing stops *every* example choosing the *same*
expert — which is what Diagnostic 1 detects. Both terms are needed and they oppose
each other:

- minimize entropy **per example** → each pair commits to an expert
- maximize entropy of the **batch-average** gate → experts stay balanced

The slides supply the first and the detector for the second. `model.py` implements
both (`sharpen_loss`, `balance_loss`);
`test_sharpen_and_balance_oppose_each_other` pins the opposition.

**Top-K is skipped.** The slide notes it isn't differentiable (fixable with
Gumbel-softmax or noisy top-k plus load balancing), but with 3 experts it buys
nothing — it's a compute optimization for the dozens-of-experts regime. Temperature
does the same job differentiably.

## Data: the frozen RRF copy

```bash
PYTHONPATH=. .venv/bin/python -m moe_reranker.import_rrf --batch-id rrf_003
PYTHONPATH=. .venv/bin/python -m moe_reranker.import_rrf --batch-id rrf_003 --verify
```

`import_rrf.py` is **read-only on the source**. It copies staged pairs into
`artifacts/moe_reranker/data/<batch_id>/pairs.json` with a `provenance.json`
recording the source path, a SHA-256 over the staged files, counts, and the
labeler. `--verify` re-checks the digest, so a later run can prove the source
didn't shift underneath it. Copying rather than reading live matters because the
RRF track is still being iterated on.

**`rrf_003` imported: 2,619 pairs (1,175 pos / 1,444 neg), 418 seekers, 337
carrying both classes → 2,773 within-seeker triplets.** The real pairs have 19,
from 6 seekers. Source verified unchanged after import.

**What these labels are.** One LLM judge's opinion
(`google/gemini-3.1-flash-lite`, naive framing), not human accept/decline
outcomes. That judge scores 0.6358 pair AUC and 0.5942 accuracy on the real hard
slice, so roughly 4 in 10 labels are wrong on hard pairs. Usable as an auxiliary
task and for within-seeker ranking structure; **never** a substitute for the 200
real pairs, and **never** promoted into `data/dataset_*.json`.

## Next step

The build's conclusion — data, not architecture — points at exactly this batch.
`rrf_003` supplies 24× the pairs and 146× the within-seeker triplets, which
unlocks both the volume the MMoE needs and the within-seeker training that cancels
the per-seeker base rate by construction. Wire the frozen copy into
`moe_reranker/data.py` as an auxiliary-task source, retrain, and re-run
`scripts/moe_cv_compare.py`. If the MMoE is going to beat a plain logistic on these
features, that is where it will show.

Open question that must be answered first: **feature compatibility.** The frozen
copy has no `voyage-4-nano` embeddings (its cached vectors are Qwen3-Embedding-8B,
a different space from the real-pair features), so either the nano-cosine feature is
dropped for the auxiliary rows, the synthetic pairs get a nano encode pass, or both
sides move to Qwen3. Mixing spaces across tasks would be incoherent.
