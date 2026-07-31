# The 69-pair holdout ranks bad models fine and good models not at all

Every baseline row in `docs/baseline-results-holdout.md` rests on 69 pairs — 29
positive queries against a 65-candidate corpus. That population had already
reversed a conclusion three times in this project. This sweep re-scores every
frozen baseline on all **200 real pairs** (100 positive queries, 178-candidate
corpus) so the leaderboard rests on 2.9× the sample.

Fourteen models are now on one population: TF-IDF, frozen BERT, Voyage-4-nano,
Voyage-4-large, five open-weight HF embedders, and five fine-tuned twotower
arms.

## The headline finding

Split the models by where they land on all 200 and correlate the two
populations' rankings:

| group | Spearman(holdout MRR, all-200 MRR) | p |
|---|---|---|
| all 14 models | **+0.886** | 0.000 |
| bottom 8 | **+0.976** | 0.000 |
| **top 6** | **−0.029** | 0.957 |

Taken as a whole the holdout looks excellent — +0.886 is a strong correlation,
and it is why nobody caught this earlier. But that number is carried almost
entirely by the bottom of the table. Among the six best models the holdout
carries **no information at all**: −0.029 is indistinguishable from random
ordering.

That is precisely the wrong place to be uninformative. Nobody in this project
has ever had to decide whether zembed-1 beats frozen BERT. Every decision that
was actually made — which baseline is the bar, whether a fine-tune helped,
whether an open-weight model can replace production — was a decision *among the
top six*, which span just 0.052 MRR on all 200.

**Practical rule:** the holdout is fine as a screen for "is this model broken".
It cannot rank models that work. Anything competitive must be scored on all 200.

## Results — all 200 real pairs (100 positive queries, 178 candidates)

Ranked by MRR. Full table, including the holdout reference and easy-negative
slices, in `docs/baseline-results-real200.md`.

| model | family | pair AUC | hard-neg | MRR | R@1 | R@10 |
|---|---|---|---|---|---|---|
| twotower `top1_ctrl` | fine-tuned | 0.5683 | 0.5484 | **0.3550** | **0.1900** | 0.6900 |
| twotower Arm A (v2) | fine-tuned | 0.5594 | 0.5558 | 0.3341 | 0.1800 | 0.6400 |
| BGE-en-ICL | open-weight | 0.5389 | 0.5226 | 0.3190 | 0.1700 | 0.6200 |
| Voyage-4-nano (frozen) | baseline | 0.5593 | 0.5046 | 0.3171 | 0.1800 | 0.5900 |
| Voyage-4-large (production) | baseline | 0.5726 | 0.5422 | 0.3102 | 0.1300 | **0.7000** |
| twotower Qwen micro-6 | fine-tuned | **0.5947** | **0.5608** | 0.3031 | 0.1400 | 0.6600 |
| twotower `top1_sharp` | fine-tuned | 0.5429 | 0.4578 | 0.3010 | 0.1400 | 0.6400 |
| twotower Qwen micro-1 | fine-tuned | 0.5604 | 0.4828 | 0.2734 | 0.1400 | 0.5600 |
| Qwen3-Embedding-8B | open-weight | 0.5529 | 0.4680 | 0.2045 | 0.0500 | 0.5500 |
| TF-IDF (lexical) | baseline | 0.5649 | 0.5164 | 0.1313 | 0.0500 | 0.2600 |
| E5-Mistral-7B-instruct | open-weight | 0.4597 | 0.3772 | 0.1159 | 0.0300 | 0.3000 |
| Frozen BERT | baseline | 0.4697 | 0.4108 | 0.0941 | 0.0200 | 0.1800 |
| NV-Embed-v2 | open-weight | 0.4841 | 0.3836 | 0.0857 | 0.0400 | 0.1600 |
| zembed-1-embedding | open-weight | 0.4707 | 0.4864 | 0.0377 | 0.0100 | 0.0800 |

## What changes

**1. "Qwen3-Embedding-8B beats Voyage-4-large" does not survive.** This is the
claim in `CLAUDE.md` and `docs/hf-embedding-baseline-findings.md` — the first
model of any kind to beat Boardy's production model, 0.6595 vs 0.6086 pair AUC.
On all 200 pairs Qwen scores **0.5529 against Voyage-4-large's 0.5726**, and is
worse on every retrieval metric (MRR 0.2045 vs 0.3102, R@1 0.0500 vs 0.1300).
Its hard-negative AUC of 0.4680 is below chance — and hard negatives are the
only population that exists in production.

This was verified through the exact code path that produced the original claim,
not a similar one: the same `baselines/hf_embedding` encoder, the same 8192
context, the same absence of Matryoshka truncation. The holdout subset of this
very run lands at 0.6543 against the published 0.6595.

**2. No open-weight model beats production overall.** BGE-en-ICL is the best of
them and genuinely good at retrieval — MRR 0.3190 and R@1 0.1700 both *beat*
Voyage-4-large's 0.3102 / 0.1300 — but it trails on pair AUC (0.5389 vs 0.5726)
and on recall@10 (0.6200 vs 0.7000). It is a real result and worth carrying
forward, but it is not the claim that was on the board, and BGE-en-ICL was never
the model the claim was about.

**3. TF-IDF's respectable holdout showing was a small-pool artifact.** Pair AUC
holds up (0.5649 on 200 vs 0.5922 on 69) but retrieval collapses as the pool
grows: MRR 0.2475 → 0.1313, R@1 0.1379 → 0.0500. This partly answers
`docs/possible-bugs.md` #3 — TF-IDF beating the fine-tunes was always a
pair-classification result, and it does not survive as a retrieval result.

**4. The fine-tuned nano arms hold the top two MRR spots.** `top1_ctrl` (0.3550)
and Arm A v2 (0.3341) lead all 14 models, ahead of Voyage-4-large. The 69-pair
view had ranked Qwen micro-6 first; on 200 it falls to sixth — the single
largest rank movement in the table (1 → 6). This strengthens, rather than
weakens, the conclusion in `docs/twotower-top1-optimised-experiment.md`.

**5. Pair AUC and retrieval disagree, and the disagreement is systematic.** Qwen
micro-6 has the best pair AUC of any model (0.5947) and sixth-best MRR;
BGE-en-ICL is ninth on AUC and third on MRR. Ranking a model requires naming the
metric first — "best model" is not well defined here.

## Reproduction fidelity — why these numbers can be trusted

Each run scored the holdout subset alongside all 200, so every model's holdout
row can be checked against its own published number. Same code, same encoders,
same `baselines/metrics` calls:

| model | published holdout AUC | this sweep | delta |
|---|---|---|---|
| TF-IDF | 0.5922 | 0.5922 | exact |
| Frozen BERT | 0.4595 | 0.4595 | exact |
| E5-Mistral-7B | 0.5664 | 0.5664 | exact |
| BGE-en-ICL | 0.5750 | 0.5750 | exact |
| zembed-1 | 0.5052 | 0.5086 | +0.0034 |
| NV-Embed-v2 | 0.5034 | 0.5034 | exact |
| Qwen3-8B | 0.6595 | 0.6543 | −0.0052 |

**Five of seven reproduce exactly.** The two that don't (zembed-1, Qwen3-8B) run
bf16, where a text's embedding depends slightly on which texts share its batch,
and this sweep's pair ordering differs from the baselines' file order. They land
within ±0.006. No protocol difference is involved anywhere.

Note NV-Embed-v2 reproduced exactly only after being moved to A100-80GB at the
published 8192 context; the same run at 4096 drifted to 0.5095.

## Two defects found along the way

**TF-IDF is not reproducible across environments, and this affects the repo's
published numbers.** `baselines/tfidf` fits with `max_features=20000`, and on
this data the vocabulary lands at *exactly* 20000 — the truncation binds.
sklearn then drops features by document frequency and the tie-break at that
cutoff is not stable across environments. Identical code, data, scikit-learn
1.9.0 and numpy 2.4.6 give holdout AUC 0.5922 / MRR 0.2475 / R@1 0.1379 in this
repo's venv but 0.5914 / 0.2653 / 0.1724 in a Modal container. Pair AUC barely
moves; rankings flip, because a different vocabulary slice changes which terms
carry IDF weight. There are no ties in the score matrix, so this is not argsort
tie-breaking — it is a genuinely different fitted vectorizer.

Consequence: TF-IDF's numbers here come from `eval_real_full/run_baseline.py`
run locally, the only environment that reproduces the published row. Anyone
re-running `baselines/tfidf` elsewhere should expect different numbers and
should not treat that as a bug in their setup.

**`max_length` must match the published run or the comparison is void.** An
initial pass ran E5-Mistral and NV-Embed-v2 at 4096 where their published runs
used 8192. E5's holdout drifted to 0.5819 from 0.5664; at the corrected 8192 it
reproduces exactly. Every published HF run used 8192 with no truncation — this
is now asserted in `modal_baseline_eval.py`'s `CONFIGS` comment.

## Caveats

- **NV-Embed-v2 is a documented approximation** and was already flagged as such
  in `docs/hf-embedding-baseline-findings.md` — it runs symmetrically because it
  ships no sentence-transformers prompts dict, which understates it. It also
  OOMs at 8192 on A100-40GB for the 200-pair text set; `--gpu A100-80GB` is
  required, which is a new constraint the holdout-only run never hit.
- Retrieval metrics are comparable **between models within one subset only**. A
  178-candidate pool is strictly harder than a 65-candidate one, so `all` and
  `holdout` columns must never be compared to each other. `n_candidates` is
  recorded in every metrics file.
- 100 positive queries still quantizes recall@1 at 0.01. Differences of one or
  two queries remain noise; the top-6 spread of 0.052 MRR is roughly five
  queries' worth.
- This sweep changes *how models are ranked*, not what the task is. Absolute
  numbers remain low because the task is residual "will these two connect",
  not topical relevance — see `docs/objective.md`.

## Reproduce

```bash
python -m pytest tests/test_eval_real_full_baselines.py -q

# neural baselines on Modal (TF-IDF deliberately excluded — see above)
modal run eval_real_full/modal_baseline_eval.py --configs bert
modal run eval_real_full/modal_baseline_eval.py --configs qwen8b --gpu A100-40GB
modal run eval_real_full/modal_baseline_eval.py --configs e5_mistral --gpu A100-40GB
modal run eval_real_full/modal_baseline_eval.py --configs zembed --gpu A100-40GB
modal run eval_real_full/modal_baseline_eval.py --configs bge_en_icl --gpu A100-40GB
modal run eval_real_full/modal_baseline_eval.py --configs nv_embed --gpu A100-80GB

modal volume get dorby-eval-real-full-results real200_baselines \
    ./artifacts/eval_real_full/

# TF-IDF locally — the only environment that reproduces its published row
python -m eval_real_full.run_baseline --config tfidf

python -m eval_real_full.export   # -> docs/baseline-results-real200.{md,json}
```

Total Modal cost for the sweep: **under $0.70**, dominated by six 7-8B models.

## What this leaves

- `docs/baseline-results-holdout.md`, `CLAUDE.md`, and
  `docs/hf-embedding-baseline-findings.md` all still assert the Qwen headline.
  They need correcting against this table.
- BGE-en-ICL beating production on retrieval is unexplored and was previously
  read as "strong retrieval, middling classification" on 69 pairs. It holds up.
- The hybrid TF-IDF+nano fusion (current #2 on the holdout table at 0.6397) has
  **not** been scored on all 200 — it needs a fitted fusion, not just an
  encoder, so it did not fit this sweep's shape.
- The LLM judge (`gemini-3.1-flash-lite`, holdout 0.6358) is also unscored here;
  it has no shared vector space, so it has no retrieval metrics to compare.
