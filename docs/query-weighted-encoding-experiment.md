# The profile is drowning the ask

The seeker side of every model in this project is one string: eight tagged
profile fields with `Search query: …` appended. On the 200 real pairs that query
averages **218 characters against a 10,178-character seeker string — about
2.1%**, roughly 55 tokens in 2,500.

That 2% is the only part saying what the person wants *right now*. The other 98%
is biography. This experiment tests whether that ratio is costing us, using
frozen `voyage-4-nano` with **no training of any kind**.

It is. Reweighting the two parts takes recall@1 on all 200 real pairs from
**0.1800 to 0.3000** and recall@10 from **0.5900 to 0.9100**.

Isolated package `query_weighted/`. Nothing under `baselines/`, `eval_real_full/`,
or any `twotower*` package was modified.

## Arms

Two independent families, so neither has to be trusted alone:

* **text-level** — change what the encoder sees: drop the query, use only the
  query, move it to the front, repeat it 3/5/10 times.
* **vector-level** — encode profile and query **separately**, then combine as
  `normalize(α·Q̂ + (1−α)·P̂)`. α=0 reproduces `profile_only` exactly and α=1
  reproduces `query_only` exactly (both test-pinned), so α is one clean knob.

`concat_baseline` is the current production seeker text and serves as the
validation gate.

## Results — all 200 real pairs (100 positive queries, 178 candidates)

| arm | pair AUC | hard-neg | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| `profile_only` | 0.5424 | 0.4862 | 0.2357 | 0.0900 | 0.3900 | 0.5000 |
| **`concat_baseline`** *(current)* | 0.5593 | 0.5046 | 0.3171 | 0.1800 | 0.4700 | 0.5900 |
| `query_first` | 0.5730 | 0.5304 | 0.3528 | 0.2100 | 0.5300 | 0.6500 |
| `query_x3_front` | 0.5843 | 0.5430 | 0.3596 | 0.2000 | 0.5500 | 0.6500 |
| `query_x5_front` | 0.5859 | 0.5464 | 0.3526 | 0.1900 | 0.5300 | 0.6500 |
| `query_x10_front` | 0.5879 | 0.5546 | 0.3599 | 0.2000 | 0.5600 | 0.6500 |
| `alpha_0.3` | 0.5785 | 0.5404 | 0.3884 | 0.2400 | 0.5700 | 0.6800 |
| `alpha_0.5` | 0.5866 | 0.5694 | 0.4309 | 0.2300 | 0.6800 | 0.8200 |
| **`alpha_0.6`** | **0.5872** | 0.5818 | 0.4649 | 0.2500 | 0.7500 | 0.8900 |
| `alpha_0.7` | 0.5799 | 0.5856 | 0.5009 | **0.3000** | 0.7600 | 0.8900 |
| `alpha_0.9` | 0.5619 | 0.5896 | 0.4924 | 0.2800 | 0.7700 | 0.9000 |
| **`query_only`** | 0.5530 | **0.5914** | **0.5019** | **0.3000** | **0.7800** | **0.9100** |

**`alpha_0.6` beats `concat_baseline` on every metric on every population** —
all 200, train-131, and holdout-69:

| arm | all-200 (AUC/hard/MRR/R@1) | train-131 | holdout-69 |
|---|---|---|---|
| `concat_baseline` | 0.559 / 0.505 / 0.317 / 0.180 | 0.551 / 0.469 / 0.355 / 0.211 | 0.579 / 0.571 / 0.461 / 0.276 |
| `alpha_0.6` | 0.587 / 0.582 / 0.465 / 0.250 | 0.562 / 0.546 / 0.532 / 0.324 | 0.631 / 0.647 / 0.737 / **0.586** |
| `query_only` | 0.553 / 0.591 / 0.502 / 0.300 | 0.523 / 0.556 / 0.574 / 0.366 | 0.602 / 0.643 / 0.700 / 0.517 |

On the holdout, recall@1 goes from **8 of 29** to **17 of 29**.

### Validation gate

`concat_baseline` reproduces frozen `voyage-4-nano`'s published all-200 row
**digit-for-digit**: pair AUC 0.5593, MRR 0.3171, recall@1 0.1800, recall@10
0.5900. The harness is not doing anything new to the baseline.

## Why it works

Jaccard token overlap with the *true* candidate versus a random distractor:

| seeker text | vs target | vs distractor | ratio |
|---|---|---|---|
| query only | 0.0314 | 0.0244 | **1.29×** |
| profile only | 0.1697 | 0.1630 | **1.04×** |

The profile overlaps everything about equally — 1.04× is almost no
discrimination. It is not neutral filler: averaged into the seeker vector it
dilutes the query's 1.29× edge with noise. `profile_only` scoring recall@1 0.0900
(worse than chance-adjacent) against `query_only`'s 0.3000 says the same thing
from the other side.

The two families agree, which matters because they have different failure modes.
Text-level arms improve only modestly (R@1 0.1800 → ~0.2000) and plateau,
because concatenation still forces one shared vector. Vector-level combination
keeps the query's direction intact and rises monotonically to α≈0.6–0.7.

## The caveat that has to be stated first

**The retrieval metric is partly circular here.** "Retrieval" means finding the
candidate who was actually introduced to this seeker — and per `docs/objective.md`
those candidates were selected **by Boardy production, using this same
searchQuery**. A query-only encoder is therefore partly re-deriving production's
own retrieval decision. Some of the recall@10 0.9100 is that.

Two things bound the concern:

* **Name leakage is ruled out.** Only 6 of 100 queries share a rare capitalised
  token (corpus df ≤ 3) with their target, and they are topic terms —
  `Vancouver`, `SEIS`, `PayPal` — not identity giveaways.
* **The accept/decline metrics are not circular at all.** Pair AUC and
  hard-negative AUC score a (seeker, candidate) pair directly with no corpus and
  no ranking, so production's selection cannot inflate them. There,
  `query_only`'s **hard-negative AUC of 0.5914** and `alpha_0.6`'s 0.5818 are the
  **best of any model measured on all 200** — ahead of Qwen micro-6's 0.5608 and
  Arm A's 0.5558. `alpha_0.6`'s pair AUC of 0.5872 is second only to Qwen
  micro-6 (0.5947) and beats production Voyage-4-large (0.5726).

So: treat the retrieval numbers as an upper bound with a circularity component,
and the hard-negative AUC as the clean signal. Both point the same way.

## Why this is cheap to ship

The vector-level form fits the <100 ms budget **better** than the current setup:

```
seeker_vector = normalize(α · encode(query) + (1−α) · encode(profile))
```

`encode(profile)` depends only on the user's profile, so it is precomputed
offline and cached exactly like candidate vectors — it changes when the profile
changes, not per search. The only online work is `encode(query)`, which is **~55
tokens instead of ~2,500**. That is a materially *faster* online encode than
today's path, not a slower one, plus one vector add and a normalize.

No fine-tuning, no new model, no extra serving dependency.

## Caveats

- One run, no replicate, no error bar. The effect is far larger than the
  one-query resolution limit (R@1 moves 12 queries of 100, 9 of 29 on the
  holdout), but the exact α optimum is not resolved — 0.6 and 0.7 differ by
  5 queries and that gap is not meaningful.
- Text-level arms confound "more weight" with "protected from truncation": at
  4096 tokens the longest seeker strings truncate, and front-loading keeps the
  query while `concat_baseline` loses it. The α arms have no such confound,
  which is why they carry the conclusion.
- Frozen `voyage-4-nano` only. Whether the same reweighting helps the fine-tuned
  adapters, Voyage-4-large, or the open-weight models is untested.
- The easy/hard negative split is pinned to the `concat_baseline` text for every
  arm. Otherwise repeating the query changes the Jaccard overlap that defines
  "hard" and the arms would be scored on different populations.
- 100 positive queries come from only 78 unique seekers, so queries are not
  fully independent.

## Reproduce

```bash
python -m pytest tests/test_query_weighted.py -q     # 11 tests, no GPU needed

modal run query_weighted/modal_eval.py --run-id qw_001 --subsets all,train,holdout
modal volume get dorby-query-weighted-results qw_001/metrics.json \
    ./artifacts/query_weighted/qw_001/metrics.json
```

Embeddings cache to the `dorby-query-weighted-cache` volume keyed by content
hash, so re-runs are free and a new α costs nothing (pure arithmetic on cached
vectors). Cost of `qw_001`: **$0.08**.

## What this leaves

1. **Replicate and resolve α.** A finer grid around 0.55–0.75 on a second run,
   and ideally α fitted on train and evaluated on holdout rather than read off
   the same table.
2. **Apply it to the fine-tuned models.** Every twotower arm trained on the
   concatenated seeker text. If dilution is this costly, the fine-tunes have
   been learning through the same noise — and `top1_ctrl` plus α-weighting is
   the obvious next combination.
3. **Re-examine the earlier conclusion.** `docs/twotower-top1-optimised-experiment.md`
   concluded the fine-tunes "pull the right person up the list without converting
   to first place." This says a large part of that ceiling was representational,
   not a limit of the loss or the labels.
4. **Escape the circularity.** The clean way to measure this is a candidate pool
   production never ranked for these queries. Nothing in the repo has one yet.
