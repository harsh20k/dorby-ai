# Static reciprocal two-tower — the cold-start slice of Ga Wu's FWP paper

**Status: run complete (`voyage-4-nano`, Modal A10G).** Adding the static
reciprocal term lifts pair ROC-AUC over the forward-only score on every
population tested (train +0.043, holdout +0.039, all-200 +0.033), with the
combined score landing at or above every frozen and fine-tuned model
previously measured on this repo's all-200 population. **Caveat up front,
before the table below reads as more than it is:** a 5,000-sample bootstrap
of the AUC delta gives 95% CIs that include zero on both holdout
(`[-0.0655, 0.1474]`) and all-200 (`[-0.0333, 0.0994]`) — `P(delta > 0)` is
0.76 and 0.84 respectively. Directionally consistent across three
independent-ish populations (train, holdout, all-200) and cheap to obtain
(zero training, one fitted scalar), but not statistically proven at 69-200
pairs. Treat as a promising lead, not a settled win — this project has
retracted a headline claim before (Qwen3-Embedding-8B vs. Voyage-4-large,
see `docs/all-200-baseline-sweep.md`) for exactly this reason.

Ga Wu (Prof., Boardy AI industry partner) shared a design proposal,
["Dynamic Reciprocal User Matching with Fast Weight Programmers"](https://) —
`Fast_Weight_Programmer_for_User_Matching.pdf`, 2026-06-07 — for a reciprocal
social-matching recommender. This experiment implements the one slice of it
that this repo's data can actually support, and tests it on the real 200-pair
dataset.

## The paper, in short

Every user has two textual profile sections: **look-for** (what they want,
`x_look`) and **background** (what they're about, `x_bg`). The paper encodes
these into two separate embeddings, `k_i = E_look(x_look_i)` and
`v_i = E_bg(x_bg_i)`, and scores a (seeker `u`, candidate `i`) pair with a
**reciprocal** combination:

```
s_forward(u, i)    = k_u . v_i     # what u wants vs. what i offers
s_reciprocal(i, u) = k_i . v_u     # what i wants vs. what u offers
S(u, i) = s_forward(u, i) + lambda * s_reciprocal(i, u)
```

The paper's main contribution sits on top of this: a **Fast Weight
Programmer** memory that adapts `k_u` into a session-dependent `q_t^(u)` from
a query user's logged (impression, skip, like, message, ..., timestamp)
history — two matrices (`A_+`, `A_-`) written by a delta rule and read to
produce a preference correction, trained end-to-end so the "slow" writer
networks learn what history should mean.

## Scope decision — why not the FWP memory

The FWP memory needs a **sequential per-user interaction log with
timestamps**: what was shown, what action was taken, when. This repo's real
dataset (`data/dataset_positive.json` / `dataset_negative.json`, see
`data/dataset_summary.md` and `docs/objective.md`) has none of that — every
pair is a single static `(seeker, candidate, accept/decline)` outcome, no
session, no sequence, no timestamp. Building the memory here would mean
fabricating interaction histories, which this experiment does not do.

The paper's own math says what to do with zero history — its eq. 121:

```
q_t^(u) = Norm(k_u)      # both A_+ and A_- are zero
```

i.e. the dynamic memory collapses to exactly the static look-for embedding.
That cold-start fallback is the only slice of the paper this dataset can
exercise, so this experiment implements *that* plus the piece nothing else in
this repo has tried: **the static reciprocal term** (paper eqs. 3, 8, 9).

## What was built

New isolated package: **`baselines/reciprocal_static/`** (per this repo's
experiment-isolation rule — nothing under `baselines/{bert_frozen,voyage_nano}`
or `twotower/` was touched).

| File | Role |
|---|---|
| `text.py` | `bg_text`/`look_text`/`seeker_look_text` — splits a profile into the paper's two views. Reuses `baselines.bert_frozen.text.PROFILE_FIELDS` read-only. |
| `eval.py` | CLI: encode both views with frozen `voyage-4-nano` (no fine-tuning — same frozen-baseline convention as every other `baselines/*`), compute forward + reciprocal cosine scores, fit `lambda`, report metrics. |

**Text split** (`baselines/reciprocal_static/text.py`):
- `look_text(profile)` = the profile's own `lookingFor` field, tagged.
- `bg_text(profile)` = every other profile field (`positioning`, `background`,
  `notes`, `locationAvailability`, `introPreferences`, `personalPreferences`,
  `meetingAndSchedulingPreferences`), tagged — i.e. `PROFILE_FIELDS` minus
  `lookingFor`.
- `seeker_look_text(profile, search_query)` = `look_text(profile)` +
  the pair's `searchQuery`. The paper has no query concept at all (its eq. 1
  is profile-only); `searchQuery` is this repo's own per-interaction demand
  signal, and CLAUDE.md's query-ablation finding says it's load-bearing, so
  it's folded into the seeker's look-for (demand) side, never the background
  side. A candidate's own `k_i` (used only for the reciprocal term) stays
  profile-only — they have no query in that direction.
- One consequence worth naming: a physical user therefore gets **two
  different k-embeddings** depending on which role a given pair puts them in
  (query-augmented as a seeker, plain as a candidate) — not the single
  canonical per-user `k_i` the paper's eq. 1 defines. This matches how every
  other baseline in this repo already treats seeker vs. candidate text
  asymmetrically (`searchQuery` only ever appears seeker-side).

**Both `E_look` and `E_bg` are the same frozen `voyage-4-nano`** applied to
two disjoint text views — not the untied, independently-weighted encoders the
paper explicitly leaves open ("potentially different"). Testing genuinely
separate-weight encoders (e.g. two LoRA adapters) is a further step, not this
one.

**Fitting `lambda`.** A single scalar, 1-D grid search (default range
`[-2, 2]`, step `0.05`) maximizing pair ROC-AUC of `s_fwd + lambda*s_recip`
on the frozen real **TRAIN** subset only (`eval_real_full`, `subset="train"`,
131 pairs) — never touching holdout or all-200 labels during fitting, same
train/holdout discipline as every other experiment here.

**Retrieval stays forward-only**, matching the paper's own design (Sec.
4.1/4.3: the ANN index holds only `v_i`, queried by `k_u` alone; reciprocal is
a rerank-only signal, Sec. 4.4). Pair-classification metrics use the combined
score; retrieval (MRR/NDCG/Recall@K) uses `s_forward` alone.

**Execution.** Ran on Modal (A10G), not local MPS — `baselines/reciprocal_static/
modal_eval.py`, mirroring `baselines/hf_embedding/modal_eval.py`'s single-model
shape (image, HF cache volume, results volume) plus the
`eval_real_full/data_frozen` manifest mount `eval_real_full/modal_baseline_eval.py`
needs. `voyage-4-nano` is small enough to run on local MPS (as every other
`baselines/*` command in this repo's README does), but this run was kept off
the local machine.

## Results

Run `real200_001`, model `voyage-4-nano`, `lambda` fit on 131 real train pairs
(grid `[-2, 2]`, step `0.05`): **fitted `lambda = 1.75`**
(train AUC forward-only 0.5512 → combined 0.5944).

| population | n (pos/neg) | corpus | pair AUC fwd-only | pair AUC combined | Δ | hard-neg AUC (combined) | easy-neg AUC (combined) | MRR (fwd-only retrieval) | R@1 |
|---|---|---|---|---|---|---|---|---|---|
| holdout | 69 (29/40) | 65 | 0.5853 | **0.6241** | +0.0388 | 0.6293 | 0.6448 | 0.4785 | 0.2414 |
| all-200 | 200 (100/100) | 178 | 0.5638 | **0.5964** | +0.0326 | 0.5478 | 0.7208 | 0.3461 | 0.1600 |

**Bootstrap uncertainty** (5,000 resamples, stratified by label, over the
cached per-pair `s_fwd`/`s_recip` scores): AUC-delta 95% CI is
`[-0.0655, 0.1474]` on holdout and `[-0.0333, 0.0994]` on all-200 —
`P(delta > 0)` = 0.76 / 0.84. Positive on average, same sign in three separate
checks (train where it was fit, holdout, all-200), but the interval crosses
zero — this is a lead, not a proof.

**Against the reference tables** (`docs/baseline-results-real200.md`, "All 200
pairs" / "Holdout 69 pairs"): the combined score's pair AUC (0.5964 all-200,
0.6241 holdout) is at or above **every** model in both tables except
`twotower Qwen micro-6`/`micro-1` — ahead of Voyage-4-large production
(0.5726 all-200 / 0.6086 holdout), every other frozen baseline, and every
other fine-tuned `twotower` arm, obtained with **zero training** and a single
fitted scalar. What it does *not* beat: retrieval. MRR (0.3461 all-200 /
0.4785 holdout) trails Voyage-4-large (0.3102 / 0.5287) and `top1_ctrl`
(0.3550 / 0.5436) — expected, since retrieval here ranks by `s_forward` alone
(paper-faithful design, Sec. 4.1/4.3) using only the short `lookingFor` +
query text, not the full profile every other baseline's seeker/candidate
encoding uses. **The gain is a reranking/classification effect, not a
retrieval effect** — which is exactly where a reciprocal term is supposed to
help (the paper reserves it for reranking, never retrieval, Sec. 4.4), and is
cheap under the <100 ms budget since it's one precomputed dot product per
already-retrieved candidate.

## Repro

```bash
# Modal (used for this run)
modal run baselines/reciprocal_static/modal_eval.py --run-id real200_001
modal volume get dorby-reciprocal-static-eval real200_001/metrics.json \
    ./artifacts/reciprocal_static/metrics.json

# local (small/cheap enough for MPS if preferred)
python -m baselines.reciprocal_static.eval --data-dir data

pytest tests/test_reciprocal_static.py -q
```

## Next steps (not done here)

- **Untie the encoders.** This run uses one frozen `voyage-4-nano` for both
  `E_look` and `E_bg`. The paper leaves them "potentially different" —
  testing two separately-trained LoRA adapters (one per view) is the natural
  next step and untested here.
- **Confidence, not just a point estimate, before trusting `lambda=1.75`.**
  The bootstrap above says the AUC gain could plausibly be noise; a larger
  eval population or a held-out re-fit (e.g. k-fold over the 131 train pairs)
  would tighten this before leaning on it for a real decision.
- **Try `lambda` as a full calibration** (paper eq. 101,
  `w1*s_fwd + w2*s_recip + b`) instead of the single-coefficient
  `s_fwd + lambda*s_recip` — more flexible, still cheap, still no per-eval-label
  leakage if fit on train only.
