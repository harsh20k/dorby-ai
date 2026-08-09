# top1_ctrl field/query grid sweep

Generated: 2026-08-09. Code: `baselines/twotower_top1_ctrl_field_sweep/`.
Isolated new experiment, not an edit of `baselines/voyage_nano_field_sweep`
(the same 105-combo grid run against the frozen model) — see the "ML /
data-science experiments" isolation rule in CLAUDE.md.

## What this tests

`docs/voyage-nano-field-sweep-experiment.md` ran the 105-combo seeker ×
candidate field / search-query grid against **frozen** Voyage-4-nano. This
experiment re-runs the identical grid — same 105 combos, same all-200-pair
population, same full metric suite — against **`top1_ctrl_001`**, this
project's best fine-tuned model on the metric/population CLAUDE.md treats
as authoritative (`docs/baseline-results-real200.md`): a LoRA adapter
(rank 8, q/k/v/o_proj) on Voyage-4-nano, trained with micro-batch 6 and a
single negative per anchor — the winning cell in "Which Lever Actually
Worked?" (`docs/twotower-ablation-verdict.md`). Full-profile top1_ctrl
scores pair AUC 0.5683 on all-200 (0.5974 on holdout), already the
project's best all-200 fine-tune number pre-sweep — this experiment asks
whether field selection helps a fine-tuned model too, the way it helped
frozen Voyage-4-nano (0.5614 → 0.6110).

Run on Modal (A100-40GB), one GPU boot for all 105 combos (22 unique encode
groups via the same dedup trick as the frozen sweep — see
`baselines/twotower_top1_ctrl_field_sweep/sweep.py`).

## Results

**Top 10 by pair AUC** (200 real pairs):

| Seeker fields | Candidate fields | Query | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR |
|---|---|---|---|---|---|---|
| background | lookingFor | yes | **0.6395** | 0.6056 | 0.6972 | 0.3492 |
| positioning | lookingFor | yes | 0.6318 | 0.5838 | 0.7012 | 0.3678 |
| positioning | background+lookingFor | yes | 0.6250 | 0.5636 | 0.7128 | 0.4288 |
| lookingFor | background+lookingFor | yes | 0.6224 | 0.5948 | 0.7376 | 0.4031 |
| None (query-only) | lookingFor | yes | 0.6220 | 0.5512 | 0.7780 | 0.3884 |
| background | positioning+lookingFor | yes | 0.6191 | 0.5740 | 0.6704 | 0.3661 |
| positioning+background | lookingFor | yes | 0.6183 | 0.6012 | 0.6684 | 0.3417 |
| positioning | positioning+lookingFor | yes | 0.6168 | 0.5488 | 0.7058 | 0.3813 |
| lookingFor | lookingFor | yes | 0.6168 | 0.5648 | 0.7244 | 0.3102 |
| lookingFor | positioning+lookingFor | yes | 0.6167 | 0.5422 | 0.7184 | 0.3377 |

**Bottom 5 by pair AUC:**

| Seeker fields | Candidate fields | Query | Pair AUC |
|---|---|---|---|
| positioning | lookingFor | no | 0.5474 |
| positioning | background+lookingFor | no | 0.5446 |
| background | positioning+background | no | 0.5431 |
| positioning | background | no | 0.5420 |
| background | positioning+background+lookingFor | no | **0.5416** |

**Best on hard-neg AUC:** seeker=`positioning+background`, candidate=
`positioning+lookingFor`, query=yes — 0.6058 hard-neg AUC (pair AUC
0.6055). The overall-best combo (`background`/`lookingFor`) is a close
second at 0.6056 hard-neg AUC — the pair-AUC winner and the hard-neg-AUC
winner are nearly the same combo here, unlike the frozen sweep where they
diverged more (0.6110 pair-AUC winner scored only 0.5396 hard-neg).

**Best on retrieval (MRR):** seeker=`None` (query-only), candidate=
`background+lookingFor` OR `positioning+background`, query=yes — MRR
0.4736 (tied). Query-only seekers dominate the MRR ranking (4 of the top 5
MRR combos), same pattern as the frozen sweep, more pronounced here.

**Query-only seeker:** best is candidate=`lookingFor`, pair AUC **0.6220**
— beats the full-profile baseline (0.5683) using zero seeker profile
fields, and is close behind the overall winner (0.6395). Stronger than the
frozen model's equivalent (0.5861).

**Query matters, consistently:**

| | Mean pair AUC |
|---|---|
| Query included (n=56) | 0.5966 |
| Query excluded (n=49) | 0.5578 |

Every one of the 5 worst combos excludes the query — same pattern as the
frozen sweep.

## Reading the result

**Field selection helps the fine-tuned model even more than it helped the
frozen one.** Full-profile top1_ctrl scores 0.5683 pair AUC on all-200; the
best combo found here (0.6395) is +0.0712 over that — a bigger absolute
gain than field selection bought the frozen model (0.5614 → 0.6110,
+0.0496). Fine-tuning did not make the model field-selection-proof; the
same "shorter, targeted text beats the full profile" pattern holds.

**This is now the best pair AUC of any embedding-family model tested in
this project on the all-200 population** — 0.6395 beats every row in
`docs/baseline-results-real200.md`'s all-200 table (previous best: Voyage-
4-large 0.5726, best fine-tune Qwen micro-6 0.5947) and every combo in the
frozen Voyage-4-nano field sweep (best 0.6110). It is within 0.006 of the
LLM judge's own best (0.6451, `docs/llm-judge-focused-prompt-experiment.md`)
— by far the closest any embedding-based model has come to LLM-judge
performance anywhere in this project, and unlike a per-candidate LLM call,
top1_ctrl's serving cost is a merged LoRA adapter with **no runtime
overhead over frozen nano** (per the project's `<100ms` latency framing in
CLAUDE.md).

**Winning fields differ from both the frozen sweep and the LLM judge.** The
frozen sweep's best was seeker=`positioning`/candidate=`lookingFor`; the
LLM judge's best is seeker=`positioning+lookingFor`/candidate=all three;
top1_ctrl's best is seeker=`background`/candidate=`lookingFor`. Three
different "best field selections" across three different models on the
same task — reinforces the earlier finding that field selection is
model-specific, not a universal property of the fields.

**Caveat: same as the frozen sweep** — 105-way search on one 200-pair
population, no held-out check. The gap between best (0.6395) and 2nd-best
(0.6318) is small enough that this should be read as "field selection
helps," not as a validated production config without a holdout check.

## Reproducing

```bash
# Modal GPU (A100-40GB; one boot runs all 105 combos via encoding dedup)
modal run baselines/twotower_top1_ctrl_field_sweep/modal_eval.py --run-id real_all
modal volume get dorby-twotower-top1-ctrl-field-sweep-eval real_all/sweep_results.json \
  ./artifacts/twotower_top1_ctrl_field_sweep_modal/real_all/sweep_results.json

# local smoke test only (adapter needs a GPU for a real run)
python -m baselines.twotower_top1_ctrl_field_sweep.eval --data-dir data --limit-combos 4
```

Full 105-row results:
`artifacts/twotower_top1_ctrl_field_sweep_modal/real_all/sweep_results.json`
/ `.csv`.
