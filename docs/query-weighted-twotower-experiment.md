# Query-weighted encoding on the fine-tuned two-tower model

## Question

`query_weighted/` found that on the **frozen** voyage-4-nano encoder, embedding
the search query and profile as separate vectors and blending them
(`normalize(alpha*query + (1-alpha)*profile)`) roughly doubles recall@1 over
simply concatenating query into the profile text. Does that hold for the
**fine-tuned** encoder too — specifically `top1_ctrl`
(`artifacts/twotower_top1_optimised/top1_ctrl_001`), the best fine-tune in the
project so far (all-200 recall@1 0.19, first fine-tune ever to beat the frozen
baseline; see `docs/twotower-top1-optimised-experiment.md`)?

## Method

New isolated package, `twotower_query_weighted/` — no file under `twotower/`,
`query_weighted/`, or `eval_real_full/` is modified; only their public API is
imported. Same design as `query_weighted/eval.py`: encode `profile_only` and
`query_only` once each through the adapter (`twotower.eval.encode_role`, LoRA
adapter loaded read-only via `twotower.eval.load_model_for_eval`), then every
alpha is a free numpy blend of the two. Text builders (`profile_only`,
`query_only`, `concat_baseline`) are imported unchanged from `query_weighted.text`
— they already wrap `baselines.bert_frozen.text`, the same serialization
`twotower.data.LabeledPair.seeker_text` uses, so `concat_baseline` text is
byte-identical to what the adapter was fine-tuned and originally evaluated on.
Scored on all 200 real pairs (`eval_real_full.data.load_real_pairs`), same
metric functions (`baselines.metrics`) as every other number in this project.

```
modal run twotower_query_weighted/modal_eval.py --run-id qw_top1_ctrl_001
```

## Results — `top1_ctrl`, all 200 real pairs

| Encoding | Pair AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|
| Original (concat, published) | 0.5683 | 0.3550 | 0.19 | 0.69 |
| profile_only | 0.5489 | 0.2800 | 0.13 | 0.59 |
| **query_only** | 0.5945 | **0.5076** | **0.32** | **0.90** |
| **alpha_0.6** | **0.6129** | 0.4818 | 0.29 | 0.87 |

**The pattern replicates on the fine-tuned model.** Query-weighting nearly
doubles recall@1 over the model's own original concat-text encoding (0.19 →
0.29–0.32) and lifts every other metric too. `alpha_0.6` on the fine-tuned
adapter (AUC 0.6129) is now the best pair-classification number of any model —
frozen or fine-tuned — measured on all 200 real pairs in this project.
`query_only` gives the single best recall@1/MRR/R@10 of anything tested here.

As with the frozen-model experiment, `profile_only` is the weakest arm by a
wide margin — the profile text dilutes rather than adds to the query's signal,
on the fine-tuned encoder just as on the frozen one.

## Serving-cost note

This is not a training change and does not add serving latency. The seeker's
*profile* embedding doesn't depend on the live query — it can be computed
once, offline, and cached, exactly like candidate embeddings already are. Only
the query itself needs a live encode, which the current concat-based system
already requires (it encodes "profile + query" as one string per request).
Switching to `query_only` or an alpha blend needs the same one live encode
call — no additional round trip.

## Which arm to prefer

`query_only` wins on retrieval (recall@1, MRR, recall@10) by throwing the
profile away entirely; `alpha_0.6` wins on pair AUC (overall accept/decline
discrimination) by keeping a fifth of the profile's contribution. Which
matters more depends on the product surface: `query_only` for "find the best
few candidates fast," `alpha_0.6` for "will this specific pair actually work
out." Full sweep across all nine alphas is in
`artifacts/twotower_query_weighted/qw_top1_ctrl_001/top1_ctrl/metrics.json`.

## Topology visualization

Published alongside: a 3D PCA embedding graph of the 200 real pairs
(`artifacts/twotower_query_weighted/embeddings_dump.json` →
`docs/html/query-weighted-topology.html`), toggling between the frozen and
fine-tuned encoder and between direct seeker→candidate edges and the 2-hop
seeker→query→candidate path, colored by accept/decline outcome. Artifact:
https://claude.ai/code/artifact/a7787d14-b5c0-415a-bce6-8f629d63e865

Caveat: PCA to 3D keeps only ~13–16% of the embedding variance for both
models (`explained_variance` in `graph_data.json`) — positions show coarse
clustering, not precise distances. It is a qualitative complement to the
metrics table above, not a substitute for it.

A second **Projection** toggle (PCA vs. raw dims 1–3) makes that concrete:
taking the model's first three raw output dimensions as-is — no PCA — keeps
under 1% of variance (`raw_dim0_variance`), roughly 50x less than PCA's 3
components. Neural embedding dimensions aren't ordered by importance the way
PCA components are; even for Matryoshka-trained models like voyage-4-nano,
where truncating to a smaller *prefix* (e.g. 1024 of the full dimension) stays
usable, that ordering holds only at the coarse checkpoints the model was
trained at, not down to individual dimensions 1–3. The toggle exists to make
that visually obvious rather than just asserted.

## Reproduce

```bash
export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1  # not needed here, no Bedrock calls
modal run twotower_query_weighted/modal_eval.py --run-id qw_top1_ctrl_001
modal volume get dorby-twotower-query-weighted-results qw_top1_ctrl_001 \
    ./artifacts/twotower_query_weighted/qw_top1_ctrl_001

modal run twotower_query_weighted/modal_dump_embeddings.py
modal volume get dorby-twotower-query-weighted-results embeddings_dump.json \
    ./artifacts/twotower_query_weighted/embeddings_dump.json
python scripts/build_query_weighted_topology.py
```
