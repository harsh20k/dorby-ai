# Experiment HTMLs — index

One flat table first: every self-contained HTML this project has produced,
local or published, in one place. Detailed per-file notes (what each one
demonstrates, why it looks the way it does) follow below for anyone who
wants the full story.

## All local + published HTML outputs

| Name                                                                                                                                                  | Where                                                                                                    | Built                                | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`pairs-comparison-graph-hub-test.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-hub-test.html)                       | local (`docs/html/`)                                                                                     | 2026-07-24                           | Dual-pane real-vs-synth pairing graph, 3-profile smoke test after routing generation prompts through LangSmith Hub                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [`pairs-comparison-graph-no-refex.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-no-refex.html)                       | local (`docs/html/`)                                                                                     | 2026-07-24                           | Dual-pane graph, 5-profile batch after dropping redundant reference examples from `generate_profile`                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [`pairs-comparison-graph-disjoint.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-disjoint.html)                       | local (`docs/html/`)                                                                                     | 2026-07-24                           | Dual-pane graph, same 5 profiles re-paired with disjoint seeker/candidate split + per-seeker cap                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [`pairs-comparison-graph-named.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-named.html)                             | local (`docs/html/`)                                                                                     | 2026-07-24                           | Dual-pane graph, 10-profile batch after the name-collision fix (10/10 unique names)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [`real-pairs-tfidf-cluster.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-tfidf-cluster.html)                                     | local (`docs/html/`)                                                                                     | 2026-07-24                           | Force-directed real-pairs graph, TF-IDF similarity force added                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [`real-pairs-voyage-lookingfor-cluster.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-cluster.html)             | local (`docs/html/`)                                                                                     | 2026-07-24                           | Force-directed real-pairs graph, voyage-4-large `lookingFor` similarity force                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`real-pairs-tfidf-pca.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-tfidf-pca.html)                                             | local (`docs/html/`)                                                                                     | 2026-07-24                           | Static PCA/SVD scatter (no physics), TF-IDF — 1.67% variance explained                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [`real-pairs-voyage-lookingfor-pca.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-pca.html)                     | local (`docs/html/`)                                                                                     | 2026-07-24                           | Static PCA scatter, voyage-4-large `lookingFor` — 12.9% variance explained                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [`real-pairs-voyage-lookingfor-3d-pca.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-3d-pca.html)               | local (`docs/html/`)                                                                                     | 2026-07-24                           | 3D PCA scatter, hand-rolled canvas projector — 17.3% cumulative (PC1–3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [`real-pairs-voyage-lookingfor-3d-manifold.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-3d-manifold.html)     | local (`docs/html/`)                                                                                     | 2026-07-24                           | 3D PCA / t-SNE / UMAP scatter with a layout selector                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [`baseline-results-holdout-browser.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/baseline-results-holdout-browser.html)                     | local (`docs/html/`)                                                                                     | 2026-07-20                           | Browser for the matched-holdout baseline comparison table                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [`holdout-embedding-space-3d.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/holdout-embedding-space-3d.html)                                 | local (`docs/html/`)                                                                                     | 2026-07-25                           | 3D PCA map of the 69 holdout contacts in voyage-4-nano space, whole-profile vs. `lookingFor`-sectioned embeddings, with good/bad match lines and a scatter (dispersion) analysis — see `scripts/build_holdout_embedding_space_3d.py` / `scripts/analyze_section_dispersion.py`                                                                                                                                                                                                                                                                                             |
| [`holdout-field-isolation-embedding-space-3d.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/holdout-field-isolation-embedding-space-3d.html) | local (`docs/html/`)                                                                                     | 2026-07-26                           | 3D PCA map of the same 115 holdout contacts, but every profile field and every `lookingFor` ask embedded **alone** (no other field present) instead of swapped into an otherwise-whole profile — see "Field isolation experiment" below                                                                                                                                                                                                                                                                                                                                    |
| [`llm-judge-comparison.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/llm-judge-comparison.html)                                             | local (`docs/html/`)                                                                                     | 2026-07-25, rebuilt 2026-07-26       | Ranked pair-AUC bar chart + hard/easy-neg breakdown for LLM-judge (model, framing) combinations against every embedding baseline, matched 69-pair holdout — see `scripts/build_llm_judge_browser.py` / `docs/llm-judge-experiment.md`. 2026-07-26 rebuild adds the new `structured_cot` variant (see "LLM judge: does forcing multi-aspect CoT help?" below); the two Bedrock combos and `calibrated` are cached artifacts from an earlier session not present in this checkout (`artifacts/` is gitignored), so the chart currently shows `naive` + `structured_cot` only |
| LLM judge vs. embedding baselines                                                                                                                     | [published](https://claude.ai/code/artifact/12f8f93b-8fc4-41e5-bc13-b05ce8ab45fa)                        | 2026-07-25                           | Published version of `llm-judge-comparison.html` above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Pairs graph — Boardy AI                                                                                                                               | [published](https://claude.ai/code/artifact/642d0a82-7784-4843-b0ad-5686cf7db24c)                        | 2026-07-24                           | Likely one of the `pairs-comparison-graph*.html` variants above, published via `--fragment` — exact source not traceable from this session                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Real pairs graph — Boardy AI                                                                                                                          | [published](https://claude.ai/code/artifact/ac74ea3a-912d-407a-a040-74d8c62d1edd)                        | 2026-07-22 (page updated 2026-07-24) | Predates the batches above; likely an early real-only single-pane build                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Holdout comparison browser — Dorby AI                                                                                                                 | [published](https://claude.ai/code/artifact/95beeed4-9a3d-4a79-906d-cf2d24d0457f)                        | 2026-07-20                           | Likely `baseline-results-holdout-browser.html` above, by date match                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Does splitting lookingFor into sections help matching?                                                                                                | [published](https://claude.ai/code/artifact/3ec8c0da-9ba1-4de9-b52d-b057507b6163)                        | 2026-07-24                           | lookingFor field-sectioning: 4 experiments (candidate- vs seeker-sectioned, softer aggregation, hybrid-fusion stacking) — see `docs/lookingfor-sectioning-findings.md`                                                                                                                                                                                                                                                                                                                                                                                                     |
| Holdout contacts in voyage-4-nano space                                                                                                               | [published](https://claude.ai/code/artifact/5bf01ecd-0731-4f7e-93f3-ee74f8688e21)                        | 2026-07-25                           | Published version of `holdout-embedding-space-3d.html` above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Query-Time Nudge vs. Joint Encoding                                                                                                                   | [published](https://claude.ai/code/artifact/d491b7db-0db8-458a-8148-78001c084e30)                        | 2026-07-25                           | Not traceable to a local `docs/html/` file from this session                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| LLM judge vs. embedding baselines                                                                                                                     | [published](https://claude.ai/code/artifact/12f8f93b-8fc4-41e5-bc13-b05ce8ab45fa)                        | 2026-07-25                           | Not traceable to a local `docs/html/` file from this session                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Which fields carry a person's identity?                                                                                                               | [published](https://claude.ai/code/artifact/c3daa30f-4b68-4312-ba4b-8e69e4c77550)                        | 2026-07-26                           | Published version of `holdout-field-isolation-embedding-space-3d.html` above (field isolation experiment)                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Synthetic Pair Pipeline — Proposed Flow                                                                                                               | [published](https://claude.ai/code/artifact/5455a3ec-2c0f-4926-8e08-bc705868a6cf)                        | 2026-07-26                           | Not traceable to a local `docs/html/` file from this session                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [`knowledge-graph-experiment.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/knowledge-graph-experiment.html)                                 | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/knowledge-graph-experiment.html) | 2026-07-27                           | One real user's profile + one accepted/one declined real intro, each decomposed into a knowledge graph by `google/gemini-3.1-flash-lite` and merged on shared concept labels, plus a type-taxonomy layer — see `docs/knowledge-graph-experiment.md`                                                                                                                                                                                                                                                                                                                        |
| [`twotower-rrf-triplet-findings.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-rrf-triplet-findings.html)                           | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-rrf-triplet-findings.html) | 2026-07-29                           | `voyage-4-nano` + `Qwen3-Embedding-8B` LoRA fine-tunes on `rrf_003` triplets (`MultipleNegativesRankingLoss`): training-loss curves for both runs, real-69-pair-holdout results table vs. Voyage-4-large/Arm A, and a plain-language explanation of why pair AUC rose while retrieval recall@1 fell — see `docs/twotower-rrf-triplet-experiment.md`                                                                                                                                                                                                                        |
| [`twotower-rrf-triplet-bigbatch-comparison.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-rrf-triplet-bigbatch-comparison.html)     | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-rrf-triplet-bigbatch-comparison.html) | 2026-07-29                           | Follow-up: isolated `voyage-4-nano` re-run (`twotower_rrf_triplet_bigbatch/`, own package) with real batch size 2→6 (GPU-probed ceiling was 8) and 2 negatives per anchor instead of 1 — closed recall@1 to exactly match frozen Voyage-4-large (0.345) at the cost of pair AUC dropping below it; four-way comparison table + next-steps vs. the original triplet runs — see `docs/twotower-rrf-triplet-bigbatch-experiment.md`                                                                                                                                           |
| [`twotower-ablation-verdict.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-ablation-verdict.html)                                   | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-ablation-verdict.html) | 2026-07-29                           | **Combined verdict** of the 2×2 ablation splitting the bigbatch run's two levers apart (`twotower_rrf_triplet_ablation/`, effective batch pinned to 12 in every arm, each arm run twice for a measured noise floor): **micro-batch size is what moved retrieval** (+0.05 MRR, +0.06 recall@1) while **the second negative hurt** (−0.03 pair AUC) because 27.5% of k=2 negative slots are duplicates. Also documents a corrected single-run claim and a dev-set-too-small-to-select finding — see `docs/twotower-rrf-triplet-ablation-experiment.md`                       |
| [`moe-reranker-review.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-reranker-review.html)                                             | local (`docs/html/`)                                                                                     | 2026-07-29                           | Design review + results for the **multi-gate mixture-of-experts re-ranker** over `lookingFor` (`moe_reranker/`, own isolated package). Settles the combine-rule question (veto-shaped aggregation lost monotonically to a temperature-sharpened gate) and reports the built MMoE's seeker-disjoint CV: 0.5536 vs 0.5282 for no model at all, fold std 0.067 — **not testable at 111 real training pairs**. Documents two self-inflicted measurement bugs (a label-leaking normalizer reporting a fake 0.8500 AUC; a saturated routing-MI diagnostic) — see `docs/moe-reranker-experiment.md` |
| [`moe-rrf003-synthetic-training.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-rrf003-synthetic-training.html)                                 | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/moe-rrf003-synthetic-training.html) | 2026-07-29                           | **Answer to the previous experiment's open question: training on synthetic data does not help.** The MoE trained on `rrf_003`'s 2,619 judge-labeled pairs (`moe_rrf/`, own isolated package) scored *below* the no-model TF-IDF floor on 131 real pairs, and two arms landed below chance. Locates the failure precisely: synthetic **vocabulary** transfers (0.5631 vs 0.5660 real-fitted) but synthetic **labels** don't, and the judge teacher's own 0.5797 on real pairs is a ceiling below what plain logistic regression already reaches. Within-seeker training was the one win (+0.131). Two more measurement bugs caught — a broken TF-IDF reimplementation and a stale-cache collision that silently merged two arms — see `docs/moe-rrf003-synthetic-training-findings.md` |
| [`moe-sectioned-plan.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-sectioned-plan.html) | local (`docs/html/`) | 2026-07-30 | **PRE-EXPERIMENT PROPOSAL, kept unedited — see the findings page for what happened.** Plan for the per-section MoE (`moe_sectioned/`, own isolated package). Closes the gap between the original proposal (explode `lookingFor`, route per ask) and what had been built (all of it collapsed into the single integer `n_sections`, while the section-scoring path ran disconnected and its output was never a feature). Unit of prediction moves from a pair to a **(pair, section) row** — 131 pairs become 708 rows. Diagrams the funnel, the proposed architecture, and one real accepted pair traced all the way through (Garrett scores 92nd percentile on the seeker's *Brand Operators* ask and 70th on *Narvar Partnerships* — the pair was accepted on one ask, and today's model averages that away). **Result (5 seeds x 5-fold CV): the first architecture in this project to beat plain logistic regression under replication** — 0.6467 vs 0.5758 mean AUC, winning 5/5 seeds. But not for the predicted reason: mean pooling (0.6404) matches learned attention, so the gain is from *scoring each ask separately*, not from deciding which ask matters. Four experts beat one (0.5446), the first time the mixture has earned its place. Qwen3-Embedding-8B section embeddings scored **worse** than TF-IDF (0.5433) even after fixing an asymmetric-prompt bug. Single-seed runs gave four different rankings in four runs before an embedding-PCA fix cut the projection layers from ~960k parameters to ~2.3k. Also documents that this repo reports two different CV estimators (pooled out-of-fold vs mean-of-folds) that disagree by ~0.09 — see `docs/moe-sectioned-experiment.md` |
| [`moe-sectioned-findings.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-sectioned-findings.html) | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/moe-sectioned-findings.html) | 2026-07-31 | **What the sectioned MoE actually produced** (`moe_sectioned/`). Headline: exploding a seeker's `lookingFor` into individual asks and scoring each separately hits **0.6467 mean AUC vs logistic regression's 0.5758, winning 5/5 seeds** — the first architecture in this project to beat plain logistic regression under replication. But a **plain average of the asks (0.6404) matches learned attention (0.6467)**, so the gain is the sectioning, not the clever pooling that motivated the design. Four experts beat one (0.5446), the first time the mixture earned its place. Scores the proposal's five predictions honestly side-by-side (2 right, 3 wrong). Qwen3-8B section embeddings **lost** to keyword matching (0.5433) — which stopped looking anomalous when the Qwen3-beats-Voyage claim was retracted a day later in `docs/all-200-baseline-sweep.md`. Also documents three defects found by results looking strange: ~960k parameters on 708 rows, an fp32 OOM, and half the embeddings computed with the wrong asymmetric prompt — see `docs/moe-sectioned-experiment.md` |
| [`twotower-abl-a-batch-only.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-abl-a-batch-only.html)                                   | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-abl-a-batch-only.html) | 2026-07-29                           | Ablation Arm A — micro-batch 6, k=1. The winning cell (pair AUC 0.5983 / MRR 0.5326 / recall@1 0.3793, `_v2` run only — its two runs shipped different epochs and are not replicates); loss curve + dev-accuracy overlay, holdout table vs. frozen Voyage-4-large                                                                                                                                                                                                                                                                                                                                                                   |
| [`twotower-abl-b-negs-only.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-abl-b-negs-only.html)                                     | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-abl-b-negs-only.html) | 2026-07-29                           | Ablation Arm B — micro-batch 2, k=2. The worst cell (pair AUC 0.5595 / recall@1 0.2931); its dev metric falls after epoch 3 while training loss keeps dropping, the clearest sign of the duplicate-negative problem                                                                                                                                                                                                                                                                                                                                                        |
| [`twotower-abl-c-baseline.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-abl-c-baseline.html)                                       | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-abl-c-baseline.html) | 2026-07-29                           | Ablation Arm C — micro-batch 2, k=1. The baseline corner every effect in the ablation is measured against (pair AUC 0.5996 / MRR 0.4902 / recall@1 0.3276)                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [`eval-real-full-200pairs.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/eval-real-full-200pairs.html) | published artifact + local (`docs/html/`) | 2026-07-30 | Frozen voyage-4-nano vs. the Arm A fine-tune re-measured on **all 200 real pairs** instead of the 69-pair holdout. Finding: the holdout's pair-AUC/recall@1 gains do not generalise (pair AUC −0.003 on the other 131 pairs), but hard-negative AUC improves on every population and grows with sample size (+0.040 / +0.059 / +0.072). Writeup: `docs/eval-real-full-experiment.md` |
| [`twotower-qwen-bigbatch.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-qwen-bigbatch.html) | published artifact + local (`docs/html/`) | 2026-07-30 | Qwen3-8B LoRA fine-tune at micro-batch 6 vs 1 (effective batch pinned to 12), plus all seven models re-measured on **all 200 real pairs**. Findings: the micro-batch effect replicates on a 22x larger backbone (+0.087 MRR, +0.103 R@1); Qwen fine-tuning generalises where nano's did not (+0.053 pair AUC on 200 vs nano's +0.007); and the 69-pair holdout flatters Qwen ~4x more than nano, so its published "beats Voyage-4-large" headline reverses on the full set (frozen Qwen 0.5420 < nano 0.5593 < Voyage-large 0.5726). Writeup: `docs/twotower-qwen-bigbatch-experiment.md` |
| [`all-200-baseline-sweep.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/all-200-baseline-sweep.html) | published artifact + local (`docs/html/`) | 2026-07-31 | **All 14 models on one population** — every frozen baseline (TF-IDF, BERT, nano, large, 5 open-weight HF) plus 5 twotower arms re-scored on all 200 real pairs. Headline is methodological: the 69-pair holdout correlates with the all-200 ranking at Spearman **+0.976 across the bottom 8 models but −0.029 across the top 6** — it screens for broken models but cannot rank working ones, which is the only comparison this project ever makes. Consequences: the "Qwen3-8B beats Voyage-4-large" claim is **retracted** (0.5529 vs 0.5726, hard-neg 0.4680 below chance); no open-weight model beats production overall (BGE-en-ICL wins retrieval, loses AUC); TF-IDF's strength was a small-pool artifact. 5 of 7 models reproduce their published holdout row exactly. Two defects found: TF-IDF's fitted vocabulary is environment-dependent (`max_features=20000` binds exactly), and `max_length` must match the published run. Writeup: `docs/all-200-baseline-sweep.md`, tables: `docs/baseline-results-real200.md` |
| [`query-weighted-encoding.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/query-weighted-encoding.html) | published artifact + local (`docs/html/`) | 2026-07-31 | **The searchQuery is only 2.1% of the seeker string** (218 of 10,178 chars). Encoding profile and query *separately* and combining as `normalize(α·Q̂+(1−α)·P̂)` on **frozen voyage-4-nano with no training** takes all-200 recall@1 **0.1800→0.3000** and recall@10 **0.5900→0.9100**; `alpha_0.6` beats the concatenated baseline on **every metric on every population** (holdout R@1 8/29→17/29). Mechanism: profile-vs-candidate Jaccard discriminates at only 1.04× (query 1.29×), so the biography is noise that dilutes the ask. Validation gate: `concat_baseline` reproduces frozen nano's published row digit-for-digit. **Caveat: retrieval is partly circular** (production selected these candidates using the same query) — but name leakage is ruled out (6/100) and the non-circular hard-negative AUC of 0.5914 is the best of any model on all 200. Serving cost *drops* (online encode ~55 tokens instead of ~2,500). Writeup: `docs/query-weighted-encoding-experiment.md` |
| [`Dorby AI — Framing & Experiment Proposal`](https://claude.ai/code/artifact/5aed8e2e-a4e4-4c2d-adbc-500c307f855a) | published artifact only (no local file) | 2026-07-27 | Framing deck for the project's objective and proposed experiment slate. Published before the two-tower ablation series; predates the 200-pair evaluation, so any accuracy figures in it are 69-pair-holdout numbers. |
| [`project-story.html`](http://dorby-project-story-411960113601.s3-website-us-east-1.amazonaws.com/) | local (`docs/html/`) + S3 static site | 2026-07-30 | Plain-language chronological walkthrough of the whole project (2026-07-16 → 07-30) as a scroll-driven slideshow. Not a Claude artifact — hosted on S3, URL recorded in `docs/project-story-url.md`. |
| [`twotower-top1-optimised.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-top1-optimised.html) | published artifact + local (`docs/html/`) | 2026-07-30 | Two changes targeting recall@1 at fixed Arm A settings. **Sharpening the loss** (MNRL scale 20→50 + `hardness_mode='hard_negatives'`) **backfired on every metric** — all-200 R@1 0.1800→0.1400, below the untrained baseline, hard-neg AUC 0.4578 (worse than chance). **Fixing checkpoint selection** to rank against a real dev corpus (`CorpusRecallDevEvaluator`, `primary_metric='recall@1'`) produced the **best MRR of any model in the project (0.3550 on all 200)** and the first fine-tune to beat frozen nano at R@1 (19 vs 18 of 100) — invisible on the 69-pair holdout. Writeup: `docs/twotower-top1-optimised-experiment.md` |
| [`query-weighted-topology.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/query-weighted-topology.html) | published artifact + local (`docs/html/`) | 2026-08-03 | **Does `query_weighted/`'s frozen-model finding (query-weighting ~doubles recall@1) hold on the fine-tuned two-tower adapter too?** New isolated package `twotower_query_weighted/`, reusing `twotower.eval.encode_role` on `top1_ctrl` (the project's best fine-tune so far) plus `query_weighted.text`'s builders, read-only. **Yes — the pattern replicates.** `top1_ctrl` all-200: concat (original, published) AUC 0.5683 / MRR 0.355 / R@1 0.19 → **query_only AUC 0.5945 / MRR 0.5076 / R@1 0.32** (best retrieval of any model, frozen or fine-tuned, in the project) and **alpha_0.6 AUC 0.6129** (best pair-classification of any model on all 200). `profile_only` is again the weakest arm (AUC 0.5489), confirming the profile dilutes rather than helps on the fine-tuned encoder too. No serving-latency cost: the seeker profile vector is cacheable per-user offline, only the query needs a live encode, same as the current concat path. Also ships a 3D PCA topology graph (200 real pairs, seeker/query/candidate nodes, direct vs 2-hop via-query edges, frozen vs fine-tuned toggle) — caveat: only ~13-16% variance retained in 3D, qualitative not quantitative. Writeup: `docs/query-weighted-twotower-experiment.md` |
| [`twotower-no-query-comparison.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-no-query-comparison.html) | published artifact + local (`docs/html/`) | 2026-08-03 | **Does retraining `top1_ctrl` with the search query removed from training entirely replicate the eval-time query-weighting win found in `twotower_query_weighted/`? No — and a user-caught eval/train-mismatch bug changed the answer.** New isolated package `twotower_no_query/`, exact `top1_ctrl` recipe (library-default loss, `recall@1` checkpoint selection, micro-batch 6/accum 2, lr 2e-4, 5 epochs, A100-80GB) retrained on rows whose anchor text is `profile_to_text` instead of `seeker_to_text` — verified row-for-row identical to the original k1 file except the anchor text (all 643 rows), confirmed via a zero-cost local `--dry-run` before any GPU spend. **Bug** (`docs/possible-bugs.md` #5): the first eval reused `eval_real_full.eval.run_eval`, which hardcodes query-included seeker text — correct for every other adapter here, wrong for this one, which never saw a query in training. Caught by the user before publishing. Corrected eval (seeker text matching training, via `twotower_query_weighted.eval`'s already-published `profile_only` path): all-200 AUC 0.5574 / MRR 0.2827 / **R@1 0.13** / R@10 0.62 — **within noise of `top1_ctrl`'s own eval-time profile-only swap** (AUC 0.5489 / MRR 0.2800 / R@1 0.13 / R@10 0.59), i.e. whether the query is present *during training* barely matters at all. What moves the numbers is what's fed in *at eval time* — the same `top1_ctrl` weights score 0.13 R@1 on profile-only text and 0.32 on query-only text. Both fine-tunes clear the **frozen, never-trained** model by a real margin on the same profile-only text (AUC 0.5424 / MRR 0.2357 / R@1 0.09 / R@10 0.50, scored via the frozen-model experiment's own `qw_001` run) — fine-tuning genuinely helps the profile encoding (+44% relative R@1), it just doesn't matter which text it trained on to get there. **The 69-pair holdout misled again** (4th+ time in this project, independent of the bug): `no_query_001`'s holdout R@1 was 0.4138, MRR 0.5574 — looked like the best fine-tune ever, reversed completely on all 200. **Extension:** new isolated package `voyage_large_query_weighted/` runs the identical concat/profile-only/query-only/alpha-blend sweep from `query_weighted/` on **voyage-4-large** (Boardy's production model) via a role-name adapter, `query_weighted.eval.run_all_arms` unmodified. **voyage-4-large's query-only arm is the best result of any model, frozen or fine-tuned, in this entire project's all-200 comparison** — pair AUC 0.5452, hard-neg AUC **0.6140**, MRR **0.5897**, recall@1 **0.42**, recall@10 0.93 — beating nano's own query-only arm and every two-tower fine-tune above. **Third leg** (`docs/twotower-query-only-experiment.md`): new isolated package `twotower_query_only/` trains the same `top1_ctrl` recipe on seeker text built from the search query alone — no profile text at all — evaluated matched-distribution *from the start* this time (learning from the bug above). Result mirrors the profile side exactly: `query_only_001` (AUC 0.5952, hard-neg **0.6492**, MRR 0.4985, R@1 0.29, R@10 0.91) lands within noise of `top1_ctrl`'s eval-time query-only swap (R@1 0.32) — no consistent direction of advantage either way. **Conclusion across both legs: training the model specifically on a seeker-text representation does not beat just handing that representation to a normally-trained model at eval time** — the two-tower adapter's usefulness comes from what it's asked to encode at inference, not from training specialization. Writeup: `docs/twotower-no-query-experiment.md` + `docs/twotower-query-only-experiment.md`, bug detail: `docs/possible-bugs.md` #5. **Addendum (2026-08-04):** does the project's standard 1024-dim truncation cost accuracy vs. each model's native width? Same four arms re-run at nano's native 2048 dims (`--truncate-dim 2048`, no code change — already a CLI flag) and large's max Matryoshka width (`voyage_large_query_weighted/run_native.py`, new sibling entrypoint, `run.py` untouched, `output_dimension=2048`). No — every arm on both models is flat or slightly *worse* at native width; large's query-only arm (this project's best result) drops from R@1 0.42/MRR 0.5897 at 1024-dim to 0.36/0.5545 at 2048. Consistent with Matryoshka training making the leading 1024 dims a complete embedding on their own, not a lossy crop — 1024 was already the stronger choice everywhere else in this project, not a compromise. |
| [`twotower-split-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-split-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-04 | **Does a genuinely split two-tower (two separate LoRA adapters, one per role, instead of one shared model reading different text) beat the single-model approach? No — it's the worst of the three query-only variants tested.** New isolated package `twotower_split/`, custom training loop (SentenceTransformerTrainer can't route different columns through different models) reusing `twotower.train`'s generic helpers read-only; same rows as `twotower_query_only/`. Verified both towers get independent nonzero gradients before any GPU spend. All-200: AUC 0.5677, hard-neg AUC **0.4832 (below chance)**, MRR 0.4844, R@1 0.28 — worse than both `top1_ctrl`'s eval-time query-only swap (R@1 0.32) and `query_only_001`'s single-model training (R@1 0.29) on every metric, sharply so on hard-neg AUC. Dev recall@1 also peaked at epoch 1 and declined every epoch after, faster overfitting than any single-model run. Likely cause: the frozen base model's query/document space is already well-aligned from Voyage's own large-scale training; splitting into two independently-initialized adapters throws that away and asks the model to relearn compatibility from only 583 rows — not enough data. Holdout again misled (AUC 0.6267, R@1 0.4483 — the best-looking holdout number of any two-tower model, reversed on all 200). **Caveat added 2026-08-04** (`docs/possible-bugs.md` #6): this run's custom training loop trained on nano's native untruncated 2048-dim embedding, not the 1024-dim one every other number in this project uses — not retroactively rerun, recorded as a methodological gap. |
| [`twotower-field-gate-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-field-gate-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-04 | **Does a learned, per-seeker gate over decomposed profile pieces beat a fixed alpha blend or plain query-only? No — it's the worst of the four on every retrieval metric.** New isolated package `twotower_field_gate/`, one shared tower (unlike `twotower_split/`), seeker side split into `query_only` + `lookingFor` + `positioning`, each encoded separately and combined by a small learned gate (`FieldGate`, 9,219 params: linear layer -> softmax -> weighted sum) instead of one fixed alpha. Found and fixed the truncation bug (`docs/possible-bugs.md` #6) *before* any GPU spend this time — verified numerically that a raw forward pass returns nano's untruncated 2048-dim embedding, not the 1024-dim one everything else uses, and truncates explicitly to match. All-200: AUC 0.5919, hard-neg AUC 0.5520, MRR 0.4204, **R@1 0.23**, R@10 0.78 — beats `split_001` on classification (AUC/hard-neg) but is the **worst of all four approaches on MRR/R@1/R@10**, including the split-towers result. Dev recall@1 peaked at epoch 1 (0.3333, the highest epoch-1 peak of any two-tower run) then declined, same overfitting shape as every other custom-loop experiment. Holdout misled again (R@1 0.5172, best-looking holdout of any two-tower model, reversed to worst on all 200). **Conclusion across three architectural attempts (split towers, learned field gate) vs. the single-representation baseline: added architecture has not beaten simply choosing what text to feed one normally-trained shared tower — that lever (`docs/query-weighted-encoding-experiment.md`) remains the strongest and cheapest found in this project.** |
| [`twotower-kl-reg-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-kl-reg-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-04 | **Does a KL penalty against the frozen base model — the standard RLHF-style fix for a model drifting from a reference policy — beat `top1_ctrl` outright, and does it stop the early-epoch overfit/decay seen in `split_001`/`field_gate_001`? Mixed: the mechanism works, but it's a net negative on the recipe it was actually tested on.** New isolated package `twotower_kl_reg/`, `top1_ctrl`'s exact recipe (same rows, LoRA shape, batch/lr schedule, `MultipleNegativesRankingLoss(scale=20)`, recall@1 checkpoint selection) plus one new loss term (`losses.py::KLRegularizedMNRL`) penalizing the LoRA-adapted model's in-batch similarity distribution for diverging from the frozen base model's same distribution — obtained by toggling the same PEFT model's adapter off/on (`disable_adapters()`), not a second model. Verified locally before GPU spend: adapted==frozen exactly at LoRA init (max diff 0.0, since LoRA's B matrix starts at zero), nonzero gradients after backward. Dev recall@1 **rose monotonically and plateaued** (0.267→0.283→0.333→0.333→0.333) instead of declining — but a same-day check of `top1_ctrl`'s own curve (no KL) showed it was *already* flat (0.30→0.333→0.333→0.317→0.333), so the collapse this mechanism targets wasn't actually present in this recipe; it belongs to the more complex custom loops (two independent towers, an extra gate module) this experiment didn't touch. All-200: AUC 0.5504, hard-neg AUC 0.5148, MRR 0.3376, R@1 0.18, R@10 0.67 — every metric slightly below plain `top1_ctrl` (AUC 0.5683, hard-neg 0.5484, MRR 0.3550, R@1 0.19, R@10 0.69). Holdout misled again (R@1 0.3793, MRR 0.5242 — best-looking holdout two-tower number in the project, reversed to a small loss on all 200). `top1_ctrl` + eval-time query-weighting remains the strongest, cheapest lever. Writeup: `docs/twotower-kl-reg-experiment.md` |
| [`nomad-drift-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/nomad-drift-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-03 | **Calibrates `query_weighted/`'s alpha (query/profile blend weight) on the 1000-profile synthetic batch `rrf_003` instead of eyeballing it off the same 200 real pairs it's reported on.** New isolated package `nomad_drift/`, reusing `query_weighted.eval.run_all_arms`/`combine` and `query_weighted.text` unmodified. **Phase 1** (2,619 judge-labeled synthetic pairs, corpus-free pair AUC): alpha rises monotonically with no interior peak, calibrated alpha = **1.0** (pure query, pairAUC 0.5471→0.6241). **Phase 2** (all 200 real pairs, same `run_all_arms` that produced the published table): the calibrated alpha does *not* win on the metric it was calibrated on — overall pair AUC peaks in the interior at alpha=0.6 (0.5872) then falls to 0.5530 at alpha=1.0 — but **hard-negative AUC keeps climbing the whole way to alpha=1.0 (0.5914), matching `query_weighted`'s own published best hard-neg AUC of any model on all 200 real pairs**, reproduced from a population 13x larger and disjoint from the real pairs. Also wins MRR/R@1/R@10 outright (0.5019/0.30/0.91). Worked example: seeker Rudraksh's query for a manufacturing-industry buyer network — `concat_baseline`'s #1 result is literally his own profile echoed from elsewhere in the corpus (cosine 0.9548) while the true accepted match sits at rank 108; the calibrated encoding drops the self-echo to #4 and pulls the true match to #3. Writeup: `docs/nomad-drift-experiment.md` |
| [`nomad-drift-sectioned-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/nomad-drift-sectioned-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-03 | **Follow-up to `nomad_drift/`: instead of blending the query with the whole profile vector, blend it with just the seeker's `lookingFor` section closest to the query.** New isolated package `nomad_drift_sectioned/`, reusing `nomad_drift.calibrate.combine_batch` and `query_weighted.eval.encode_everything`/`score_arm` unmodified, section-splitter shared with `baselines/voyage_nano_sectioned` (`synth_pipeline.pairing_rrf.sections.seeker_vectors`). **Phase 1** (rrf_003 calibration): section-selection accuracy against ground truth is only 48.8% — barely better than chance — and the pair-AUC sweep is nearly identical to the whole-profile version, still picking alpha=1.0. **Phase 2** (all 200 real pairs, full alpha grid): despite that, section-selection beats whole-profile blending on **4 of 5 metrics' best value anywhere in the sweep** — pair AUC 0.5927 vs 0.5872, hard-neg AUC 0.5958 vs 0.5914, MRR 0.5159 vs 0.5019, R@1 0.3200 vs 0.3000 — and at alpha=0.4/0.7 specifically it wins *every* metric simultaneously, not a trade-off. **The catch: the calibration process never found this** — pair AUC on synthetic data is monotonic to the alpha=1.0 boundary in both mechanisms, and at that boundary whole-profile vs. section-selected is mathematically identical (multiplied by zero either way), so the improvement lives entirely in the interior the calibration curve can't discriminate. A genuine limitation of boundary-monotonic calibration, caught only by comparing the two full real-200 sweeps side by side. Writeup: `docs/nomad-drift-sectioned-experiment.md` |
| [`rrf_qwen_full_001_browser.html`](file:///Users/harsh/Artifacts/dorby-ai/artifacts/pairing_rrf_qwen_judge/rrf_qwen_full_001/_browser.html) | local (`artifacts/pairing_rrf_qwen_judge/`, gitignored — not published, 87MB) | 2026-08-04 | **First batch anywhere in this project judged with `qwen.qwen3-32b-v1:0` on Bedrock instead of `gemini-3.1-flash-lite` on OpenRouter, and the largest synthetic pairing batch to date.** New isolated package `synth_pipeline/pairing_rrf_qwen_judge/` (duplicated from `pairing_rrf/` per the isolation rule — swapping the judge model changes what a batch means), reusing `baselines/llm_judge/bedrock_backend.py`'s structured-output/fallback logic read-only. Pool: all 9,659 profiles from `bedrock_synth/run_20260804_023936` ($26.07, Gemma 3 27B), 4,153 seekers / 5,506 candidates, 12,091 `lookingFor`-section queries. Embedding (Qwen3-Embedding-8B, Modal A100) split across two separate Modal accounts running concurrently to roughly halve wall-clock time — a one-off operational script (`scripts/embed_two_accounts.py`), not a pipeline stage. Two real bugs found and fixed at this new scale: query generation was fully serial (~42/min, would have taken ~5h for 12,091 queries) — `query_gen.py` gained a `concurrency` param, threaded like the judge phase already was; and Chroma's `collection.add()` has an undocumented-until-hit max batch size (5,461) — `store.py` now chunks against `client.get_max_batch_size()`. **25,445 pairs judged, $15.34, 14,749 pos / 10,696 neg.** Qwen3-32B costs ~3x less per call than flash-lite but scores lower on the reference holdout (pair AUC 0.5802 vs 0.6358, hard-neg AUC 0.6224 vs 0.6466 — a much smaller gap than the headline number suggests). **Same base-rate leak as `rrf_002`, more pronounced at scale: seeker-identity alone predicts the label at 0.739 AUC, while within-seeker RRF AUC is only 0.546 (near chance)** — 24,453 (anchor, +, −) triplets available for within-seeker training, which cancels the base rate by construction; do not train a plain pairwise classifier on this batch. Labels are a model's opinion, not real accept/decline outcomes — **not promoted** into `data/dataset_*.json`, exported to `exports/rrf_datasets_qwen_judge/rrf_qwen_full_001/` (git-tracked path, but now gitignored directly since this export alone is too large). |
| [`twotower-queryonly-back-look-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-queryonly-back-look-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-06 | **New best two-tower fine-tune in the project — beats `top1_ctrl` on every metric and sets a new project-wide hard-negative AUC record.** A 105-way field/query sweep against the frozen `top1_ctrl` checkpoint (`baselines/twotower_top1_ctrl_field_sweep/`) found its best recall@1 combo scored frozen: seeker = search query only (zero profile fields), candidate = `background`+`lookingFor`. New isolated package `twotower_queryonly_back_look/`, `top1_ctrl`'s exact recipe otherwise (same 643-row `rrf_003` population, LoRA shape, batch/lr schedule, `MultipleNegativesRankingLoss(scale=20)`, recall@1 checkpoint selection), text built from two already-existing unmodified builders (`query_weighted.text.query_only`, `field_pairs_sweep.text.background_lookingfor`). All-200: pair AUC 0.5983 (top1_ctrl 0.5683), hard-neg AUC **0.6564** (0.5484), MRR 0.4791 (0.3550), recall@1 0.30 (0.19, +11 correct matches), recall@10 0.86 (0.69). **The hard/easy-neg AUC ordering inverted** (hard 0.6564 > easy 0.5700) — previously only the LLM judge showed this signature; 0.6564 also beats the judge's own previous project-record hard-neg AUC of 0.6466, at merged-LoRA serving cost instead of a per-candidate API call. **The holdout did not mislead this time** — first custom-loop run in the project where holdout and all-200 point the same direction. Not a clean sweep against every zero-training lever though: `top1_ctrl` + eval-time query-weighting still holds higher MRR/R@1 (query_only swap) and higher pair AUC (alpha_0.6 blend) at zero training cost. Writeup: `docs/twotower-queryonly-back-look-experiment.md` |
| [`twotower-field-pairs-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-field-pairs-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-06 | **Do two-field seeker+candidate fine-tunes beat `top1_ctrl`'s full-profile recipe? No, on any of the three field pairs — but all three win on hard-negative resistance, and the ranking flips once training and a trimmed candidate side enter the picture.** Three isolated packages (`twotower_field_pos_bg/`, `twotower_field_pos_look/`, `twotower_field_bg_look/`), each `top1_ctrl`'s exact recipe (LoRA rank 8, plain `MultipleNegativesRankingLoss(scale=20)`, recall@1 checkpoint selection, same 643-row `rrf_003` population) with **both** seeker and candidate text trimmed to one identity-field pair (no query, no other field) — the fields the field-isolation experiment found carry real person-identity signal alone. Motivated by a free eval-time sweep (`field_pairs_sweep/`, frozen nano, seeker side only) that ranked `pos_lookingfor` best of the three; once the candidate side is also trimmed and the model actually fine-tuned, **`background_lookingfor` wins instead, on every metric** — the frozen seeker-only ranking didn't predict the trained-both-sides ranking. All-200: `field_bg_look_001` AUC 0.5610 / MRR 0.2674 / R@1 0.12 / R@10 0.58 (closest of the three to `top1_ctrl`'s 0.5683/0.3550/0.19/0.69, still behind on every metric except hard-neg AUC); `field_pos_bg_001` AUC 0.5496 / MRR 0.2497 / R@1 0.13; `field_pos_look_001` AUC 0.5467 / MRR 0.2407 / R@1 0.13. **All three beat `top1_ctrl` on hard-neg AUC** (0.572-0.579 vs 0.548), traded against a large easy-neg AUC drop (0.593-0.602 vs 0.684) — trimming to two fields sharpens resistance to plausible-but-wrong candidates at the cost of general ranking. Training helped every arm recover ground over its own frozen counterpart, just not enough to close the gap. Holdout misled again on all three (`field_bg_look_001` holdout AUC 0.6224 vs its own all-200 0.5610). `top1_ctrl` + eval-time query-weighting remains the strongest, cheapest lever in this project. Writeup: `docs/twotower-field-pairs-experiment.md` |
| [Judge Prompt Evolution — evo_001 → evo_009 + focused](https://claude.ai/code/artifact/1e089702-6f90-4676-98e4-6c7e69813119) | published artifact only (no local file) | 2026-08-03, updated 2026-08-08 | **Ten variants of automatic LLM-judge prompt optimization — the sixth closes almost the entire gap to naive; the seventh shows more examples + all-200 sampling doesn't help and introduces a new failure mode; the eighth isolates that failure mode with a code-level fix and recovers most of the gap; the ninth reruns the eighth's recipe from a stronger seed and does worse.** `judge_prompt_evolution/` (own isolated package): an optimizer LLM revises the judge's system prompt each round against a fresh batch of real labeled example pairs, no accuracy feedback inside the loop. **evo_001** (Sonnet → Deepseek, train-split examples) grew 1,011→24,905 chars via 32 hand-accreted, example-specific rules — pair AUC **0.5734** vs. naive's 0.6177. A rewritten meta-prompt (v2: "revise the rubric, generalize don't copy specifics") produced **evo_002** (naive seed, 0.5918) and **evo_003** (seeded from `structured_cot` instead, converged to the **identical 0.5918**, evidence the result is a process attractor, not a seed property). **evo_004** (v2 + forced *aggressive* distillation every 5 rounds) dropped the JSON contract twice at its most-compressed pass and scored **0.5700**, worst of the clean runs. **evo_005** (same as evo_004 but examples sampled from all 200 real pairs, deliberately breaking the holdout split) scored the highest nominal AUC of any run, 0.6016 — a leakage artifact, excluded from ranking (flagged via a `run.py`-recorded `leakage_warning`, surfaced by `eval_evolved.py`); a sub-experiment scoring its **unsummarized** round-20 prompt needed a Bedrock/MiniMax detour (fixing a real content-block-ordering bug in reasoning-model responses) before landing on a clean Gemini-API number, 0.5790 — below its own summarized final, meaning that summarize step helped even in a contaminated run. **evo_006** tested whether the fix was the *idea* of periodic summarization or its *aggressive wording*: same setup as evo_004 but with a gentler summarizer explicitly told brevity isn't the goal, confirmed via LangSmith push before running — cut only 2-35% per pass (vs. evo_004's 55-74%), never dropped the contract, and scored **pair AUC 0.6105 — the closest any evolution run has gotten, beating even hand-designed `structured_cot` (0.6100)**. Clean ranking: naive (0.6177) > evo_006 (0.6105) > structured_cot (0.6100) > evo_002 ≈ evo_003 (0.5918) > evo_001 (0.5734) > evo_004 (0.5700). Also added a direct-Gemini-API eval backend (OpenRouter credits ran out mid-project) and pushed every fixed prompt (meta-optimizer, both summarizer variants) to LangSmith Hub with pull-first loading at call time. **evo_007** then tested scaling up evo_006's recipe: 6 examples/round (up from 4) sampled from all 200 real pairs (contaminated, same tradeoff as evo_005, accepted deliberately), gentle summarizer every 5 rounds. OpenRouter ran out of credits again at round 6 (same reserved-max_tokens issue as the LLM-judge experiment's early cost incident) — resumed on a new direct-Gemini-API optimizer backend (`gemini-3.1-flash-lite`, `optimizer_backend="gemini"` added to `optimizer.py`/`config.py`). That backend then silently dropped the required JSON output contract starting round 10 (13/20 iterations flagged) — patched by hand (canonical `RESPONSE_CONTRACT` appended verbatim, original preserved as `final_prompt_raw_broken`), pushed to Hub as `evo_007--final-patched`, then scored **0.5739** — worse than naive and worse than every clean run except evo_004, despite more examples and a larger sampling pool. Not comparable to the clean ranking, but directionally clear: neither change helped, and switching optimizers introduced a contract-dropping failure mode Deepseek never showed in six prior runs. **evo_008** then isolated the optimizer-model question from the mid-run resume: gemini-3.1-flash-lite as optimizer from round 1 (not resumed partway), same recipe otherwise (6 ex/round, now 2/2/2 balanced, all-200 sampling, gentle summarizer/5). Also added `optimizer.py::repair_contract()` — a code-level fix that automatically appends the canonical `RESPONSE_CONTRACT` block back whenever `validate_contract()` flags a problem, every round, instead of relying on wording or a one-off hand patch. Only 1/20 rounds needed a repair (vs. evo_007's 13/20), self-healed instantly, and **scored 0.6037** — a large recovery over evo_007's 0.5739, using nearly the same recipe. Still contaminated/not comparable, but shows most of evo_007's damage was the compounding unrepaired contract loss, not Gemini as an optimizer per se. **evo_009** reran evo_008's exact recipe seeded from `structured_cot` instead of naive — ran perfectly clean (zero contract repairs) but scored only **0.5814**, well below evo_008's 0.6037 despite structured_cot's own un-evolved score (0.6100) sitting close to naive. Same pattern as evo_002/evo_003 on the earlier recipe: a stronger starting prompt doesn't carry through the loop into a better result. **evo_focused_001** (2026-08-08) is the tenth run and the first on a different seed family: the **focused** judge prompt (given the `searchQuery`, shown only seeker `positioning`/`lookingFor` + candidate `positioning`/`background`/`lookingFor`, 0.6451 published pair AUC), in its own isolated package `judge_prompt_evolution_focused/`. The substantive change beyond the seed is that the **optimizer is shown the same trimmed field set plus the query** — previously it saw complete profiles, which would have had it writing rules about fields the judge never sees. Recipe otherwise matches evo_006 (4 examples/round at 2/1/1, gentle summarizer every 5 rounds), with gemini-3.1-flash-lite as optimizer from round 1 via the direct Google API and examples drawn from all 200 real pairs by choice (this prompt's downstream job is labeling new synthetic pairs, not scoring these 200). Ran cleanly — 1,388→1,772 chars, one auto-repaired contract drop at round 9 — and **scored 0.5885 against a same-code-path seed control of 0.6474 (published: 0.6451), a −0.0589 loss: ten runs, ten losses, and the widest gap yet**. The informative part is *where* it lost: the yes/no decision was untouched (0.5900 vs 0.5950), but the confidence score went dead — worth **+0.0524 AUC** on top of the raw yes/no in the seed and **−0.0015** after evolution, because the evolved prompt compressed the output contract into one sentence and dropped the definition of what `confidence` means. `validate_contract` passed it (all three keys present), so this is a new failure mode: a contract that degrades while staying syntactically valid. Writeups: `docs/judge-prompt-evolution-experiment.md`, `docs/judge-prompt-evolution-focused-experiment.md` |
| [`reciprocal-static-findings.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/reciprocal-static-findings.html) | published artifact + local (`docs/html/`) | 2026-08-06 | **Does scoring the candidate's own stated preference against the seeker's background — the static half of Ga Wu's Fast-Weight-Programmer reciprocal-matching proposal — improve accept/decline prediction? Directionally yes, on zero training, but not statistically proven.** New isolated package `baselines/reciprocal_static/`. The paper's actual contribution (a session-history-adapted preference memory) needs per-user interaction logs with timestamps this dataset doesn't have, so only the paper's own cold-start reduction (`q_t = k_u`, i.e. no memory) plus its untested reciprocal term were built: `S = s_forward + λ·s_reciprocal` where `s_forward = k_u·v_i` (seeker's look-for vs. candidate's background) and `s_reciprocal = k_i·v_u` (candidate's own look-for vs. seeker's background), both embeddings from the same frozen `voyage-4-nano`, `λ=1.75` fit by grid search on the 131 real train pairs only. Pair AUC rose on every population checked: train 0.5512→0.5944, holdout (69 pairs) 0.5853→**0.6241**, all-200 0.5638→**0.5964** — the combined score sits at or above every frozen baseline and fine-tuned `twotower` arm previously measured except two Qwen3-Embedding fine-tunes, for zero training. **Caveat that keeps this a lead, not a result:** a 5,000-sample bootstrap of the AUC delta gives 95% CIs that cross zero on both holdout (`[-0.0655, 0.1474]`) and all-200 (`[-0.0333, 0.0994]`), `P(Δ>0)` = 0.76 / 0.84. Retrieval (MRR) did not improve — expected and by design, since retrieval ranks by `s_forward` alone (paper-faithful: reciprocal is reranking-only, never retrieval), so the gain is a cheap rerank effect, not a retrieval one. Ran on Modal (A10G), not local. Writeup: `docs/reciprocal-static-experiment.md` |
| [`reciprocal-static-rrf003-experiment.md`](file:///Users/harsh/Artifacts/dorby-ai/docs/reciprocal-static-rrf003-experiment.md) | local (no HTML — null result) | 2026-08-10 | **Follow-up to the above: calibrate λ on rrf_003's 2,619 judge-labeled synthetic pairs instead of the 131 real train pairs, and narrow the background view to `positioning`+`background` only — does it hold up?** New isolated package `baselines/reciprocal_static_rrf003/`. No: the grid search on rrf_003 lands at **λ=0.05** (vs. the original's λ=1.75), so combined ≈ forward-only by construction, and applying that frozen λ to the real 200 gives pair AUC 0.5578→0.5573 (a small loss, not a gain). Forward-only itself is also lower than the original run on the identical real-200 population (0.5578 vs. 0.5638), most likely from the narrower background field set — but this run changed the fitting population *and* the field set at once, so it can't isolate which change did what. Reads as consistent with the original finding's own bootstrap caveat (its 95% CI already crossed zero) rather than a real reversal. Writeup: `docs/reciprocal-static-rrf003-experiment.md` |
| [`reciprocal-lambda-grid.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/reciprocal-lambda-grid.html) | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/reciprocal-lambda-grid.html) | 2026-08-10, updated 2026-08-10 | **Instead of fitting λ, just sweep it from -2 to +2 and plot pair AUC directly (`baselines/reciprocal_lambda_grid/`, new isolated package, `bg_text` narrowed like the rrf003 run above) — does the curve back up either earlier claim?** Both curves (holdout, all-200) are smooth, single-peaked, and clearly positive-sided: holdout peaks at λ=0.25 (AUC 0.5966→**0.6181**), all-200 peaks at λ=0.55 (AUC 0.5578→**0.5663**) — the reciprocal term visibly carries signal in the same direction as the original λ=1.75 finding, and the rrf003 run's λ=0.05 undershot this curve's own peak on the identical field set. **But every point on this curve is read directly off the same labels being scored, so "best λ" here is optimistic by construction (train-accuracy-shaped, not test-accuracy-shaped)** — it doesn't validate a deployable number, it just shows the earlier null result was a bad fit rather than evidence the reciprocal term is worthless. Field set and fitting population are still confounded across all three reciprocal experiments; next step is the original train→holdout recipe re-run with the narrowed field set to isolate that. Writeup: `docs/reciprocal-lambda-grid-experiment.md`. **Update (same page, second chart added):** the same no-fit sweep re-run on the fine-tuned `top1_ctrl_001` LoRA checkpoint (`baselines/reciprocal_lambda_grid_top1ctrl/`, new isolated package) instead of frozen voyage-4-nano. Forward-only alone is already far stronger (holdout 0.6655, all-200 0.5890) than any combined frozen-model score, and **the holdout curve's peak flips to negative λ** (−0.25 → 0.6853) — the first case in this project where the "expected" positive-λ direction hurts. All-200 keeps the same positive sign as frozen (λ=0.50 → 0.5961). Caveat: `top1_ctrl` was trained on full-profile text, never this narrower look/bg split, so this measures an out-of-distribution scoring, not a clean fine-tuning test. Writeup: `docs/reciprocal-lambda-grid-top1ctrl-experiment.md`. **Update (same page, third chart added):** re-run again on `voyage_gemini_ctrl_001` — `top1_ctrl`'s exact recipe retrained on a bigger, newer, but measurably leakier synthetic batch (`baselines/reciprocal_lambda_grid_voyage_gemini_ctrl/`, new isolated package). **Forward-only alone hits 0.7164 holdout AUC — the best number of any kind recorded anywhere in this project so far** — and unlike `top1_ctrl`, both populations keep the same positive λ-sign as frozen (holdout peak λ=0.35 → 0.7345; all-200 rises to a flat plateau ≈0.656–0.659 from λ≈1.0 onward, not a real peak despite the reported best point at λ=1.90). Same out-of-distribution caveat as `top1_ctrl`, plus its training batch's own leakage checks flagged it leakier than `top1_ctrl`'s (candidate-only AUC 0.758 vs. 0.634) — a live possibility worth flagging before treating the gain as settled. Writeup: `docs/reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md`. **Update (same page, fourth chart added, 2026-08-11):** same no-fit sweep on `ask_offer_001` — the first fine-tune with two *independent* LoRA towers (Ask on `lookingFor`+`searchQuery`, Offer on `positioning`+`background`) jointly trained on the combined score at a fixed λ=1.75, instead of one shared encoder (`reciprocal_lambda_grid_ask_offer/`, new isolated package, reusing `twotower_ask_offer`'s model/text code read-only). Forward-only alone is weak (holdout 0.6138, all-200 0.5126 — below frozen voyage-4-nano's own 0.5638) — two narrow, single-field towers didn't out-encode the frozen zero-training split. Holdout curve has a small genuine peak at λ=0.15 (→0.6293); all-200 rises to the grid boundary at λ=1.90 (→0.5723) with no real interior peak, same "shelf not a peak" pattern as `voyage_gemini_ctrl`'s all-200 curve. Net: two dedicated jointly-trained towers still don't beat one shared tower (`voyage_gemini_ctrl`, 0.7345 holdout / 0.6587 all-200 combined) plus a free post-hoc sweep. Writeup: `docs/twotower-ask-offer-experiment.md` |
| [`bilinear-mf-results.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/bilinear-mf-results.html) | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bilinear-mf-results.html) | 2026-08-06 | **Does matrix factorization help? Factoring the *text* (LSA) robustly lifts hard-negative discrimination on both backbones; factoring the *scoring function* (low-rank bilinear on frozen embeddings) does not work at 131 real training pairs.** `bilinear_mf/` (own isolated package, nothing under `baselines/`/`eval_real_full/`/`twotower/` touched). Classic recsys `users x items` MF doesn't transfer here — 129 seekers x 178 candidates with only 200 filled cells, and free per-contact vectors generalize to nobody — so two variants that read text were tested instead. **Arm 1 `lsa`** — truncated SVD of the `documents x terms` matrix, then cosine in the compressed space; label-free, so nothing can leak and there is no model to overfit. **Compression improves hard-negative AUC on both a lexical and a neural backbone at every rank tested** — Voyage-4-large 0.5422 -> 0.579-0.590 (k=32/64/128), TF-IDF 0.5164 -> 0.616-0.647 (k=16-128). A plateau, not a lucky `k`, and hard negatives are the only negative population that exists in production. Voyage at k=128 beats production Voyage-4-large on nearly everything at once (AUC 0.5726->0.5978, hard-neg 0.5422->0.5902, MRR 0.3102->0.3118, R@1 0.13->0.14) and its 0.5978 would be **the best all-200 pair AUC in this project** (prev. twotower Qwen micro-6, 0.5947) for zero training and zero serving cost — **but k=128 was picked by reading the all-200 column, so it is a hypothesis, not a result**; under train-split selection Voyage picks k=32 and the honest gain shrinks to +0.009 AUC / +0.037 hard-neg / -0.037 MRR. **Arm 2 `bilinear`** — `score(s,c) = s.c + (As).(Bc)`, i.e. `s^T(I + A^T B)c`, the content-based form of MF (per-contact vectors computed from text, not looked up), motivated by cosine being stuck at `W=I` and unable to express asymmetric complementarity. Selected by inner seeker-disjoint CV on train pairs only, scored by seeker-disjoint 10-fold CV over all 200 against a 50-draw label-permutation null. **It fails three separate ways**: on the production backbone it loses 0.032 all-200 AUC and drops the holdout below chance (0.4845 vs cosine 0.5802) and doesn't clear its own null (p=0.196); where it appears to win (TF-IDF +0.044) the gain is entirely easy-negative (0.5604->0.5984) while hard-neg goes backwards (0.6294->0.6100); and retrieval degrades on both (Voyage MRR 0.323->0.098) because the residual is fit on 200 observed pairs but applied to a 178-candidate ranking it never constrains. **The instructive number: inner CV said 0.7399 on TF-IDF, honest out-of-fold was 0.6193 (Voyage 0.6218 -> 0.5410)** — selection reports the max over 64 configs on 131 pairs, so at this data size hyperparameter selection is itself enough overfitting to invent a result; with SVD width fixed at 128 the head regularized to numerically zero (residual norm 7e-08, the model correctly chose 'keep cosine') and only widening the grid produced a nonzero head at all. Third experiment to hit this wall after `moe_reranker` and `twotower_rrf_triplet_ablation`. Anchors: with the head disabled the package reproduces the published frozen-cosine rows digit-for-digit (TF-IDF 0.5649/0.1313, Voyage 0.5726/0.3102), asserted in `tests/test_bilinear_mf.py` (9 passing) — which is how all three of its measurement bugs were caught (a TF-IDF fit-set that shifted every IDF weight, a stale content-hashed cache serving the pre-fix vectors, and retrieval ranked in the un-reduced space). Voyage runs cost $0 (578/578 cache hits). Writeup: `docs/bilinear-mf-experiment.md` |
| [`twotower-voyage-gemini-ctrl-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-voyage-gemini-ctrl-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-11 | `top1_ctrl`'s exact recipe retrained on a newer, much larger synthetic batch (`pairing_voyage_gemini/smoke_test_002`, measurably leakier than `rrf_002` on pre-training checks) — beats `top1_ctrl` on every all-200 metric (pair AUC 0.6081, hard-neg AUC 0.6264) but doesn't beat the project's current record (`queryonly_back_look_001`). Writeup: `docs/twotower-voyage-gemini-ctrl-experiment.md` |
| [`twotower-voyage-gemini-kl-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-voyage-gemini-kl-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-11 | Retested `kl_reg_ctrl_001`'s KL-divergence-against-frozen-base penalty on `voyage_gemini_ctrl_001`'s bigger, leakier batch — loses on every all-200 metric again (pair AUC 0.5479 vs 0.6081, hard-neg AUC 0.5230 vs 0.6264), a wider margin than the small-batch retest, and unusually the 69-pair holdout agreed with the all-200 verdict this time instead of overstating it. Writeup: `docs/twotower-voyage-gemini-kl-experiment.md` |
| [`twotower-qwen-voyage-gemini-experiment.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-qwen-voyage-gemini-experiment.html) | published artifact + local (`docs/html/`) | 2026-08-11 | Fine-tuned Qwen3-Embedding-8B (`twotower_qwen_bigbatch`'s winning micro-6 recipe) on `voyage_gemini_ctrl_001`'s same batch — **new project-wide records on all-200 pair AUC (0.6446, ties the LLM judge's own best, 0.6451) and hard-neg AUC (0.6862, beats the prior field-selection-swept record, 0.6732)**, without any field-selection trick; nano still leads every retrieval metric on this batch. Same leakage caveat as `voyage_gemini_ctrl_001` applies. Writeup: `docs/twotower-qwen-voyage-gemini-experiment.md` |
| [`twotower-ask-offer-findings.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-ask-offer-findings.html) | published artifact + local (`docs/html/`) | 2026-08-11 | **First actual training run of the reciprocal two-tower design (`twotower_ask_offer/`, new isolated package): two independently-trained LoRA towers — Ask (lookingFor) and Offer (positioning+background) — jointly optimized on the combined score S = s_fwd + λ·s_rev via a hand-rolled in-batch-negative loss, on the exact same 3008 rows `voyage_gemini_ctrl_001` trained on.** Mixed, genuinely different result from the zero-training version: on all-200, the reciprocal term adds real signal (pair AUC 0.5126→**0.5714**, +0.059), but on the 69-pair holdout it does nothing (0.6138→0.6121). The bigger finding: the trained forward-only tower itself (0.5126 all-200) is *weaker* than frozen Voyage-4-nano's own forward score (0.5638, `reciprocal_static`) — narrowing each tower to a single field likely cost more than untying the weights gained, consistent with this project's field-pair findings. Does not beat `voyage_gemini_ctrl`'s combined score (0.6587 all-200) on either population. Full writeup: `docs/twotower-ask-offer-experiment.md`. Design plan this followed: [`reciprocal-two-tower-training-plan.html`](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/reciprocal-two-tower-training-plan.html) (now updated to link back and marked built). |
| [`bdata-voyage-nano-experiment.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/bdata-voyage-nano-experiment.html) | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-voyage-nano-experiment.html) | 2026-08-12 | **Frozen Voyage-4-nano on B-data holdout:** pair AUC **0.4691** (below chance); hard-neg AUC **0.3626**; REJECT cosine slightly above ACCEPT. Same model was 0.56–0.58 on the old 200-pair seed set; loses even to TF-IDF (0.512) on the identical matched holdout. Package: `bdata_voyage_nano/`. Writeup: `docs/bdata-voyage-nano-experiment.md`. |
| [`bdata-unique-contacts.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/bdata-unique-contacts.html) | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-unique-contacts.html) | 2026-08-12 | **Unique-person catalog from locked B-data:** 29,923 contacts keyed on `positioning` hash (fallback background/lookingFor). 10,333 appear as both seeker and candidate; 10,835 are candidate-only with no id. JSON at `data/unique_contacts_B_data.json`. Writeup: `docs/bdata-unique-contacts.md`. |
| [`bdata-tfidf-experiment.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/bdata-tfidf-experiment.html) | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-tfidf-experiment.html) | 2026-08-12 | **First isolated experiment on locked `data/B-data.json`:** unsupervised TF-IDF cosine (`bdata_tfidf/`) predicting `ACCEPT` vs `REJECT` (PENDING dropped). Seeker-disjoint 70/30 freeze (`bdata_tfidf/split.json`). Holdout pair AUC **0.5121** (near chance); hard-neg AUC **0.4026** (below chance); within-seeker mean AUC **0.4970**. Lexical overlap does not separate Boardy accept/decline on this production dump. Also summarizes B-data shape (mostly PENDING, yes-skew when resolved, no candidate contactIds). Writeup: `docs/bdata-tfidf-experiment.md`. |

Local file links above are absolute `file://` paths pinned to this repo's
checkout location on this machine (`/Users/harsh/Artifacts/dorby-ai/docs/html/`)
so they open directly in a browser on click, no server needed — they will
not resolve on another machine or from a different clone path; use
`open docs/html/<file>.html` there instead. Published links work from any
device but are private unless shared from the page's share menu.

---

Every `docs/html/pairs-comparison-graph*.html` file is a self-contained,
two-pane dual graph built by `scripts/build_real_pairs_graph.py`: left
pane is the real dataset (200 genuinely-real pairs / 297 contacts —
`real_only()` excludes the 460 `cmsynth*` promoted-synthetic pairs also
present in `data/dataset_positive.json`/`dataset_negative.json`), right
pane is one `synth_pipeline.pairing` batch from `artifacts/pairing/`.
Details on each pairing batch (profile source, config, findings) are in
`docs/profile-generation-local-and-bedrock.md`.

## Local files (this repo, `docs/`)

| file | built | synth batch | profile run | synth nodes/edges | git status |
|---|---|---|---|---|---|
| `pairs-comparison-graph.html` | 2026-07-23 | `pair_test_001` | `run_20260723_212205` (20 profiles) | 20 / 104 | **not tracked** — `.gitignore` excludes this exact filename by name (line 42); every later variant uses a different filename and isn't caught by that rule |
| `pairs-comparison-graph-hub-test.html` | 2026-07-24 | `pair_hub_test_001` | `run_20260724_101605` (3 profiles) | 3 / 4 | committed (`0a9b8e1`) |
| `pairs-comparison-graph-no-refex.html` | 2026-07-24 | `pair_no_refex_001` | `run_20260724_111235` (5 profiles) | 5 / 12 | committed, latest revision `4b45005` (physics/polarize fix) |
| `pairs-comparison-graph-disjoint.html` | 2026-07-24 | `pair_disjoint_001` | `run_20260724_111235` (same 5 profiles, re-paired) | 4 / 3 | committed (`8a57a4f`) |
| `pairs-comparison-graph-named.html` | 2026-07-24 | `pair_named_001` | `run_20260724_123452` (10 profiles) | 8 / 6 | pending commit as of this writing |

Open any of these directly in a browser (`open docs/html/<file>.html`) — fully
self-contained, no server needed.

### What each one demonstrates

- **`pairs-comparison-graph.html`** — first dual-pane build, `pair_test_001`
  (20 profiles). Established the "topology is nothing like real data"
  finding: 5.2 vs 0.67 edges/node, 80% both-role vs 5%.
- **`pairs-comparison-graph-hub-test.html`** — 3-profile smoke test after
  routing generation prompts through LangSmith Hub (hub-only, no local
  fallback). Confirms the hub-routing change is behavior-neutral.
- **`pairs-comparison-graph-no-refex.html`** — 5-profile batch after
  dropping the 2 real-profile reference examples from `generate_profile`
  (redundant with `style_refresh`/`archetype_refresh`). Also the pane
  where the real-pairs-pane label-polarization physics force (positive
  edges pull left, negative pull right) was added and fixed — see
  `docs/profile-generation-local-and-bedrock.md` and the git log on
  `scripts/build_real_pairs_graph.py` for the "constant force too weak,
  switched to proportional target-x" story.
- **`pairs-comparison-graph-disjoint.html`** — same 5 profiles as above,
  re-paired with the new disjoint seeker/candidate split + per-seeker pair
  cap (`--seeker-frac 0.48 --max-pairs-per-seeker 1`). Density dropped
  2.4 → 0.6 edges/node on the identical profile pool, right in line with
  real data's 0.673 — direct empirical confirmation the structural fix
  works.
- **`pairs-comparison-graph-named.html`** — first batch generated with
  the programmatic random-name injection fix (`generate_profile` v3):
  10/10 unique names, 10/10 compliance, vs. the prior ~50-80% Anya/Aris
  collapse rate. Paired with the same disjoint-split + cap settings.

## Real-only single-pane similarity visualizations

Two more `build_real_pairs_graph.py` outputs, single-pane (real data only,
no synth comparison), both keeping the label-polarization force
(positive-labeled contacts drift left, negative drift right) and adding a
second physics force that pulls content-similar contacts closer together
and pushes dissimilar ones further apart — a signed spring over every pair
of nodes, not just edges. Force strength (`SIM_K=0.01`, `SIM_MAX_DIST=500`
in `scripts/build_real_pairs_graph.py`) was tuned against a headless port
of the exact physics loop before shipping (job tmp dir, not committed):
correlation(similarity, final on-screen distance) ≈ -0.39, top-50-most-similar
pairs end up ~2.8x closer than bottom-50-least-similar, while the left/right
label split still holds 99%+ correct with both forces active together.

| file | similarity source | text embedded | cache | git status |
|---|---|---|---|---|
| `real-pairs-tfidf-cluster.html` | `--similarity-mode tfidf` | whole-profile TF-IDF (`baselines.bert_frozen.text.candidate_to_text`, same serialization the pairing scorer fits on) | none needed — refit each run, cheap (sklearn, no API) | pending commit |
| `real-pairs-voyage-lookingfor-cluster.html` | `--similarity-mode voyage_large_lookingfor` | `lookingFor` field only, `voyage-4-large` | `artifacts/voyage_large_lookingfor/emb/*.npy`, content-hash keyed (via `VoyageLargeEncoder`'s own per-text disk cache) — confirmed 297/297 cache hits, 0 new API calls on a second run | pending commit |

The Voyage run cost 297 texts / 111,537 tokens (`artifacts/voyage_large_lookingfor/usage.json`), one-time — every later rebuild of this file or any other visualization using the same cache dir reuses the embeddings for free.

**Why these two barely show visible clustering:** checked directly by fitting
PCA/SVD and reading `explained_variance_ratio_` before building anything
further — TF-IDF's first 2 components explain only **1.67%** of variance
(3 components: 2.69%); voyage-4-large's `lookingFor` embedding is better but
still modest, **12.9%** (2 components) / **17.3%** (3 components). A force
layout with 3 competing forces (repulsion, edge springs, polarize,
similarity) will visually dilute a signal this size into "no obvious
clusters" even when the underlying correlation is real (-0.39, verified
above) — force-directed graphs are simply not a reliable tool for confirming
whether clustering structure exists. See the direct PCA/SVD scatter plots
below for the more trustworthy diagnostic.

### Direct PCA/SVD scatter (no physics) — the actual clustering diagnostic

`--layout pca`: nodes placed at the embedding's own first 2 components,
static, no simulation at all — the label-polarization force is also off here
(position is 100% determined by the embedding, not by pos/neg label), so
these are a real, unbiased look at whether the raw embedding space has
structure. Colored by pairing polarity instead (green = all-positive,
gray = mixed/no pairs, red = all-negative) so label correlation can still be
eyeballed against position.

| file | similarity source | components shown | cumulative variance explained |
|---|---|---|---|
| `real-pairs-tfidf-pca.html` | TF-IDF (`TruncatedSVD`) | PC1 vs PC2 | 1.67% |
| `real-pairs-voyage-lookingfor-pca.html` | voyage-4-large `lookingFor` (`PCA`) | PC1 vs PC2 | 12.9% |

**Honest read:** neither should be expected to show dramatic, obvious
clusters — the cumulative variance numbers above say the real structure
(whatever it is) lives mostly in dimensions beyond the first 2, especially
for TF-IDF. The Voyage version is the more informative of the two (7x more
variance captured) and is the one worth actually looking at for any real
signal; TF-IDF's is close to a null result. If clearer clustering is the
goal, the next lever is 3 components (mild further gain) or a nonlinear
reduction (UMAP/t-SNE) rather than more physics tuning — those are built to
preserve local neighborhood structure that linear PCA/SVD compresses away.

### Direct 3D PCA scatter (`scripts/build_real_pairs_3d_scatter.py`)

A separate, standalone script (not a `build_real_pairs_graph.py` mode — that
file's renderer is 2D SVG, a genuine 3D scatter needed its own hand-rolled
canvas perspective projector). `real-pairs-voyage-lookingfor-3d-pca.html`:
297 real contacts placed at the first 3 PCA components of their cached
voyage-4-large `lookingFor` embeddings (`artifacts/voyage_large_lookingfor/`,
same cache as the 2D version — this run was 297/297 cache hits, 0 new API
calls). Positions are entirely fixed, no simulation of any kind; mouse drag
only rotates the camera (yaw/pitch), scroll only zooms — neither moves a
node. The 200 seeker→candidate pairs are drawn as thin dotted directed
arrows (green = positive, red = negative) between their fixed points, nodes
colored by pairing polarity as in the 2D version. PC1/PC2/PC3 explain
7.9%/5.0%/4.4% of variance (17.3% cumulative, matching the 2D version's
PC1+PC2 of 12.9% plus PC3's further 4.4%); expect the same "no dramatic
clusters" honest read, now with one more (weak) axis to look along.

Full per-component breakdown (PCA fit to 10 components on the same 297
embeddings, for reference — decay is gradual with no elbow, meaning the
real structure is spread thin across many directions rather than
concentrated in the first few, which is exactly the bad case for any
linear projection):

| component | variance explained | cumulative |
|---|---|---|
| PC1 | 7.90% | 7.90% |
| PC2 | 4.96% | 12.86% |
| PC3 | 4.41% | 17.27% |
| PC4 | 3.12% | 20.39% |
| PC5 | 2.78% | 23.17% |
| PC6 | 2.75% | 25.92% |
| PC7 | 2.48% | 28.40% |
| PC8 | 2.33% | 30.73% |
| PC9 | 2.18% | 32.91% |
| PC10 | 2.04% | 34.95% |

### PCA / t-SNE / UMAP 3D comparison with a layout selector (`scripts/build_real_pairs_3d_manifold.py`)

The gradual-decay table above means more PCA components won't fix the
"no obvious clusters" problem — it's the wrong tool, since it can only ever
show whichever axes happen to carry the most raw variance, not axes chosen
to make a good picture. `real-pairs-voyage-lookingfor-3d-manifold.html`
reuses the 3D canvas scatter from the PCA-only build above (same fixed
positions, no simulation, same dotted directed pos/neg arrows) but computes
**three** layouts up front from the same cached embeddings — PCA (linear,
baseline for comparison), t-SNE (`sklearn.manifold.TSNE`, perplexity 30),
and UMAP (`umap-learn`, `n_neighbors=15`, `min_dist=0.1`) — and embeds all
three in one file with a button selector top-left, so switching is instant
(positions swap, camera angle stays put). New dependency: `umap-learn`,
added to `requirements.txt`. Still 297/297 cache hits, 0 new API calls.

**Caveat carried in the UI itself** (shown per-layout under the selector):
t-SNE/UMAP positions are nonlinear — they optimize for "things that were
near in the real 1024-dim space stay near here," which is exactly what you
want for an honest "is there visual clustering" read, but unlike PCA there
is no axis meaning and no variance-explained number to quote; only relative
neighborhoods in the picture are trustworthy, not absolute distances or
directions.

## Field isolation experiment (`holdout-field-isolation-embedding-space-3d.html`)

Sibling to the `holdout-embedding-space-3d.html` experiment above, isolated
in its own package: `baselines/voyage_nano_field_isolation/`. That earlier
run swapped one `lookingFor` section into an otherwise-unchanged profile, so
every embedded row still carried the whole profile's context and the finding
was "splitting barely moves the point" (whole↔section cosine 0.89-0.90). This
run tests the opposite condition: what if a field carries **no** other
context at all?

For each of the 115 unique contacts in the frozen 69-pair real holdout:

- one **whole** embedding (unchanged, `profile_to_text(profile)`)
- one **field-alone** embedding per non-empty profile field — just
  `"positioning: ..."` alone, nothing else (up to 8 per contact)
- one **section-alone** embedding per `lookingFor` paragraph — just
  `"lookingFor: <that one paragraph>"` alone (only when `lookingFor` has more
  than one paragraph; single-paragraph contacts are already covered by the
  `lookingFor` field-alone row)

All 1,808 texts (115 whole + 755 field-alone + 938 section-alone) were
encoded in one `voyage-4-nano` pass on Modal (`modal_embed_space.py`, L4 GPU;
`batch_size=16` OOM'd on 24GB with this row mix, `batch_size=4` ran clean in
~1 min). Unlike the sibling experiment, embeddings are then pulled from the
Modal volume and loaded into a **local persistent Chroma collection**
(`scripts/load_field_isolation_to_chroma.py` →
`artifacts/voyage_nano_field_isolation/chroma/`, open-source, no server) —
the visualization script reads vectors back out of Chroma rather than the
raw `.npy`, so the DB is a real link in the pipeline, not just an extra copy.

**Finding: isolating a field moves it much further than swapping one did.**
Whole↔field-alone cosine averages **0.705** (vs. 0.89-0.90 in the sectioned
run) and whole↔section-alone averages **0.684** — both far looser, since
there's no shared profile text left to anchor the vector near its owner. A
person's own fields spread out to ~2.3x the average inter-person distance
(`constellationRatio`), vs. a tight little knot in the sectioned run.

**A second, new question this isolation makes possible:** do same-named
fields cluster by *topic* (e.g. all `notes` fields resemble each other)
regardless of whose they are, or does person identity still dominate even in
isolation? Per-field breakdown (`meanCosToWhole` vs.
`meanCosAcrossContacts`, both cosine on raw 1024-d vectors):

| field | n | vs. own whole profile | vs. same field, other people |
|---|---|---|---|
| positioning | 115 | 0.890 | 0.549 |
| background | 115 | 0.850 | 0.580 |
| lookingFor | 115 | 0.852 | 0.595 |
| notes | 114 | 0.700 | 0.563 |
| introPreferences | 100 | 0.731 | 0.680 |
| locationAvailability | 114 | 0.450 | 0.676 |
| meetingAndSchedulingPreferences | 15 | 0.391 | 0.751 |
| personalPreferences | 67 | 0.366 | 0.756 |

`positioning`/`background`/`lookingFor` stay clearly person-specific even
alone (own-profile cosine well above the across-people baseline of ~0.58).
`locationAvailability`, `personalPreferences`, and
`meetingAndSchedulingPreferences` invert that: they're *more* similar to
other people's same field than to their own owner's whole profile — those
fields read as boilerplate/scheduling logistics once isolated, carrying
almost no person-identifying signal on their own. That's a genuinely new
result the sectioned-swap experiment couldn't surface, since those fields
were never isolated there.

Rerun with:

```bash
modal run baselines/voyage_nano_field_isolation/modal_embed_space.py --batch-size 4
modal volume get dorby-sectioning-eval embed_space_fields_holdout/embeddings.npy \
    artifacts/voyage_nano_field_isolation/embeddings.npy --force
modal volume get dorby-sectioning-eval embed_space_fields_holdout/meta.json \
    artifacts/voyage_nano_field_isolation/meta.json --force
python scripts/load_field_isolation_to_chroma.py --reset
python scripts/build_field_isolation_embedding_space_3d.py
```

## LLM judge: does forcing multi-aspect CoT help? (`llm-judge-comparison.html`, `structured_cot`)

Run before scaling `rrf_002`'s synthetic profile pool to 500, to check whether
the judge that labels pairs could simply be made more accurate by asking it to
reason harder — score six fixed-weight aspects (location/availability,
ask-offer alignment, skill/domain evidence, seniority/stage fit,
domain/industry fit, practical constraints) with cited evidence, then
aggregate to a verdict, instead of `naive`'s direct yes/no. Same model
(`google/gemini-3.1-flash-lite`), same profiles, same missing `searchQuery` —
only the prompt changes. Full design and mechanism discussion in
`docs/llm-judge-experiment.md`'s `structured_cot` section;
`baselines/llm_judge/structured.py` recomputes the verdict from the six raw
scores using fixed canonical weights, not whatever weight the model echoes
back, so the model can't quietly reweight an aspect to swing its own answer.

**Finding: it didn't help — a small, uniform step backward.** Both variants
run back-to-back in the same session on the identical matched 69-pair holdout:

| | naive | structured_cot |
|---|---|---|
| Pair ROC-AUC | **0.6409** | 0.6336 |
| Decision accuracy | **0.6087** | 0.5507 |
| Hard-negative AUC | **0.6543** | 0.6267 |
| Says "yes" | 55.1% | 75.4% |

Naive wins on every metric. The mechanism shows up in the yes-rate jump:
averaging six independently-scored 0-5 aspects **regresses toward the
decision boundary** rather than sharpening judgment — a pair with one weak
aspect and five middling ones still lands close to 0.5, so `structured_cot`
says "yes" far more often (75.4% vs 55.1%) without being more often right.
Decomposition bought per-pair audit evidence (six cited justifications
instead of 2-4 sentences) at ~2.5× the output tokens, not accuracy.
**Decision: `naive` stays the labeling judge** for the next synthetic batch;
`structured_cot` is not adopted.

Rerun with:

```bash
python -m baselines.llm_judge.push_prompts --tag v1   # only needed after editing the prompt
python -m baselines.llm_judge.eval --data-dir data --variant naive --split holdout
python -m baselines.llm_judge.eval --data-dir data --variant structured_cot --split holdout
python3 scripts/build_llm_judge_browser.py
```

## Two-tower distillation: LLM-judge soft labels vs. hard 0/1 labels (2026-07-26)

No HTML for this one — a training-run comparison, not a visualization.
Logged here anyway since it's a small, complete experiment: same recipe as
`arm_a_real_only` (LoRA on voyage-4-nano, 5 epochs, 111 real-only train
pairs), with the training label swapped from the hard accept/decline 0/1 to
the naive LLM judge's continuous confidence-signed score. Full writeup,
mechanism, and caveats in
[`twotower-run-001-findings.md`](twotower-run-001-findings.md#distillation-experiment-llm-judge-soft-labels-instead-of-hard-01-2026-07-26).

**Finding: distilling the judge's soft score beat every twotower run to
date on real-holdout pair AUC** (0.604 vs. Arm A's 0.579, `run_001`'s
0.578), also ahead of TF-IDF's 0.592 — traded off against a drop in
retrieval MRR (0.359 vs. Arm A's 0.388). One caveat: checkpoint selection
fell back to the final epoch rather than a validated best (the best
train-dev checkpoint had been pruned by `save_total_limit`), so treat this
as a promising lead rather than a confirmed win.

## MoE re-ranker over `lookingFor` (`moe-reranker-review.html`, 2026-07-29)

**Isolated package: `moe_reranker/`.** Nothing in `baselines/` or `twotower/` was
modified. The three relevance-shaped aggregation modes are *reimplemented* in
`moe_reranker/aggregation.py` rather than imported, specifically so the veto-shaped
modes could be added without touching
`baselines/voyage_nano_sectioned/aggregate.py`, which the lookingFor-sectioning
experiment owns; `tests/test_moe_aggregation.py::test_matches_shared_baseline_on_shared_modes`
pins the duplicate against the original so it cannot drift. Shared baseline
helpers are imported read-only via `moe_reranker/section_scoring.py`.

Two experiments, both on the matched 69-pair real holdout:

1. **Which shape should expert opinions combine in?** Eleven aggregations over one
   cached embedding pass (`scripts/compare_section_aggregation.py`). Ordered
   most-veto to most-relevance, pair AUC climbs monotonically at every step —
   `min` 0.5836 → `softmin(τ=.20)` 0.5931 → `mean` 0.5940 → `softmax(τ=.05)`
   **0.5983** — and hard-negative AUC follows the identical ladder. No individual
   gap is significant (paired bootstrap softmax−min = +0.0146, 95% CI
   [−0.021, +0.051]); the ordered ladder across six configurations and two metrics
   is what carries it. **Verdict: temperature-sharpened relevance gate, not a
   soft-min veto.** Note τ=0.05 beats τ→0, so sharper is not automatically better.

2. **Does an MMoE earn its parameters here?** Built to the MMoE slides: 3 experts,
   shared bottom over 14 engineered features, two gates (real accept/decline +
   LLM-judge auxiliary), τ tunable, per-example sharpening *and* batch-average
   balancing entropy terms, expert dropout, three diagnostics. Seeker-disjoint
   5-fold CV over the 111 real train pairs (`scripts/moe_cv_compare.py`):

   | model | mean AUC | fold std |
   |---|---|---|
   | nano cosine, no model | 0.5282 | 0.138 |
   | logistic regression | 0.5251 | 0.144 |
   | MoE, single task | 0.5434 | 0.065 |
   | MMoE, multi-task | **0.5536** | 0.067 |

   Everything sits within one fold-to-fold standard deviation of everything else.
   Train AUC reaches 0.861 while train-dev AUC falls to 0.345. **The MMoE is not
   testable at this data size — the bottleneck is data, not architecture.** Two
   things did survive: multi-task beat single-task in 3 of 5 folds, and both MoE
   variants were half as volatile across folds (regularization buys stability, not
   accuracy). **The real 69-pair holdout was deliberately not spent** — CV says
   nothing is separable, so the one-shot check would burn the project's one clean
   measurement to confirm a null.

**Two measurement bugs found, both self-inflicted, both now pinned by tests.**
A first `noisy_or` rescaled by the min/max of the matrix it was passed — and since
positives and negatives are aggregated in *separate* calls, that made the transform
label-dependent and reported pair AUC **0.8500** against a ~0.60 field. Fixed with
a data-independent `p = (1+cos)/2` map. Separately, the routing-vs-seeker-identity
diagnostic first reported normalized MI 0.815, which reads alarming but is near the
floor: *random* routing scores 0.706 on this data (111 rows over 75 seekers), so the
real excess is +0.109. It now reports a permutation null, excess, and p-value.

**Data.** `moe_reranker/import_rrf.py` freezes a read-only copy of a `pairing_rrf`
batch into `artifacts/moe_reranker/data/<batch_id>/` with a SHA-256 of the source
and a `--verify` mode. `rrf_003` is imported: **2,619 judge-labeled pairs, 337 of
418 seekers carrying both classes → 2,773 within-seeker triplets** (the real pairs
have 19, from 6 seekers). Those labels are one LLM judge's opinion, not human
outcomes, so they are auxiliary-task and ranking-structure material only — never
promoted into `data/dataset_*.json`.


## Training the MoE on rrf_003's synthetic pairs (`moe-rrf003-synthetic-training.html`, 2026-07-29)

**Isolated package: `moe_rrf/`.** Nothing in `moe_reranker/` or `baselines/` was
modified — `moe_reranker.model` and `moe_reranker.diagnostics` are imported
unchanged, and only the feature layer (which genuinely differs) was copied.

This closes the question the MoE review left open. That experiment concluded the
MMoE was untestable on 111 real pairs and that the bottleneck was "data, not
architecture." `rrf_003` supplies 24× the pairs and 146× the within-seeker
triplets. **The conclusion turned out to be wrong, and the corrected version is
more useful: the bottleneck is label validity, not data volume.**

Evaluation design worth reusing: **when training uses only synthetic pairs, every
real pair outside the frozen holdout becomes a legitimate test set** — 131 pairs
instead of 69, SE ±0.0505 instead of ±0.0709 (29% tighter), and the one-shot
holdout stays unspent. Populations asserted disjoint (923 vs 1,217 contact ids,
zero overlap).

| arm | real AUC | fold std | n_train |
|---|---|---|---|
| TF-IDF alone, real vocabulary (**the floor**) | 0.5660 | — | 0 |
| TF-IDF alone, synthetic vocabulary | 0.5631 | — | 0 |
| **logistic regression, real pairs (best)** | **0.6398** | 0.177 | 105 |
| MoE, real pairs | 0.3934 | 0.158 | 105 |
| logistic regression, synthetic | 0.4270 | — | 2,619 |
| MoE, synthetic — pairwise | 0.4066 | — | 2,619 |
| MoE, synthetic — within-seeker triplets | 0.5380 | — | 2,773 |
| synth pretrain → real fine-tune | 0.4730 | 0.145 | 105 |

Findings:

- **Nothing trained on synthetic data beats the no-model floor**, and two arms are
  *below chance* — the synthetic feature→label relationship is anti-correlated with
  real outcomes, so training on it inverts the decision.
- **The failure is in the labels, not the text.** A vocabulary learned entirely
  from synthetic profiles scores real pairs within 0.003 of a real-fitted one. The
  words transfer; the labels don't. This rules out lexical distribution shift.
- **It is a transfer failure, not a training failure.** Synthetic-internal AUC is
  0.6086–0.6637, so fitting works fine.
- **The distillation ceiling is the reframe.** The judge teacher scores **0.5797**
  on the same 131 real pairs, so even a perfect imitator lands below
  `logistic_real`'s 0.6398 — and the students come in *below their own teacher*,
  because they learn the judge's decision function as expressed on synthetic
  profiles, which is not the function it applies to real ones. More labels at this
  teacher quality is not the lever.
- **Within-seeker training was the one prediction that held**: +0.131 over pairwise
  (0.4066 → 0.5380), the largest single effect measured, enabled by the 2,773
  triplets. Keep it; it only rescues to ~chance here, but the mechanism is real.
- **Plain logistic regression on ~105 real pairs is still the best model** — the
  third independent time the MoE machinery has failed to pay for itself.

Two more measurement bugs found, both caught by an implausible number rather than a
failing test. A **hand-rolled TF-IDF** (unigrams + sublinear vs the repo encoder's
bigrams) scored 0.4366 where the repo encoder scored 0.5660, correlation 0.52 — the
strongest single feature was simply wrong; fixed by reusing
`baselines.tfidf.encode.TfidfEncoder` unchanged and verified against the documented
holdout figure. And a **stale-cache collision**: `TfidfEncoder.encode()` keys its
cache on the texts but **not the fitted vocabulary**, so encoding the same real rows
under a real-fitted and a synth-fitted vectorizer collided and returned identical
vectors — the tell was two arms reporting byte-identical 0.5660. Worth knowing for
any experiment that re-encodes identical text under a new fit.

The real 69-pair holdout was **not spent** — nothing came close to earning it.


## Per-ask MoE: sectioned re-ranker (`moe-sectioned-findings.html`, 2026-07-31)

Package: `moe_sectioned/`. Full tables: `docs/moe-sectioned-experiment.md`.
Two pages, deliberately kept separate:

- [`moe-sectioned-plan.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-sectioned-plan.html)
  — the **pre-experiment proposal**, written and published before anything was
  built, kept **unedited** and banner-marked as superseded. It predicted five
  things and got three wrong; it is retained so the predictions stay auditable
  beside the outcomes rather than being quietly revised into agreement.
- [`moe-sectioned-findings.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-sectioned-findings.html)
  — what actually happened.
  [Published artifact](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/moe-sectioned-findings.html).

### What it closed

The original MoE proposal was to explode a seeker's `lookingFor` field and reason
about each ask separately. What had been built collapsed all of it into the single
integer `n_sections` — which does not even count asks correctly, it counts
blank-line-separated blocks (13 where the first real seeker wrote 4) — while the
section-scoring machinery in `moe_reranker/section_scoring.py` ran as a
disconnected parallel path whose output was a report, never a feature. This moves
the unit of prediction from a pair to a **(pair, section) row**: 131 training
pairs become 708 rows, median 5 asks per seeker.

### Results (5 seeds x seeker-disjoint 5-fold CV, pooled out-of-fold AUC)

| arm | mean | sd | vs logistic | wins |
|---|---|---|---|---|
| `logistic_pair` | 0.5758 | 0.024 | — | — |
| **`moe_attention`** (TF-IDF) | **0.6467** | 0.029 | **+0.0709** | **5/5** |
| `moe_mean` (TF-IDF) | 0.6404 | 0.019 | +0.0646 | 5/5 |
| `moe_softmax` (TF-IDF) | 0.5568 | 0.056 | −0.0190 | 2/5 |
| `mlp_attention` (TF-IDF) | 0.5446 | 0.048 | −0.0312 | 2/5 |
| `moe_attention` (Qwen3) | 0.5433 | 0.050 | −0.0324 | 1/5 |

**First architecture in this project to beat plain logistic regression under
replication** — but a plain average of the asks matches learned attention, so the
gain is the *sectioning*, not the pooling that motivated the whole design. Four
experts beat one, the first time the mixture has earned its place here.

### Why single-seed numbers from this experiment are worthless

Runs `sec_001`–`sec_004` produced **four different arm rankings in four runs**.
Root cause: the two "reduction" projections held ~960k parameters (TF-IDF,
20,000-d) or ~197k (Qwen3, 4,096-d) against 708 rows, so the projections *were*
the model. Fixed by `EmbeddingReducer`, a train-rows-only PCA to 48 dims, cutting
them to ~2.3k parameters. Anything quoted from a single seed of this package
predates or ignores that fix.

### Two more paid-run defects, both caught by comparison rather than by failure

- The Modal encode OOMed on an A100-40GB by loading 8B params in **fp32** (36.68
  GiB allocated, 942 MiB free). `baselines/hf_embedding/encode.py` already used
  bfloat16 and capped `max_seq_length`; this script was written fresh instead of
  copied.
- **Qwen3-Embedding is asymmetric** (`query_prompt_name="query"` in the model
  registry) and the first encode applied no prompt to either side. Cache keys are
  now role-qualified so a query and a document vector cannot collide. Fixing it
  did **not** rescue the score.

### Relationship to the all-200 sweep

Landed one commit later (`81b4821`) and it revises two things here:

1. The Qwen3 loss stopped being anomalous. The belief it contradicted — Qwen3-8B
   beats Voyage-4-large, 0.6595 vs 0.6086 — was **retracted**; on all 200 pairs
   Qwen loses on every metric. Two unrelated measurements now agree that dense
   Qwen3 embeddings are not the lever on this data.
2. **"Spend the holdout" was withdrawn as the next step.** The 69-pair holdout has
   Spearman −0.029 against the all-200 ranking among the top 6 models. This leaves
   a genuine evaluation gap: all-200 is the right population, but this model trains
   on 131 of those 200, so it cannot be scored there without leakage. The honest
   position is to report the cross-validated number and label it as such.

Total Modal spend across three A100 runs: ~$0.55 of a $3 budget.

## Judge prompt evolution: nine automatic-optimization variants — the sixth nearly closes the gap, the seventh shows scale doesn't help, the eighth isolates why, the ninth shows seed choice doesn't rescue it either (2026-07-31, updated 2026-08-03)

**Isolated package: `judge_prompt_evolution/`.** No files under
`baselines/llm_judge/` or `data/` were edited — the seed judge prompt is
duplicated (not imported) into `judge_prompt_evolution/seed_prompt.py`, the
`structured_cot` seed source is imported read-only from
`baselines.llm_judge.prompt`, and the AUC check is a new file
(`eval_evolved.py`) that imports the original experiment's call/scoring
machinery read-only.

`docs/llm-judge-experiment.md`'s naive judge prompt (pair AUC 0.6177 on all
200 real pairs) had already beaten two hand-designed alternatives
(`calibrated`, `structured_cot`, both holdout-only at the time). This
experiment asked whether a fully automatic prompt-optimization loop — no
human writing the prompt at all — could do what manual engineering couldn't,
then iterated on the loop itself across four runs. Each round: an optimizer
LLM sees the current prompt plus a fresh batch of real labeled examples
(train split only, sampled without replacement) and proposes a revision.
Deliberately no accuracy feedback inside any loop — every AUC check ran once,
after all 20 rounds, as an explicitly separate confirmed step.

**Run 1 (`evo_001`) lost on every metric** — pair AUC 0.5734 vs. 0.6177.
Prompt length went 1,011 → 24,905 chars under Sonnet 4.5 (rounds 1-9, purely
additive — 32 numbered rules, several citing exact dollar figures and sector
names lifted from single training examples) → Deepseek-v4-pro's first round
(10→11) cut that 74% unprompted ("condensed... to improve generalization")
→ grew back to 18,463 by round 20.

**The meta-prompt was rewritten based on that failure** (dropped the
hard/easy-negative framing, replaced "incremental sharpening" with an
explicit instruction to revise the rubric and generalize rather than append
and copy specifics), pushed to Hub as `v2`. **Run 2 (`evo_002`)**, Deepseek
only, grew smoothly to 13,142 chars — pair AUC **0.5918**. Better, roughly
half the gap closed, still a loss.

**Confirming `structured_cot` on all 200 real pairs** puts it at **0.6100**
— closer to naive than either evolution run. That motivated **Run 3
(`evo_003`)**: same v2 process, seeded from `structured_cot` instead of
naive. Round 1 immediately collapsed the six-aspect seed to 1,960 chars (the
v2 meta-prompt's plain-JSON contract forced discarding the weighted-aspect
structure), then grew to 8,669 by round 20 — scoring pair AUC **0.5918**,
identical to `evo_002` to four decimal places despite a 77-point-higher,
structurally different starting point. The cleanest evidence yet that this
particular ~0.59 result is an attractor of the loop process itself, not a
property of the seed.

**Run 4 (`evo_004`)** tested whether *forcing* periodic consolidation (a
separate distillation-only prompt every 5 rounds, no example batch, just
"merge redundant rules, cut restated points") beats relying on the
meta-prompt's instruction alone. It worked correctly 3 of 4 times — real
distillation, e.g. round 10 cut 3,552→1,496 chars by merging genuinely
redundant paragraphs. But the most aggressive pass (round 20, compressing
4,017 chars down to ~1,125) **dropped the required JSON output contract
twice in a row**, despite the summarizer prompt explicitly listing it as a
hard constraint — a third, less extreme attempt (2,071 chars) finally kept
it. Scored pair AUC **0.5700** using that valid attempt — worse than both
non-summarized v2 runs, the second-worst result of anything tried.

**Clean ranking across four attempts:** naive (0.6177) > structured_cot
(0.6100) > evo_002 ≈ evo_003 (0.5918) > evo_001 (0.5734) > evo_004 (0.5700).
Neither fix — generalizing away from example-specific rules, or forcing
periodic consolidation — closed the gap; the second opened it slightly
wider than the first fix already had.

**Run 5 (`evo_005`) tested the discipline itself: what if the optimizer just
sees all 200 pairs instead of train-only?** Same process as `evo_004`, only
`sampling.py`'s `ExampleBank` now draws from the full 200-pair pool (100
pos/50 hard-neg/50 easy-neg) instead of train's 71/30/30 — meaning the
holdout is no longer held out. Flagged as exploratory and non-comparable
*before* running (`run.py` prints and records a loud `leakage_warning`
whenever `split != "train"`, surfaced again by `eval_evolved.py`). The run
itself was the cleanest mechanically of any so far — all 4 summarize
checkpoints worked on the first try, ending at 2,307 chars — and scored the
**highest nominal pair AUC of any evolution run at the time, 0.6016**.
That's not a real result: the optimizer had already seen labeled examples
from roughly a quarter of the exact population it was later scored against.
Excluded from the ranking, exists as a documented control confirming *why*
the train/holdout discipline matters. A follow-up sub-experiment scored
`evo_005`'s **unsummarized** round-20 prompt (3,629 chars) on its own: first
attempt used AWS Bedrock (`minimax.minimax-m2.5`, since OpenRouter credits
had run out) and found a real bug — Bedrock's Converse API puts reasoning
models' actual answer in `content[1]`, not `content[0]`, which
`bedrock_backend.py::call_bedrock_verdict` doesn't handle (`KeyError`,
worked around locally, not fixed in the shared file) — and scored a
near-chance 0.5095, uninterpretable on its own since it used a different
judge model than everything else here. Rerun via a new **direct Gemini API**
backend (`GEMINI_API_KEY`, bypassing OpenRouter, same `gemini-3.1-flash-lite`
as every other number in this doc) scored **0.5790** — below `evo_005`'s own
summarized final (0.6016), meaning that particular summarize step helped
even inside a contaminated run, unlike `evo_004`'s experience with the same
wording. That observation motivated Run 6.

**Run 6 (`evo_006`) tested whether the aggressive summarizer's *wording*,
not the *idea* of periodic summarization, was what hurt `evo_004`.** Drafted
a second summarizer prompt collaboratively — same clarify/generalize goal,
but every phrase pushing toward brevity removed and replaced with explicit
permission to stay long ("a result close to its starting size is fine")
— confirmed with the user and pushed to LangSmith Hub
(`judge-prompt-evolution-summarizer-gentle`) before running. Otherwise
identical to `evo_004`: naive seed, v2 meta-prompt, Deepseek, distill every
5 rounds, train-split examples (clean). The gentle summarizer cut visibly
less at every checkpoint (2-35% vs. `evo_004`'s 55-74%) and never dropped
the JSON contract, including at round 20 where the aggressive version had
failed twice. **Scored pair AUC 0.6105 — the closest any evolution run has
come, beating even hand-designed `structured_cot` (0.6100)**, closing the
gap to naive from `evo_004`'s −0.0477 to just −0.0072.

**Clean ranking across six attempts:** naive (0.6177) > **evo_006 (0.6105)**
> structured_cot (0.6100) > evo_002 ≈ evo_003 (0.5918) > evo_001 (0.5734) >
evo_004 (0.5700). The mechanism of the fix mattered as much as its
existence: generalizing away from example-specific rules alone recovered
about half the original overfitting gap; adding *aggressive* periodic
consolidation on top made it worse; adding *gentle* periodic consolidation
instead recovered nearly all the rest.

**Run 7 (`evo_007`) tested scaling evo_006's recipe up: more examples per
round, and sampling from all 200 pairs instead of train-only.** 6 examples
(3 pos/2 hard-neg/1 easy-neg, up from 4) drawn from the full 200-pair pool
(contaminated by design, same accepted tradeoff as `evo_005`), gentle
summarizer every 5 rounds, otherwise identical to `evo_006`. Rounds 1-5 ran
on Deepseek as usual; **OpenRouter then ran out of credits at round 6**
(the same reserved-against-`max_tokens` mechanism documented in the
LLM-judge experiment's cost incident, tripped here because the bigger
6-example batches pushed the affordable completion budget below the fixed
8,000-token cap). Rather than shrink `max_tokens`, added a second optimizer
backend — direct Gemini API calls (`optimizer_backend="gemini"` in
`config.py`/`optimizer.py`, reusing the raw-`urllib` pattern from
`eval_evolved.py`'s existing Gemini eval path) — and resumed on
`gemini-3.1-flash-lite` for rounds 6-20.

**The Gemini optimizer then introduced a new failure mode**: starting round
10, 13 of the last 15 records (rounds 10-20 plus both later summarize steps)
came back missing the required `reasoning`/`confidence` keys and any
mention of JSON output — the meta-prompt's hard constraints notwithstanding.
The final round-20 rubric (1,614 chars, four clearly-stated principles:
constraint/exclusion compliance, direct reciprocal utility, strategic
compatibility, high-signal intent) was coherent but literally never told the
judge to return JSON. Patched by hand — the canonical `RESPONSE_CONTRACT`
block appended verbatim, original preserved as `final_prompt_raw_broken` in
`summary.json` — pushed to LangSmith Hub as a new commit
(`judge-prompt-evolution` tag `evo_007--final-patched`).

**Scored pair AUC 0.5739** on all 200 real pairs — worse than naive
(0.6177, Δ −0.0438) and worse than every clean or contaminated run except
`evo_004`, despite the larger batches and wider (structurally favorable, per
`evo_005`) sampling pool. Not comparable to the clean ranking, but the
directional finding stands on its own: scaling up example count and
sampling breadth bought nothing here, and the optimizer-backend switch cost
a genuine new failure mode that six prior Deepseek-optimized runs never hit.

**Run 8 (`evo_008`) isolated whether that new failure mode was really about
Gemini-as-optimizer, or about `evo_007`'s mid-run resume compounding an
unrepaired contract loss.** Same recipe as `evo_007` — naive seed, v2
meta-prompt, all-200 sampling, gentle summarizer every 5 rounds — but
`gemini-3.1-flash-lite` as optimizer from **round 1**, a more balanced
2/2/2 example split (up from 3/2/1), and a genuine fix instead of a
one-time hand patch: `optimizer.py::repair_contract()` now appends the
canonical `RESPONSE_CONTRACT` block back automatically, every round,
whenever `validate_contract()` flags a problem — recorded via
`contract_repaired`/`prompt_after_raw` on the affected record, no manual
intervention required. **Only 1 of 20 rounds needed a repair** (round 15,
vs. `evo_007`'s 13 of 20), and it self-healed immediately instead of
compounding. Scored **pair AUC 0.6037** — a large recovery over `evo_007`'s
0.5739 on nearly the same recipe, close to several clean Deepseek-optimized
runs (though still contaminated/not comparable to the clean ranking). The
takeaway: most of `evo_007`'s damage traces to the mid-run resume and
uncorrected compounding contract loss, not Gemini being a categorically
worse optimizer — once the contract is guaranteed to self-heal, Gemini
performs respectably in this role.

**Run 9 (`evo_009`) reran `evo_008`'s exact recipe with one variable
changed: `structured_cot` as the seed instead of naive.** Same v2
meta-prompt, gemini-3.1-flash-lite optimizer from round 1, 6 examples/round
2/2/2, all-200 sampling, gentle summarizer every 5 rounds, contract
auto-repair on — this mirrors the earlier `evo_002`/`evo_003` comparison
(same process, different seed) but on the newer, scaled-up recipe. Ran
perfectly clean: **zero contract repairs across all 20 rounds and both
summarize steps** (`evo_008` needed exactly one). Scored **pair AUC
0.5814** — worse than `evo_008`'s 0.6037 on the identical recipe, and worse
than `structured_cot`'s own un-evolved score (0.6100), despite that seed
starting closer to naive than the naive seed's own 0.6177. Same conclusion
`evo_002`/`evo_003` reached on the older 4-example/train-only recipe: a
stronger starting prompt does not carry forward into a better evolved
result — the loop appears to erase seed-prompt quality differences (or
worse) rather than build on them. Across `evo_006`–`evo_009`, four
independent variables were tried against the process that produced
`evo_006`'s best clean result (more examples, wider sampling, a different
optimizer model, a stronger seed) and none of them improved on it.

Bugs found along the way: a `json.loads` strict-mode failure (Deepseek
emitting raw newlines inside JSON string values, deterministic — same error,
same location, across separate calls) fixed with a local lenient parser
rather than touching shared `synth_pipeline/llm.py`; an `optimizer.py` call
site that never threaded through `OPENROUTER_API_KEY`; a fixed
`max_tokens=3000` too small once the prompt grew; LangSmith commit tags
colliding across iterations sharing an index (optimize + its summarize
step); the summarizer contract-dropping failure mode (originally aggressive-
summarizer only, later also seen from the Gemini optimizer itself in
`evo_007`); the Bedrock reasoning-model content-block bug above; and
OpenRouter's reserved-against-`max_tokens` credit check stranding `evo_007`
mid-run once its larger example batches grew the prompt, worked around by
adding a second optimizer backend (`optimizer_backend="gemini"`). `run.py
--resume` (extended to understand interleaved summarize steps, and now also
mid-run optimizer-backend switches) meant none of these crashes cost
already-completed rounds. Every fixed prompt (meta v1/v2, both summarizer
variants) is now pushed to and pull-loaded from LangSmith Hub at call time,
not just read from the local file. See
`docs/judge-prompt-evolution-experiment.md`
for the full writeup, every iteration's prompt text, and repro commands.

## Published Artifacts (claude.ai, this account)

See the unified table at the top of this doc for all 9 currently-published
pages (checked live via `Artifact` `action: "list"` on 2026-07-26). The
first 3 (`Pairs graph`, `Real pairs graph`, `Holdout comparison browser`)
predate this worktree's sessions, so their exact source file among the
local HTMLs above is a best guess from filename/date proximity, not a
verified fact. `Does splitting lookingFor into sections help matching?`
and `Holdout contacts in voyage-4-nano space` were published directly from
sessions in this worktree and are definitively traceable to
`docs/lookingfor-sectioning-findings.md` and
`holdout-embedding-space-3d.html` respectively, as is `Which fields carry
a person's identity?` (the field isolation experiment above). `Query-Time
Nudge vs. Joint Encoding`, `LLM judge vs. embedding baselines`, and
`Synthetic Pair Pipeline — Proposed Flow` are not traceable to a specific
local file from this session's history. If you want a guaranteed link
between a specific experiment and a shareable URL going forward, publish
the file explicitly (`Artifact` with `--fragment` output for local HTMLs,
per commit `0ade0ca`) rather than relying on this guesswork.
