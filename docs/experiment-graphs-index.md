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
| [`knowledge-graph-experiment.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/knowledge-graph-experiment.html)                                 | local (`docs/html/`) + [published](https://claude.ai/code/artifact/af0a622e-ba61-488f-b999-cf555f61d2ac) | 2026-07-27                           | One real user's profile + one accepted/one declined real intro, each decomposed into a knowledge graph by `google/gemini-3.1-flash-lite` and merged on shared concept labels, plus a type-taxonomy layer — see `docs/knowledge-graph-experiment.md`                                                                                                                                                                                                                                                                                                                        |
| [`twotower-rrf-triplet-findings.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-rrf-triplet-findings.html)                           | local (`docs/html/`) + [published](https://claude.ai/code/artifact/3511253c-fb74-429e-820c-d30dbd8c4816) | 2026-07-29                           | `voyage-4-nano` + `Qwen3-Embedding-8B` LoRA fine-tunes on `rrf_003` triplets (`MultipleNegativesRankingLoss`): training-loss curves for both runs, real-69-pair-holdout results table vs. Voyage-4-large/Arm A, and a plain-language explanation of why pair AUC rose while retrieval recall@1 fell — see `docs/twotower-rrf-triplet-experiment.md`                                                                                                                                                                                                                        |
| [`twotower-rrf-triplet-bigbatch-comparison.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-rrf-triplet-bigbatch-comparison.html)     | local (`docs/html/`) + [published](https://claude.ai/code/artifact/6d2ea3d7-0f90-42b6-ac2e-ac807dcaedb1) | 2026-07-29                           | Follow-up: isolated `voyage-4-nano` re-run (`twotower_rrf_triplet_bigbatch/`, own package) with real batch size 2→6 (GPU-probed ceiling was 8) and 2 negatives per anchor instead of 1 — closed recall@1 to exactly match frozen Voyage-4-large (0.345) at the cost of pair AUC dropping below it; four-way comparison table + next-steps vs. the original triplet runs — see `docs/twotower-rrf-triplet-bigbatch-experiment.md`                                                                                                                                           |
| [`twotower-ablation-verdict.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-ablation-verdict.html)                                   | local (`docs/html/`) + [published](https://claude.ai/code/artifact/4ab4000d-f7a6-4477-8a2e-3b578da25cdc) | 2026-07-29                           | **Combined verdict** of the 2×2 ablation splitting the bigbatch run's two levers apart (`twotower_rrf_triplet_ablation/`, effective batch pinned to 12 in every arm, each arm run twice for a measured noise floor): **micro-batch size is what moved retrieval** (+0.05 MRR, +0.06 recall@1) while **the second negative hurt** (−0.03 pair AUC) because 27.5% of k=2 negative slots are duplicates. Also documents a corrected single-run claim and a dev-set-too-small-to-select finding — see `docs/twotower-rrf-triplet-ablation-experiment.md`                       |
| [`moe-reranker-review.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-reranker-review.html)                                             | local (`docs/html/`)                                                                                     | 2026-07-29                           | Design review + results for the **multi-gate mixture-of-experts re-ranker** over `lookingFor` (`moe_reranker/`, own isolated package). Settles the combine-rule question (veto-shaped aggregation lost monotonically to a temperature-sharpened gate) and reports the built MMoE's seeker-disjoint CV: 0.5536 vs 0.5282 for no model at all, fold std 0.067 — **not testable at 111 real training pairs**. Documents two self-inflicted measurement bugs (a label-leaking normalizer reporting a fake 0.8500 AUC; a saturated routing-MI diagnostic) — see `docs/moe-reranker-experiment.md` |
| [`moe-rrf003-synthetic-training.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/moe-rrf003-synthetic-training.html)                                 | local (`docs/html/`) + [published](https://claude.ai/code/artifact/9f8af37b-d294-43db-bf14-229df33a48d2) | 2026-07-29                           | **Answer to the previous experiment's open question: training on synthetic data does not help.** The MoE trained on `rrf_003`'s 2,619 judge-labeled pairs (`moe_rrf/`, own isolated package) scored *below* the no-model TF-IDF floor on 131 real pairs, and two arms landed below chance. Locates the failure precisely: synthetic **vocabulary** transfers (0.5631 vs 0.5660 real-fitted) but synthetic **labels** don't, and the judge teacher's own 0.5797 on real pairs is a ceiling below what plain logistic regression already reaches. Within-seeker training was the one win (+0.131). Two more measurement bugs caught — a broken TF-IDF reimplementation and a stale-cache collision that silently merged two arms — see `docs/moe-rrf003-synthetic-training-findings.md` |
| [`twotower-abl-a-batch-only.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-abl-a-batch-only.html)                                   | local (`docs/html/`) + [published](https://claude.ai/code/artifact/9dcd8dc1-97ba-4183-b77b-717380d1966b) | 2026-07-29                           | Ablation Arm A — micro-batch 6, k=1. The winning cell (pair AUC 0.5983 / MRR 0.5326 / recall@1 0.3793, `_v2` run only — its two runs shipped different epochs and are not replicates); loss curve + dev-accuracy overlay, holdout table vs. frozen Voyage-4-large                                                                                                                                                                                                                                                                                                                                                                   |
| [`twotower-abl-b-negs-only.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-abl-b-negs-only.html)                                     | local (`docs/html/`) + [published](https://claude.ai/code/artifact/187a3df4-362b-4373-98fe-0b34f00bbb61) | 2026-07-29                           | Ablation Arm B — micro-batch 2, k=2. The worst cell (pair AUC 0.5595 / recall@1 0.2931); its dev metric falls after epoch 3 while training loss keeps dropping, the clearest sign of the duplicate-negative problem                                                                                                                                                                                                                                                                                                                                                        |
| [`twotower-abl-c-baseline.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/twotower-abl-c-baseline.html)                                       | local (`docs/html/`) + [published](https://claude.ai/code/artifact/d2407099-83f8-4e65-a5a3-f8125ae57381) | 2026-07-29                           | Ablation Arm C — micro-batch 2, k=1. The baseline corner every effect in the ablation is measured against (pair AUC 0.5996 / MRR 0.4902 / recall@1 0.3276)                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [`eval-real-full-200pairs.html`](https://claude.ai/code/artifact/d807fdd2-afe5-40ad-bafa-04423b9ccf87) | published artifact + local (`docs/html/`) | 2026-07-30 | Frozen voyage-4-nano vs. the Arm A fine-tune re-measured on **all 200 real pairs** instead of the 69-pair holdout. Finding: the holdout's pair-AUC/recall@1 gains do not generalise (pair AUC −0.003 on the other 131 pairs), but hard-negative AUC improves on every population and grows with sample size (+0.040 / +0.059 / +0.072). Writeup: `docs/eval-real-full-experiment.md` |
| [`twotower-qwen-bigbatch.html`](https://claude.ai/code/artifact/f9283fec-b9dd-46f1-846f-9280523feb18) | published artifact + local (`docs/html/`) | 2026-07-30 | Qwen3-8B LoRA fine-tune at micro-batch 6 vs 1 (effective batch pinned to 12), plus all seven models re-measured on **all 200 real pairs**. Findings: the micro-batch effect replicates on a 22x larger backbone (+0.087 MRR, +0.103 R@1); Qwen fine-tuning generalises where nano's did not (+0.053 pair AUC on 200 vs nano's +0.007); and the 69-pair holdout flatters Qwen ~4x more than nano, so its published "beats Voyage-4-large" headline reverses on the full set (frozen Qwen 0.5420 < nano 0.5593 < Voyage-large 0.5726). Writeup: `docs/twotower-qwen-bigbatch-experiment.md` |
| [`Dorby AI — Framing & Experiment Proposal`](https://claude.ai/code/artifact/5aed8e2e-a4e4-4c2d-adbc-500c307f855a) | published artifact only (no local file) | 2026-07-27 | Framing deck for the project's objective and proposed experiment slate. Published before the two-tower ablation series; predates the 200-pair evaluation, so any accuracy figures in it are 69-pair-holdout numbers. |
| [`project-story.html`](http://dorby-project-story-411960113601.s3-website-us-east-1.amazonaws.com/) | local (`docs/html/`) + S3 static site | 2026-07-30 | Plain-language chronological walkthrough of the whole project (2026-07-16 → 07-30) as a scroll-driven slideshow. Not a Claude artifact — hosted on S3, URL recorded in `docs/project-story-url.md`. |
| [`twotower-top1-optimised.html`](https://claude.ai/code/artifact/6caec2cf-5434-462e-a29a-a55dd13018f1) | published artifact + local (`docs/html/`) | 2026-07-30 | Two changes targeting recall@1 at fixed Arm A settings. **Sharpening the loss** (MNRL scale 20→50 + `hardness_mode='hard_negatives'`) **backfired on every metric** — all-200 R@1 0.1800→0.1400, below the untrained baseline, hard-neg AUC 0.4578 (worse than chance). **Fixing checkpoint selection** to rank against a real dev corpus (`CorpusRecallDevEvaluator`, `primary_metric='recall@1'`) produced the **best MRR of any model in the project (0.3550 on all 200)** and the first fine-tune to beat frozen nano at R@1 (19 vs 18 of 100) — invisible on the 69-pair holdout. Writeup: `docs/twotower-top1-optimised-experiment.md` |

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
