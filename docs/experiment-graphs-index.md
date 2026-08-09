# Experiment HTMLs — index

One flat table first: every self-contained HTML this project has produced,
local or published, in one place. Detailed per-file notes (what each one
demonstrates, why it looks the way it does) follow below for anyone who
wants the full story.

## All local + published HTML outputs

| Name | Where | Built | What it is |
|---|---|---|---|
| [`pairs-comparison-graph-hub-test.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-hub-test.html) | local (`docs/html/`) | 2026-07-24 | Dual-pane real-vs-synth pairing graph, 3-profile smoke test after routing generation prompts through LangSmith Hub |
| [`pairs-comparison-graph-no-refex.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-no-refex.html) | local (`docs/html/`) | 2026-07-24 | Dual-pane graph, 5-profile batch after dropping redundant reference examples from `generate_profile` |
| [`pairs-comparison-graph-disjoint.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-disjoint.html) | local (`docs/html/`) | 2026-07-24 | Dual-pane graph, same 5 profiles re-paired with disjoint seeker/candidate split + per-seeker cap |
| [`pairs-comparison-graph-named.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/pairs-comparison-graph-named.html) | local (`docs/html/`) | 2026-07-24 | Dual-pane graph, 10-profile batch after the name-collision fix (10/10 unique names) |
| [`real-pairs-tfidf-cluster.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-tfidf-cluster.html) | local (`docs/html/`) | 2026-07-24 | Force-directed real-pairs graph, TF-IDF similarity force added |
| [`real-pairs-voyage-lookingfor-cluster.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-cluster.html) | local (`docs/html/`) | 2026-07-24 | Force-directed real-pairs graph, voyage-4-large `lookingFor` similarity force |
| [`real-pairs-tfidf-pca.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-tfidf-pca.html) | local (`docs/html/`) | 2026-07-24 | Static PCA/SVD scatter (no physics), TF-IDF — 1.67% variance explained |
| [`real-pairs-voyage-lookingfor-pca.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-pca.html) | local (`docs/html/`) | 2026-07-24 | Static PCA scatter, voyage-4-large `lookingFor` — 12.9% variance explained |
| [`real-pairs-voyage-lookingfor-3d-pca.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-3d-pca.html) | local (`docs/html/`) | 2026-07-24 | 3D PCA scatter, hand-rolled canvas projector — 17.3% cumulative (PC1–3) |
| [`real-pairs-voyage-lookingfor-3d-manifold.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/real-pairs-voyage-lookingfor-3d-manifold.html) | local (`docs/html/`) | 2026-07-24 | 3D PCA / t-SNE / UMAP scatter with a layout selector |
| [`baseline-results-holdout-browser.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/baseline-results-holdout-browser.html) | local (`docs/html/`) | 2026-07-20 | Browser for the matched-holdout baseline comparison table |
| [`holdout-embedding-space-3d.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/holdout-embedding-space-3d.html) | local (`docs/html/`) | 2026-07-25 | 3D PCA map of the 69 holdout contacts in voyage-4-nano space, whole-profile vs. `lookingFor`-sectioned embeddings, with good/bad match lines and a scatter (dispersion) analysis — see `scripts/build_holdout_embedding_space_3d.py` / `scripts/analyze_section_dispersion.py` |
| [`llm-judge-comparison.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/llm-judge-comparison.html) | local (`docs/html/`) | 2026-07-25, updated 2026-08-09 | Two full sections (69-pair holdout + all-200-pair), each with a ranked pair-AUC bar chart + hard/easy-neg breakdown, for nine LLM-judge (model, framing, backend) combinations on the holdout (twelve on the 200-pair section, including three candidate-field-permutation runs) plus every embedding baseline, including three field-selected rows (Voyage-4-nano/large, Qwen3-Embedding-8B) matching the focused judge's exact fields — hover any bar for a seeker-fields / candidate-fields / query-yes-no tooltip. **Qwen3-Embedding-8B field-selected is the best overall pair AUC in the project (0.6862 holdout), though the LLM judge still leads hard-negative AUC; on the seeker side, trimming to `lookingFor` alone beats the two-field focused prompt on the holdout (0.6698, best hard-neg AUC in the project); on the candidate side no two-field permutation beats the full three-field candidate (0.6451 all-200), and dropping `background` hurts hard-neg AUC most** — see `scripts/build_llm_judge_browser.py` / `docs/llm-judge-experiment.md` / `docs/llm-judge-focused-prompt-experiment.md` / `docs/voyage-field-selected-experiment.md` / `docs/llm-judge-seeker-background-experiment.md` / `docs/qwen3-embedding-field-selected-experiment.md` / `docs/llm-judge-seeker-field-isolation-experiment.md` / `docs/llm-judge-candidate-field-permutation-experiment.md` |
| LLM judge vs. embedding baselines | [published](https://claude.ai/code/artifact/12f8f93b-8fc4-41e5-bc13-b05ce8ab45fa) | 2026-07-25, updated 2026-08-09 | Published version of `llm-judge-comparison.html` above |
| [`experiment-index.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/experiment-index.html) | local (`docs/html/`) | 2026-07-20, updated 2026-08-08 | Chronological index of every visualization this project has produced (published Claude artifacts + local self-contained HTML), most recent first |
| Dorby AI — Experiment Index | [published](https://claude.ai/code/artifact/d0b86eb9-4eda-4d03-9c76-9c79201f4ebb) | 2026-07-20, updated 2026-08-09 | Published version of `experiment-index.html` above |
| Pairs graph — Boardy AI | [published](https://claude.ai/code/artifact/642d0a82-7784-4843-b0ad-5686cf7db24c) | 2026-07-24 | Likely one of the `pairs-comparison-graph*.html` variants above, published via `--fragment` — exact source not traceable from this session |
| Real pairs graph — Boardy AI | [published](https://claude.ai/code/artifact/ac74ea3a-912d-407a-a040-74d8c62d1edd) | 2026-07-22 (page updated 2026-07-24) | Predates the batches above; likely an early real-only single-pane build |
| Holdout comparison browser — Dorby AI | [published](https://claude.ai/code/artifact/95beeed4-9a3d-4a79-906d-cf2d24d0457f) | 2026-07-20 | Likely `baseline-results-holdout-browser.html` above, by date match |
| Does splitting lookingFor into sections help matching? | [published](https://claude.ai/code/artifact/3ec8c0da-9ba1-4de9-b52d-b057507b6163) | 2026-07-24 | lookingFor field-sectioning: 4 experiments (candidate- vs seeker-sectioned, softer aggregation, hybrid-fusion stacking) — see `docs/lookingfor-sectioning-findings.md` |
| Holdout contacts in voyage-4-nano space | [published](https://claude.ai/code/artifact/5bf01ecd-0731-4f7e-93f3-ee74f8688e21) | 2026-07-25 | Published version of `holdout-embedding-space-3d.html` above |

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

## Published Artifacts (claude.ai, this account)

See the unified table at the top of this doc for all 4 currently-published
pages. The first 3 predate this worktree's sessions, so their exact source
file among the local HTMLs above is a best guess from filename/date
proximity, not a verified fact — the 4th (lookingFor sectioning findings)
was published directly from this session and is definitively traceable to
`docs/lookingfor-sectioning-findings.md`. If you want a guaranteed link
between a specific experiment and a shareable URL going forward, publish
the file explicitly (`Artifact` with `--fragment` output for local HTMLs,
per commit `0ade0ca`) rather than relying on this guesswork.
