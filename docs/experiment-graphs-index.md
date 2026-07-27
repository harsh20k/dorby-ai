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
| [`holdout-field-isolation-embedding-space-3d.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/holdout-field-isolation-embedding-space-3d.html) | local (`docs/html/`) | 2026-07-26 | 3D PCA map of the same 115 holdout contacts, but every profile field and every `lookingFor` ask embedded **alone** (no other field present) instead of swapped into an otherwise-whole profile — see "Field isolation experiment" below |
| [`llm-judge-comparison.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/llm-judge-comparison.html) | local (`docs/html/`) | 2026-07-25, rebuilt 2026-07-26 | Ranked pair-AUC bar chart + hard/easy-neg breakdown for LLM-judge (model, framing) combinations against every embedding baseline, matched 69-pair holdout — see `scripts/build_llm_judge_browser.py` / `docs/llm-judge-experiment.md`. 2026-07-26 rebuild adds the new `structured_cot` variant (see "LLM judge: does forcing multi-aspect CoT help?" below); the two Bedrock combos and `calibrated` are cached artifacts from an earlier session not present in this checkout (`artifacts/` is gitignored), so the chart currently shows `naive` + `structured_cot` only |
| LLM judge vs. embedding baselines | [published](https://claude.ai/code/artifact/12f8f93b-8fc4-41e5-bc13-b05ce8ab45fa) | 2026-07-25 | Published version of `llm-judge-comparison.html` above |
| Pairs graph — Boardy AI | [published](https://claude.ai/code/artifact/642d0a82-7784-4843-b0ad-5686cf7db24c) | 2026-07-24 | Likely one of the `pairs-comparison-graph*.html` variants above, published via `--fragment` — exact source not traceable from this session |
| Real pairs graph — Boardy AI | [published](https://claude.ai/code/artifact/ac74ea3a-912d-407a-a040-74d8c62d1edd) | 2026-07-22 (page updated 2026-07-24) | Predates the batches above; likely an early real-only single-pane build |
| Holdout comparison browser — Dorby AI | [published](https://claude.ai/code/artifact/95beeed4-9a3d-4a79-906d-cf2d24d0457f) | 2026-07-20 | Likely `baseline-results-holdout-browser.html` above, by date match |
| Does splitting lookingFor into sections help matching? | [published](https://claude.ai/code/artifact/3ec8c0da-9ba1-4de9-b52d-b057507b6163) | 2026-07-24 | lookingFor field-sectioning: 4 experiments (candidate- vs seeker-sectioned, softer aggregation, hybrid-fusion stacking) — see `docs/lookingfor-sectioning-findings.md` |
| Holdout contacts in voyage-4-nano space | [published](https://claude.ai/code/artifact/5bf01ecd-0731-4f7e-93f3-ee74f8688e21) | 2026-07-25 | Published version of `holdout-embedding-space-3d.html` above |
| Query-Time Nudge vs. Joint Encoding | [published](https://claude.ai/code/artifact/d491b7db-0db8-458a-8148-78001c084e30) | 2026-07-25 | Not traceable to a local `docs/html/` file from this session |
| LLM judge vs. embedding baselines | [published](https://claude.ai/code/artifact/12f8f93b-8fc4-41e5-bc13-b05ce8ab45fa) | 2026-07-25 | Not traceable to a local `docs/html/` file from this session |
| Which fields carry a person's identity? | [published](https://claude.ai/code/artifact/c3daa30f-4b68-4312-ba4b-8e69e4c77550) | 2026-07-26 | Published version of `holdout-field-isolation-embedding-space-3d.html` above (field isolation experiment) |
| Synthetic Pair Pipeline — Proposed Flow | [published](https://claude.ai/code/artifact/5455a3ec-2c0f-4926-8e08-bc705868a6cf) | 2026-07-26 | Not traceable to a local `docs/html/` file from this session |

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
