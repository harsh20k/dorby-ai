# Experiment graph HTMLs — index

Every `docs/pairs-comparison-graph*.html` file is a self-contained,
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

Open any of these directly in a browser (`open docs/<file>.html`) — fully
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

## Published Artifacts (claude.ai, this account)

`Artifact` publishing (`action: "list"`, `scope: "mine"`) shows 3
currently-published pages relevant to this project:

| title | url | last updated | likely source |
|---|---|---|---|
| Pairs graph — Boardy AI | https://claude.ai/code/artifact/642d0a82-7784-4843-b0ad-5686cf7db24c | 2026-07-24 | uncertain — not published by this session; possibly one of the `pairs-comparison-graph*.html` variants above via `--fragment` mode (added in commit `0ade0ca`), but this session made no `Artifact` publish calls, so the exact source file is not traceable from here |
| Real pairs graph — Boardy AI | https://claude.ai/code/artifact/ac74ea3a-912d-407a-a040-74d8c62d1edd | 2026-07-22 | uncertain — predates every batch documented above; likely an early real-only single-pane build (no local file with that exact shape exists in this worktree currently) |
| Holdout comparison browser — Dorby AI | https://claude.ai/code/artifact/95beeed4-9a3d-4a79-906d-cf2d24d0457f | 2026-07-20 | likely `docs/baseline-results-holdout-browser.html` (`scripts/build_holdout_browser.py`, generated 2026-07-20 per file mtime, committed to `main` at `ef9fd8d`/`bdd1631`) — plausible by date match but not confirmed |

**Caveat on the "likely source" column:** this session never called
`Artifact` to publish anything — all 7 graphs above (5 comparison, 2
similarity) were only opened locally (`open docs/<file>.html`). The 3
published pages listed
were published in earlier sessions this account doesn't have visibility
into from here, so the source-file mapping is a best guess from filename/
date proximity, not a verified fact. If you want a definitive link between
a specific experiment and a shareable URL, the reliable path is to
explicitly publish the file you want (`Artifact` with `--fragment` output,
per commit `0ade0ca`) rather than trust this table's guesses.
