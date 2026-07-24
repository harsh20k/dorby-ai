#!/usr/bin/env python3
"""Build a self-contained HTML force-directed graph of Boardy pair data.

Default: the original 200 real pairs (data/dataset_positive.json +
dataset_negative.json, excluding any cmsynth* synthetic contacts). Nodes are
contacts; directed edges are searchQuery-labeled seeker->candidate interactions,
green for positive pairs and red for negative. Click a node to see its full
profile and every edge touching it.

With --compare <pairing-batch-dir>, renders two panes side by side: real pairs on
the left, a synthetic pairing batch (artifacts/pairing/<batch_id>/) on the right.
The batch is read directly from its own directory, not from the promoted dataset —
pairing batches are deliberately never promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POS = ROOT / "data" / "dataset_positive.json"
DEFAULT_NEG = ROOT / "data" / "dataset_negative.json"
DEFAULT_OUT = ROOT / "docs" / "real-pairs-graph.html"

PROFILE_FIELDS = [
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
]


def is_synthetic(contact_id: str) -> bool:
    return contact_id.startswith("cmsynth")


def real_only(contact_id: str) -> bool:
    return not is_synthetic(contact_id)


def build_graph(
    pos_pairs: list[dict],
    neg_pairs: list[dict],
    *,
    include: Callable[[str], bool] = real_only,
    node_meta: dict[str, dict] | None = None,
) -> dict:
    """Nodes = contacts, edges = seeker->candidate pairs.

    `include` decides which contact ids are kept; a pair is dropped unless BOTH
    sides pass. `node_meta` merges extra per-node fields (archetype, profile_id)
    for panes that have them.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    node_meta = node_meta or {}

    def touch_node(contact_id: str, file: dict, role: str) -> None:
        node = nodes.setdefault(
            contact_id,
            {"id": contact_id, "profile": {}, "roles": set(), "pairCount": 0},
        )
        node["roles"].add(role)
        node["pairCount"] += 1
        if not node["profile"]:
            node["profile"] = {k: file.get(k) for k in PROFILE_FIELDS}
        meta = node_meta.get(contact_id)
        if meta:
            node.setdefault("archetype", meta.get("archetype"))
            node.setdefault("profileId", meta.get("profile_id"))

    for label, pairs in (("pos", pos_pairs), ("neg", neg_pairs)):
        for p in pairs:
            seeker_id = p["userContactId"]
            cand_id = p["matchContactId"]
            if not include(seeker_id) or not include(cand_id):
                continue
            touch_node(seeker_id, p.get("userContactFile") or {}, "seeker")
            touch_node(cand_id, p.get("matchContactFile") or {}, "candidate")
            edges.append(
                {
                    "source": seeker_id,
                    "target": cand_id,
                    "label": label,
                    "searchQuery": p.get("searchQuery"),
                }
            )

    out_nodes = []
    for node in nodes.values():
        roles = node.pop("roles")
        node["role"] = "both" if len(roles) == 2 else next(iter(roles))
        out_nodes.append(node)

    return {"nodes": out_nodes, "edges": edges}


def load_pairing_batch(batch_dir: Path) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Read a synth_pipeline.pairing batch into (pos_pairs, neg_pairs, node_meta)."""
    batch_dir = Path(batch_dir)
    pairs_dir = batch_dir / "pairs"
    if not pairs_dir.is_dir():
        raise FileNotFoundError(f"no pairs/ directory under {batch_dir}")

    pos_pairs: list[dict] = []
    neg_pairs: list[dict] = []
    for path in sorted(pairs_dir.glob("*.json")):
        env = json.loads(path.read_text(encoding="utf-8"))
        pair = env.get("pair")
        if not pair:
            continue
        (pos_pairs if env.get("label") == "pos" else neg_pairs).append(pair)

    node_meta: dict[str, dict] = {}
    profiles_path = batch_dir / "profiles.json"
    if profiles_path.exists():
        node_meta = json.loads(profiles_path.read_text(encoding="utf-8"))

    return pos_pairs, neg_pairs, node_meta


def graph_stats(graph: dict) -> dict:
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "pos": sum(1 for e in graph["edges"] if e["label"] == "pos"),
        "neg": sum(1 for e in graph["edges"] if e["label"] == "neg"),
        "both_role": sum(1 for n in graph["nodes"] if n["role"] == "both"),
        "mean_degree": round(n_edges / n_nodes, 2) if n_nodes else 0.0,
    }


def suggest_physics(graph: dict) -> dict:
    """Scale repulsion/spring off mean degree.

    Baseline constants were tuned on the real graph (~1.35 edges/node — a forest
    of tiny stars). A pairing batch runs 5-10x denser, and reusing those constants
    there collapses the cluster into an unreadable ball.
    """
    stats = graph_stats(graph)
    ratio = max(1.0, stats["mean_degree"] / 1.35) if stats["nodes"] else 1.0
    return {
        "repulsion": round(4200 * ratio, 1),
        "springLen": round(90 * (ratio**0.5), 1),
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Pairs graph — Boardy AI</title>
<style>
  :root {
    --paper: #f4f6f3;
    --panel: #ffffff;
    --panel-glass: rgba(255, 255, 255, 0.82);
    --ink: #16211d;
    --muted: #5c6b64;
    --line: #ccd5cf;
    --accent: #2b6f77;
    --accent-soft: #d9e9ea;
    --pos: #2f9e5c;
    --pos-soft: #e0f2e6;
    --neg: #d1483f;
    --neg-soft: #f7e2df;
    --seeker: #3b6fb5;
    --candidate: #b8862c;
    --both: #8a3fb0;
    --shadow: 0 10px 30px rgba(20, 32, 28, 0.10);
    --sans: "Avenir Next", Avenir, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #121a17;
      --panel: #1a2320;
      --panel-glass: rgba(26, 35, 32, 0.82);
      --ink: #e7ece9;
      --muted: #93a39b;
      --line: #2d3934;
      --accent: #6ec0c9;
      --accent-soft: #223b3d;
      --pos: #52c581;
      --pos-soft: #1c3527;
      --neg: #ea7168;
      --neg-soft: #3a2320;
      --seeker: #7ea8e0;
      --candidate: #dcab5e;
      --both: #b787dd;
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
  }
  :root[data-theme="dark"] {
    --paper: #121a17;
    --panel: #1a2320;
    --panel-glass: rgba(26, 35, 32, 0.82);
    --ink: #e7ece9;
    --muted: #93a39b;
    --line: #2d3934;
    --accent: #6ec0c9;
    --accent-soft: #223b3d;
    --pos: #52c581;
    --pos-soft: #1c3527;
    --neg: #ea7168;
    --neg-soft: #3a2320;
    --seeker: #7ea8e0;
    --candidate: #dcab5e;
    --both: #b787dd;
    --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  }
  :root[data-theme="light"] {
    --paper: #f4f6f3;
    --panel: #ffffff;
    --panel-glass: rgba(255, 255, 255, 0.82);
    --ink: #16211d;
    --muted: #5c6b64;
    --line: #ccd5cf;
    --accent: #2b6f77;
    --accent-soft: #d9e9ea;
    --pos: #2f9e5c;
    --pos-soft: #e0f2e6;
    --neg: #d1483f;
    --neg-soft: #f7e2df;
    --seeker: #3b6fb5;
    --candidate: #b8862c;
    --both: #8a3fb0;
    --shadow: 0 10px 30px rgba(20, 32, 28, 0.10);
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    font-family: var(--sans);
    color: var(--ink);
    background: var(--paper);
    overflow: hidden;
  }
  .app { display: flex; height: 100vh; }
  .panes { flex: 1 1 auto; min-width: 0; display: grid; grid-template-columns: 1fr; }
  .panes.split { grid-template-columns: 1fr 1fr; }
  .pane { position: relative; min-width: 0; overflow: hidden; }
  .panes.split .pane + .pane { border-left: 1px solid var(--line); }
  .pane svg { width: 100%; height: 100%; display: block; cursor: grab; background: var(--paper); }
  .pane svg.panning { cursor: grabbing; }

  header.hero {
    position: absolute;
    top: 0; left: 0; right: 0;
    padding: 1rem 1.25rem 0.6rem;
    pointer-events: none;
    z-index: 5;
  }
  header.hero .masthead {
    display: inline-flex;
    flex-direction: column;
    gap: 0.35rem;
    background: var(--panel-glass);
    backdrop-filter: blur(6px);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    box-shadow: var(--shadow);
    max-width: min(46rem, 88%);
  }
  header.hero h1 {
    font-family: var(--serif);
    font-size: 1.2rem;
    line-height: 1.15;
    margin: 0;
    letter-spacing: -0.01em;
    text-wrap: balance;
  }
  header.hero p { margin: 0; color: var(--muted); font-size: 0.8rem; line-height: 1.4; }
  header.hero .stats {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.01em;
  }

  .toolbar {
    position: absolute;
    top: 1rem; right: 1.25rem;
    display: flex;
    gap: 0.75rem;
    align-items: center;
    background: var(--panel-glass);
    backdrop-filter: blur(6px);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.45rem 0.8rem;
    box-shadow: var(--shadow);
    z-index: 6;
    pointer-events: auto;
    flex-wrap: wrap;
    justify-content: flex-end;
    max-width: 70%;
  }
  .toolbar label {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    color: var(--ink);
    white-space: nowrap;
    cursor: pointer;
  }
  .toolbar input[type="checkbox"] { accent-color: var(--accent); width: 0.9rem; height: 0.9rem; }
  .toolbar input[type="search"] {
    border: 1px solid var(--line);
    background: var(--panel);
    color: var(--ink);
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font: inherit;
    font-size: 0.78rem;
    width: 9rem;
  }
  .toolbar .fit-btn {
    border: 1px solid var(--line);
    background: var(--panel);
    color: var(--ink);
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font: inherit;
    font-size: 0.76rem;
    cursor: pointer;
  }
  .toolbar .fit-btn:hover { border-color: var(--accent); color: var(--accent); }
  .toolbar input:focus-visible,
  .panel button:focus-visible,
  .toolbar .fit-btn:focus-visible,
  .node:focus-visible circle {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .legend-dot {
    display: inline-block;
    width: 0.55rem; height: 0.55rem;
    border-radius: 50%;
    flex: 0 0 auto;
  }
  .legend-archetypes {
    position: absolute;
    left: 1.25rem;
    bottom: 3.2rem;
    z-index: 5;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    background: var(--panel-glass);
    backdrop-filter: blur(6px);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    font-size: 0.72rem;
    color: var(--muted);
    max-width: 20rem;
    max-height: 40vh;
    overflow-y: auto;
  }
  .legend-archetypes:empty { display: none; }
  .legend-archetypes div { display: flex; align-items: center; gap: 0.4rem; }
  .legend-archetypes span.txt { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .panel {
    width: min(400px, 34vw);
    flex: 0 0 auto;
    background: var(--panel);
    border-left: 1px solid var(--line);
    overflow-y: auto;
    padding: 1.25rem 1.3rem 2rem;
    display: none;
  }
  .panel.open { display: block; }
  .panel h2 {
    font-family: var(--serif);
    font-size: 1.2rem;
    line-height: 1.25;
    margin: 0 0 0.3rem;
    text-wrap: balance;
  }
  .panel .id {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    word-break: break-all;
    margin-bottom: 0.9rem;
  }
  .panel .role-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    margin-bottom: 0.9rem;
  }
  .panel .source-chip {
    display: inline-flex;
    font-family: var(--mono);
    font-size: 0.68rem;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    margin: 0 0 0.9rem 0.35rem;
    background: var(--accent-soft);
    color: var(--accent);
  }
  .panel .close-btn {
    float: right;
    border: 1px solid var(--line);
    background: var(--paper);
    color: var(--ink);
    border-radius: 999px;
    width: 1.9rem; height: 1.9rem;
    cursor: pointer;
    font-size: 1.05rem;
    line-height: 1;
  }
  .field { margin-bottom: 1rem; max-width: 34rem; }
  .field h3 {
    margin: 0 0 0.3rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    font-weight: 700;
  }
  .field p { margin: 0; font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .edge-list { display: grid; gap: 0.55rem; padding: 0; margin: 0; list-style: none; }
  .edge-list li {
    border-left: 3px solid var(--line);
    background: var(--paper);
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 0.7rem;
    font-size: 0.84rem;
    line-height: 1.45;
  }
  .edge-list li.pos { border-color: var(--pos); background: var(--pos-soft); }
  .edge-list li.neg { border-color: var(--neg); background: var(--neg-soft); }
  .edge-list .dir {
    font-family: var(--mono);
    font-weight: 600;
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
  }

  .tooltip {
    position: absolute;
    pointer-events: none;
    background: var(--ink);
    color: var(--paper);
    font-size: 0.78rem;
    padding: 0.4rem 0.6rem;
    border-radius: 8px;
    max-width: 320px;
    z-index: 10;
    display: none;
    line-height: 1.35;
  }

  .credit {
    position: absolute;
    left: 1.25rem;
    bottom: 1rem;
    z-index: 5;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    background: var(--panel-glass);
    backdrop-filter: blur(6px);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.35rem 0.8rem;
    pointer-events: auto;
  }
  .credit a { color: var(--accent); text-decoration: none; }
  .credit a:hover,
  .credit a:focus-visible { text-decoration: underline; }

  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="panes" id="panes"></div>
  <aside class="panel" id="panel">
    <button class="close-btn" id="panelClose">&times;</button>
    <h2 id="panelTitle">Contact</h2>
    <div class="id" id="panelId"></div>
    <div id="panelRole"></div>
    <div id="panelBody"></div>
  </aside>
</div>

<template id="pane-tpl">
  <div class="pane">
    <header class="hero">
      <div class="masthead">
        <h1 class="title"></h1>
        <p class="subtitle"></p>
        <div class="stats"></div>
      </div>
    </header>
    <div class="toolbar">
      <label><input type="checkbox" class="showPos" checked /> <span class="legend-dot" style="background:var(--pos)"></span> positive</label>
      <label><input type="checkbox" class="showNeg" checked /> <span class="legend-dot" style="background:var(--neg)"></span> negative</label>
      <span class="role-legend"></span>
      <input type="search" class="search" placeholder="Filter…" autocomplete="off" />
      <button class="fit-btn" type="button">Fit</button>
    </div>
    <svg class="graph"></svg>
    <div class="legend-archetypes"></div>
    <div class="tooltip"></div>
  </div>
</template>

<script id="graph-data" type="application/json">__EMBEDDED_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById("graph-data").textContent);
const PANES = DATA.panes;
const svgNS = "http://www.w3.org/2000/svg";
const ROLE_COLOR = { seeker: "var(--seeker)", candidate: "var(--candidate)", both: "var(--both)" };
// Categorical ramp for archetype coloring — the synth pane's `role` is degenerate
// (almost every node is "both"), so role coloring carries no information there.
const ARCHETYPE_COLORS = [
  "#3b6fb5", "#b8862c", "#8a3fb0", "#2f9e5c",
  "#d1483f", "#2b6f77", "#c2557f", "#7a6a3a",
];

const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// --- shared detail panel (one for the whole app; two 380px asides in a split
// view would be unusable) ---
const panel = document.getElementById("panel");
document.getElementById("panelClose").addEventListener("click", () => panel.classList.remove("open"));

function fieldHtml(label, value) {
  const v = (value || "").toString().trim();
  if (!v) return "";
  return `<div class="field"><h3>${escapeHtml(label)}</h3><p>${escapeHtml(v)}</p></div>`;
}

function openPanel(n, spec, edges) {
  document.getElementById("panelTitle").textContent =
    (n.profile.positioning || n.id).toString().replace(/\s+/g, " ").slice(0, 60);
  document.getElementById("panelId").textContent = n.id;
  const roleLabel = { seeker: "Seeker only", candidate: "Candidate only", both: "Both seeker & candidate" }[n.role];
  const roleVar = ROLE_COLOR[n.role];
  const archetypeChip = n.archetype
    ? `<span class="source-chip">${escapeHtml(n.archetype)}</span>` : "";
  document.getElementById("panelRole").innerHTML =
    `<span class="role-chip" style="background:color-mix(in srgb, ${roleVar} 18%, transparent);color:${roleVar}">${roleLabel}</span>`
    + `<span class="source-chip">${escapeHtml(spec.title)}</span>` + archetypeChip;

  const asSeeker = edges.filter(e => e.source === n.id);
  const asCandidate = edges.filter(e => e.target === n.id);
  const edgeItem = (e, dirLabel) => `
    <li class="${e.label}">
      <div class="dir">${escapeHtml(dirLabel)}</div>
      ${escapeHtml(e.searchQuery || "")}
    </li>`;

  document.getElementById("panelBody").innerHTML = [
    fieldHtml("Positioning", n.profile.positioning),
    fieldHtml("Background", n.profile.background),
    fieldHtml("Looking for", n.profile.lookingFor),
    fieldHtml("Location / availability", n.profile.locationAvailability),
    fieldHtml("Notes", n.profile.notes),
    fieldHtml("Intro preferences", n.profile.introPreferences),
    fieldHtml("Personal preferences", n.profile.personalPreferences),
    fieldHtml("Meeting & scheduling", n.profile.meetingAndSchedulingPreferences),
    asSeeker.length ? `<div class="field"><h3>As seeker (${asSeeker.length})</h3><ul class="edge-list">${asSeeker.map(e => edgeItem(e, "→ " + e.target)).join("")}</ul></div>` : "",
    asCandidate.length ? `<div class="field"><h3>As candidate (${asCandidate.length})</h3><ul class="edge-list">${asCandidate.map(e => edgeItem(e, e.source + " →")).join("")}</ul></div>` : "",
  ].join("");
  panel.classList.add("open");
}

// --- one independent graph instance per pane ---
function createGraph(root, spec) {
  const nodes = spec.nodes.map(n => ({...n, x: (Math.random()-0.5)*900, y: (Math.random()-0.5)*900, vx: 0, vy: 0}));
  const nodeById = new Map(nodes.map(n => [n.id, n]));
  const edges = spec.edges.filter(e => nodeById.has(e.source) && nodeById.has(e.target));

  // --- connected components (undirected) so separate clusters repel each other ---
  const compOf = new Map();
  {
    const adj = new Map(nodes.map(n => [n.id, []]));
    for (const e of edges) { adj.get(e.source).push(e.target); adj.get(e.target).push(e.source); }
    let compId = 0;
    for (const n of nodes) {
      if (compOf.has(n.id)) continue;
      const stack = [n.id];
      compOf.set(n.id, compId);
      while (stack.length) {
        const cur = stack.pop();
        for (const nb of adj.get(cur)) {
          if (!compOf.has(nb)) { compOf.set(nb, compId); stack.push(nb); }
        }
      }
      compId++;
    }
  }

  // --- physics (per-pane: a dense pairing batch needs more repulsion than the
  // sparse real graph, or it collapses into an unreadable ball) ---
  const P = spec.physics || {};
  const REPULSION = P.repulsion ?? 4200;
  const SPRING_LEN = P.springLen ?? 90;
  const CROSS_COMPONENT_BOOST = 3.2;
  const SPRING = 0.012;
  const CENTER = 0.0006;
  const DAMP = 0.85;
  const ALPHA_DECAY = 0.996;
  const ALPHA_MIN = 0.02;

  let alpha = 1;
  let simRunning = false;

  function tickPhysics(strength) {
    for (const n of nodes) { n.fx = 0; n.fy = 0; }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx*dx + dy*dy + 0.01;
        let d = Math.sqrt(d2);
        const boost = compOf.get(a.id) === compOf.get(b.id) ? 1 : CROSS_COMPONENT_BOOST;
        const f = (REPULSION * boost) / d2;
        dx /= d; dy /= d;
        a.fx += dx * f; a.fy += dy * f;
        b.fx -= dx * f; b.fy -= dy * f;
      }
    }
    for (const e of edges) {
      const a = nodeById.get(e.source), b = nodeById.get(e.target);
      let dx = b.x - a.x, dy = b.y - a.y;
      let d = Math.sqrt(dx*dx + dy*dy) || 0.01;
      const f = (d - SPRING_LEN) * SPRING;
      dx /= d; dy /= d;
      a.fx += dx * f; a.fy += dy * f;
      b.fx -= dx * f; b.fy -= dy * f;
    }
    for (const n of nodes) {
      n.fx -= n.x * CENTER;
      n.fy -= n.y * CENTER;
      if (n.pinned) { n.vx = 0; n.vy = 0; continue; }
      n.vx = (n.vx + n.fx * strength) * DAMP;
      n.vy = (n.vy + n.fy * strength) * DAMP;
      n.x += n.vx;
      n.y += n.vy;
    }
  }

  function runSimulationLoop() {
    if (!simRunning) return;
    tickPhysics(alpha);
    render();
    alpha *= ALPHA_DECAY;
    if (alpha < ALPHA_MIN) { simRunning = false; fitView(); return; }
    requestAnimationFrame(runSimulationLoop);
  }

  function reheat(amount = 0.7) {
    alpha = Math.max(alpha, amount);
    if (!simRunning) { simRunning = true; requestAnimationFrame(runSimulationLoop); }
  }

  // --- svg scaffold ---
  const svg = root.querySelector("svg.graph");
  let viewBox = { x: -600, y: -600, w: 1200, h: 1200 };
  function applyViewBox() {
    svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
  }
  applyViewBox();

  // Marker ids must be pane-unique: url(#id) resolves document-wide to the FIRST
  // match, so a shared id would silently render pane 2 with pane 1's arrowheads.
  const posMarker = `arrow-pos-${spec.key}`;
  const negMarker = `arrow-neg-${spec.key}`;
  svg.innerHTML = `
    <defs>
      <marker id="${posMarker}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--pos)"></path>
      </marker>
      <marker id="${negMarker}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--neg)"></path>
      </marker>
    </defs>
    <g class="edgeLayer"></g>
    <g class="nodeLayer"></g>
  `;
  const edgeLayer = svg.querySelector(".edgeLayer");
  const nodeLayer = svg.querySelector(".nodeLayer");

  // --- parallel-edge curvature: A->B and B->A both exist often on a dense pane,
  // and as straight lines they overlap exactly, making the picture understate the
  // edge count. Offset each member of a group onto its own arc. ---
  const groupIndex = new Map();
  for (const e of edges) {
    const key = [e.source, e.target].sort().join("|");
    const list = groupIndex.get(key) || [];
    list.push(e);
    groupIndex.set(key, list);
  }
  const curveOf = new Map();
  for (const list of groupIndex.values()) {
    const m = list.length;
    list.forEach((e, k) => curveOf.set(e, m === 1 ? 0 : (k - (m - 1) / 2) * 18));
  }

  function edgePath(e) {
    const a = nodeById.get(e.source), b = nodeById.get(e.target);
    const curve = curveOf.get(e) || 0;
    if (!curve) return `M${a.x},${a.y} L${b.x},${b.y}`;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    let dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx*dx + dy*dy) || 1;
    const nx = -dy / d, ny = dx / d;
    return `M${a.x},${a.y} Q${mx + nx*curve},${my + ny*curve} ${b.x},${b.y}`;
  }

  const edgeEls = edges.map(e => {
    const path = document.createElementNS(svgNS, "path");
    path.style.stroke = e.label === "pos" ? "var(--pos)" : "var(--neg)";
    path.setAttribute("fill", "none");
    path.setAttribute("stroke-width", "2.4");
    path.setAttribute("stroke-opacity", "0.55");
    path.setAttribute("marker-end", `url(#${e.label === "pos" ? posMarker : negMarker})`);
    path.style.pointerEvents = "none";
    edgeLayer.appendChild(path);

    // invisible wide hit-path so hovering is easy without a visually fat edge
    const hit = document.createElementNS(svgNS, "path");
    hit.setAttribute("stroke", "transparent");
    hit.setAttribute("fill", "none");
    hit.setAttribute("stroke-width", "14");
    hit.style.cursor = "pointer";
    hit.addEventListener("mouseenter", (ev) => showTooltip(ev, e));
    hit.addEventListener("mousemove", moveTooltip);
    hit.addEventListener("mouseleave", hideTooltip);
    edgeLayer.appendChild(hit);
    return { e, path, hit };
  });

  // --- node coloring ---
  const archetypes = [...new Set(nodes.map(n => n.archetype).filter(Boolean))].sort();
  const archetypeColor = new Map(archetypes.map((a, i) => [a, ARCHETYPE_COLORS[i % ARCHETYPE_COLORS.length]]));
  const colorBy = spec.colorBy === "archetype" && archetypes.length ? "archetype" : "role";
  function nodeColor(n) {
    if (colorBy === "archetype") return archetypeColor.get(n.archetype) || "var(--muted)";
    return ROLE_COLOR[n.role];
  }

  let draggingNode = null;
  const nodeEls = nodes.map(n => {
    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("class", "node");
    g.setAttribute("tabindex", "0");
    g.setAttribute("role", "button");
    g.setAttribute("aria-label", n.id);
    g.style.cursor = "pointer";
    const r = 5 + Math.min(n.pairCount, 8);
    const circle = document.createElementNS(svgNS, "circle");
    circle.setAttribute("r", r);
    circle.style.fill = nodeColor(n);
    circle.style.stroke = "var(--panel)";
    circle.setAttribute("stroke-width", "1.2");
    g.appendChild(circle);
    g.addEventListener("click", (ev) => { ev.stopPropagation(); openPanel(n, spec, edges); });
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openPanel(n, spec, edges); } });
    g.addEventListener("mouseenter", (ev) => showTooltip(ev, null, n));
    g.addEventListener("mousemove", moveTooltip);
    g.addEventListener("mouseleave", hideTooltip);
    g.addEventListener("mousedown", (ev) => {
      draggingNode = n;
      n.pinned = true;
      n.vx = 0; n.vy = 0;
      reheat();
      ev.stopPropagation();
    });
    nodeLayer.appendChild(g);
    return { n, g };
  });

  // --- header + legends ---
  root.querySelector(".title").textContent = spec.title;
  root.querySelector(".subtitle").textContent = spec.subtitle || "";
  root.querySelector(".stats").textContent = spec.stats;

  const roleLegend = root.querySelector(".role-legend");
  if (colorBy === "role") {
    roleLegend.innerHTML = ["seeker", "candidate", "both"].map(rk =>
      `<label><span class="legend-dot" style="background:${ROLE_COLOR[rk]}"></span> ${rk}</label>`
    ).join("");
    roleLegend.style.display = "inline-flex";
    roleLegend.style.gap = "0.75rem";
  }
  const archLegend = root.querySelector(".legend-archetypes");
  if (colorBy === "archetype") {
    archLegend.innerHTML = archetypes.map(a =>
      `<div><span class="legend-dot" style="background:${archetypeColor.get(a)}"></span><span class="txt" title="${escapeHtml(a)}">${escapeHtml(a)}</span></div>`
    ).join("");
  }

  // --- filter ---
  const showPos = root.querySelector(".showPos");
  const showNeg = root.querySelector(".showNeg");
  const search = root.querySelector(".search");
  function haystack(n) {
    const p = n.profile || {};
    return [n.id, n.archetype, p.positioning, p.background, p.lookingFor, p.notes].filter(Boolean).join("\n").toLowerCase();
  }
  function passesFilter(n) {
    const term = search.value.trim().toLowerCase();
    if (!term) return true;
    return haystack(n).includes(term);
  }
  showPos.addEventListener("change", render);
  showNeg.addEventListener("change", render);
  search.addEventListener("input", render);

  // --- tooltip ---
  const tooltip = root.querySelector(".tooltip");
  function showTooltip(ev, edge, node) {
    if (edge) {
      const dir = `${escapeHtml(edge.source)} &rarr; ${escapeHtml(edge.target)}`;
      tooltip.innerHTML = `<strong>${edge.label === "pos" ? "Positive" : "Negative"}</strong> · ${dir}<br>${escapeHtml(edge.searchQuery || "")}`;
    } else if (node) {
      const arch = node.archetype ? `<br>${escapeHtml(node.archetype)}` : "";
      tooltip.innerHTML = `<strong>${escapeHtml(node.id)}</strong><br>role: ${node.role} · ${node.pairCount} pair edge(s)${arch}`;
    }
    tooltip.style.display = "block";
    moveTooltip(ev);
  }
  function moveTooltip(ev) {
    const rect = root.getBoundingClientRect();
    tooltip.style.left = (ev.clientX - rect.left + 14) + "px";
    tooltip.style.top = (ev.clientY - rect.top + 14) + "px";
  }
  function hideTooltip() { tooltip.style.display = "none"; }

  function render() {
    for (const { e, path, hit } of edgeEls) {
      const a = nodeById.get(e.source), b = nodeById.get(e.target);
      const d = edgePath(e);
      path.setAttribute("d", d);
      hit.setAttribute("d", d);
      const visible = (e.label === "pos" ? showPos.checked : showNeg.checked) && passesFilter(a) && passesFilter(b);
      path.style.display = visible ? "" : "none";
      hit.style.display = visible ? "" : "none";
    }
    for (const { n, g } of nodeEls) {
      g.setAttribute("transform", `translate(${n.x},${n.y})`);
      g.style.display = passesFilter(n) ? "" : "none";
    }
  }

  // --- fit to content: with the viewBox tracking the actual node bbox, the
  // ABSOLUTE scale of REPULSION stops mattering (only its ratio to SPRING does),
  // which is what lets two very differently-sized graphs share one code path. ---
  function fitView(pad = 90) {
    if (!nodes.length) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x > maxX) maxX = n.x;
      if (n.y > maxY) maxY = n.y;
    }
    const w = Math.max(maxX - minX, 1) + pad * 2;
    const h = Math.max(maxY - minY, 1) + pad * 2;
    const rect = svg.getBoundingClientRect();
    const aspect = (rect.width && rect.height) ? rect.width / rect.height : 1;
    let vw = w, vh = h;
    if (w / h > aspect) vh = w / aspect; else vw = h * aspect;
    viewBox = { x: (minX + maxX) / 2 - vw / 2, y: (minY + maxY) / 2 - vh / 2, w: vw, h: vh };
    applyViewBox();
  }
  root.querySelector(".fit-btn").addEventListener("click", () => fitView());

  // --- pan / zoom ---
  let panning = false, panStart = null, viewStart = null;
  svg.addEventListener("mousedown", (ev) => {
    if (ev.target.closest(".node")) return;
    panning = true;
    svg.classList.add("panning");
    panStart = { x: ev.clientX, y: ev.clientY };
    viewStart = { ...viewBox };
  });
  // One delegated pair of listeners per pane, not one pair per node — the old
  // per-node binding fired every node's handler on every mouse move.
  root.addEventListener("mousemove", (ev) => {
    if (draggingNode) {
      const rect = svg.getBoundingClientRect();
      const scale = viewBox.w / rect.width;
      draggingNode.x += ev.movementX * scale;
      draggingNode.y += ev.movementY * scale;
      if (!simRunning) render();
      return;
    }
    if (!panning) return;
    const rect = svg.getBoundingClientRect();
    const scale = viewBox.w / rect.width;
    viewBox.x = viewStart.x - (ev.clientX - panStart.x) * scale;
    viewBox.y = viewStart.y - (ev.clientY - panStart.y) * scale;
    applyViewBox();
  });
  window.addEventListener("mouseup", () => {
    if (draggingNode) { draggingNode.pinned = false; draggingNode = null; reheat(); }
    panning = false;
    svg.classList.remove("panning");
  });
  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const rect = svg.getBoundingClientRect();
    const mx = viewBox.x + (ev.clientX - rect.left) / rect.width * viewBox.w;
    const my = viewBox.y + (ev.clientY - rect.top) / rect.height * viewBox.h;
    const factor = ev.deltaY > 0 ? 1.1 : 0.9;
    viewBox.x = mx - (mx - viewBox.x) * factor;
    viewBox.y = my - (my - viewBox.y) * factor;
    viewBox.w *= factor;
    viewBox.h *= factor;
    applyViewBox();
  }, { passive: false });

  // Small graphs settle instantly; only the big pane is worth animating.
  if (REDUCED_MOTION || nodes.length < 60) {
    for (let it = 0; it < 400; it++) tickPhysics(1);
    render();
    requestAnimationFrame(() => fitView());
  } else {
    render();
    reheat(1);
  }

  return { nodes, edges, render, reheat, fitView };
}

const panesEl = document.getElementById("panes");
if (PANES.length > 1) panesEl.classList.add("split");
const tpl = document.getElementById("pane-tpl");
for (const spec of PANES) {
  const frag = tpl.content.cloneNode(true);
  const root = frag.querySelector(".pane");
  panesEl.appendChild(frag);
  createGraph(root, spec);
}
</script>
</body>
</html>
"""


def _pane_spec(
    key: str,
    title: str,
    subtitle: str,
    graph: dict,
    *,
    color_by: str = "role",
) -> dict:
    stats = graph_stats(graph)
    return {
        "key": key,
        "title": title,
        "subtitle": subtitle,
        "colorBy": color_by,
        "physics": suggest_physics(graph),
        "stats": (
            f"{stats['nodes']} contacts · {stats['edges']} pairs "
            f"({stats['pos']} pos / {stats['neg']} neg) · "
            f"{stats['mean_degree']} edges/node · {stats['both_role']} both-role"
        ),
        "nodes": graph["nodes"],
        "edges": graph["edges"],
    }


def build_panes(specs: list[dict], out_path: Path) -> dict:
    payload = json.dumps({"panes": specs}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {
        "panes": [
            {"key": s["key"], "nodes": len(s["nodes"]), "edges": len(s["edges"])}
            for s in specs
        ]
    }


def build(
    pos_path: Path,
    neg_path: Path,
    out_path: Path,
    *,
    compare: Path | None = None,
) -> dict:
    pos_pairs = json.loads(pos_path.read_text(encoding="utf-8"))
    neg_pairs = json.loads(neg_path.read_text(encoding="utf-8"))
    real = build_graph(pos_pairs, neg_pairs)

    specs = [
        _pane_spec(
            "real",
            "Real pairs",
            "Original Boardy seeker→candidate pairs, synthetic contacts excluded.",
            real,
        )
    ]

    if compare is not None:
        s_pos, s_neg, node_meta = load_pairing_batch(compare)
        synth = build_graph(
            s_pos, s_neg, include=lambda cid: True, node_meta=node_meta
        )
        specs.append(
            _pane_spec(
                "synth",
                f"Synthetic pairs — {Path(compare).name}",
                "Profile-first generated contacts, paired and labeled by the "
                "TF-IDF+Voyage-nano scorer. Colored by archetype.",
                synth,
                color_by="archetype",
            )
        )

    return build_panes(specs, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=DEFAULT_POS)
    parser.add_argument("--neg", type=Path, default=DEFAULT_NEG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="a synth_pipeline.pairing batch dir (artifacts/pairing/<batch_id>) "
        "to render as a second pane",
    )
    args = parser.parse_args()
    stats = build(args.pos, args.neg, args.out, compare=args.compare)
    summary = ", ".join(
        f"{p['key']}: {p['nodes']} nodes/{p['edges']} edges" for p in stats["panes"]
    )
    print(f"Wrote {args.out} — {summary}.")


if __name__ == "__main__":
    main()
