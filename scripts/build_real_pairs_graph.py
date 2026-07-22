#!/usr/bin/env python3
"""Build a self-contained HTML force-directed graph of the original 200 real
pairs (data/dataset_positive.json + dataset_negative.json, excluding any
cmsynth* synthetic contacts). Nodes are contacts; directed edges are
searchQuery-labeled seeker->candidate interactions, green for positive pairs
and red for negative. Click a node to see its full profile and every edge
touching it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def build_graph(pos_pairs: list[dict], neg_pairs: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def touch_node(contact_id: str, file: dict, role: str) -> None:
        node = nodes.setdefault(
            contact_id,
            {"id": contact_id, "profile": {}, "roles": set(), "pairCount": 0},
        )
        node["roles"].add(role)
        node["pairCount"] += 1
        if not node["profile"]:
            node["profile"] = {k: file.get(k) for k in PROFILE_FIELDS}

    for label, pairs in (("pos", pos_pairs), ("neg", neg_pairs)):
        for p in pairs:
            seeker_id = p["userContactId"]
            cand_id = p["matchContactId"]
            if is_synthetic(seeker_id) or is_synthetic(cand_id):
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

    node_list = []
    for n in nodes.values():
        roles = n["roles"]
        role = "both" if len(roles) == 2 else next(iter(roles))
        node_list.append(
            {
                "id": n["id"],
                "profile": n["profile"],
                "role": role,
                "pairCount": n["pairCount"],
            }
        )

    return {"nodes": node_list, "edges": edges}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Real pairs graph — Dorby AI</title>
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
  #app { display: flex; height: 100vh; }
  #canvas-wrap { position: relative; flex: 1 1 auto; min-width: 0; }
  svg { width: 100%; height: 100%; display: block; cursor: grab; background: var(--paper); }
  svg.panning { cursor: grabbing; }

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
    max-width: min(46rem, 62vw);
  }
  header.hero h1 {
    font-family: var(--serif);
    font-size: 1.35rem;
    line-height: 1.15;
    margin: 0;
    letter-spacing: -0.01em;
    text-wrap: balance;
  }
  header.hero p { margin: 0; color: var(--muted); font-size: 0.83rem; line-height: 1.4; }
  header.hero .stats {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.74rem;
    color: var(--accent);
    letter-spacing: 0.01em;
  }

  .toolbar {
    position: absolute;
    top: 1rem; right: 1.25rem;
    display: flex;
    gap: 0.9rem;
    align-items: center;
    background: var(--panel-glass);
    backdrop-filter: blur(6px);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.5rem 0.85rem;
    box-shadow: var(--shadow);
    z-index: 6;
    pointer-events: auto;
    flex-wrap: wrap;
    max-width: min(48vw, 580px);
  }
  .toolbar label {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    color: var(--ink);
    white-space: nowrap;
    cursor: pointer;
  }
  .toolbar input[type="checkbox"] { accent-color: var(--accent); width: 0.95rem; height: 0.95rem; }
  .toolbar input[type="search"] {
    border: 1px solid var(--line);
    background: var(--panel);
    color: var(--ink);
    border-radius: 999px;
    padding: 0.32rem 0.75rem;
    font: inherit;
    font-size: 0.82rem;
    width: 12rem;
  }
  .toolbar input:focus-visible,
  #panel button:focus-visible,
  .node:focus-visible circle {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .legend-dot {
    display: inline-block;
    width: 0.6rem; height: 0.6rem;
    border-radius: 50%;
    flex: 0 0 auto;
  }

  #panel {
    width: min(400px, 38vw);
    flex: 0 0 auto;
    background: var(--panel);
    border-left: 1px solid var(--line);
    overflow-y: auto;
    padding: 1.25rem 1.3rem 2rem;
    display: none;
  }
  #panel.open { display: block; }
  #panel h2 {
    font-family: var(--serif);
    font-size: 1.2rem;
    line-height: 1.25;
    margin: 0 0 0.3rem;
    text-wrap: balance;
  }
  #panel .id {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    word-break: break-all;
    margin-bottom: 0.9rem;
  }
  #panel .role-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    margin-bottom: 0.9rem;
  }
  #panel .close-btn {
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

  #tooltip {
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

  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
</style>
</head>
<body>
<div id="app">
  <div id="canvas-wrap">
    <header class="hero">
      <div class="masthead">
        <h1>Real pairs graph</h1>
        <p>Original real seeker&rarr;candidate pairs, synthetic contacts excluded. Drag nodes, scroll to zoom, click a node for its profile.</p>
        <div class="stats" id="statsLine"></div>
      </div>
    </header>
    <div class="toolbar">
      <label><input type="checkbox" id="showPos" checked /> <span class="legend-dot" style="background:var(--pos)"></span> positive</label>
      <label><input type="checkbox" id="showNeg" checked /> <span class="legend-dot" style="background:var(--neg)"></span> negative</label>
      <label><span class="legend-dot" style="background:var(--seeker)"></span> seeker</label>
      <label><span class="legend-dot" style="background:var(--candidate)"></span> candidate</label>
      <label><span class="legend-dot" style="background:var(--both)"></span> both</label>
      <input type="search" id="search" placeholder="Filter by id or text…" autocomplete="off" />
    </div>
    <svg id="svg"></svg>
    <div id="tooltip"></div>
  </div>
  <aside id="panel">
    <button class="close-btn" id="panelClose">&times;</button>
    <h2 id="panelTitle">Contact</h2>
    <div class="id" id="panelId"></div>
    <div id="panelRole"></div>
    <div id="panelBody"></div>
  </aside>
</div>

<script id="graph-data" type="application/json">__EMBEDDED_JSON__</script>
<script>
const GRAPH = JSON.parse(document.getElementById("graph-data").textContent);
const nodes = GRAPH.nodes.map(n => ({...n, x: (Math.random()-0.5)*900, y: (Math.random()-0.5)*900, vx: 0, vy: 0}));
const nodeById = new Map(nodes.map(n => [n.id, n]));
const edges = GRAPH.edges.filter(e => nodeById.has(e.source) && nodeById.has(e.target));

const ROLE_COLOR = { seeker: "var(--seeker)", candidate: "var(--candidate)", both: "var(--both)" };

// --- connected components (undirected) so separate clusters can repel each other, Obsidian-style ---
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

// --- live force simulation: attraction along edges, repulsion between all nodes
// (boosted between different connected components so isolated clusters drift apart),
// mild global centering so the whole graph doesn't fly off-screen. Alpha decays to
// rest, and reheats when a node is dragged, so it behaves like a real force field
// rather than a one-shot static layout. ---
const REPULSION = 4200;
const CROSS_COMPONENT_BOOST = 3.2;
const SPRING = 0.012;
const SPRING_LEN = 90;
const CENTER = 0.0006;
const DAMP = 0.85;
const ALPHA_DECAY = 0.996;
const ALPHA_MIN = 0.02;
const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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
  if (alpha < ALPHA_MIN) { simRunning = false; return; }
  requestAnimationFrame(runSimulationLoop);
}

function reheat(amount = 0.7) {
  alpha = Math.max(alpha, amount);
  if (!simRunning) { simRunning = true; requestAnimationFrame(runSimulationLoop); }
}

if (REDUCED_MOTION) {
  // settle instantly, no animation loop
  for (let it = 0; it < 400; it++) tickPhysics(1);
} else {
  reheat(1);
}

// --- svg setup ---
const svg = document.getElementById("svg");
const svgNS = "http://www.w3.org/2000/svg";
let viewBox = { x: -600, y: -600, w: 1200, h: 1200 };
function applyViewBox() {
  svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
}
applyViewBox();

svg.innerHTML = `
  <defs>
    <marker id="arrow-pos" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" style="fill:var(--pos)"></path>
    </marker>
    <marker id="arrow-neg" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" style="fill:var(--neg)"></path>
    </marker>
  </defs>
  <g id="edgeLayer"></g>
  <g id="nodeLayer"></g>
`;
const edgeLayer = document.getElementById("edgeLayer");
const nodeLayer = document.getElementById("nodeLayer");

const edgeEls = edges.map(e => {
  // visible edge: thicker stroke + a large arrowhead marks the search direction (seeker -> candidate)
  const line = document.createElementNS(svgNS, "line");
  line.style.stroke = e.label === "pos" ? "var(--pos)" : "var(--neg)";
  line.setAttribute("stroke-width", "2.4");
  line.setAttribute("stroke-opacity", "0.55");
  line.setAttribute("marker-end", e.label === "pos" ? "url(#arrow-pos)" : "url(#arrow-neg)");
  line.style.pointerEvents = "none";
  edgeLayer.appendChild(line);

  // invisible wide hit-line layered on top so hovering is easy without a visually fat edge
  const hit = document.createElementNS(svgNS, "line");
  hit.setAttribute("stroke", "transparent");
  hit.setAttribute("stroke-width", "14");
  hit.style.cursor = "pointer";
  hit.addEventListener("mouseenter", (ev) => showTooltip(ev, e));
  hit.addEventListener("mousemove", (ev) => moveTooltip(ev));
  hit.addEventListener("mouseleave", hideTooltip);
  edgeLayer.appendChild(hit);

  return { e, line, hit };
});

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
  circle.style.fill = ROLE_COLOR[n.role];
  circle.style.stroke = "var(--panel)";
  circle.setAttribute("stroke-width", "1.2");
  g.appendChild(circle);
  g.addEventListener("click", (ev) => { ev.stopPropagation(); openPanel(n); });
  g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openPanel(n); } });
  g.addEventListener("mouseenter", (ev) => showTooltip(ev, null, n));
  g.addEventListener("mousemove", (ev) => moveTooltip(ev));
  g.addEventListener("mouseleave", hideTooltip);
  makeDraggable(g, n);
  nodeLayer.appendChild(g);
  return { n, g, r };
});

document.getElementById("statsLine").textContent =
  `${nodes.length} contacts · ${edges.length} pairs · ${nodes.filter(n => n.role === "both").length} both-role`;

function render() {
  for (const { e, line, hit } of edgeEls) {
    const a = nodeById.get(e.source), b = nodeById.get(e.target);
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    hit.setAttribute("x1", a.x); hit.setAttribute("y1", a.y);
    hit.setAttribute("x2", b.x); hit.setAttribute("y2", b.y);
    const visible = (e.label === "pos" ? showPos.checked : showNeg.checked) && passesFilter(a) && passesFilter(b);
    line.style.display = visible ? "" : "none";
    hit.style.display = visible ? "" : "none";
  }
  for (const { n, g } of nodeEls) {
    g.setAttribute("transform", `translate(${n.x},${n.y})`);
    g.style.display = passesFilter(n) ? "" : "none";
  }
}

// --- filter ---
const showPos = document.getElementById("showPos");
const showNeg = document.getElementById("showNeg");
const search = document.getElementById("search");
function haystack(n) {
  const p = n.profile || {};
  return [n.id, p.positioning, p.background, p.lookingFor, p.notes].filter(Boolean).join("\n").toLowerCase();
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
const tooltip = document.getElementById("tooltip");
function showTooltip(ev, edge, node) {
  if (edge) {
    const dir = `${escapeHtml(edge.source)} &rarr; ${escapeHtml(edge.target)}`;
    tooltip.innerHTML = `<strong>${edge.label === "pos" ? "Positive" : "Negative"}</strong> · ${dir}<br>${escapeHtml(edge.searchQuery || "")}`;
  } else if (node) {
    tooltip.innerHTML = `<strong>${escapeHtml(node.id)}</strong><br>role: ${node.role} · ${node.pairCount} pair edge(s)`;
  }
  tooltip.style.display = "block";
  moveTooltip(ev);
}
function moveTooltip(ev) {
  const wrap = document.getElementById("canvas-wrap").getBoundingClientRect();
  tooltip.style.left = (ev.clientX - wrap.left + 14) + "px";
  tooltip.style.top = (ev.clientY - wrap.top + 14) + "px";
}
function hideTooltip() { tooltip.style.display = "none"; }

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// --- panel ---
const panel = document.getElementById("panel");
document.getElementById("panelClose").addEventListener("click", () => panel.classList.remove("open"));

function fieldHtml(label, value) {
  const v = (value || "").toString().trim();
  if (!v) return "";
  return `<div class="field"><h3>${escapeHtml(label)}</h3><p>${escapeHtml(v)}</p></div>`;
}

function openPanel(n) {
  document.getElementById("panelTitle").textContent = (n.profile.positioning || n.id).toString().replace(/\s+/g, " ").slice(0, 60);
  document.getElementById("panelId").textContent = n.id;
  const roleLabel = { seeker: "Seeker only", candidate: "Candidate only", both: "Both seeker & candidate" }[n.role];
  const roleVar = ROLE_COLOR[n.role];
  document.getElementById("panelRole").innerHTML = `<span class="role-chip" style="background:color-mix(in srgb, ${roleVar} 18%, transparent);color:${roleVar}">${roleLabel}</span>`;

  const asSeeker = edges.filter(e => e.source === n.id);
  const asCandidate = edges.filter(e => e.target === n.id);

  const edgeItem = (e, dirLabel) => `
    <li class="${e.label}">
      <div class="dir">${dirLabel}</div>
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

// --- pan / zoom ---
let panning = false, panStart = null, viewStart = null;
svg.addEventListener("mousedown", (ev) => {
  if (ev.target.closest(".node")) return;
  panning = true;
  svg.classList.add("panning");
  panStart = { x: ev.clientX, y: ev.clientY };
  viewStart = { ...viewBox };
});
window.addEventListener("mousemove", (ev) => {
  if (!panning) return;
  const rect = svg.getBoundingClientRect();
  const scale = viewBox.w / rect.width;
  viewBox.x = viewStart.x - (ev.clientX - panStart.x) * scale;
  viewBox.y = viewStart.y - (ev.clientY - panStart.y) * scale;
  applyViewBox();
});
window.addEventListener("mouseup", () => { panning = false; svg.classList.remove("panning"); });
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

function makeDraggable(g, n) {
  let dragging = false;
  g.addEventListener("mousedown", (ev) => {
    dragging = true;
    n.pinned = true;
    n.vx = 0; n.vy = 0;
    reheat();
    ev.stopPropagation();
  });
  window.addEventListener("mousemove", (ev) => {
    if (!dragging) return;
    const rect = svg.getBoundingClientRect();
    const scale = viewBox.w / rect.width;
    n.x += ev.movementX * scale;
    n.y += ev.movementY * scale;
    if (!simRunning) render();
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    n.pinned = false;
    reheat();
  });
}

render();
</script>
</body>
</html>
"""


def build(pos_path: Path, neg_path: Path, out_path: Path) -> dict:
    pos_pairs = json.loads(pos_path.read_text(encoding="utf-8"))
    neg_pairs = json.loads(neg_path.read_text(encoding="utf-8"))
    graph = build_graph(pos_pairs, neg_pairs)

    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=DEFAULT_POS)
    parser.add_argument("--neg", type=Path, default=DEFAULT_NEG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    stats = build(args.pos, args.neg, args.out)
    print(f"Wrote {args.out} — {stats['nodes']} nodes, {stats['edges']} edges.")


if __name__ == "__main__":
    main()
