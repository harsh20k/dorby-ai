#!/usr/bin/env python3
"""3D scatter of real pairs at voyage-4-large `lookingFor` embeddings, with a
layout selector toggling between PCA, t-SNE, and UMAP — all three computed
up front and embedded, so switching is instant and doesn't recompute anything.

Unlike PCA (linear, ranked by raw variance), t-SNE and UMAP are nonlinear
manifold methods that explicitly optimize for a low-dimensional layout where
nearby points stay nearby — the direct answer to "PCA only captured 17% of
variance in 3 components, is there visual clustering to find at all?" (see
docs/experiment-graphs-index.md). Node positions are always fixed once
computed; mouse drag/scroll only move the camera, never a node.

Reuses the same voyage-4-large disk cache as build_real_pairs_3d_scatter.py
(artifacts/voyage_large_lookingfor/) — free if that cache already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_POS = ROOT / "data" / "dataset_positive.json"
DEFAULT_NEG = ROOT / "data" / "dataset_negative.json"
DEFAULT_OUT = ROOT / "docs" / "real-pairs-voyage-lookingfor-3d-manifold.html"
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "voyage_large_lookingfor"

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


def build_nodes_and_edges(pos_pairs: list[dict], neg_pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def touch(cid: str, file: dict) -> None:
        node = nodes.setdefault(cid, {"id": cid, "profile": {}, "pairCount": 0})
        node["pairCount"] += 1
        if not node["profile"]:
            node["profile"] = {k: file.get(k) for k in PROFILE_FIELDS}

    for label, pairs in (("pos", pos_pairs), ("neg", neg_pairs)):
        for p in pairs:
            seeker_id = p["userContactId"]
            cand_id = p["matchContactId"]
            if is_synthetic(seeker_id) or is_synthetic(cand_id):
                continue
            touch(seeker_id, p.get("userContactFile") or {})
            touch(cand_id, p.get("matchContactFile") or {})
            edges.append({"source": seeker_id, "target": cand_id, "label": label, "searchQuery": p.get("searchQuery")})

    return list(nodes.values()), edges


def fetch_embeddings(nodes: list[dict], *, cache_dir: Path):
    from baselines.voyage_large.encode import VoyageLargeEncoder

    texts = [(n["profile"].get("lookingFor") or "").strip() or " " for n in nodes]
    encoder = VoyageLargeEncoder(cache_dir=cache_dir)
    emb = encoder.encode(texts, input_type="document", label="real_lookingfor_manifold")
    encoder.write_usage_meta()
    return emb


def normalize_cube(coords):
    import numpy as np

    coords = np.asarray(coords, dtype=np.float64)
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = np.abs(coords).max() or 1.0
    return (coords / scale).tolist()


def compute_pca3d(emb):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(emb)
    ratios = pca.explained_variance_ratio_.tolist()
    note = (
        f"PC1 {ratios[0]*100:.1f}% / PC2 {ratios[1]*100:.1f}% / PC3 {ratios[2]*100:.1f}% "
        f"of variance (cumulative {sum(ratios)*100:.1f}%). Linear projection — distances "
        "here are only trustworthy along these 3 fixed axes."
    )
    return normalize_cube(coords), note


def compute_tsne3d(emb, *, perplexity: float, seed: int):
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(emb)
    note = (
        f"perplexity={perplexity}, KL divergence={tsne.kl_divergence_:.3f} at convergence. "
        "Nonlinear — optimizes so nearby points in the original 1024-dim space stay nearby "
        "here; absolute distances/axis directions are not meaningful, only relative "
        "neighborhoods are."
    )
    return normalize_cube(coords), note


def compute_umap3d(emb, *, n_neighbors: int, min_dist: float, seed: int):
    import umap

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    coords = reducer.fit_transform(emb)
    note = (
        f"n_neighbors={n_neighbors}, min_dist={min_dist}. Nonlinear, like t-SNE but better "
        "preserves some global structure between clusters — same caveat: axis directions "
        "and absolute distances aren't meaningful, only relative neighborhoods."
    )
    return normalize_cube(coords), note


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Real pairs — voyage-4-large lookingFor, PCA / t-SNE / UMAP 3D scatter</title>
<style>
  :root {
    --paper: #f4f6f3; --panel: #ffffff; --panel-glass: rgba(255,255,255,0.85);
    --ink: #16211d; --muted: #5c6b64; --line: #ccd5cf; --accent: #2b6f77;
    --pos: #2f9e5c; --neg: #d1483f; --shadow: 0 10px 30px rgba(20,32,28,0.10);
    --sans: "Avenir Next", Avenir, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #121a17; --panel: #1a2320; --panel-glass: rgba(26,35,32,0.85);
      --ink: #e7ece9; --muted: #93a39b; --line: #2d3934; --accent: #6ec0c9;
      --pos: #52c581; --neg: #ea7168; --shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
  }
  :root[data-theme="dark"] {
    --paper: #121a17; --panel: #1a2320; --panel-glass: rgba(26,35,32,0.85);
    --ink: #e7ece9; --muted: #93a39b; --line: #2d3934; --accent: #6ec0c9;
    --pos: #52c581; --neg: #ea7168; --shadow: 0 10px 30px rgba(0,0,0,0.35);
  }
  :root[data-theme="light"] {
    --paper: #f4f6f3; --panel: #ffffff; --panel-glass: rgba(255,255,255,0.85);
    --ink: #16211d; --muted: #5c6b64; --line: #ccd5cf; --accent: #2b6f77;
    --pos: #2f9e5c; --neg: #d1483f; --shadow: 0 10px 30px rgba(20,32,28,0.10);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body { font-family: var(--sans); color: var(--ink); background: var(--paper); overflow: hidden; }
  #stage { position: relative; width: 100vw; height: 100vh; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; background: var(--paper); }
  canvas.dragging { cursor: grabbing; }

  header.hero { position: absolute; top: 0; left: 0; right: 0; padding: 1rem 1.25rem 0; pointer-events: none; z-index: 5; }
  .masthead {
    display: inline-flex; flex-direction: column; gap: 0.5rem;
    background: var(--panel-glass); backdrop-filter: blur(6px);
    border: 1px solid var(--line); border-radius: 14px;
    padding: 0.85rem 1.1rem; box-shadow: var(--shadow); max-width: min(52rem, 90%);
    pointer-events: auto;
  }
  .masthead h1 { font-family: var(--serif); font-size: 1.15rem; margin: 0; line-height: 1.2; text-wrap: balance; }
  .masthead p.stats { margin: 0; color: var(--muted); font-size: 0.8rem; line-height: 1.45; }
  .masthead p.note { margin: 0; color: var(--accent); font-family: var(--mono); font-size: 0.7rem; line-height: 1.45; }

  .layout-selector { display: flex; gap: 0.4rem; }
  .layout-selector button {
    font: inherit; font-size: 0.78rem; font-weight: 600;
    border: 1px solid var(--line); background: var(--panel); color: var(--muted);
    border-radius: 999px; padding: 0.3rem 0.85rem; cursor: pointer;
  }
  .layout-selector button:hover { border-color: var(--accent); color: var(--accent); }
  .layout-selector button.active { background: var(--accent); border-color: var(--accent); color: var(--panel); }

  .legend {
    position: absolute; left: 1.25rem; bottom: 1rem; z-index: 5;
    display: flex; gap: 1rem; align-items: center;
    background: var(--panel-glass); backdrop-filter: blur(6px);
    border: 1px solid var(--line); border-radius: 999px;
    padding: 0.4rem 0.9rem; font-size: 0.76rem; color: var(--muted);
    pointer-events: none;
  }
  .legend .dot { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; margin-right: 0.3rem; }
  .legend .line { display: inline-block; width: 1.1rem; height: 0; border-top: 2px dotted currentColor; vertical-align: middle; margin-right: 0.3rem; }

  .hint {
    position: absolute; right: 1.25rem; bottom: 1rem; z-index: 5;
    font-family: var(--mono); font-size: 0.72rem; color: var(--muted);
    background: var(--panel-glass); backdrop-filter: blur(6px);
    border: 1px solid var(--line); border-radius: 999px; padding: 0.35rem 0.8rem;
    pointer-events: none;
  }

  .tooltip {
    position: absolute; pointer-events: none; background: var(--ink); color: var(--paper);
    font-size: 0.78rem; padding: 0.4rem 0.6rem; border-radius: 8px; max-width: 320px;
    z-index: 10; display: none; line-height: 1.4;
  }
</style>
</head>
<body>
<div id="stage">
  <canvas id="c"></canvas>
  <header class="hero">
    <div class="masthead">
      <h1>Real pairs — voyage-4-large <code>lookingFor</code>, 3D scatter</h1>
      <div class="layout-selector" id="layoutSelector"></div>
      <p class="stats" id="statsLine">__STATS__</p>
      <p class="note" id="noteLine"></p>
    </div>
  </header>
  <div class="legend">
    <span><span class="dot" style="background:var(--pos)"></span>positive pair</span>
    <span><span class="dot" style="background:var(--neg)"></span>negative pair</span>
    <span style="color:var(--pos)"><span class="line"></span>seeker → candidate (pos)</span>
    <span style="color:var(--neg)"><span class="line"></span>seeker → candidate (neg)</span>
  </div>
  <div class="hint">drag to rotate · scroll to zoom</div>
  <div class="tooltip" id="tooltip"></div>
</div>

<script id="graph-data" type="application/json">__EMBEDDED_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById("graph-data").textContent);
const nodes = DATA.nodes;   // [{id, profile, pairCount, layouts: {pca:[x,y,z], tsne:[...], umap:[...]}}]
const edges = DATA.edges;   // [{source, target, label, searchQuery}]
const LAYOUT_META = DATA.layoutMeta; // {pca: {label, note}, tsne: {...}, umap: {...}}
const LAYOUT_ORDER = ["pca", "tsne", "umap"];
let currentLayout = "tsne";

for (const n of nodes) {
  [n.x, n.y, n.z] = n.layouts[currentLayout];
}

// polarity per node (for coloring)
for (const n of nodes) {
  let pos = 0, neg = 0;
  for (const e of edges) {
    if (e.source !== n.id && e.target !== n.id) continue;
    if (e.label === "pos") pos++; else neg++;
  }
  const total = pos + neg;
  n.polarity = total ? (neg - pos) / total : 0;
}

const nodeById = new Map(nodes.map(n => [n.id, n]));

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const stage = document.getElementById("stage");
const statsLine = document.getElementById("statsLine");
const noteLine = document.getElementById("noteLine");

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = stage.clientWidth * dpr;
  canvas.height = stage.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);
resize();

// --- layout selector buttons ---
const selectorEl = document.getElementById("layoutSelector");
for (const key of LAYOUT_ORDER) {
  const btn = document.createElement("button");
  btn.textContent = LAYOUT_META[key].label;
  btn.dataset.key = key;
  btn.addEventListener("click", () => setLayout(key));
  selectorEl.appendChild(btn);
}
function setLayout(key) {
  currentLayout = key;
  for (const n of nodes) { [n.x, n.y, n.z] = n.layouts[key]; }
  for (const btn of selectorEl.querySelectorAll("button")) {
    btn.classList.toggle("active", btn.dataset.key === key);
  }
  noteLine.textContent = LAYOUT_META[key].note;
  draw();
}

// --- fixed 3D positions per layout: camera state only, never moves a node ---
let yaw = 0.6, pitch = -0.35, zoom = 1, panX = 0, panY = 0;
let dragging = false, lastX = 0, lastY = 0;

canvas.addEventListener("mousedown", (ev) => {
  dragging = true;
  canvas.classList.add("dragging");
  lastX = ev.clientX; lastY = ev.clientY;
});
window.addEventListener("mouseup", () => { dragging = false; canvas.classList.remove("dragging"); });
window.addEventListener("mousemove", (ev) => {
  if (!dragging) { handleHover(ev); return; }
  const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
  lastX = ev.clientX; lastY = ev.clientY;
  yaw += dx * 0.006;
  pitch += dy * 0.006;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
  draw();
});
canvas.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  zoom *= ev.deltaY > 0 ? 0.92 : 1.08;
  zoom = Math.max(0.2, Math.min(6, zoom));
  draw();
}, { passive: false });

function project(p) {
  let x = p.x, y = p.y, z = p.z;
  let cx = Math.cos(yaw), sx = Math.sin(yaw);
  let x1 = x * cx + z * sx;
  let z1 = -x * sx + z * cx;
  let cy = Math.cos(pitch), sy = Math.sin(pitch);
  let y2 = y * cy - z1 * sy;
  let z2 = y * sy + z1 * cy;
  const CAM_DIST = 4.2;
  const f = CAM_DIST / (CAM_DIST - z2);
  const scale = 260 * zoom;
  return {
    sx: x1 * f * scale + panX,
    sy: y2 * f * scale + panY,
    depth: z2,
    persp: f,
  };
}

let cx0 = 0, cy0 = 0;

function drawAxes() {
  const axLen = 1.15;
  const axes = [
    { from: [-axLen,0,0], to: [axLen,0,0], label: "1" },
    { from: [0,-axLen,0], to: [0,axLen,0], label: "2" },
    { from: [0,0,-axLen], to: [0,0,axLen], label: "3" },
  ];
  ctx.save();
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--line");
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.font = "11px var(--mono)";
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted");
  for (const ax of axes) {
    const a = project({x: ax.from[0], y: ax.from[1], z: ax.from[2]});
    const b = project({x: ax.to[0], y: ax.to[1], z: ax.to[2]});
    ctx.beginPath();
    ctx.moveTo(cx0 + a.sx, cy0 + a.sy);
    ctx.lineTo(cx0 + b.sx, cy0 + b.sy);
    ctx.stroke();
    ctx.fillText(ax.label, cx0 + b.sx + 4, cy0 + b.sy);
  }
  ctx.restore();
}

function colorForLabel(label) {
  const styles = getComputedStyle(document.documentElement);
  return label === "pos" ? styles.getPropertyValue("--pos").trim() : styles.getPropertyValue("--neg").trim();
}

function drawArrow(x1, y1, x2, y2, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 0.8;
  ctx.globalAlpha = 0.55;
  ctx.setLineDash([2, 4]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 0.85;
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 6;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(ang - Math.PI / 7), y2 - headLen * Math.sin(ang - Math.PI / 7));
  ctx.lineTo(x2 - headLen * Math.cos(ang + Math.PI / 7), y2 - headLen * Math.sin(ang + Math.PI / 7));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

let lastProjected = [];

function draw() {
  const w = stage.clientWidth, h = stage.clientHeight;
  cx0 = w / 2;
  cy0 = h / 2;
  ctx.clearRect(0, 0, w, h);

  drawAxes();

  const proj = new Map();
  for (const n of nodes) {
    const p = project(n);
    proj.set(n.id, { ...p, screenX: cx0 + p.sx, screenY: cy0 + p.sy });
  }

  const edgesSorted = edges.slice().sort((a, b) => {
    const da = (proj.get(a.source).depth + proj.get(a.target).depth) / 2;
    const db = (proj.get(b.source).depth + proj.get(b.target).depth) / 2;
    return da - db;
  });
  for (const e of edgesSorted) {
    const a = proj.get(e.source), b = proj.get(e.target);
    if (!a || !b) continue;
    drawArrow(a.screenX, a.screenY, b.screenX, b.screenY, colorForLabel(e.label));
  }

  const nodesSorted = nodes.slice().sort((a, b) => proj.get(a.id).depth - proj.get(b.id).depth);
  lastProjected = [];
  for (const n of nodesSorted) {
    const p = proj.get(n.id);
    const r = (3 + Math.min(n.pairCount, 6)) * Math.max(0.5, p.persp);
    ctx.beginPath();
    ctx.arc(p.screenX, p.screenY, r, 0, Math.PI * 2);
    const negPct = Math.round(((n.polarity + 1) / 2) * 100);
    ctx.fillStyle = `color-mix(in srgb, ${colorForLabel("neg")} ${negPct}%, ${colorForLabel("pos")} ${100 - negPct}%)`;
    ctx.globalAlpha = Math.min(1, 0.55 + p.persp * 0.4);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--panel");
    ctx.stroke();
    lastProjected.push({ n, screenX: p.screenX, screenY: p.screenY, r });
  }
}

function handleHover(ev) {
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  let hit = null;
  for (let i = lastProjected.length - 1; i >= 0; i--) {
    const it = lastProjected[i];
    const dx = mx - it.screenX, dy = my - it.screenY;
    if (dx * dx + dy * dy <= (it.r + 2) * (it.r + 2)) { hit = it; break; }
  }
  if (hit) {
    const n = hit.n;
    tooltip.innerHTML = `<strong>${(n.profile.positioning || n.id).toString().slice(0, 60)}</strong><br>` +
      `${n.pairCount} pair edge(s)<br>` +
      `<em>${(n.profile.lookingFor || "").toString().slice(0, 140)}</em>`;
    tooltip.style.display = "block";
    tooltip.style.left = (ev.clientX + 14) + "px";
    tooltip.style.top = (ev.clientY + 14) + "px";
  } else {
    tooltip.style.display = "none";
  }
}

setLayout(currentLayout);
window.addEventListener("resize", () => { resize(); draw(); });
</script>
</body>
</html>
"""


def build(
    pos_path: Path,
    neg_path: Path,
    out_path: Path,
    *,
    cache_dir: Path,
    tsne_perplexity: float,
    umap_neighbors: int,
    umap_min_dist: float,
    seed: int,
) -> dict:
    pos_pairs = json.loads(pos_path.read_text(encoding="utf-8"))
    neg_pairs = json.loads(neg_path.read_text(encoding="utf-8"))
    nodes, edges = build_nodes_and_edges(pos_pairs, neg_pairs)

    emb = fetch_embeddings(nodes, cache_dir=cache_dir)

    pca_coords, pca_note = compute_pca3d(emb)
    tsne_coords, tsne_note = compute_tsne3d(emb, perplexity=tsne_perplexity, seed=seed)
    umap_coords, umap_note = compute_umap3d(
        emb, n_neighbors=umap_neighbors, min_dist=umap_min_dist, seed=seed
    )

    for i, n in enumerate(nodes):
        n["layouts"] = {
            "pca": [round(v, 4) for v in pca_coords[i]],
            "tsne": [round(v, 4) for v in tsne_coords[i]],
            "umap": [round(v, 4) for v in umap_coords[i]],
        }

    layout_meta = {
        "pca": {"label": "PCA", "note": pca_note},
        "tsne": {"label": "t-SNE", "note": tsne_note},
        "umap": {"label": "UMAP", "note": umap_note},
    }

    n_pos = sum(1 for e in edges if e["label"] == "pos")
    n_neg = sum(1 for e in edges if e["label"] == "neg")
    stats = (
        f"{len(nodes)} contacts · {len(edges)} pairs ({n_pos} pos / {n_neg} neg) · "
        "position from cached voyage-4-large lookingFor embeddings, fixed — no force simulation, "
        "camera drag/zoom only"
    )

    payload = json.dumps(
        {"nodes": nodes, "edges": edges, "layoutMeta": layout_meta},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = payload.replace("<", "\\u003c")
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", payload)
    html = html.replace("__STATS__", stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {"nodes": len(nodes), "edges": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", type=Path, default=DEFAULT_POS)
    parser.add_argument("--neg", type=Path, default=DEFAULT_NEG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = build(
        args.pos,
        args.neg,
        args.out,
        cache_dir=args.cache_dir,
        tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        seed=args.seed,
    )
    print(f"Wrote {args.out} — {stats['nodes']} nodes / {stats['edges']} edges, layouts: pca/tsne/umap.")


if __name__ == "__main__":
    main()
