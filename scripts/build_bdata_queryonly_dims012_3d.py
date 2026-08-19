#!/usr/bin/env python3
"""3D scatter of B-data queryonly vectors using embedding dims 0,1,2 (not PCA).

Reads artifacts/bdata_queryonly_back_look/vectors/{corpus,queries}.npy and
emits a self-contained Three.js Points viewer. Does not touch prior 3D
builders or the vector caches.

  python scripts/build_bdata_queryonly_dims012_3d.py
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS = ROOT / "artifacts" / "bdata_queryonly_back_look" / "vectors"
DEFAULT_OUT = ROOT / "docs" / "html" / "bdata-queryonly-dims012-3d.html"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>B-data queryonly — dims 0,1,2</title>
<style>
  :root {
    --paper: #0e1412; --panel: #1a2320; --panel-glass: rgba(18,26,23,0.82);
    --ink: #e7ece9; --muted: #93a39b; --line: #2d3934; --accent: #6ec0c9;
    --cand: #7aa2c8; --pos: #52c581; --neg: #ea7168;
    --shadow: 0 10px 30px rgba(0,0,0,0.35);
    --sans: "Avenir Next", Avenir, -apple-system, "Segoe UI", sans-serif;
    --serif: "Iowan Old Style", Palatino, Georgia, serif;
    --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body { font-family: var(--sans); color: var(--ink); background: var(--paper); overflow: hidden; }
  #stage { position: relative; width: 100vw; height: 100vh; }
  canvas { display: block; width: 100%; height: 100%; }

  header.hero { position: absolute; top: 0; left: 0; right: 0; padding: 1rem 1.25rem 0; pointer-events: none; z-index: 5; }
  .masthead {
    display: inline-flex; flex-direction: column; gap: 0.35rem;
    background: var(--panel-glass); backdrop-filter: blur(8px);
    border: 1px solid var(--line); border-radius: 14px;
    padding: 0.85rem 1.1rem; box-shadow: var(--shadow); max-width: min(42rem, 92%);
    pointer-events: auto;
  }
  .masthead h1 { font-family: var(--serif); font-size: 1.12rem; margin: 0; line-height: 1.25; }
  .masthead p { margin: 0; color: var(--muted); font-size: 0.8rem; line-height: 1.45; }
  .masthead .stats { font-family: var(--mono); font-size: 0.72rem; color: var(--accent); }

  .controls {
    position: absolute; left: 1.25rem; bottom: 1rem; z-index: 5;
    display: flex; gap: 0.55rem; flex-wrap: wrap; align-items: center;
    background: var(--panel-glass); backdrop-filter: blur(8px);
    border: 1px solid var(--line); border-radius: 999px;
    padding: 0.4rem 0.85rem; font-size: 0.76rem; color: var(--muted);
  }
  .controls label { display: inline-flex; align-items: center; gap: 0.3rem; cursor: pointer; user-select: none; }
  .controls input { accent-color: var(--accent); }
  .dot { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; }
  .dot.cand { background: var(--cand); }
  .dot.pos { background: var(--pos); }
  .dot.neg { background: var(--neg); }

  .hint {
    position: absolute; right: 1.25rem; bottom: 1rem; z-index: 5;
    font-family: var(--mono); font-size: 0.72rem; color: var(--muted);
    background: var(--panel-glass); backdrop-filter: blur(8px);
    border: 1px solid var(--line); border-radius: 999px; padding: 0.35rem 0.8rem;
    pointer-events: none;
  }

  .tooltip {
    position: absolute; pointer-events: none; background: #e7ece9; color: #121a17;
    font-size: 0.78rem; padding: 0.45rem 0.65rem; border-radius: 8px; max-width: 360px;
    z-index: 10; display: none; line-height: 1.4; box-shadow: var(--shadow);
  }
  .tooltip strong { font-family: var(--mono); font-size: 0.72rem; }
  .tooltip .role { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: #5c6b64; }
</style>
</head>
<body>
<div id="stage">
  <div id="c"></div>
  <header class="hero">
    <div class="masthead">
      <h1>B-data in the first three embedding dimensions</h1>
      <p>
        Not PCA. x/y/z = dims 0/1/2 of voyage-4-nano +
        <span class="stats">queryonly_back_look_001</span> LoRA.
        Seeker = query only; candidate = background + lookingFor.
        A 3-dim slice of a 1024-d vector — most of the space is off-axis.
      </p>
      <p class="stats" id="stats"></p>
    </div>
  </header>
  <div class="controls">
    <label><input type="checkbox" id="showCand" checked /> <span class="dot cand"></span> candidates</label>
    <label><input type="checkbox" id="showPos" checked /> <span class="dot pos"></span> ACCEPT queries</label>
    <label><input type="checkbox" id="showNeg" checked /> <span class="dot neg"></span> REJECT queries</label>
  </div>
  <div class="hint">drag orbit · scroll zoom · right-drag pan</div>
  <div class="tooltip" id="tip"></div>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA = __PAYLOAD__;

function b64ToF32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

const xyz = b64ToF32(DATA.xyzB64);
const nC = DATA.nCorpus;
const nA = DATA.nAccept;
const nR = DATA.nReject;
const offA = nC * 3;
const offR = offA + nA * 3;

document.getElementById("stats").textContent =
  `${nC.toLocaleString()} candidates · ${nA.toLocaleString()} ACCEPT · ${nR.toLocaleString()} REJECT · ${DATA.generated}`;

const stage = document.getElementById("c");
const tip = document.getElementById("tip");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0e1412, 1);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0e1412, 4.5, 11);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.05, 40);
camera.position.set(2.15, 1.45, 2.55);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.4;
controls.maxDistance = 12;
controls.target.set(0, 0, 0);

function axisLine(from, to, color) {
  const g = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(...from), new THREE.Vector3(...to),
  ]);
  return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 }));
}
scene.add(axisLine([-1.15,0,0], [1.15,0,0], 0x93a39b));
scene.add(axisLine([0,-1.15,0], [0,1.15,0], 0x93a39b));
scene.add(axisLine([0,0,-1.15], [0,0,1.15], 0x93a39b));

function makeLabel(text, pos) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#93a39b";
  ctx.font = "28px ui-monospace, SF Mono, Menlo, monospace";
  ctx.fillText(text, 8, 42);
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const s = new THREE.Sprite(mat);
  s.position.set(...pos);
  s.scale.set(0.42, 0.105, 1);
  scene.add(s);
}
makeLabel("dim0", [1.28, 0, 0]);
makeLabel("dim1", [0, 1.28, 0]);
makeLabel("dim2", [0, 0, 1.28]);

function cloud(offset, count, color, ids) {
  const positions = xyz.subarray(offset, offset + count * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.computeBoundingSphere();
  const mat = new THREE.PointsMaterial({
    color,
    size: 2.4,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.78,
    depthWrite: false,
  });
  const pts = new THREE.Points(geo, mat);
  pts.frustumCulled = false;
  pts.userData.ids = ids;
  scene.add(pts);
  return pts;
}

const clouds = {
  cand: cloud(0, nC, 0x7aa2c8, DATA.corpusIds),
  pos: cloud(offA, nA, 0x52c581, DATA.acceptIds),
  neg: cloud(offR, nR, 0xea7168, DATA.rejectIds),
};

document.getElementById("showCand").addEventListener("change", (e) => { clouds.cand.visible = e.target.checked; });
document.getElementById("showPos").addEventListener("change", (e) => { clouds.pos.visible = e.target.checked; });
document.getElementById("showNeg").addEventListener("change", (e) => { clouds.neg.visible = e.target.checked; });

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.035;
const mouse = new THREE.Vector2();
const roleName = { cand: "candidate", pos: "ACCEPT query", neg: "REJECT query" };

function onMove(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(
    Object.values(clouds).filter((c) => c.visible),
    false,
  );
  if (!hits.length) {
    tip.style.display = "none";
    renderer.domElement.style.cursor = "grab";
    return;
  }
  hits.sort((a, b) => a.distanceToRay - b.distanceToRay);
  const hit = hits[0];
  const key = Object.keys(clouds).find((k) => clouds[k] === hit.object);
  const id = hit.object.userData.ids[hit.index] || "";
  const p = hit.point;
  tip.innerHTML = `<div class="role">${roleName[key] || key}</div><strong>${id}</strong><br>x ${p.x.toFixed(3)} · y ${p.y.toFixed(3)} · z ${p.z.toFixed(3)}`;
  tip.style.display = "block";
  tip.style.left = (ev.clientX + 14) + "px";
  tip.style.top = (ev.clientY + 14) + "px";
  renderer.domElement.style.cursor = "pointer";
}
renderer.domElement.addEventListener("pointermove", onMove);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
</script>
</body>
</html>
"""


def _b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def build(vectors_dir: Path, out_path: Path) -> dict:
    # Unique document pool + pair-aligned seekers (queries.npy is ACCEPT-only).
    corpus = np.load(vectors_dir / "corpus.npy", mmap_mode="r")
    pos_s = np.load(vectors_dir / "pos_seeker.npy", mmap_mode="r")
    neg_s = np.load(vectors_dir / "neg_seeker.npy", mmap_mode="r")
    corpus_ids = json.loads((vectors_dir / "corpus_ids.json").read_text(encoding="utf-8"))
    pos_ids = json.loads((vectors_dir / "pos_ids.json").read_text(encoding="utf-8"))
    neg_ids = json.loads((vectors_dir / "neg_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((vectors_dir / "manifest.json").read_text(encoding="utf-8"))

    if len(corpus_ids) != corpus.shape[0]:
        raise SystemExit("corpus ids/rows mismatch")
    if len(pos_ids) != pos_s.shape[0]:
        raise SystemExit("pos ids/rows mismatch")
    if len(neg_ids) != neg_s.shape[0]:
        raise SystemExit("neg ids/rows mismatch")

    c3 = np.asarray(corpus[:, :3], dtype=np.float32)
    a3 = np.asarray(pos_s[:, :3], dtype=np.float32)
    r3 = np.asarray(neg_s[:, :3], dtype=np.float32)
    stacked = np.concatenate([c3, a3, r3], axis=0)
    scale = float(np.abs(stacked).max() or 1.0)
    stacked = stacked / scale

    payload = {
        "source": str(vectors_dir.relative_to(ROOT)),
        "model": manifest.get("model_name"),
        "adapter": Path(str(manifest.get("adapter_dir") or "")).name,
        "packing": manifest.get("packing"),
        "axes": ["dim0", "dim1", "dim2"],
        "scale": scale,
        "nCorpus": int(c3.shape[0]),
        "nAccept": int(a3.shape[0]),
        "nReject": int(r3.shape[0]),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "xyzB64": _b64_f32(stacked),
        "corpusIds": list(corpus_ids),
        "acceptIds": list(pos_ids),
        "rejectIds": list(neg_ids),
    }
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {
        "out": str(out_path),
        "n_corpus": payload["nCorpus"],
        "n_accept": payload["nAccept"],
        "n_reject": payload["nReject"],
        "bytes": out_path.stat().st_size,
        "scale": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not (args.vectors / "corpus.npy").is_file():
        raise SystemExit(f"Missing vectors at {args.vectors}")
    stats = build(args.vectors, args.out)
    print(
        f"Wrote {stats['out']} ({stats['bytes']:,} bytes) "
        f"cand={stats['n_corpus']} accept={stats['n_accept']} reject={stats['n_reject']} "
        f"scale={stats['scale']:.4f}"
    )


if __name__ == "__main__":
    main()
