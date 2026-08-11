#!/usr/bin/env python3
"""Build a self-contained 3D map of holdout contacts, isolated fields, and
isolated lookingFor sections in voyage-4-nano space.

Sibling to scripts/build_holdout_embedding_space_3d.py. That experiment swapped
one lookingFor section into an otherwise-unchanged profile, so every row still
carried the whole profile's context and points barely moved. This one reads
vectors back out of the local Chroma collection built by
scripts/load_field_isolation_to_chroma.py, where each row is *only* one field's
text (or one lookingFor section's text) with nothing else -- no shared context
left to dominate the vector. Emits one HTML file: a canvas 3D scatter with each
contact's whole-profile anchor tethered to its own isolated-field dots, and
each lookingFor field tethered on to its own isolated-section dots.

Usage:
  python scripts/build_field_isolation_embedding_space_3d.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = ROOT / "artifacts" / "voyage_nano_field_isolation" / "chroma"
DEFAULT_COLLECTION = "holdout_field_isolation_voyage_nano"
DEFAULT_META = ROOT / "artifacts" / "voyage_nano_field_isolation" / "meta.json"
DEFAULT_OUT = ROOT / "docs" / "html" / "holdout-field-isolation-embedding-space-3d.html"

SECTION_CHARS = 420
FIELD_CHARS = 320
POSITIONING_CHARS = 260

FIELD_ORDER = [
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
]


def truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_payload(db_dir: Path, collection_name: str, meta_path: Path) -> dict:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(collection_name)
    got = collection.get(include=["embeddings", "metadatas", "documents"])
    ids = got["ids"]
    emb = np.asarray(got["embeddings"], dtype=np.float32)
    metadatas = got["metadatas"]
    documents = got["documents"]

    meta = json.loads(meta_path.read_text())
    contacts_meta = {c["id"]: c for c in meta["contacts"]}

    # One joint PCA over whole + field + section vectors, so an offset from a
    # contact's own anchor is on the same scale as the gap between contacts.
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(emb)
    coords = coords / (np.abs(coords).max() or 1.0)
    evr = pca.explained_variance_ratio_

    whole_idx: dict[str, int] = {}
    field_idx: dict[str, dict[str, int]] = {}
    section_idx: dict[str, list[int]] = {}
    for i, md in enumerate(metadatas):
        cid = md["contactId"]
        if md["kind"] == "whole":
            whole_idx[cid] = i
        elif md["kind"] == "field":
            field_idx.setdefault(cid, {})[md["field"]] = i
        else:  # section_alone
            section_idx.setdefault(cid, []).append(i)

    # ---- cosine geometry on the raw 1024-d vectors ----
    whole_field_cos: list[float] = []
    field_field_same_contact_cos: list[float] = []
    whole_section_cos: list[float] = []
    section_section_cos: list[float] = []
    per_field_whole_cos: dict[str, list[float]] = {f: [] for f in FIELD_ORDER}
    per_field_vecs: dict[str, list[np.ndarray]] = {f: [] for f in FIELD_ORDER}

    nodes: list[dict] = []
    for cid, w in whole_idx.items():
        cmeta = contacts_meta[cid]
        roles = cmeta["roles"]
        role = "both" if len(roles) > 1 else roles[0]

        fidx = field_idx.get(cid, {})
        field_ids = list(fidx.values())
        cos_to_whole_fields = (emb[field_ids] @ emb[w]).tolist() if field_ids else []
        whole_field_cos.extend(cos_to_whole_fields)
        if len(field_ids) > 1:
            sim = emb[field_ids] @ emb[field_ids].T
            iu = np.triu_indices(len(field_ids), 1)
            field_field_same_contact_cos.extend(sim[iu].tolist())
        for fname, fi in fidx.items():
            if fname in per_field_whole_cos:
                per_field_whole_cos[fname].append(float(emb[fi] @ emb[w]))
                per_field_vecs[fname].append(emb[fi])

        secs = section_idx.get(cid, [])
        cos_to_whole_secs = (emb[secs] @ emb[w]).tolist() if secs else []
        whole_section_cos.extend(cos_to_whole_secs)
        dispersion = 0.0
        if len(secs) > 1:
            sim = emb[secs] @ emb[secs].T
            iu = np.triu_indices(len(secs), 1)
            section_section_cos.extend(sim[iu].tolist())
            dispersion = 1.0 - float(sim[iu].mean())

        looking_for_field_idx = fidx.get("lookingFor")

        nodes.append(
            {
                "id": cid,
                "role": role,
                "pairCount": cmeta["pairCount"],
                "dispersion": round(dispersion, 4),
                "positioning": truncate(cmeta["profile"].get("positioning"), POSITIONING_CHARS),
                "whole": [round(float(v), 5) for v in coords[w]],
                "fields": [
                    {
                        "field": fname,
                        "text": truncate(documents[fi], FIELD_CHARS),
                        "cos": round(float(c), 4),
                        "xyz": [round(float(v), 5) for v in coords[fi]],
                    }
                    for fname, fi, c in zip(fidx.keys(), fidx.values(), cos_to_whole_fields)
                ],
                "sections": [
                    {
                        "text": truncate(documents[s], SECTION_CHARS),
                        "cos": round(float(c), 4),
                        "xyz": [round(float(v), 5) for v in coords[s]],
                        "anchor": (
                            [round(float(v), 5) for v in coords[looking_for_field_idx]]
                            if looking_for_field_idx is not None
                            else [round(float(v), 5) for v in coords[w]]
                        ),
                    }
                    for s, c in zip(secs, cos_to_whole_secs)
                ],
            }
        )

    nodes.sort(key=lambda n: -(len(n["fields"]) + len(n["sections"])))

    present_fields = {f["field"] for n in nodes for f in n["fields"]}
    field_order_present = [f for f in FIELD_ORDER if f in present_fields]

    whole_vecs = emb[[whole_idx[c] for c in whole_idx]]
    across = whole_vecs @ whole_vecs.T
    iu = np.triu_indices(len(whole_vecs), 1)

    # Do same-named fields from *different* people cluster together (field type
    # dominates) or do a person's own fields stay closer to each other (person
    # identity dominates)? Compare the two directly.
    per_field_stats = {}
    for fname, vecs in per_field_vecs.items():
        if len(vecs) < 2:
            continue
        v = np.stack(vecs)
        sim = v @ v.T
        fiu = np.triu_indices(len(v), 1)
        per_field_stats[fname] = {
            "n": len(vecs),
            "meanCosToWhole": round(float(np.mean(per_field_whole_cos[fname])), 4),
            "meanCosAcrossContacts": round(float(sim[fiu].mean()), 4),
        }

    radii = [
        float(
            np.linalg.norm(
                np.concatenate(
                    [
                        coords[list(field_idx.get(c, {}).values())]
                        if field_idx.get(c)
                        else np.zeros((0, 3)),
                        coords[section_idx[c]] if section_idx.get(c) else np.zeros((0, 3)),
                    ]
                )
                - coords[whole_idx[c]],
                axis=1,
            ).mean()
        )
        for c in whole_idx
        if field_idx.get(c) or section_idx.get(c)
    ]
    cloud_pts = coords[[whole_idx[c] for c in whole_idx]]
    cloud_radius = float(np.linalg.norm(cloud_pts - cloud_pts.mean(axis=0), axis=1).mean())

    edges = [
        {"s": e["source"], "t": e["target"], "l": e["label"], "q": truncate(e.get("searchQuery"), 220)}
        for e in meta["edges"]
    ]

    stats = {
        "nContacts": len(nodes),
        "nFields": sum(len(n["fields"]) for n in nodes),
        "nSections": sum(len(n["sections"]) for n in nodes),
        "nPairs": meta["n_pairs"],
        "nPositives": meta["n_positives"],
        "nNegatives": meta["n_negatives"],
        "nSeekers": sum(1 for n in nodes if n["role"] in ("seeker", "both")),
        "nCandidates": sum(1 for n in nodes if n["role"] in ("candidate", "both")),
        "wholeFieldCos": round(float(np.mean(whole_field_cos)), 4),
        "fieldFieldSameContactCos": round(float(np.mean(field_field_same_contact_cos)), 4),
        "wholeSectionCos": round(float(np.mean(whole_section_cos)), 4),
        "sectionSectionCos": round(float(np.mean(section_section_cos)), 4),
        "acrossContactCos": round(float(np.mean(across[iu])), 4),
        "constellationRatio": round(float(np.mean(radii) / cloud_radius), 4),
        "evr": [round(float(v), 4) for v in evr],
        "evrSum": round(float(evr[:3].sum()), 4),
        "model": meta["model_name"],
        "dim": meta["truncate_dim"],
        "maxFields": max(len(n["fields"]) for n in nodes),
        "maxSections": max((len(n["sections"]) for n in nodes), default=0),
        "fieldOrder": field_order_present,
        "perField": [
            {"field": f, **per_field_stats[f]} for f in FIELD_ORDER if f in per_field_stats
        ],
    }
    return {"nodes": nodes, "edges": edges, "stats": stats}


HTML = r"""<title>Holdout contacts, isolated fields, in voyage-4-nano space</title>
<style>
  :root {
    --ground: #ecefee; --card: #ffffff; --glass: rgba(255,255,255,0.88);
    --ink: #141d20; --ink-soft: #3b4a4d; --muted: #647476; --hairline: #ccd4d2;
    --seeker: #1f7a8c; --candidate: #c2603f; --both: #a97f1f;
    --pos: #2e8b57; --neg: #3a52b0; --field: #7a5cb8; --section: #b8547c;
    --shadow: 0 8px 28px rgba(20,32,34,0.10);
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    --sans: "Avenir Next", Avenir, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground: #0f1618; --card: #171f21; --glass: rgba(23,31,33,0.88);
      --ink: #e6edeb; --ink-soft: #b6c5c3; --muted: #8b9c9a; --hairline: #26312f;
      --seeker: #58c8db; --candidate: #f0906a; --both: #ddb455;
      --pos: #5cc98a; --neg: #8098ee; --field: #b39ce6; --section: #e69ab8;
      --shadow: 0 8px 28px rgba(0,0,0,0.42);
    }
  }
  :root[data-theme="dark"] {
    --ground: #0f1618; --card: #171f21; --glass: rgba(23,31,33,0.88);
    --ink: #e6edeb; --ink-soft: #b6c5c3; --muted: #8b9c9a; --hairline: #26312f;
    --seeker: #58c8db; --candidate: #f0906a; --both: #ddb455;
    --pos: #5cc98a; --neg: #8098ee; --field: #b39ce6; --section: #e69ab8;
    --shadow: 0 8px 28px rgba(0,0,0,0.42);
  }
  :root[data-theme="light"] {
    --ground: #ecefee; --card: #ffffff; --glass: rgba(255,255,255,0.88);
    --ink: #141d20; --ink-soft: #3b4a4d; --muted: #647476; --hairline: #ccd4d2;
    --seeker: #1f7a8c; --candidate: #c2603f; --both: #a97f1f;
    --pos: #2e8b57; --neg: #3a52b0; --field: #7a5cb8; --section: #b8547c;
    --shadow: 0 8px 28px rgba(20,32,34,0.10);
  }

  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ground); color: var(--ink);
    font-family: var(--sans); -webkit-font-smoothing: antialiased;
  }
  .eyebrow {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
  }
  .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  .bar {
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
    padding: 14px 22px; border-bottom: 1px solid var(--hairline); background: var(--card);
  }
  .bar h1 {
    margin: 0; font-family: var(--serif); font-weight: 600;
    font-size: clamp(17px, 2.4vw, 22px); letter-spacing: -0.01em; text-wrap: balance;
  }
  .bar .sub { color: var(--muted); font-size: 13px; }
  .bar .spacer { flex: 1 1 auto; }

  .stage {
    position: relative; height: min(74vh, 780px); min-height: 460px;
    border-bottom: 1px solid var(--hairline); overflow: hidden;
  }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }

  .rail {
    position: absolute; top: 16px; left: 16px; width: 240px; max-width: calc(100% - 32px);
    background: var(--glass); backdrop-filter: blur(9px);
    border: 1px solid var(--hairline); border-radius: 3px; box-shadow: var(--shadow);
    padding: 14px 15px; display: flex; flex-direction: column; gap: 13px;
    max-height: calc(100% - 32px); overflow-y: auto;
  }
  .stat .k { display: block; margin-bottom: 3px; }
  .stat .v { font-family: var(--mono); font-size: 22px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
  .stat .n { display: block; font-size: 11.5px; color: var(--muted); line-height: 1.45; margin-top: 3px; }
  .rule { height: 1px; background: var(--hairline); }
  .legend { display: flex; flex-direction: column; gap: 7px; font-size: 12.5px; }
  .legend .row { display: flex; align-items: center; gap: 8px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
  .dot.hollow { background: none; border: 2px solid currentColor; }
  .dot.tiny { width: 6px; height: 6px; }
  .tether { width: 18px; height: 0; border-top: 1.5px dotted var(--muted); flex: none; }
  .wire { width: 18px; height: 2px; border-radius: 1px; flex: none; }

  .fieldpicker { display: flex; flex-direction: column; gap: 7px; }
  .fp-head { display: flex; align-items: center; justify-content: space-between; }
  .fp-actions { display: flex; gap: 6px; }
  button.fp-btn {
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.06em; text-transform: uppercase;
    background: none; border: 1px solid var(--hairline); color: var(--muted);
    padding: 2px 7px; border-radius: 2px; cursor: pointer;
  }
  button.fp-btn:hover { border-color: var(--field); color: var(--field); }
  .field-row { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--ink-soft); cursor: pointer; }
  .field-row input { accent-color: var(--field); }
  .field-swatch { width: 9px; height: 9px; border-radius: 50%; flex: none; }

  .controls {
    position: absolute; left: 16px; right: 16px; bottom: 16px;
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    background: var(--glass); backdrop-filter: blur(9px);
    border: 1px solid var(--hairline); border-radius: 3px; box-shadow: var(--shadow);
    padding: 11px 15px;
  }
  .ctl { display: flex; align-items: center; gap: 9px; }
  .ctl label { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
  input[type="range"] { width: 140px; accent-color: var(--field); }
  .amp-val { font-family: var(--mono); font-size: 13px; min-width: 34px; font-variant-numeric: tabular-nums; }
  .toggle { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-soft); cursor: pointer; }
  button.reset {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    background: none; border: 1px solid var(--hairline); color: var(--ink-soft);
    padding: 5px 11px; border-radius: 2px; cursor: pointer;
  }
  button.reset:hover { border-color: var(--field); color: var(--field); }
  :focus-visible { outline: 2px solid var(--field); outline-offset: 2px; }
  .hint { margin-left: auto; font-size: 11.5px; color: var(--muted); }

  .tip {
    position: absolute; pointer-events: none; z-index: 9; max-width: 330px;
    background: var(--card); border: 1px solid var(--hairline); border-radius: 3px;
    box-shadow: var(--shadow); padding: 11px 13px; font-size: 12.5px; line-height: 1.5;
    opacity: 0; transition: opacity .1s;
  }
  .tip.on { opacity: 1; }
  .tip .who { font-family: var(--mono); font-size: 10.5px; color: var(--muted); letter-spacing: 0.06em; }
  .tip .hd { font-family: var(--serif); font-size: 15px; margin: 3px 0 5px; }
  .tip .body { color: var(--ink-soft); }
  .tip .meta { margin-top: 7px; padding-top: 6px; border-top: 1px solid var(--hairline); color: var(--muted); font-size: 11.5px; }

  .read { max-width: 68ch; margin: 0 auto; padding: 40px 22px 64px; }
  .read h2 {
    font-family: var(--serif); font-weight: 600; font-size: 21px; letter-spacing: -0.01em;
    margin: 34px 0 10px; text-wrap: balance;
  }
  .read h2:first-of-type { margin-top: 8px; }
  .read p { font-size: 15px; line-height: 1.66; color: var(--ink-soft); margin: 0 0 13px; }
  .read strong { color: var(--ink); font-weight: 600; }
  .read code { font-family: var(--mono); font-size: 0.88em; background: var(--card); padding: 1px 5px; border-radius: 2px; }
  .tablewrap { overflow-x: auto; margin: 16px 0 20px; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  th, td { text-align: left; padding: 8px 12px 8px 0; border-bottom: 1px solid var(--hairline); }
  th { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
  td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  .note { font-size: 12.5px; color: var(--muted); line-height: 1.55; }

  @media (max-width: 720px) {
    .rail { position: static; width: auto; max-height: none; margin: 12px; }
    .stage { height: auto; }
    canvas { height: 60vh; }
    .controls { position: static; margin: 0 12px 12px; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="bar">
  <h1>Our test profiles, split into isolated fields</h1>
  <span class="sub">115 people &middot; each profile field embedded alone &middot; each lookingFor ask embedded alone</span>
  <span class="spacer"></span>
  <span class="eyebrow" id="modelTag"></span>
</div>

<div class="stage" id="stage">
  <canvas id="c"></canvas>
  <div class="rail">
    <div class="stat">
      <span class="k eyebrow">One field, alone, vs. the whole profile</span>
      <span class="v" id="sWholeField"></span>
      <span class="n">A field embedded with nothing else, compared to that person's full-profile point. Two <em>different</em> people score <span class="num" id="sAcross"></span>.</span>
    </div>
    <div class="rule"></div>
    <div class="stat">
      <span class="k eyebrow">How wide one person spreads</span>
      <span class="v" id="sRatio"></span>
      <span class="n">A person&rsquo;s cluster of isolated fields and asks, measured against the distance between people.</span>
    </div>
    <div class="rule"></div>
    <div class="legend">
      <div class="row"><span class="dot" style="background:var(--seeker)"></span> Person looking <span class="num" id="lSeek" style="color:var(--muted)"></span></div>
      <div class="row"><span class="dot" style="background:var(--candidate)"></span> Person suggested <span class="num" id="lCand" style="color:var(--muted)"></span></div>
      <div class="row"><span class="dot" style="background:var(--both)"></span> Both at once</div>
      <div class="row"><span class="dot hollow tiny" style="color:var(--section)"></span> One ask, alone (lookingFor only)</div>
      <div class="row"><span class="tether"></span> Tied back to its parent</div>
      <div class="row"><span class="wire" style="background:var(--pos)"></span> Good match <span class="num" id="lPos" style="color:var(--muted)"></span></div>
      <div class="row"><span class="wire" style="background:var(--neg)"></span> Bad match <span class="num" id="lNeg" style="color:var(--muted)"></span></div>
    </div>
    <div class="rule"></div>
    <div class="fieldpicker">
      <div class="fp-head">
        <span class="eyebrow">Fields shown</span>
        <span class="fp-actions">
          <button type="button" class="fp-btn" id="fieldsAll">all</button>
          <button type="button" class="fp-btn" id="fieldsNone">none</button>
        </span>
      </div>
      <div id="fieldChecks"></div>
    </div>
    <div class="rule"></div>
    <div class="note">The model describes each person with 1,024 numbers. This is the flattest 3D shadow of that, keeping <span class="num" id="sEvr"></span> of the detail &mdash; so treat it as a sketch, not a measurement.</div>
  </div>

  <div class="controls">
    <div class="ctl">
      <label for="amp">Spread apart</label>
      <input type="range" id="amp" min="1" max="12" step="0.5" value="1" />
      <span class="amp-val" id="ampVal">1.0&times;</span>
    </div>
    <label class="toggle"><input type="checkbox" id="showWhole" checked /> Profile points</label>
    <label class="toggle"><input type="checkbox" id="showSections" checked /> Asks</label>
    <label class="toggle"><input type="checkbox" id="showPairs" checked /> Match lines</label>
    <button class="reset" id="reset">Reset view</button>
    <span class="hint">drag to rotate &middot; scroll to zoom &middot; click a person to isolate them</span>
  </div>

  <div class="tip" id="tip"></div>
</div>

<div class="read">
  <h2>What's different this time</h2>
  <p>The earlier version of this page (<code>holdout-embedding-space-3d.html</code>) split a
  person&rsquo;s <code>lookingFor</code> field into separate asks, but kept every other field of
  their profile in the text each time. The one thing that changed barely moved the point,
  because everything unchanged still dominated the vector.</p>
  <p>This page does the opposite: every dot here comes from <strong>exactly one field&rsquo;s
  text and nothing else</strong>. A person&rsquo;s <code>positioning</code> alone, their
  <code>notes</code> alone, one <code>lookingFor</code> ask alone &mdash; no shared context left
  to hold anything in place.</p>

  <h2>Isolating a field moves it a lot more</h2>
  <p>A lone field scores <strong id="pWholeField"></strong> similar to that same person&rsquo;s
  whole-profile point &mdash; far looser than the earlier experiment&rsquo;s near-identical
  0.89&ndash;0.90. Two different people&rsquo;s whole profiles still score
  <strong id="pAcross"></strong>, so a lone field is now much closer to the boundary between
  &ldquo;still recognizably this person&rdquo; and &ldquo;just floating on its own topic.&rdquo;
  A person&rsquo;s own fields spread out to about <strong id="pRatio"></strong> of the gap between
  people &mdash; noticeably wider than the earlier constellation.</p>

  <div class="tablewrap">
    <table>
      <thead><tr><th>Comparing</th><th>Similarity</th><th>Meaning</th></tr></thead>
      <tbody>
        <tr><td>A field, alone, vs. that person&rsquo;s whole profile</td><td class="n" id="tWholeField"></td><td>loosely related</td></tr>
        <tr><td>One person&rsquo;s own fields vs. each other</td><td class="n" id="tFieldField"></td><td>only loosely related</td></tr>
        <tr><td>A lookingFor ask, alone, vs. that person&rsquo;s whole profile</td><td class="n" id="tWholeSection"></td><td>loosely related</td></tr>
        <tr><td>One person&rsquo;s own asks vs. each other</td><td class="n" id="tSectionSection"></td><td>loosely related</td></tr>
        <tr><td>One person vs. a different person (whole profiles)</td><td class="n" id="tAcross"></td><td>clearly different</td></tr>
      </tbody>
    </table>
  </div>

  <h2>Do same-named fields cluster by topic, or by person?</h2>
  <p>A different question this isolation makes possible: take everyone&rsquo;s
  <code>notes</code> field alone, ignore whose it is, and check how similar they are to each
  other &mdash; then compare that to how similar each one is to its own owner&rsquo;s whole
  profile. If same-named fields cluster together regardless of person, embedding a field
  &ldquo;alone&rdquo; risks losing whose profile it belongs to.</p>

  <div class="tablewrap">
    <table>
      <thead><tr><th>Field</th><th>Count</th><th>vs. its own whole profile</th><th>vs. same field, other people</th></tr></thead>
      <tbody id="fieldRows"></tbody>
    </table>
  </div>
  <p class="note">When the right column is close to the middle one, that field&rsquo;s text
  reads as generic across people &mdash; the model can barely tell whose it is once it&rsquo;s
  alone. When the middle column is clearly higher, the field still carries person-specific
  signal even in isolation.</p>

  <p class="note">Every point on this page comes from one field&rsquo;s text only, no search
  query, no other field. Similarity is measured on the full 1,024-number vectors, not the
  flattened 3D view. Vectors are read out of a local Chroma collection
  (<code>artifacts/voyage_nano_field_isolation/chroma</code>), built by
  <code>scripts/load_field_isolation_to_chroma.py</code> from embeddings computed on Modal by
  <code>baselines/voyage_nano_field_isolation/modal_embed_space.py</code>.</p>
</div>

<script>
const DATA = __DATA__;
const S = DATA.stats;

const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
set("modelTag", S.model + " · " + S.dim + "d");
set("sWholeField", S.wholeFieldCos.toFixed(3));
set("sAcross", S.acrossContactCos.toFixed(3));
set("sRatio", (S.constellationRatio * 100).toFixed(0) + "%");
set("sEvr", (S.evrSum * 100).toFixed(1) + "%");
set("lSeek", S.nSeekers);
set("lCand", S.nCandidates);
set("lPos", S.nPositives);
set("lNeg", S.nNegatives);
set("pWholeField", S.wholeFieldCos.toFixed(4));
set("pAcross", S.acrossContactCos.toFixed(4));
set("pRatio", (S.constellationRatio * 100).toFixed(0) + "%");
set("tWholeField", S.wholeFieldCos.toFixed(4));
set("tFieldField", S.fieldFieldSameContactCos.toFixed(4));
set("tWholeSection", S.wholeSectionCos.toFixed(4));
set("tSectionSection", S.sectionSectionCos.toFixed(4));
set("tAcross", S.acrossContactCos.toFixed(4));

document.getElementById("fieldRows").innerHTML = S.perField.map(f =>
  "<tr><td>" + f.field + '</td><td class="n">' + f.n +
  '</td><td class="n">' + f.meanCosToWhole.toFixed(3) +
  '</td><td class="n">' + f.meanCosAcrossContacts.toFixed(3) + "</td></tr>"
).join("");

const css = getComputedStyle(document.documentElement);
const hue = { seeker: "--seeker", candidate: "--candidate", both: "--both" };
function colorOf(role) { return css.getPropertyValue(hue[role]).trim() || "#888"; }

// One distinct, theme-stable color per field (golden-angle hue spacing so
// adjacent fields never land on similar hues regardless of list length).
const fieldColors = new Map(S.fieldOrder.map((f, i) => [f, `hsl(${(i * 137.508) % 360}deg 55% 55%)`]));
function fieldColorOf(field) { return fieldColors.get(field) || "#888"; }

const selectedFields = new Set(S.fieldOrder);
const fieldChecksEl = document.getElementById("fieldChecks");
fieldChecksEl.innerHTML = S.fieldOrder.map(f =>
  '<label class="field-row"><input type="checkbox" data-field="' + f + '" checked />' +
  '<span class="field-swatch" style="background:' + fieldColorOf(f) + '"></span>' + f + "</label>"
).join("");
fieldChecksEl.querySelectorAll("input[data-field]").forEach(inp => {
  inp.addEventListener("change", () => {
    const f = inp.dataset.field;
    if (inp.checked) selectedFields.add(f); else selectedFields.delete(f);
    draw();
  });
});
document.getElementById("fieldsAll").addEventListener("click", () => {
  fieldChecksEl.querySelectorAll("input[data-field]").forEach(inp => { inp.checked = true; selectedFields.add(inp.dataset.field); });
  draw();
});
document.getElementById("fieldsNone").addEventListener("click", () => {
  fieldChecksEl.querySelectorAll("input[data-field]").forEach(inp => { inp.checked = false; selectedFields.delete(inp.dataset.field); });
  draw();
});

const nodeById = new Map();
DATA.nodes.forEach((n, i) => { n.i = i; nodeById.set(n.id, n); });

const pairsByNode = new Map();
DATA.edges.forEach(e => {
  for (const id of [e.s, e.t]) {
    if (!pairsByNode.has(id)) pairsByNode.set(id, []);
    pairsByNode.get(id).push(e);
  }
});

let yaw = 0.6, pitch = -0.28, zoom = 1, focus = null, hover = null;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const stage = document.getElementById("stage");
const tip = document.getElementById("tip");
let W = 0, H = 0, dpr = 1;

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = canvas.clientWidth; H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
new ResizeObserver(resize).observe(stage);

function project(p) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = p[0] * cy - p[2] * sy;
  const z1 = p[0] * sy + p[2] * cy;
  const y1 = p[1] * cp - z1 * sp;
  const z2 = p[1] * sp + z1 * cp;
  const scale = Math.min(W, H) * 0.40 * zoom;
  const persp = 2.6 / (2.6 + z2);
  return { x: W / 2 + x1 * scale * persp, y: H / 2 + y1 * scale * persp, z: z2, s: persp };
}

let dragging = false, lx = 0, ly = 0, moved = false;
canvas.addEventListener("mousedown", e => {
  dragging = true; moved = false; lx = e.clientX; ly = e.clientY;
  canvas.classList.add("dragging");
});
window.addEventListener("mouseup", () => { dragging = false; canvas.classList.remove("dragging"); });
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  const dx = e.clientX - lx, dy = e.clientY - ly;
  if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
  yaw += dx * 0.006; pitch += dy * 0.006;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  lx = e.clientX; ly = e.clientY;
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  zoom = Math.max(0.35, Math.min(9, zoom * (e.deltaY > 0 ? 0.9 : 1.11)));
  draw();
}, { passive: false });

let hits = [];
canvas.addEventListener("mousemove", e => {
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = null, bd = 13 * 13;
  for (const h of hits) {
    const d = (h.x - mx) * (h.x - mx) + (h.y - my) * (h.y - my);
    if (d < bd) { bd = d; best = h; }
  }
  if (best !== hover) { hover = best; draw(); }
  if (best) showTip(best, mx, my); else tip.classList.remove("on");
});
canvas.addEventListener("mouseleave", () => { hover = null; tip.classList.remove("on"); draw(); });
canvas.addEventListener("click", () => {
  if (moved) return;
  focus = (hover && hover.node && focus !== hover.node.id) ? hover.node.id : null;
  draw();
});

function esc(s) { return String(s == null ? "" : s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

function showTip(h, mx, my) {
  const n = h.node;
  const roleWord = n.role === "both" ? "seeker &amp; candidate" : n.role;
  let html = '<div class="who">' + esc(n.id) + " &middot; " + roleWord + "</div>";
  if (h.kind === "whole") {
    html += '<div class="hd">Full profile</div>';
    html += '<div class="body">' + esc(n.positioning || "—") + "</div>";
    const mine = pairsByNode.get(n.id) || [];
    const good = mine.filter(e => e.l === "pos").length;
    html += '<div class="meta">' + n.fields.length + " field" + (n.fields.length === 1 ? "" : "s") +
      " &middot; " + n.sections.length + " ask" + (n.sections.length === 1 ? "" : "s") + "<br />" +
      '<span style="color:var(--pos)">' + good + " good</span> &middot; " +
      '<span style="color:var(--neg)">' + (mine.length - good) + " bad</span> match" +
      (mine.length === 1 ? "" : "es") + "</div>";
  } else if (h.kind === "field") {
    const f = n.fields[h.si];
    html += '<div class="hd">' + esc(f.field) + " (alone)</div>";
    html += '<div class="body">' + esc(f.text) + "</div>";
    html += '<div class="meta"><span class="num">' + f.cos.toFixed(3) + "</span> similar to the whole profile</div>";
  } else {
    const s = n.sections[h.si];
    html += '<div class="hd">Ask (alone)</div>';
    html += '<div class="body">' + esc(s.text) + "</div>";
    html += '<div class="meta">ask ' + (h.si + 1) + " of " + n.sections.length +
      " &middot; <span class=\"num\">" + s.cos.toFixed(3) + "</span> similar to the whole profile</div>";
  }
  tip.innerHTML = html;
  tip.classList.add("on");
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let tx = mx + 16, ty = my + 16;
  if (tx + tw > W - 8) tx = mx - tw - 16;
  if (ty + th > H - 8) ty = Math.max(8, my - th - 16);
  tip.style.left = tx + "px"; tip.style.top = ty + "px";
}

const ampEl = document.getElementById("amp");
const ampVal = document.getElementById("ampVal");
const showWhole = document.getElementById("showWhole");
const showSections = document.getElementById("showSections");
const showPairs = document.getElementById("showPairs");
let amp = 1;
ampEl.addEventListener("input", () => {
  amp = parseFloat(ampEl.value);
  ampVal.textContent = amp.toFixed(1) + "×";
  draw();
});
showWhole.addEventListener("change", draw);
showSections.addEventListener("change", draw);
showPairs.addEventListener("change", draw);
document.getElementById("reset").addEventListener("click", () => {
  yaw = 0.6; pitch = -0.28; zoom = 1; focus = null;
  amp = 1; ampEl.value = "1"; ampVal.textContent = "1.0×";
  showWhole.checked = true; showSections.checked = true; showPairs.checked = true;
  fieldChecksEl.querySelectorAll("input[data-field]").forEach(inp => { inp.checked = true; selectedFields.add(inp.dataset.field); });
  draw();
});

function offsetPoint(anchor, xyz) {
  return [
    anchor[0] + (xyz[0] - anchor[0]) * amp,
    anchor[1] + (xyz[1] - anchor[1]) * amp,
    anchor[2] + (xyz[2] - anchor[2]) * amp,
  ];
}

function draw() {
  if (!W || !H) return;
  ctx.clearRect(0, 0, W, H);
  hits = [];

  const drawWhole = showWhole.checked;
  const lookingForOn = selectedFields.has("lookingFor");
  const drawSections = showSections.checked && lookingForOn;
  const dim = id => focus && focus !== id;
  const items = [];

  for (const n of DATA.nodes) {
    const wp = project(n.whole);
    const faded = dim(n.id);
    const col = colorOf(n.role);
    if (drawWhole) {
      items.push({ t: "node", kind: "whole", node: n, x: wp.x, y: wp.y, z: wp.z, s: wp.s, col, faded });
    }

    // field dots hang off the whole-profile anchor, one at a time per checkbox
    for (let si = 0; si < n.fields.length; si++) {
      const f = n.fields[si];
      if (!selectedFields.has(f.field)) continue;
      const fcol = fieldColorOf(f.field);
      const p = offsetPoint(n.whole, f.xyz);
      const fp = project(p);
      items.push({ t: "tether", x0: wp.x, y0: wp.y, x: fp.x, y: fp.y, z: (wp.z + fp.z) / 2, col: fcol, faded });
      items.push({ t: "node", kind: "field", node: n, si, x: fp.x, y: fp.y, z: fp.z, s: fp.s, col: fcol, faded });
    }

    // section dots hang off the lookingFor field dot, only when lookingFor is selected
    if (drawSections) {
      const anchor = offsetPoint(n.whole, n.fields.find(f => f.field === "lookingFor")?.xyz ?? n.whole);
      const anchorProj = project(anchor);
      for (let si = 0; si < n.sections.length; si++) {
        const s = n.sections[si];
        const p = offsetPoint(anchor, s.xyz);
        const sp = project(p);
        items.push({ t: "tether", x0: anchorProj.x, y0: anchorProj.y, x: sp.x, y: sp.y, z: (anchorProj.z + sp.z) / 2, col: "var(--section)", faded });
        items.push({ t: "node", kind: "section", node: n, si, x: sp.x, y: sp.y, z: sp.z, s: sp.s, col: "var(--section)", faded });
      }
    }
  }

  if (showPairs.checked) {
    for (const e of DATA.edges) {
      const a = nodeById.get(e.s), b = nodeById.get(e.t);
      if (!a || !b) continue;
      const faded = focus ? (focus !== e.s && focus !== e.t) : false;
      const lit = !!(hover && (hover.node.id === e.s || hover.node.id === e.t));
      const pa = project(a.whole), pb = project(b.whole);
      items.push({
        t: "pair", x0: pa.x, y0: pa.y, x: pb.x, y: pb.y, z: (pa.z + pb.z) / 2,
        col: css.getPropertyValue(e.l === "pos" ? "--pos" : "--neg").trim(), faded, lit,
      });
    }
  }

  items.sort((a, b) => b.z - a.z);

  for (const it of items) {
    if (it.t === "tether") {
      ctx.save();
      ctx.globalAlpha = it.faded ? 0.05 : 0.32;
      ctx.strokeStyle = it.col.startsWith("var") ? css.getPropertyValue(it.col.slice(4, -1)).trim() : it.col;
      ctx.lineWidth = 1; ctx.setLineDash([1.5, 3]);
      ctx.beginPath(); ctx.moveTo(it.x0, it.y0); ctx.lineTo(it.x, it.y); ctx.stroke();
      ctx.restore();
    } else if (it.t === "pair") {
      ctx.save();
      ctx.globalAlpha = it.faded ? 0.04 : (it.lit ? 0.95 : 0.6);
      ctx.strokeStyle = it.col; ctx.lineWidth = it.lit ? 2.2 : 1.4;
      ctx.beginPath(); ctx.moveTo(it.x0, it.y0); ctx.lineTo(it.x, it.y); ctx.stroke();
      ctx.restore();
    } else {
      const isWhole = it.kind === "whole";
      const isField = it.kind === "field";
      const on = hover && hover.node === it.node &&
        (hover.kind === it.kind) && (isWhole || hover.si === it.si);
      const baseR = isWhole ? 5.2 : (isField ? 3.0 : 1.9);
      const r = baseR * it.s * (on ? 1.7 : 1);
      const col = it.col.startsWith("var") ? css.getPropertyValue(it.col.slice(4, -1)).trim() : it.col;
      ctx.save();
      ctx.globalAlpha = it.faded ? 0.07 : (isWhole ? 0.95 : 0.55);
      if (isWhole) {
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(it.x, it.y, r, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = it.faded ? 0.1 : 1;
        ctx.strokeStyle = css.getPropertyValue("--card").trim();
        ctx.lineWidth = 1.2; ctx.stroke();
      } else {
        ctx.strokeStyle = col; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.arc(it.x, it.y, r, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.restore();
      hits.push({ x: it.x, y: it.y, kind: it.kind, node: it.node, si: it.si });
    }
  }
}

const mq = window.matchMedia("(prefers-color-scheme: dark)");
mq.addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
resize();
</script>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    payload = build_payload(args.db_dir, args.collection, args.meta)
    stats = payload["stats"]
    html = HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    print(f"contacts        {stats['nContacts']}  ({stats['nSeekers']} seeker / {stats['nCandidates']} candidate)")
    print(f"fields          {stats['nFields']}  (max {stats['maxFields']} on one contact)")
    print(f"sections        {stats['nSections']}  (max {stats['maxSections']} on one contact)")
    print(f"whole<->field   {stats['wholeFieldCos']}")
    print(f"field<->field   {stats['fieldFieldSameContactCos']}")
    print(f"whole<->section {stats['wholeSectionCos']}")
    print(f"section<->sect  {stats['sectionSectionCos']}")
    print(f"across contacts {stats['acrossContactCos']}")
    print(f"constellation   {stats['constellationRatio']} of cloud radius")
    print(f"PCA3 variance   {stats['evrSum']}")
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
