#!/usr/bin/env python3
"""Build a self-contained 3D map of the holdout contacts in voyage-4-nano space.

Consumes the embeddings written by
baselines/voyage_nano_sectioned/modal_embed_space.py (one whole-profile vector
per contact + one vector per lookingFor section, all encoded in a single pass so
they share a space), projects all of them jointly to 3 PCA components, and emits
one HTML file with a canvas 3D scatter: each contact's whole-profile anchor is
tethered to its own section points by dotted lines.

Usage:
  python scripts/build_holdout_embedding_space_3d.py
  python scripts/build_holdout_embedding_space_3d.py --amp 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_RUN = ROOT / "artifacts" / "voyage_nano_sectioned_modal" / "embed_space_holdout"
DEFAULT_OUT = ROOT / "docs" / "holdout-embedding-space-3d.html"

SECTION_CHARS = 420
POSITIONING_CHARS = 260


def truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def section_label(text: str) -> str:
    """First markdown-ish heading line of a section, else its opening words."""
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    first = first.lstrip("#").strip(" *_-–—:")
    if first and len(first) <= 60:
        return first
    words = " ".join(text.split())
    return truncate(words, 48)


def build_payload(run_dir: Path) -> dict:
    emb = np.load(run_dir / "embeddings.npy")
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = meta["rows"]
    if len(rows) != emb.shape[0]:
        raise ValueError(f"meta rows ({len(rows)}) != embeddings ({emb.shape[0]})")

    # One joint PCA over whole-profile *and* section vectors, so a section's
    # offset from its own anchor is on the same scale as the gap between contacts.
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(emb)
    coords = coords / (np.abs(coords).max() or 1.0)
    evr = pca.explained_variance_ratio_

    contacts_meta = {c["id"]: c for c in meta["contacts"]}
    whole_idx: dict[str, int] = {}
    section_idx: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row["kind"] == "whole":
            whole_idx[row["contactId"]] = i
        else:
            section_idx.setdefault(row["contactId"], []).append(i)

    # Cosine geometry, reported on the raw 1024-d vectors (not the 3D shadow).
    whole_sec_cos: list[float] = []
    sec_sec_cos: list[float] = []
    nodes: list[dict] = []
    for cid, w in whole_idx.items():
        cmeta = contacts_meta[cid]
        roles = cmeta["roles"]
        role = "both" if len(roles) > 1 else roles[0]
        secs = section_idx.get(cid, [])

        cos_to_whole = (emb[secs] @ emb[w]).tolist() if secs else []
        whole_sec_cos.extend(cos_to_whole)
        dispersion = 0.0
        if len(secs) > 1:
            sim = emb[secs] @ emb[secs].T
            iu = np.triu_indices(len(secs), 1)
            sec_sec_cos.extend(sim[iu].tolist())
            # 1 - mean pairwise cosine among this contact's own sections: how
            # unlike each other their separate asks are. See
            # scripts/analyze_section_dispersion.py.
            dispersion = 1.0 - float(sim[iu].mean())

        nodes.append(
            {
                "id": cid,
                "role": role,
                "pairCount": cmeta["pairCount"],
                "dispersion": round(dispersion, 4),
                "positioning": truncate(cmeta["profile"].get("positioning"), POSITIONING_CHARS),
                "whole": [round(float(v), 5) for v in coords[w]],
                "sections": [
                    {
                        "label": section_label(rows[s]["text"] or ""),
                        "text": truncate(rows[s]["text"], SECTION_CHARS),
                        "cos": round(float(c), 4),
                        "xyz": [round(float(v), 5) for v in coords[s]],
                    }
                    for s, c in zip(secs, cos_to_whole)
                ],
            }
        )

    nodes.sort(key=lambda n: -len(n["sections"]))

    whole_vecs = emb[[whole_idx[c] for c in whole_idx]]
    across = whole_vecs @ whole_vecs.T
    iu = np.triu_indices(len(whole_vecs), 1)

    # How big is a contact's own constellation next to the whole cloud? Answers
    # "would sectioning even move this point far enough to matter?"
    radii = [
        float(np.linalg.norm(coords[section_idx[c]] - coords[whole_idx[c]], axis=1).mean())
        for c in whole_idx
        if section_idx.get(c)
    ]
    cloud_pts = coords[[whole_idx[c] for c in whole_idx]]
    cloud_radius = float(np.linalg.norm(cloud_pts - cloud_pts.mean(axis=0), axis=1).mean())

    edges = [
        {"s": e["source"], "t": e["target"], "l": e["label"], "q": truncate(e.get("searchQuery"), 220)}
        for e in meta["edges"]
    ]

    stats = {
        "nContacts": len(nodes),
        "nSections": sum(len(n["sections"]) for n in nodes),
        "nPairs": meta["n_pairs"],
        "nPositives": meta["n_positives"],
        "nNegatives": meta["n_negatives"],
        "nSeekers": sum(1 for n in nodes if n["role"] in ("seeker", "both")),
        "nCandidates": sum(1 for n in nodes if n["role"] in ("candidate", "both")),
        "wholeSectionCos": round(float(np.mean(whole_sec_cos)), 4),
        "wholeSectionCosMin": round(float(np.min(whole_sec_cos)), 4),
        "secSecCos": round(float(np.mean(sec_sec_cos)), 4),
        "acrossContactCos": round(float(np.mean(across[iu])), 4),
        "constellationRatio": round(float(np.mean(radii) / cloud_radius), 4),
        "evr": [round(float(v), 4) for v in evr],
        "evrSum": round(float(evr[:3].sum()), 4),
        "model": meta["model_name"],
        "dim": meta["truncate_dim"],
        "maxSections": max(len(n["sections"]) for n in nodes),
    }
    payload = {"nodes": nodes, "edges": edges, "stats": stats}

    # Optional: the dispersion write-up renders only if the analysis has been run.
    analysis_path = run_dir / "dispersion_analysis.json"
    if analysis_path.exists():
        payload["analysis"] = json.loads(analysis_path.read_text())
    else:
        print(f"note: {analysis_path.name} missing — run scripts/analyze_section_dispersion.py")

    return payload


HTML = r"""<title>Holdout contacts in voyage-4-nano space</title>
<style>
  :root {
    --ground: #ecefee; --card: #ffffff; --glass: rgba(255,255,255,0.88);
    --ink: #141d20; --ink-soft: #3b4a4d; --muted: #647476; --hairline: #ccd4d2;
    --seeker: #1f7a8c; --candidate: #c2603f; --both: #a97f1f;
    --pos: #2e8b57; --neg: #3a52b0;
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
      --pos: #5cc98a; --neg: #8098ee;
      --shadow: 0 8px 28px rgba(0,0,0,0.42);
    }
  }
  :root[data-theme="dark"] {
    --ground: #0f1618; --card: #171f21; --glass: rgba(23,31,33,0.88);
    --ink: #e6edeb; --ink-soft: #b6c5c3; --muted: #8b9c9a; --hairline: #26312f;
    --seeker: #58c8db; --candidate: #f0906a; --both: #ddb455;
    --pos: #5cc98a; --neg: #8098ee;
    --shadow: 0 8px 28px rgba(0,0,0,0.42);
  }
  :root[data-theme="light"] {
    --ground: #ecefee; --card: #ffffff; --glass: rgba(255,255,255,0.88);
    --ink: #141d20; --ink-soft: #3b4a4d; --muted: #647476; --hairline: #ccd4d2;
    --seeker: #1f7a8c; --candidate: #c2603f; --both: #a97f1f;
    --pos: #2e8b57; --neg: #3a52b0;
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

  /* ---- instrument bar ---- */
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

  /* ---- stage ---- */
  .stage {
    position: relative; height: min(74vh, 780px); min-height: 460px;
    border-bottom: 1px solid var(--hairline); overflow: hidden;
  }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }

  .rail {
    position: absolute; top: 16px; left: 16px; width: 232px; max-width: calc(100% - 32px);
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
  .tether { width: 18px; height: 0; border-top: 1.5px dotted var(--muted); flex: none; }
  .wire { width: 18px; height: 2px; border-radius: 1px; flex: none; }

  /* ---- controls ---- */
  .controls {
    position: absolute; left: 16px; right: 16px; bottom: 16px;
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    background: var(--glass); backdrop-filter: blur(9px);
    border: 1px solid var(--hairline); border-radius: 3px; box-shadow: var(--shadow);
    padding: 11px 15px;
  }
  .ctl { display: flex; align-items: center; gap: 9px; }
  .ctl label { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
  input[type="range"] { width: 140px; accent-color: var(--seeker); }
  .amp-val { font-family: var(--mono); font-size: 13px; min-width: 34px; font-variant-numeric: tabular-nums; }
  .toggle { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-soft); cursor: pointer; }
  button.reset {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    background: none; border: 1px solid var(--hairline); color: var(--ink-soft);
    padding: 5px 11px; border-radius: 2px; cursor: pointer;
  }
  button.reset:hover { border-color: var(--seeker); color: var(--seeker); }
  :focus-visible { outline: 2px solid var(--seeker); outline-offset: 2px; }
  .hint { margin-left: auto; font-size: 11.5px; color: var(--muted); }

  /* ---- tooltip ---- */
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

  /* ---- prose ---- */
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
  <h1>Where the holdout lives in voyage-4-nano space</h1>
  <span class="sub">115 contacts &middot; 951 <code style="font-family:var(--mono);font-size:12px">lookingFor</code> sections &middot; profile text only, no search query</span>
  <span class="spacer"></span>
  <span class="eyebrow" id="modelTag"></span>
</div>

<div class="stage" id="stage">
  <canvas id="c"></canvas>
  <div class="rail">
    <div class="stat">
      <span class="k eyebrow">Anchor &rarr; section similarity</span>
      <span class="v" id="sWholeSec"></span>
      <span class="n">Mean cosine between a contact&rsquo;s whole-profile vector and each of its own section vectors. Across <em>different</em> contacts it is only <span class="num" id="sAcross"></span>.</span>
    </div>
    <div class="rule"></div>
    <div class="stat">
      <span class="k eyebrow">Constellation vs. cloud</span>
      <span class="v" id="sRatio"></span>
      <span class="n">A contact&rsquo;s section spread as a fraction of the distance between contacts. Sections stay firmly inside their own contact&rsquo;s neighbourhood.</span>
    </div>
    <div class="rule"></div>
    <div class="legend">
      <div class="row"><span class="dot" style="background:var(--seeker)"></span> Seeker anchor <span class="num" id="lSeek" style="color:var(--muted)"></span></div>
      <div class="row"><span class="dot" style="background:var(--candidate)"></span> Candidate anchor <span class="num" id="lCand" style="color:var(--muted)"></span></div>
      <div class="row"><span class="dot" style="background:var(--both)"></span> Both roles</div>
      <div class="row"><span class="dot hollow" style="color:var(--muted)"></span> One <code style="font-size:11px">lookingFor</code> section</div>
      <div class="row"><span class="tether"></span> Section tethered to its anchor</div>
      <div class="row"><span class="wire" style="background:var(--pos)"></span> Good match <span class="num" id="lPos" style="color:var(--muted)"></span></div>
      <div class="row"><span class="wire" style="background:var(--neg)"></span> Bad match <span class="num" id="lNeg" style="color:var(--muted)"></span></div>
    </div>
    <div class="rule"></div>
    <div class="note">First 3 PCA components of the shared 1024-d space, holding <span class="num" id="sEvr"></span> of total variance.</div>
  </div>

  <div class="controls">
    <div class="ctl">
      <label for="amp">Section spread</label>
      <input type="range" id="amp" min="1" max="12" step="0.5" value="1" />
      <span class="amp-val" id="ampVal">1.0&times;</span>
    </div>
    <label class="toggle"><input type="checkbox" id="showSections" checked /> Sections</label>
    <label class="toggle"><input type="checkbox" id="showPairs" checked /> Match lines</label>
    <button class="reset" id="reset">Reset view</button>
    <span class="hint">drag to rotate &middot; scroll to zoom &middot; click an anchor to isolate</span>
  </div>

  <div class="tip" id="tip"></div>
</div>

<div class="read">
  <h2>What this is</h2>
  <p>Every contact in the frozen 69-pair real holdout appears twice over. Once as an
  <strong>anchor</strong> &mdash; their whole profile, serialised exactly as the baselines
  serialise it, with the search query deliberately left out. And once per
  <code>lookingFor</code> section &mdash; the same full profile with <code>lookingFor</code>
  swapped down to a single one of its asks. That is the precise text variant the
  sectioning experiment scores against, so this map shows the geometry that experiment
  is actually operating on.</p>
  <p>All 1,066 vectors were encoded in one <code>voyage-4-nano</code> pass on an L4, then
  projected together through a single PCA so that a section&rsquo;s offset from its own
  anchor is drawn on the same scale as the gap between two different contacts.</p>

  <h2>The thing the map shows</h2>
  <p>Sectioning barely moves a profile. A section vector sits at a mean cosine of
  <strong id="pWholeSec"></strong> to its own anchor, while two different contacts sit at
  <strong id="pAcross"></strong>. Swapping <code>lookingFor</code> down to one ask changes a
  profile&rsquo;s position by a rounding error next to the distance separating one person
  from another &mdash; a contact&rsquo;s sections form a tight knot roughly
  <strong id="pRatio"></strong> the radius of the cloud they sit in. At <span class="num">1.0&times;</span>
  spread the constellations are almost invisible; that is the honest picture, and the
  slider only inflates the offsets so the tether structure can be read.</p>
  <p>The reason is structural. A section variant keeps <em>the entire rest of the
  profile</em> &mdash; positioning, background, notes, preferences &mdash; and edits one field.
  The shared text dominates the embedding, so every section inherits almost all of its
  coordinates from the profile it came out of.</p>

  <div class="tablewrap">
    <table>
      <thead><tr><th>Measurement</th><th>Cosine</th><th>Reading</th></tr></thead>
      <tbody>
        <tr><td>Anchor &harr; its own sections</td><td class="n" id="tWholeSec"></td><td>near-identical</td></tr>
        <tr><td>Section &harr; section, same contact</td><td class="n" id="tSecSec"></td><td>near-identical</td></tr>
        <tr><td>Anchor &harr; anchor, different contacts</td><td class="n" id="tAcross"></td><td>genuinely far apart</td></tr>
      </tbody>
    </table>
  </div>

  <h2>Why it matters for the sectioning result</h2>
  <p>Seeker-sectioning measurably helps at the top of the ranking &mdash; top-1 retrieval
  went from 27.6% to 34.5% &mdash; and it does that on offsets this small. The signal it
  adds is real but it is a fine adjustment layered on a position that is set almost
  entirely by the rest of the profile, not a relocation of the contact.</p>
  <p>That also frames the Recall@10 question. If every section sits within
  <span class="num" id="pWholeSec2"></span> cosine of its anchor, the max-over-sections score
  is being taken over a set of very similar candidates, and which one wins is decided by
  small differences. Softening the aggregation could not recover Recall@10 partly
  because there was never much spread to aggregate over.</p>

  <h2>Do vague profiles get worse matches?</h2>
  <p>A contact whose sections disagree with each other is carrying several unrelated
  asks at once. Call that <strong>dispersion</strong> &mdash; one minus the mean cosine
  among a contact&rsquo;s own sections. It is worth measuring separately from sheer
  section count, and it is: the two are almost uncorrelated (<span class="num">r&nbsp;=&nbsp;0.06</span>),
  so dispersion tracks breadth of intent rather than volume of text.</p>

  <p>Asked directly &mdash; does dispersion tell you whether a pair was labelled good or
  bad? &mdash; the answer is <strong>no</strong>. Across both sides of the pair and four
  different shape measures, every ROC-AUC lands between
  <span class="num" id="q1lo"></span> and <span class="num" id="q1hi"></span>, and nothing
  survives a permutation test. That is the sensible result: the label describes a
  <em>relationship</em> between two people, so a property of one profile on its own has
  no particular reason to predict it.</p>

  <p>Asked the other way &mdash; does dispersion predict how <em>hard</em> a seeker is to
  serve? &mdash; there is a real effect. Ranking each positive pair&rsquo;s true match
  against all <span class="num" id="q2corpus"></span> holdout candidates, a seeker&rsquo;s
  dispersion correlates with how far down their true match falls at
  <strong id="q2rho"></strong> (Spearman, permutation <span class="num" id="q2p"></span>,
  95% CI <span class="num" id="q2ci"></span>). It holds at
  <span class="num" id="q2partial"></span> after controlling for section count, so this is
  not just &ldquo;longer profiles are harder&rdquo;. <strong>Scattered seekers are harder to
  match &mdash; not more likely to be given a bad match, but more likely to have their
  right match buried.</strong></p>

  <p>And that is where sectioning earns its keep. Splitting the holdout&rsquo;s positive
  queries at the median dispersion:</p>

  <div class="tablewrap">
    <table>
      <thead><tr><th>Seeker group</th><th>Queries</th><th>MRR, whole profile</th><th>MRR, sectioned</th><th>Change</th></tr></thead>
      <tbody id="q3rows"></tbody>
    </table>
  </div>

  <p>Essentially <strong>all</strong> of sectioning&rsquo;s benefit goes to the multi-intent
  seekers. Focused seekers gain nothing, because there was nothing blurred together to
  separate. This is the mechanism the earlier follow-ups were reaching for: sectioning is
  not a general improvement to the encoder, it is a targeted repair for profiles that
  carry several live threads at once &mdash; which also predicts it would be wasted effort
  on single-topic fields like <code>locationAvailability</code>.</p>

  <p class="note">Sample-size caution: 29 positive queries, split 17 / 12. The correlation&rsquo;s
  bootstrap interval reaches close to zero at the low end and this was one of several
  measures examined, so treat it as a well-formed lead rather than a settled number. The
  retrieval figures on this page are computed from profile text alone, without
  <code>searchQuery</code>, and so are not comparable to
  <code>docs/baseline-results-holdout.md</code>; what is comparable is whole-profile
  against sectioned within this page, which is the contrast the argument rests on.
  Reproduce with <code>scripts/analyze_section_dispersion.py</code>.</p>

  <p class="note">Profile text only &mdash; <code>searchQuery</code> is excluded from every vector
  on this page, on both sides. Cosines are computed on the raw 1024-d vectors, not on the
  3D projection. Sections are split on the blank-line paragraph breaks already present in
  the data, the same literal split
  <code>baselines/voyage_nano_sectioned/text.py</code> uses.</p>
</div>

<script>
const DATA = __DATA__;
const S = DATA.stats;

/* ---------- fill in the readings ---------- */
const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
set("modelTag", S.model + " · " + S.dim + "d");
set("sWholeSec", S.wholeSectionCos.toFixed(3));
set("sAcross", S.acrossContactCos.toFixed(3));
set("sRatio", (S.constellationRatio * 100).toFixed(0) + "%");
set("sEvr", (S.evrSum * 100).toFixed(1) + "%");
set("lSeek", S.nSeekers);
set("lCand", S.nCandidates);
set("lPos", S.nPositives);
set("lNeg", S.nNegatives);
set("pWholeSec", S.wholeSectionCos.toFixed(4));
set("pWholeSec2", S.wholeSectionCos.toFixed(3));
set("pAcross", S.acrossContactCos.toFixed(4));
set("pRatio", (S.constellationRatio * 100).toFixed(0) + "%");
set("tWholeSec", S.wholeSectionCos.toFixed(4));
set("tSecSec", S.secSecCos.toFixed(4));
set("tAcross", S.acrossContactCos.toFixed(4));

/* ---------- dispersion analysis readings ---------- */
const A = DATA.analysis;
if (A) {
  const aucs = A.q1_label_prediction.map(r => r.auc);
  set("q1lo", Math.min(...aucs).toFixed(3));
  set("q1hi", Math.max(...aucs).toFixed(3));
  const q2 = A.q2_retrieval_difficulty;
  set("q2corpus", A.corpus_size);
  set("q2rho", "ρ = " + q2.spearman_rho.toFixed(3));
  set("q2p", "p = " + q2.perm_p.toFixed(3));
  set("q2ci", "[" + q2.bootstrap_ci95.map(v => v.toFixed(2)).join(", ") + "]");
  set("q2partial", "ρ = " + q2.partial_rho_controlling_section_count.toFixed(3));

  const nameOf = { focused: "Focused (low dispersion)", multi_intent: "Multi-intent (high dispersion)" };
  document.getElementById("q3rows").innerHTML = A.q3_who_sectioning_helps.groups.map(g =>
    "<tr><td>" + nameOf[g.name] + '</td><td class="n">' + g.n +
    '</td><td class="n">' + g.mrr_whole.toFixed(3) +
    '</td><td class="n">' + g.mrr_sectioned.toFixed(3) +
    '</td><td class="n" style="color:' + (g.delta > 0.01 ? "var(--pos)" : "var(--muted)") + '">' +
    (g.delta >= 0 ? "+" : "") + g.delta.toFixed(3) + "</td></tr>"
  ).join("");
}

/* ---------- flatten to draw lists ---------- */
const css = getComputedStyle(document.documentElement);
const hue = { seeker: "--seeker", candidate: "--candidate", both: "--both" };
function colorOf(role) { return css.getPropertyValue(hue[role]).trim() || "#888"; }

const nodeById = new Map();
DATA.nodes.forEach((n, i) => { n.i = i; nodeById.set(n.id, n); });

const pairsByNode = new Map();
DATA.edges.forEach(e => {
  for (const id of [e.s, e.t]) {
    if (!pairsByNode.has(id)) pairsByNode.set(id, []);
    pairsByNode.get(id).push(e);
  }
});

/* ---------- camera ---------- */
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

/* ---------- interaction ---------- */
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
    html += '<div class="hd">Whole profile</div>';
    html += '<div class="body">' + esc(n.positioning || "—") + "</div>";
    const mine = pairsByNode.get(n.id) || [];
    const good = mine.filter(e => e.l === "pos").length;
    html += '<div class="meta">' + n.sections.length + " lookingFor section" +
      (n.sections.length === 1 ? "" : "s") +
      ' &middot; dispersion <span class="num">' + n.dispersion.toFixed(4) + "</span><br />" +
      '<span style="color:var(--pos)">' + good + " good</span> &middot; " +
      '<span style="color:var(--neg)">' + (mine.length - good) + " bad</span> match" +
      (mine.length === 1 ? "" : "es") + "</div>";
  } else {
    const s = n.sections[h.si];
    html += '<div class="hd">' + esc(s.label) + "</div>";
    html += '<div class="body">' + esc(s.text) + "</div>";
    html += '<div class="meta">section ' + (h.si + 1) + " of " + n.sections.length +
      " &middot; cosine to whole profile <span class=\"num\">" + s.cos.toFixed(4) + "</span></div>";
  }
  tip.innerHTML = html;
  tip.classList.add("on");
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let tx = mx + 16, ty = my + 16;
  if (tx + tw > W - 8) tx = mx - tw - 16;
  if (ty + th > H - 8) ty = Math.max(8, my - th - 16);
  tip.style.left = tx + "px"; tip.style.top = ty + "px";
}

/* ---------- controls ---------- */
const ampEl = document.getElementById("amp");
const ampVal = document.getElementById("ampVal");
const showSections = document.getElementById("showSections");
const showPairs = document.getElementById("showPairs");
let amp = 1;
ampEl.addEventListener("input", () => {
  amp = parseFloat(ampEl.value);
  ampVal.textContent = amp.toFixed(1) + "×";
  draw();
});
showSections.addEventListener("change", draw);
showPairs.addEventListener("change", draw);
document.getElementById("reset").addEventListener("click", () => {
  yaw = 0.6; pitch = -0.28; zoom = 1; focus = null;
  amp = 1; ampEl.value = "1"; ampVal.textContent = "1.0×";
  showSections.checked = true; showPairs.checked = true;
  draw();
});

/* ---------- render ---------- */
function draw() {
  if (!W || !H) return;
  ctx.clearRect(0, 0, W, H);
  hits = [];

  const drawSec = showSections.checked;
  const dim = id => focus && focus !== id;
  const items = [];

  for (const n of DATA.nodes) {
    const wp = project(n.whole);
    const faded = dim(n.id);
    const col = colorOf(n.role);
    items.push({ t: "node", kind: "whole", node: n, x: wp.x, y: wp.y, z: wp.z, s: wp.s, col, faded });

    if (!drawSec) continue;
    for (let si = 0; si < n.sections.length; si++) {
      const s = n.sections[si];
      // Inflate the offset from the anchor, never the anchor itself: the
      // constellation grows in place instead of the whole cloud stretching.
      const p = [
        n.whole[0] + (s.xyz[0] - n.whole[0]) * amp,
        n.whole[1] + (s.xyz[1] - n.whole[1]) * amp,
        n.whole[2] + (s.xyz[2] - n.whole[2]) * amp,
      ];
      const sp = project(p);
      items.push({ t: "tether", x0: wp.x, y0: wp.y, x: sp.x, y: sp.y, z: (wp.z + sp.z) / 2, col, faded });
      items.push({ t: "node", kind: "section", node: n, si, x: sp.x, y: sp.y, z: sp.z, s: sp.s, col, faded });
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

  items.sort((a, b) => b.z - a.z);  // painter's algorithm: far first

  for (const it of items) {
    if (it.t === "tether") {
      ctx.save();
      ctx.globalAlpha = it.faded ? 0.05 : 0.32;
      ctx.strokeStyle = it.col; ctx.lineWidth = 1; ctx.setLineDash([1.5, 3]);
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
      const on = hover && hover.node === it.node &&
        (hover.kind === it.kind) && (isWhole || hover.si === it.si);
      const r = (isWhole ? 5.2 : 2.3) * it.s * (on ? 1.7 : 1);
      ctx.save();
      ctx.globalAlpha = it.faded ? 0.07 : (isWhole ? 0.95 : 0.5);
      if (isWhole) {
        ctx.fillStyle = it.col;
        ctx.beginPath(); ctx.arc(it.x, it.y, r, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = it.faded ? 0.1 : 1;
        ctx.strokeStyle = css.getPropertyValue("--card").trim();
        ctx.lineWidth = 1.2; ctx.stroke();
      } else {
        ctx.strokeStyle = it.col; ctx.lineWidth = 1.3;
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
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    payload = build_payload(args.run_dir)
    stats = payload["stats"]
    html = HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    print(f"contacts        {stats['nContacts']}  ({stats['nSeekers']} seeker / {stats['nCandidates']} candidate)")
    print(f"sections        {stats['nSections']}  (max {stats['maxSections']} on one contact)")
    print(f"whole<->section {stats['wholeSectionCos']}  (min {stats['wholeSectionCosMin']})")
    print(f"section<->sect  {stats['secSecCos']}")
    print(f"across contacts {stats['acrossContactCos']}")
    print(f"constellation   {stats['constellationRatio']} of cloud radius")
    print(f"PCA3 variance   {stats['evrSum']}")
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
