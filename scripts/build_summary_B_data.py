#!/usr/bin/env python3
"""Build data/summary_B_data.html — quantitative overview of locked B-data.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "B-data.json"
DEFAULT_OUT = ROOT / "data" / "summary_B_data.html"

PROFILE_FIELDS = (
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
)


def _pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def _len_stats(vals: list[int]) -> dict[str, float | int]:
    if not vals:
        return {"n": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    s = sorted(vals)
    n = len(s)

    def pct(p: float) -> int:
        i = min(n - 1, max(0, int(math.ceil(p * n) - 1)))
        return s[i]

    return {
        "n": n,
        "min": s[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": s[-1],
        "mean": round(statistics.fmean(s), 1),
    }


def _profile_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj if isinstance(obj, dict) else {}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _identity_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer candidate identity without contactIds via exact contactFile hashes."""
    seeker_hash_to_ids: dict[str, set[str]] = defaultdict(set)
    match_hash_counts: Counter[str] = Counter()
    same_row_self = 0

    for r in rows:
        cid = str(r.get("contactId") or "")
        sh = _profile_hash(r.get("contactFile"))
        if cid:
            seeker_hash_to_ids[sh].add(cid)

    for r in rows:
        sh = _profile_hash(r.get("contactFile"))
        for m in r.get("matches") or []:
            if not isinstance(m, dict):
                continue
            mh = _profile_hash(m.get("contactFile"))
            match_hash_counts[mh] += 1
            if mh == sh:
                same_row_self += 1

    n_match_slots = sum(match_hash_counts.values())
    unique_match_profiles = len(match_hash_counts)
    repeated_profiles = sum(1 for c in match_hash_counts.values() if c > 1)
    slots_from_repeats = sum(c for c in match_hash_counts.values() if c > 1)
    singleton_profiles = sum(1 for c in match_hash_counts.values() if c == 1)

    seeker_hashes = set(seeker_hash_to_ids)
    overlap_hashes = seeker_hashes & set(match_hash_counts)
    overlap_match_slots = sum(match_hash_counts[h] for h in overlap_hashes)
    seekers_as_match: set[str] = set()
    for h in overlap_hashes:
        seekers_as_match |= seeker_hash_to_ids[h]

    # reuse distribution for chart (bucketed)
    reuse_buckets = Counter()
    for c in match_hash_counts.values():
        if c == 1:
            reuse_buckets["1×"] += 1
        elif c == 2:
            reuse_buckets["2×"] += 1
        elif c <= 5:
            reuse_buckets["3–5×"] += 1
        elif c <= 10:
            reuse_buckets["6–10×"] += 1
        else:
            reuse_buckets["11×+"] += 1

    return {
        "match_contact_ids_present": False,
        "method": "exact sha256 of matches[].contactFile JSON",
        "caveat": (
            "Inferred only — identical profile text suggests same person, "
            "but without contactIds this is not authoritative."
        ),
        "match_slots": n_match_slots,
        "unique_match_profiles": unique_match_profiles,
        "singleton_match_profiles": singleton_profiles,
        "repeated_match_profiles": repeated_profiles,
        "match_slots_from_repeated_profiles": slots_from_repeats,
        "pct_slots_from_repeats": round(_pct(slots_from_repeats, n_match_slots), 2),
        "seeker_profiles_also_seen_as_match": len(overlap_hashes),
        "unique_seekers_appearing_as_match": len(seekers_as_match),
        "match_slots_equal_to_some_seeker_profile": overlap_match_slots,
        "pct_match_slots_equal_seeker": round(_pct(overlap_match_slots, n_match_slots), 2),
        "same_row_seeker_equals_match": same_row_self,
        "reuse_bucket_profile_counts": dict(reuse_buckets),
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_rows = len(rows)
    seeker_ids = [r.get("contactId") for r in rows]
    unique_seekers = len(set(seeker_ids))

    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    matches_per_row: list[int] = []
    status_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    query_lens: list[int] = []
    seeker_field_present: Counter[str] = Counter()
    match_field_present: Counter[str] = Counter()
    seeker_field_lens: dict[str, list[int]] = defaultdict(list)
    match_field_lens: dict[str, list[int]] = defaultdict(list)

    rows_with_accept = 0
    rows_with_reject = 0
    rows_with_pending_only = 0
    rows_resolved = 0  # at least one ACCEPT or REJECT
    seeker_row_counts: Counter[str] = Counter()
    seeker_has_accept: set[str] = set()
    seeker_has_reject: set[str] = set()

    total_matches = 0
    for r in rows:
        cid = r.get("contactId") or ""
        seeker_row_counts[cid] += 1
        q = r.get("query") or ""
        if isinstance(q, str):
            query_lens.append(len(q))

        cf = r.get("contactFile") if isinstance(r.get("contactFile"), dict) else {}
        for f in PROFILE_FIELDS:
            if f in cf and isinstance(cf[f], str) and cf[f].strip():
                seeker_field_present[f] += 1
                seeker_field_lens[f].append(len(cf[f]))

        matches = r.get("matches") if isinstance(r.get("matches"), list) else []
        matches_per_row.append(len(matches))
        total_matches += len(matches)

        row_statuses = Counter()
        for m in matches:
            if not isinstance(m, dict):
                continue
            st = str(m.get("status") or "UNKNOWN")
            mt = str(m.get("matchType") or "UNKNOWN")
            status_counts[st] += 1
            type_counts[mt] += 1
            status_by_type[mt][st] += 1
            row_statuses[st] += 1

            mcf = m.get("contactFile") if isinstance(m.get("contactFile"), dict) else {}
            for f in PROFILE_FIELDS:
                if f in mcf and isinstance(mcf[f], str) and mcf[f].strip():
                    match_field_present[f] += 1
                    match_field_lens[f].append(len(mcf[f]))

        if row_statuses.get("ACCEPT", 0) > 0:
            rows_with_accept += 1
            seeker_has_accept.add(cid)
        if row_statuses.get("REJECT", 0) > 0:
            rows_with_reject += 1
            seeker_has_reject.add(cid)
        if (row_statuses.get("ACCEPT", 0) + row_statuses.get("REJECT", 0)) > 0:
            rows_resolved += 1
        elif row_statuses.get("PENDING", 0) > 0 and sum(row_statuses.values()) == row_statuses.get("PENDING", 0):
            rows_with_pending_only += 1

    multi_query_seekers = sum(1 for c in seeker_row_counts.values() if c > 1)
    resolved = status_counts.get("ACCEPT", 0) + status_counts.get("REJECT", 0)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "data/B-data.json",
        "n_rows": n_rows,
        "unique_seekers": unique_seekers,
        "total_matches": total_matches,
        "avg_matches_per_row": round(total_matches / n_rows, 3) if n_rows else 0,
        "matches_per_row_dist": dict(sorted(Counter(matches_per_row).items())),
        "status_counts": dict(status_counts),
        "type_counts": dict(type_counts),
        "status_by_type": {k: dict(v) for k, v in sorted(status_by_type.items())},
        "resolved_matches": resolved,
        "accept_rate_among_resolved": round(
            _pct(status_counts.get("ACCEPT", 0), resolved), 2
        ),
        "reject_rate_among_resolved": round(
            _pct(status_counts.get("REJECT", 0), resolved), 2
        ),
        "pending_rate_all": round(_pct(status_counts.get("PENDING", 0), total_matches), 2),
        "rows_with_accept": rows_with_accept,
        "rows_with_reject": rows_with_reject,
        "rows_resolved": rows_resolved,
        "rows_pending_only": rows_with_pending_only,
        "seekers_with_accept": len(seeker_has_accept),
        "seekers_with_reject": len(seeker_has_reject),
        "multi_query_seekers": multi_query_seekers,
        "single_query_seekers": unique_seekers - multi_query_seekers,
        "rows_per_seeker": _len_stats(list(seeker_row_counts.values())),
        "query_len": _len_stats(query_lens),
        "seeker_field_coverage": {
            f: {
                "n": seeker_field_present[f],
                "pct": round(_pct(seeker_field_present[f], n_rows), 2),
                "len": _len_stats(seeker_field_lens[f]),
            }
            for f in PROFILE_FIELDS
        },
        "match_field_coverage": {
            f: {
                "n": match_field_present[f],
                "pct": round(_pct(match_field_present[f], total_matches), 2),
                "len": _len_stats(match_field_lens[f]),
            }
            for f in PROFILE_FIELDS
        },
        "vs_original": {
            "original_label": "file membership (dataset_positive vs dataset_negative)",
            "b_data_label": "matches[].status (ACCEPT / REJECT / PENDING)",
            "notes_field": "absent in B-data (present in original)",
            "match_contact_id": "absent on B-data matches",
        },
        "identity": _identity_stats(rows),
    }


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(stats: dict[str, Any]) -> str:
    payload = json.dumps(stats, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>B-data summary — Dorby AI</title>
<style>
  :root {{
    --ink: #1a2421;
    --muted: #5d6b66;
    --line: #cfd8d3;
    --paper: #f4f7f5;
    --panel: #ffffff;
    --accent: #1f6b55;
    --accent2: #b45309;
    --accent3: #1d4ed8;
    --pending: #64748b;
    --accept: #15803d;
    --reject: #b91c1c;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    --sans: "Avenir Next", Avenir, "Century Gothic", "Gill Sans", sans-serif;
    --mono: "SF Mono", Menlo, Consolas, monospace;
    --shadow: 0 10px 28px rgba(26, 36, 33, 0.07);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; min-height: 100%; }}
  body {{
    font-family: var(--sans);
    color: var(--ink);
    background:
      radial-gradient(1000px 480px at 0% -10%, #dff0e8 0%, transparent 55%),
      radial-gradient(900px 420px at 100% 0%, #e8eee9 0%, transparent 50%),
      var(--paper);
    line-height: 1.45;
  }}
  .wrap {{
    width: min(1100px, calc(100% - 1.5rem));
    margin: 0 auto;
    padding: 1.75rem 0 3.5rem;
  }}
  h1 {{
    font-family: var(--serif);
    font-size: clamp(1.8rem, 3.5vw, 2.5rem);
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem;
  }}
  .lede {{
    margin: 0 0 1.25rem;
    color: var(--muted);
    max-width: 42rem;
  }}
  .meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
  }}
  .meta code {{
    font-family: var(--mono);
    font-size: 0.8rem;
    background: #e8efeb;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.25rem;
  }}
  .kpi {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    box-shadow: var(--shadow);
  }}
  .kpi .label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 700;
  }}
  .kpi .value {{
    font-family: var(--serif);
    font-size: 1.65rem;
    margin-top: 0.15rem;
  }}
  .kpi .sub {{
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.15rem;
  }}
  section {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.1rem 1.2rem 1.25rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
  }}
  section h2 {{
    font-family: var(--serif);
    font-size: 1.25rem;
    margin: 0 0 0.35rem;
  }}
  section p.note {{
    margin: 0 0 0.9rem;
    color: var(--muted);
    font-size: 0.9rem;
  }}
  .chart-row {{
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 1.25rem;
    align-items: start;
  }}
  @media (max-width: 800px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
  }}
  .bars {{ display: grid; gap: 0.55rem; }}
  .bar-row {{
    display: grid;
    grid-template-columns: 9.5rem 1fr 4.2rem;
    gap: 0.55rem;
    align-items: center;
    font-size: 0.88rem;
  }}
  .bar-row .name {{
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .track {{
    height: 12px;
    background: #e8eeea;
    border-radius: 999px;
    overflow: hidden;
  }}
  .fill {{
    height: 100%;
    border-radius: 999px;
    background: var(--accent);
  }}
  .fill.accept {{ background: var(--accept); }}
  .fill.reject {{ background: var(--reject); }}
  .fill.pending {{ background: var(--pending); }}
  .fill.type0 {{ background: var(--accent); }}
  .fill.type1 {{ background: var(--accent3); }}
  .fill.type2 {{ background: var(--accent2); }}
  .fill.type3 {{ background: #7c3aed; }}
  .count {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
    font-size: 0.82rem;
  }}
  .donut-wrap {{
    display: grid;
    place-items: center;
    gap: 0.75rem;
  }}
  .donut {{
    width: 180px;
    height: 180px;
    border-radius: 50%;
    position: relative;
  }}
  .donut::after {{
    content: "";
    position: absolute;
    inset: 42px;
    background: var(--panel);
    border-radius: 50%;
  }}
  .legend {{
    display: grid;
    gap: 0.35rem;
    font-size: 0.85rem;
  }}
  .legend span {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 0.4rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  th, td {{
    text-align: left;
    padding: 0.45rem 0.5rem;
    border-bottom: 1px solid var(--line);
    font-variant-numeric: tabular-nums;
  }}
  th {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }}
  td.mono {{ font-family: var(--mono); font-size: 0.8rem; }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }}
  @media (max-width: 800px) {{
    .two-col {{ grid-template-columns: 1fr; }}
  }}
  details {{
    margin-top: 0.75rem;
    border-top: 1px solid var(--line);
    padding-top: 0.75rem;
  }}
  summary {{
    cursor: pointer;
    color: var(--accent);
    font-weight: 600;
  }}
  pre {{
    background: #0f1714;
    color: #d7e7df;
    padding: 0.85rem;
    border-radius: 10px;
    overflow: auto;
    font-size: 0.72rem;
    max-height: 320px;
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>B-data summary</h1>
  <p class="lede">
    Quantitative overview of the locked Boardy B-dataset: seekers, queries,
    candidate matches, and ACCEPT / REJECT / PENDING outcomes.
  </p>
  <div class="meta">
    <span>Source <code id="src"></code></span>
    <span>Generated <span id="gen"></span></span>
    <span>Read-only (B-data.json locked)</span>
  </div>

  <div class="grid" id="kpis"></div>

  <section>
    <h2>Match status</h2>
    <p class="note">Label lives on <code>matches[].status</code>, not as separate pos/neg files.</p>
    <div class="chart-row">
      <div class="bars" id="status-bars"></div>
      <div class="donut-wrap">
        <div class="donut" id="status-donut"></div>
        <div class="legend" id="status-legend"></div>
      </div>
    </div>
  </section>

  <section>
    <h2>Match type</h2>
    <p class="note">Channel / source of the intro (<code>matches[].matchType</code>).</p>
    <div class="bars" id="type-bars"></div>
  </section>

  <section>
    <h2>Status × match type</h2>
    <p class="note">How outcomes break down within each channel.</p>
    <div id="status-by-type"></div>
  </section>

  <section>
    <h2>Matches per seeker row</h2>
    <p class="note">Each row is one (seeker, query) with 1–4 attached matches.</p>
    <div class="bars" id="mpr-bars"></div>
  </section>

  <section>
    <h2>Seeker multiplicity</h2>
    <p class="note">Same <code>contactId</code> can appear in multiple rows (different queries).</p>
    <div class="chart-row">
      <div class="bars" id="multi-bars"></div>
      <div>
        <table>
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody id="multi-table"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <h2>Candidate identity (no contactIds)</h2>
    <p class="note" id="identity-note"></p>
    <div class="grid" id="identity-kpis"></div>
    <div class="chart-row" style="margin-top:1rem">
      <div>
        <h3 style="font-size:1rem;margin:0 0 0.6rem;font-family:var(--serif)">Match-profile reuse</h3>
        <p class="note" style="margin-top:0">How often the same exact <code>contactFile</code> appears across match slots.</p>
        <div class="bars" id="reuse-bars"></div>
      </div>
      <div>
        <h3 style="font-size:1rem;margin:0 0 0.6rem;font-family:var(--serif)">Seeker ↔ candidate overlap</h3>
        <table>
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody id="identity-table"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <h2>Profile field coverage</h2>
    <p class="note">Share of profiles with a non-empty string for each field. <code>notes</code> is absent in B-data.</p>
    <div class="two-col">
      <div>
        <h3 style="font-size:1rem;margin:0 0 0.6rem;font-family:var(--serif)">Seeker contactFile</h3>
        <div class="bars" id="seeker-cov"></div>
      </div>
      <div>
        <h3 style="font-size:1rem;margin:0 0 0.6rem;font-family:var(--serif)">Match contactFile</h3>
        <div class="bars" id="match-cov"></div>
      </div>
    </div>
  </section>

  <section>
    <h2>Text length stats</h2>
    <p class="note">Character lengths for queries and profile fields (non-empty only).</p>
    <div class="two-col">
      <table>
        <thead>
          <tr><th>Field</th><th>n</th><th>p50</th><th>p90</th><th>max</th><th>mean</th></tr>
        </thead>
        <tbody id="len-seeker"></tbody>
      </table>
      <table>
        <thead>
          <tr><th>Field</th><th>n</th><th>p50</th><th>p90</th><th>max</th><th>mean</th></tr>
        </thead>
        <tbody id="len-match"></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Vs original seed pairs</h2>
    <table>
      <thead><tr><th>Aspect</th><th>Original (pos/neg)</th><th>B-data</th></tr></thead>
      <tbody>
        <tr>
          <td>Unit</td>
          <td>Labeled pair</td>
          <td>Seeker + query + matches[]</td>
        </tr>
        <tr>
          <td>Label</td>
          <td>File membership</td>
          <td><code>matches[].status</code></td>
        </tr>
        <tr>
          <td>Candidate id</td>
          <td><code>matchContactId</code></td>
          <td>Absent</td>
        </tr>
        <tr>
          <td><code>notes</code> field</td>
          <td>Present</td>
          <td>Absent</td>
        </tr>
        <tr>
          <td>Unresolved</td>
          <td>None (all labeled)</td>
          <td>Mostly <code>PENDING</code></td>
        </tr>
        <tr>
          <td>Candidate uniqueness</td>
          <td>Exact via <code>matchContactId</code></td>
          <td>Inferred only via profile-text hash</td>
        </tr>
        <tr>
          <td>Seeker as candidate</td>
          <td>Checkable via ids</td>
          <td>Only via profile-hash overlap</td>
        </tr>
      </tbody>
    </table>
    <details>
      <summary>Raw stats JSON</summary>
      <pre id="raw"></pre>
    </details>
  </section>
</div>
<script>
const STATS = {payload};

function fmt(n) {{
  return Number(n).toLocaleString("en-US");
}}

function barList(el, items, max, classFor) {{
  el.innerHTML = items.map(([name, count]) => {{
    const pct = max ? (100 * count / max) : 0;
    const cls = classFor ? classFor(name) : "fill";
    return `<div class="bar-row">
      <div class="name" title="${{name}}">${{name}}</div>
      <div class="track"><div class="fill ${{cls}}" style="width:${{pct.toFixed(2)}}%"></div></div>
      <div class="count">${{fmt(count)}}</div>
    </div>`;
  }}).join("");
}}

function statusClass(name) {{
  const n = String(name).toUpperCase();
  if (n === "ACCEPT") return "accept";
  if (n === "REJECT") return "reject";
  if (n === "PENDING") return "pending";
  return "";
}}

const typeClassMap = {{}};
Object.keys(STATS.type_counts || {{}}).sort((a,b) => STATS.type_counts[b]-STATS.type_counts[a])
  .forEach((k, i) => {{ typeClassMap[k] = `type${{i % 4}}`; }});

document.getElementById("src").textContent = STATS.source;
document.getElementById("gen").textContent = STATS.generated_at;
document.getElementById("raw").textContent = JSON.stringify(STATS, null, 2);

const kpis = [
  ["Rows", STATS.n_rows, "seeker × query"],
  ["Unique seekers", STATS.unique_seekers, `${{STATS.multi_query_seekers.toLocaleString()}} with >1 row`],
  ["Matches", STATS.total_matches, `avg ${{STATS.avg_matches_per_row}} / row`],
  ["ACCEPT", STATS.status_counts.ACCEPT || 0, `${{STATS.accept_rate_among_resolved}}% of resolved`],
  ["REJECT", STATS.status_counts.REJECT || 0, `${{STATS.reject_rate_among_resolved}}% of resolved`],
  ["PENDING", STATS.status_counts.PENDING || 0, `${{STATS.pending_rate_all}}% of all matches`],
];
document.getElementById("kpis").innerHTML = kpis.map(([label, value, sub]) => `
  <div class="kpi">
    <div class="label">${{label}}</div>
    <div class="value">${{fmt(value)}}</div>
    <div class="sub">${{sub}}</div>
  </div>`).join("");

const statusItems = Object.entries(STATS.status_counts).sort((a,b) => b[1]-a[1]);
const statusMax = Math.max(...statusItems.map(x => x[1]), 1);
barList(document.getElementById("status-bars"), statusItems, statusMax, statusClass);

const totalStatus = statusItems.reduce((s, [,c]) => s + c, 0) || 1;
const colors = {{ ACCEPT: "#15803d", REJECT: "#b91c1c", PENDING: "#64748b", UNKNOWN: "#94a3b8" }};
let acc = 0;
const stops = statusItems.map(([name, count]) => {{
  const start = acc;
  const pct = 100 * count / totalStatus;
  acc += pct;
  return `${{colors[name] || "#94a3b8"}} ${{start.toFixed(2)}}% ${{acc.toFixed(2)}}%`;
}});
document.getElementById("status-donut").style.background = `conic-gradient(${{stops.join(", ")}})`;
document.getElementById("status-legend").innerHTML = statusItems.map(([name, count]) => `
  <div><span style="background:${{colors[name] || "#94a3b8"}}"></span>
  ${{name}} — ${{fmt(count)}} (${{(100*count/totalStatus).toFixed(1)}}%)</div>`).join("");

const typeItems = Object.entries(STATS.type_counts).sort((a,b) => b[1]-a[1]);
barList(document.getElementById("type-bars"), typeItems, Math.max(...typeItems.map(x => x[1]), 1),
  (name) => typeClassMap[name] || "type0");

const sbt = STATS.status_by_type;
const sbtHtml = Object.entries(sbt).sort((a,b) => {{
  const sa = Object.values(a[1]).reduce((x,y)=>x+y,0);
  const sb = Object.values(b[1]).reduce((x,y)=>x+y,0);
  return sb - sa;
}}).map(([mt, st]) => {{
  const total = Object.values(st).reduce((x,y)=>x+y,0) || 1;
  const parts = ["ACCEPT","REJECT","PENDING"].map(k => {{
    const c = st[k] || 0;
    return `<div class="bar-row">
      <div class="name">${{k}}</div>
      <div class="track"><div class="fill ${{statusClass(k)}}" style="width:${{(100*c/total).toFixed(2)}}%"></div></div>
      <div class="count">${{fmt(c)}}</div>
    </div>`;
  }}).join("");
  return `<div style="margin-bottom:1rem">
    <div style="font-family:var(--mono);font-size:0.8rem;margin-bottom:0.35rem">${{mt}}
      <span style="color:var(--muted)"> · ${{fmt(total)}}</span></div>
    ${{parts}}
  </div>`;
}}).join("");
document.getElementById("status-by-type").innerHTML = sbtHtml;

const mprItems = Object.entries(STATS.matches_per_row_dist).map(([k,v]) => [`${{k}} match${{k==="1"?"":"es"}}`, v]);
barList(document.getElementById("mpr-bars"), mprItems, Math.max(...mprItems.map(x => x[1]), 1));

barList(document.getElementById("multi-bars"), [
  ["Single-query seekers", STATS.single_query_seekers],
  ["Multi-query seekers", STATS.multi_query_seekers],
], Math.max(STATS.single_query_seekers, STATS.multi_query_seekers, 1));

const rps = STATS.rows_per_seeker;
document.getElementById("multi-table").innerHTML = `
  <tr><td>Rows with ≥1 ACCEPT</td><td>${{fmt(STATS.rows_with_accept)}}</td></tr>
  <tr><td>Rows with ≥1 REJECT</td><td>${{fmt(STATS.rows_with_reject)}}</td></tr>
  <tr><td>Rows with any resolved</td><td>${{fmt(STATS.rows_resolved)}}</td></tr>
  <tr><td>Rows pending-only</td><td>${{fmt(STATS.rows_pending_only)}}</td></tr>
  <tr><td>Seekers with ACCEPT</td><td>${{fmt(STATS.seekers_with_accept)}}</td></tr>
  <tr><td>Seekers with REJECT</td><td>${{fmt(STATS.seekers_with_reject)}}</td></tr>
  <tr><td>Rows/seeker p50 / p90 / max</td><td>${{rps.p50}} / ${{rps.p90}} / ${{rps.max}}</td></tr>
  <tr><td>Query length p50 / p90 / max</td><td>${{STATS.query_len.p50}} / ${{STATS.query_len.p90}} / ${{STATS.query_len.max}}</td></tr>
`;

const ID = STATS.identity || {{}};
document.getElementById("identity-note").textContent =
  (ID.match_contact_ids_present === false
    ? "Matches have no contactId. "
    : "") + (ID.caveat || "") +
  (ID.method ? ` Method: ${{ID.method}}.` : "");

document.getElementById("identity-kpis").innerHTML = [
  ["Unique match profiles", ID.unique_match_profiles || 0, `of ${{fmt(ID.match_slots || 0)}} match slots`],
  ["Repeated profiles", ID.repeated_match_profiles || 0, `${{ID.pct_slots_from_repeats || 0}}% of slots`],
  ["Seekers seen as matches", ID.unique_seekers_appearing_as_match || 0, "by exact profile hash"],
  ["Self-match on same row", ID.same_row_seeker_equals_match || 0, "seeker profile = own match"],
].map(([label, value, sub]) => `
  <div class="kpi">
    <div class="label">${{label}}</div>
    <div class="value">${{fmt(value)}}</div>
    <div class="sub">${{sub}}</div>
  </div>`).join("");

const reuseOrder = ["1×", "2×", "3–5×", "6–10×", "11×+"];
const reuseItems = reuseOrder
  .filter(k => (ID.reuse_bucket_profile_counts || {{}})[k])
  .map(k => [k, ID.reuse_bucket_profile_counts[k]]);
barList(
  document.getElementById("reuse-bars"),
  reuseItems,
  Math.max(...reuseItems.map(x => x[1]), 1),
  () => "type2"
);

document.getElementById("identity-table").innerHTML = `
  <tr><td>Match contactIds present</td><td>${{ID.match_contact_ids_present ? "yes" : "no"}}</td></tr>
  <tr><td>Match slots</td><td>${{fmt(ID.match_slots || 0)}}</td></tr>
  <tr><td>Unique match profiles (exact hash)</td><td>${{fmt(ID.unique_match_profiles || 0)}}</td></tr>
  <tr><td>Singleton profiles (appear once)</td><td>${{fmt(ID.singleton_match_profiles || 0)}}</td></tr>
  <tr><td>Repeated profiles (appear ≥2×)</td><td>${{fmt(ID.repeated_match_profiles || 0)}}</td></tr>
  <tr><td>Slots from repeated profiles</td><td>${{fmt(ID.match_slots_from_repeated_profiles || 0)}} (${{ID.pct_slots_from_repeats || 0}}%)</td></tr>
  <tr><td>Seeker profiles also seen as a match</td><td>${{fmt(ID.seeker_profiles_also_seen_as_match || 0)}}</td></tr>
  <tr><td>Unique seekers appearing as a match</td><td>${{fmt(ID.unique_seekers_appearing_as_match || 0)}}</td></tr>
  <tr><td>Match slots equal to some seeker profile</td><td>${{fmt(ID.match_slots_equal_to_some_seeker_profile || 0)}} (${{ID.pct_match_slots_equal_seeker || 0}}%)</td></tr>
  <tr><td>Same-row seeker == match profile</td><td>${{fmt(ID.same_row_seeker_equals_match || 0)}}</td></tr>
`;

function covBars(el, cov, colorClass) {{
  const items = Object.entries(cov).map(([k,v]) => [k, v.pct]);
  const max = 100;
  el.innerHTML = items.map(([name, pct]) => `
    <div class="bar-row">
      <div class="name" title="${{name}}">${{name}}</div>
      <div class="track"><div class="fill ${{colorClass}}" style="width:${{pct}}%"></div></div>
      <div class="count">${{pct.toFixed(1)}}%</div>
    </div>`).join("");
}}
covBars(document.getElementById("seeker-cov"), STATS.seeker_field_coverage, "type0");
covBars(document.getElementById("match-cov"), STATS.match_field_coverage, "type1");

function lenRows(el, cov, extraFirst) {{
  const rows = [];
  if (extraFirst) rows.push(extraFirst);
  for (const [name, v] of Object.entries(cov)) {{
    const L = v.len;
    rows.push(`<tr>
      <td class="mono">${{name}}</td>
      <td>${{fmt(L.n)}}</td>
      <td>${{fmt(L.p50)}}</td>
      <td>${{fmt(L.p90)}}</td>
      <td>${{fmt(L.max)}}</td>
      <td>${{fmt(L.mean)}}</td>
    </tr>`);
  }}
  el.innerHTML = rows.join("");
}}
const q = STATS.query_len;
lenRows(document.getElementById("len-seeker"), STATS.seeker_field_coverage,
  `<tr><td class="mono">query</td><td>${{fmt(q.n)}}</td><td>${{fmt(q.p50)}}</td><td>${{fmt(q.p90)}}</td><td>${{fmt(q.max)}}</td><td>${{fmt(q.mean)}}</td></tr>`);
lenRows(document.getElementById("len-match"), STATS.match_field_coverage, null);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = args.source
    if not source.is_file():
        raise SystemExit(f"Missing source: {source}")
    if source.stat().st_mode & 0o222:
        raise SystemExit(
            f"Refusing to proceed: {source} is writable. Lock it first (chmod 444)."
        )

    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{source} must be a JSON array")

    stats = analyze(rows)
    html = render_html(stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(
        f"rows={stats['n_rows']} seekers={stats['unique_seekers']} "
        f"matches={stats['total_matches']} "
        f"ACCEPT={stats['status_counts'].get('ACCEPT', 0)} "
        f"REJECT={stats['status_counts'].get('REJECT', 0)} "
        f"PENDING={stats['status_counts'].get('PENDING', 0)}"
    )


if __name__ == "__main__":
    main()
