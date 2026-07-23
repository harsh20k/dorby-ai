#!/usr/bin/env python3
"""Embed artifacts/local_gemma_synth/<run>/profiles/*.json into one browsable HTML.

Scans a local_gemma_profile_gen.py run directory and writes a single
self-contained HTML file with a run selector, so you can flip between runs
without regenerating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "artifacts" / "local_gemma_synth"
DEFAULT_OUT = ROOT / "artifacts" / "local_gemma_synth" / "_browser.html"

PROFILE_FIELDS = [
    "positioning", "background", "lookingFor", "notes",
    "locationAvailability", "introPreferences", "personalPreferences",
    "meetingAndSchedulingPreferences",
]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Local gemma profile browser — Dorby AI</title>
<style>
  :root {
    --ink: #14201c; --muted: #5a6b64; --line: #c9d4ce; --paper: #f3f6f4;
    --panel: #ffffff; --accent: #1f6b55; --accent-soft: #d7ebe3;
    --bad: #a4392a; --bad-soft: #f6ded9;
    --shadow: 0 10px 30px rgba(20, 32, 28, 0.08); --radius: 14px;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    --sans: "Avenir Next", Avenir, "Century Gothic", "Gill Sans", "Trebuchet MS", sans-serif;
    --mono: "SF Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: var(--sans); color: var(--ink);
    background: radial-gradient(1200px 600px at 10% -10%, #dff0e8 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #e7eee9 0%, transparent 50%), var(--paper);
    line-height: 1.45;
  }
  .wrap { width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }
  header.hero { display: grid; gap: 1rem; margin-bottom: 1.5rem; }
  .brand { font-family: var(--serif); font-size: clamp(1.8rem, 4vw, 2.6rem); letter-spacing: -0.02em; margin: 0; }
  .lede { margin: 0; max-width: 46rem; color: var(--muted); font-size: 1.02rem; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
  select, .search, .stat-pill { background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); }
  select { font: inherit; color: var(--ink); border-radius: 999px; padding: 0.65rem 1rem; cursor: pointer; }
  .search { flex: 1 1 260px; display: flex; align-items: center; gap: 0.6rem; border-radius: 999px; padding: 0.65rem 1rem; }
  .search input { border: 0; outline: 0; width: 100%; font: inherit; background: transparent; color: var(--ink); }
  .chips { display: flex; gap: 0.4rem; flex-wrap: wrap; }
  .toggle { border: 1px solid var(--line); background: var(--panel); border-radius: 999px; padding: 0.45rem 0.85rem; font-size: 0.85rem; font-weight: 600; cursor: pointer; color: var(--muted); }
  .toggle.active { background: var(--accent); border-color: var(--accent); color: white; }
  .stats { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.9rem; }
  .stat-pill { border-radius: 10px; padding: 0.5rem 0.8rem; font-size: 0.8rem; color: var(--muted); }
  .stat-pill b { color: var(--ink); font-size: 0.95rem; }
  .meta { color: var(--muted); font-size: 0.9rem; white-space: nowrap; margin-left: auto; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; margin-top: 1.25rem; }
  .card { appearance: none; text-align: left; width: 100%; border: 1px solid var(--line); background: var(--panel); border-radius: var(--radius); padding: 1.05rem 1.15rem 1.15rem; box-shadow: var(--shadow); cursor: pointer; display: grid; gap: 0.6rem; transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease; }
  .card:hover, .card:focus-visible { transform: translateY(-2px); border-color: #9fb7ac; box-shadow: 0 14px 34px rgba(20, 32, 28, 0.12); outline: none; }
  .card-top { display: flex; justify-content: space-between; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
  .chip { display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.02em; border-radius: 999px; padding: 0.2rem 0.55rem; }
  .chip.ok { color: var(--accent); background: var(--accent-soft); }
  .chip.failed { color: var(--bad); background: var(--bad-soft); }
  .chip.local { color: #8a6d1f; background: #f4ecd2; }
  .chip.remote { color: var(--muted); background: #eef1ef; }
  .id { font-size: 0.72rem; color: var(--muted); font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .name { margin: 0; font-size: 0.98rem; font-weight: 700; }
  .snippet { margin: 0; font-size: 0.86rem; color: var(--muted); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .archetype { margin: 0; font-size: 0.78rem; color: var(--muted); }
  .empty { grid-column: 1 / -1; padding: 2.5rem 1rem; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: var(--radius); background: rgba(255,255,255,0.55); }
  dialog { width: min(880px, calc(100% - 1.5rem)); max-height: min(90vh, 960px); border: 1px solid var(--line); border-radius: 18px; padding: 0; box-shadow: 0 24px 60px rgba(20, 32, 28, 0.22); background: var(--panel); color: var(--ink); }
  dialog::backdrop { background: rgba(20, 32, 28, 0.45); backdrop-filter: blur(2px); }
  .modal-head { display: flex; justify-content: space-between; gap: 1rem; align-items: start; padding: 1.2rem 1.35rem 1rem; border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--panel); }
  .modal-head h2 { margin: 0; font-family: var(--serif); font-size: 1.4rem; letter-spacing: -0.01em; }
  .modal-head .sub { margin: 0.35rem 0 0; color: var(--muted); font-size: 0.85rem; }
  .close { border: 1px solid var(--line); background: #f7faf8; border-radius: 999px; width: 2.2rem; height: 2.2rem; font-size: 1.2rem; cursor: pointer; color: var(--ink); }
  .modal-body { padding: 1.1rem 1.35rem 1.5rem; overflow: auto; max-height: calc(min(90vh, 960px) - 5.5rem); display: grid; gap: 1.3rem; }
  .field { margin-bottom: 0.85rem; }
  .field h4 { margin: 0 0 0.25rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 700; }
  .field p { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; }
  .qc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.5rem; }
  .qc-item { border: 1px solid var(--line); border-radius: 10px; padding: 0.5rem 0.7rem; font-size: 0.82rem; background: #f7faf8; }
  .qc-item b { display: block; color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.15rem; }
  @media (max-width: 640px) { .wrap { width: min(100%, calc(100% - 1.25rem)); padding-top: 1.25rem; } .card { padding: 0.9rem; } .modal-head, .modal-body { padding-left: 1rem; padding-right: 1rem; } }
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">Local gemma profile browser</h1>
      <p class="lede">Browse profiles from <code>artifacts/local_gemma_synth/&lt;run&gt;/profiles/</code>.</p>
      <div class="toolbar">
        <select id="run"></select>
        <div class="chips">
          <button class="toggle active" data-status="all" type="button">All</button>
          <button class="toggle" data-status="ok" type="button">OK</button>
          <button class="toggle" data-status="failed" type="button">Failed</button>
        </div>
        <div class="chips">
          <button class="toggle active" data-endpoint="all" type="button">Both endpoints</button>
          <button class="toggle" data-endpoint="local" type="button">Local</button>
          <button class="toggle" data-endpoint="remote" type="button">Remote</button>
        </div>
        <label class="search" for="q">
          <span aria-hidden="true">⌕</span>
          <input id="q" type="search" placeholder="Filter by archetype, reasoning, field text…" autocomplete="off" />
        </label>
        <div class="meta" id="count"></div>
      </div>
      <div class="stats" id="stats"></div>
    </header>
    <main class="grid" id="grid" aria-live="polite"></main>
  </div>

  <dialog id="detail" aria-labelledby="detail-title">
    <div class="modal-head">
      <div>
        <h2 id="detail-title">Profile</h2>
        <p class="sub" id="detail-sub"></p>
      </div>
      <button class="close" type="button" id="close" aria-label="Close">×</button>
    </div>
    <div class="modal-body" id="detail-body"></div>
  </dialog>

  <script id="runs-data" type="application/json">__EMBEDDED_JSON__</script>
  <script>
    const RUNS = JSON.parse(document.getElementById("runs-data").textContent);
    const runNames = Object.keys(RUNS).sort();

    const runSel = document.getElementById("run");
    const grid = document.getElementById("grid");
    const countEl = document.getElementById("count");
    const statsEl = document.getElementById("stats");
    const q = document.getElementById("q");
    const dialog = document.getElementById("detail");
    const detailTitle = document.getElementById("detail-title");
    const detailSub = document.getElementById("detail-sub");
    const detailBody = document.getElementById("detail-body");
    document.getElementById("close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });

    let statusFilter = "all";
    let endpointFilter = "all";

    for (const name of runNames) {
      const opt = document.createElement("option");
      opt.value = name;
      const recs = RUNS[name].records || [];
      const ok = recs.filter((r) => r.success).length;
      opt.textContent = `${name} (${ok}/${recs.length} ok)`;
      runSel.appendChild(opt);
    }

    function text(v) { return v == null ? "" : String(v); }
    function snippet(s, n = 200) {
      const t = text(s).replace(/\s+/g, " ").trim();
      if (!t) return "—";
      return t.length > n ? t.slice(0, n - 1) + "…" : t;
    }
    function escapeHtml(s) {
      return text(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
    function haystack(r) {
      const p = r.profile || {};
      return [r.archetype, p.reasoning, p.positioning, p.background, p.lookingFor, p.notes]
        .map(text).join("\n").toLowerCase();
    }
    function field(label, value) {
      const v = text(value).trim();
      if (!v) return "";
      return `<div class="field"><h4>${escapeHtml(label)}</h4><p>${escapeHtml(v)}</p></div>`;
    }
    const FIELD_LABELS = {
      reasoning: "Reasoning", positioning: "Positioning", background: "Background",
      lookingFor: "Looking for", notes: "Notes", locationAvailability: "Location / availability",
      introPreferences: "Intro preferences", personalPreferences: "Personal preferences",
      meetingAndSchedulingPreferences: "Meeting & scheduling",
    };
    function profileFields(p) {
      if (!p) return "<p>—</p>";
      return Object.entries(FIELD_LABELS).map(([k, label]) => field(label, p[k])).join("");
    }
    function qcItem(label, value) {
      if (value === undefined) return "";
      return `<div class="qc-item"><b>${escapeHtml(label)}</b>${escapeHtml(String(value))}</div>`;
    }

    function openDetail(r) {
      const p = r.profile || {};
      detailTitle.textContent = r.archetype || "Profile";
      detailSub.textContent = `#${r.id} · endpoint=${r.endpoint} · ${r.success ? "ok" : "FAILED"} · attempts=${r.n_attempts}`;
      const qcHtml = [
        qcItem("Style version", r.style_version),
        qcItem("Archetype version", r.archetypes_version),
        qcItem("Total elapsed (s)", r.elapsed_s?.toFixed?.(1)),
      ].join("");
      detailBody.innerHTML = `
        <div class="qc-grid">${qcHtml}</div>
        <div class="side">${profileFields(p)}</div>
      `;
      dialog.showModal();
    }

    function currentRecords() {
      const r = RUNS[runSel.value];
      return r ? (r.records || []) : [];
    }

    function render() {
      let list = currentRecords();
      if (statusFilter !== "all") list = list.filter((r) => (r.success ? "ok" : "failed") === statusFilter);
      if (endpointFilter !== "all") list = list.filter((r) => r.endpoint === endpointFilter);
      const term = q.value.trim().toLowerCase();
      if (term) list = list.filter((r) => haystack(r).includes(term));

      countEl.textContent = `${list.length} of ${currentRecords().length}`;

      const run = RUNS[runSel.value];
      const meta = run?.meta;
      statsEl.innerHTML = meta ? [
        `<div class="stat-pill">Total <b>${meta.total ?? 0}</b></div>`,
        `<div class="stat-pill">Success <b>${meta.success ?? 0}</b></div>`,
        `<div class="stat-pill">Failed <b>${meta.failed ?? 0}</b></div>`,
        `<div class="stat-pill">Style versions <b>${meta.style_versions ?? 0}</b></div>`,
        `<div class="stat-pill">Archetype versions <b>${meta.archetype_versions ?? 0}</b></div>`,
      ].join("") : "";

      if (!list.length) {
        grid.innerHTML = `<div class="empty">No profiles match that filter.</div>`;
        return;
      }
      grid.innerHTML = "";
      for (const r of list) {
        const p = r.profile || {};
        const name = (p.positioning || "").split(/[.,\n]/)[0] || r.archetype || `#${r.id}`;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "card";
        btn.innerHTML = `
          <div class="card-top">
            <span class="chip ${r.success ? "ok" : "failed"}">${r.success ? "ok" : "failed"}</span>
            <span class="chip ${r.endpoint}">${escapeHtml(r.endpoint)}</span>
            <span class="id">#${r.id} · attempts=${r.n_attempts}</span>
          </div>
          <p class="name">${escapeHtml(snippet(name, 80))}</p>
          <p class="archetype">${escapeHtml(snippet(r.archetype, 90))}</p>
          <p class="snippet">${escapeHtml(snippet(p.positioning, 180))}</p>
        `;
        btn.addEventListener("click", () => openDetail(r));
        grid.appendChild(btn);
      }
    }

    document.querySelectorAll(".toggle[data-status]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".toggle[data-status]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        statusFilter = btn.dataset.status;
        render();
      });
    });
    document.querySelectorAll(".toggle[data-endpoint]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".toggle[data-endpoint]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        endpointFilter = btn.dataset.endpoint;
        render();
      });
    });
    runSel.addEventListener("change", render);
    q.addEventListener("input", render);

    if (runNames.length) {
      runSel.value = runNames[runNames.length - 1];
      render();
    }
  </script>
</body>
</html>
"""


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    profiles_dir = run_dir / "profiles"
    if not profiles_dir.exists():
        return None

    records = []
    success = 0
    failed = 0
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        attempts = rec.get("attempts", [])
        rec["n_attempts"] = len(attempts)
        rec["elapsed_s"] = sum(a.get("elapsed_s", 0) for a in attempts)
        records.append(rec)
        if rec.get("success"):
            success += 1
        else:
            failed += 1

    style_versions = len(list((run_dir / "specs").glob("style_v*.json"))) if (run_dir / "specs").exists() else 0
    archetype_versions = len(list((run_dir / "specs").glob("archetypes_v*.json"))) if (run_dir / "specs").exists() else 0

    return {
        "records": records,
        "meta": {
            "total": len(records),
            "success": success,
            "failed": failed,
            "style_versions": style_versions,
            "archetype_versions": archetype_versions,
        },
    }


def build(runs_dir: Path, out_path: Path) -> dict[str, int]:
    runs: dict[str, Any] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        loaded = _load_run(run_dir)
        if loaded is not None:
            runs[run_dir.name] = loaded

    payload = json.dumps(runs, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")  # avoid </script> breakout
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return {name: len(r["records"]) for name, r in runs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    counts = build(args.runs_dir, args.out)
    for name, n in counts.items():
        print(f"  {name}: {n} profiles")
    print(f"Wrote {args.out} with {len(counts)} runs embedded.")


if __name__ == "__main__":
    main()
