#!/usr/bin/env python3
"""Embed artifacts/synth/<batch>/{manifest,staged,dropped} into one browsable HTML.

Scans every batch directory under --synth-dir, reads its manifest.json plus
every staged/dropped pair JSON, and writes a single self-contained HTML file
with a batch selector so you can flip between batches without regenerating.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTH_DIR = ROOT / "artifacts" / "synth"
DEFAULT_OUT = ROOT / "artifacts" / "synth" / "_browser.html"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Synth batches — Dorby AI</title>
<style>
  :root {
    --ink: #14201c;
    --muted: #5a6b64;
    --line: #c9d4ce;
    --paper: #f3f6f4;
    --panel: #ffffff;
    --accent: #1f6b55;
    --accent-soft: #d7ebe3;
    --bad: #a4392a;
    --bad-soft: #f6ded9;
    --shadow: 0 10px 30px rgba(20, 32, 28, 0.08);
    --radius: 14px;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    --sans: "Avenir Next", Avenir, "Century Gothic", "Gill Sans", "Trebuchet MS", sans-serif;
    --mono: "SF Mono", Menlo, Consolas, monospace;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: var(--sans);
    color: var(--ink);
    background:
      radial-gradient(1200px 600px at 10% -10%, #dff0e8 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #e7eee9 0%, transparent 50%),
      var(--paper);
    line-height: 1.45;
  }

  .wrap { width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }

  header.hero { display: grid; gap: 1rem; margin-bottom: 1.5rem; }

  .brand { font-family: var(--serif); font-size: clamp(1.8rem, 4vw, 2.6rem); letter-spacing: -0.02em; margin: 0; }
  .lede { margin: 0; max-width: 46rem; color: var(--muted); font-size: 1.02rem; }

  .toolbar { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }

  select, .search, .stat-pill {
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
  }

  select {
    font: inherit;
    color: var(--ink);
    border-radius: 999px;
    padding: 0.65rem 1rem;
    cursor: pointer;
  }

  .search {
    flex: 1 1 260px;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    border-radius: 999px;
    padding: 0.65rem 1rem;
  }
  .search input { border: 0; outline: 0; width: 100%; font: inherit; background: transparent; color: var(--ink); }

  .chips { display: flex; gap: 0.4rem; flex-wrap: wrap; }
  .toggle {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 999px;
    padding: 0.45rem 0.85rem;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--muted);
  }
  .toggle.active { background: var(--accent); border-color: var(--accent); color: white; }

  .stats { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.9rem; }
  .stat-pill {
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    font-size: 0.8rem;
    color: var(--muted);
  }
  .stat-pill b { color: var(--ink); font-size: 0.95rem; }

  .meta { color: var(--muted); font-size: 0.9rem; white-space: nowrap; margin-left: auto; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; margin-top: 1.25rem; }

  .card {
    appearance: none; text-align: left; width: 100%;
    border: 1px solid var(--line); background: var(--panel);
    border-radius: var(--radius); padding: 1.05rem 1.15rem 1.15rem;
    box-shadow: var(--shadow); cursor: pointer; display: grid; gap: 0.6rem;
    transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
  }
  .card:hover, .card:focus-visible { transform: translateY(-2px); border-color: #9fb7ac; box-shadow: 0 14px 34px rgba(20, 32, 28, 0.12); outline: none; }

  .card-top { display: flex; justify-content: space-between; gap: 0.5rem; align-items: center; flex-wrap: wrap; }

  .chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.02em;
    border-radius: 999px; padding: 0.2rem 0.55rem;
  }
  .chip.pos { color: var(--accent); background: var(--accent-soft); }
  .chip.neg { color: #8a6d1f; background: #f4ecd2; }
  .chip.staged { color: var(--accent); background: var(--accent-soft); }
  .chip.dropped { color: var(--bad); background: var(--bad-soft); }
  .chip.mode { color: var(--muted); background: #eef1ef; }
  .chip.sample { color: #8a6d1f; background: #f4ecd2; }
  .chip.verdict-yes { color: var(--accent); background: var(--accent-soft); }
  .chip.verdict-no { color: var(--bad); background: var(--bad-soft); }

  .review-bar { display: flex; gap: 0.6rem; align-items: center; padding: 0.9rem 1.35rem; border-top: 1px solid var(--line); position: sticky; bottom: 0; background: var(--panel); }
  .review-btn { border: 1px solid var(--line); border-radius: 999px; padding: 0.55rem 1.1rem; font: inherit; font-weight: 700; cursor: pointer; }
  .review-btn.approve { background: var(--accent); border-color: var(--accent); color: white; }
  .review-btn.reject { background: var(--bad); border-color: var(--bad); color: white; }
  .review-btn:disabled { opacity: 0.5; cursor: default; }
  .review-status { color: var(--muted); font-size: 0.85rem; margin-left: auto; }

  .id { font-size: 0.72rem; color: var(--muted); font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .query { margin: 0; font-size: 0.94rem; }
  .query::before { content: "Query · "; color: var(--muted); font-weight: 600; font-size: 0.8rem; }

  .snippet { margin: 0; font-size: 0.86rem; color: var(--muted); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }

  .reason { margin: 0; font-size: 0.8rem; color: var(--bad); }

  .empty {
    grid-column: 1 / -1; padding: 2.5rem 1rem; text-align: center; color: var(--muted);
    border: 1px dashed var(--line); border-radius: var(--radius); background: rgba(255,255,255,0.55);
  }

  dialog {
    width: min(880px, calc(100% - 1.5rem)); max-height: min(90vh, 960px);
    border: 1px solid var(--line); border-radius: 18px; padding: 0;
    box-shadow: 0 24px 60px rgba(20, 32, 28, 0.22); background: var(--panel); color: var(--ink);
  }
  dialog::backdrop { background: rgba(20, 32, 28, 0.45); backdrop-filter: blur(2px); }

  .modal-head {
    display: flex; justify-content: space-between; gap: 1rem; align-items: start;
    padding: 1.2rem 1.35rem 1rem; border-bottom: 1px solid var(--line);
    position: sticky; top: 0; background: var(--panel);
  }
  .modal-head h2 { margin: 0; font-family: var(--serif); font-size: 1.4rem; letter-spacing: -0.01em; }
  .modal-head .sub { margin: 0.35rem 0 0; color: var(--muted); font-size: 0.85rem; }
  .close {
    border: 1px solid var(--line); background: #f7faf8; border-radius: 999px;
    width: 2.2rem; height: 2.2rem; font-size: 1.2rem; cursor: pointer; color: var(--ink);
  }

  .modal-body { padding: 1.1rem 1.35rem 1.5rem; overflow: auto; max-height: calc(min(90vh, 960px) - 5.5rem); display: grid; gap: 1.3rem; }

  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 720px) { .cols { grid-template-columns: 1fr; } }

  .side h3 { margin: 0 0 0.6rem; font-family: var(--serif); font-size: 1.05rem; }
  .field { margin-bottom: 0.85rem; }
  .field h4 { margin: 0 0 0.25rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 700; }
  .field p { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; }

  .qc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.5rem; }
  .qc-item { border: 1px solid var(--line); border-radius: 10px; padding: 0.5rem 0.7rem; font-size: 0.82rem; background: #f7faf8; }
  .qc-item b { display: block; color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.15rem; }

  @media (max-width: 640px) {
    .wrap { width: min(100%, calc(100% - 1.25rem)); padding-top: 1.25rem; }
    .card { padding: 0.9rem; }
    .modal-head, .modal-body { padding-left: 1rem; padding-right: 1rem; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">Synth batch browser</h1>
      <p class="lede">Browse generated pairs from <code>artifacts/synth/&lt;batch&gt;/</code> — staged and dropped, with QC/judge details.</p>
      <div class="toolbar">
        <select id="batch"></select>
        <div class="chips">
          <button class="toggle active" data-status="all" type="button">All</button>
          <button class="toggle" data-status="staged" type="button">Staged</button>
          <button class="toggle" data-status="dropped" type="button">Dropped</button>
        </div>
        <div class="chips">
          <button class="toggle active" data-label="all" type="button">Pos + Neg</button>
          <button class="toggle" data-label="pos" type="button">Pos only</button>
          <button class="toggle" data-label="neg" type="button">Neg only</button>
        </div>
        <div class="chips">
          <button class="toggle active" id="sample-toggle" type="button">Review sample only</button>
        </div>
        <label class="search" for="q">
          <span aria-hidden="true">⌕</span>
          <input id="q" type="search" placeholder="Filter by query, seed id, failure mode, drop reason…" autocomplete="off" />
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
        <h2 id="detail-title">Pair</h2>
        <p class="sub" id="detail-sub"></p>
      </div>
      <button class="close" type="button" id="close" aria-label="Close">×</button>
    </div>
    <div class="modal-body" id="detail-body"></div>
    <div class="review-bar" id="review-bar" hidden>
      <button class="review-btn approve" id="approve-btn" type="button">Approve</button>
      <button class="review-btn reject" id="reject-btn" type="button">Reject</button>
      <span class="review-status" id="review-status"></span>
    </div>
  </dialog>

  <script id="batches-data" type="application/json">__EMBEDDED_JSON__</script>
  <script>
    const BATCHES = JSON.parse(document.getElementById("batches-data").textContent);
    const batchNames = Object.keys(BATCHES).sort();

    const batchSel = document.getElementById("batch");
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
    let labelFilter = "all";
    let sampleOnly = true;
    let currentRecord = null;

    const sampleToggle = document.getElementById("sample-toggle");
    sampleToggle.addEventListener("click", () => {
      sampleOnly = !sampleOnly;
      sampleToggle.classList.toggle("active", sampleOnly);
      render();
    });

    const reviewBar = document.getElementById("review-bar");
    const approveBtn = document.getElementById("approve-btn");
    const rejectBtn = document.getElementById("reject-btn");
    const reviewStatus = document.getElementById("review-status");

    async function submitVerdict(verdict) {
      if (!currentRecord) return;
      reviewStatus.textContent = "Saving…";
      approveBtn.disabled = true;
      rejectBtn.disabled = true;
      try {
        const res = await fetch("/api/review", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rel_path: currentRecord.rel_path, verdict }),
        });
        if (!res.ok) throw new Error(await res.text());
        currentRecord.human_review = { verdict };
        reviewStatus.textContent = verdict === "yes" ? "Approved ✓" : "Rejected ✗";
        render();
      } catch (err) {
        reviewStatus.textContent = "Save failed — is scripts/serve_synth_review.py running?";
      } finally {
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
      }
    }
    approveBtn.addEventListener("click", () => submitVerdict("yes"));
    rejectBtn.addEventListener("click", () => submitVerdict("no"));

    for (const name of batchNames) {
      const opt = document.createElement("option");
      opt.value = name;
      const n = BATCHES[name].manifest?.counts;
      opt.textContent = n ? `${name} (${n.staged ?? 0} staged / ${n.dropped ?? 0} dropped)` : name;
      batchSel.appendChild(opt);
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
      return [
        r.seed_pair_id, r.seed_user_id, r.failure_mode, r.drop_reason,
        r.pair?.searchQuery, r.status, r.label,
      ].map(text).join("\n").toLowerCase();
    }

    function field(label, value) {
      const v = text(value).trim();
      if (!v) return "";
      return `<div class="field"><h4>${escapeHtml(label)}</h4><p>${escapeHtml(v)}</p></div>`;
    }

    function profileFields(f) {
      if (!f) return "<p>—</p>";
      return [
        field("Positioning", f.positioning),
        field("Looking for", f.lookingFor),
        field("Intro preferences", f.introPreferences),
        field("Personal preferences", f.personalPreferences),
        field("Meeting & scheduling", f.meetingAndSchedulingPreferences),
        field("Background", f.background),
        field("Location / availability", f.locationAvailability),
        field("Notes", f.notes),
      ].join("");
    }

    function qcItem(label, value) {
      if (value === undefined) return "";
      return `<div class="qc-item"><b>${escapeHtml(label)}</b>${escapeHtml(String(value))}</div>`;
    }

    function openDetail(r) {
      currentRecord = r;
      const pair = r.pair || {};
      detailTitle.textContent = snippet(pair.searchQuery, 90).replace(/…$/, "") || "Pair";
      detailSub.textContent = `${r.label}${r.failure_mode ? " · " + r.failure_mode : ""} · ${r.status}` +
        (r.drop_reason ? ` · ${r.drop_reason}` : "");

      if (r.in_sample) {
        reviewBar.hidden = false;
        const verdict = r.human_review?.verdict;
        reviewStatus.textContent = verdict === "yes" ? "Approved ✓" : verdict === "no" ? "Rejected ✗" : "";
      } else {
        reviewBar.hidden = true;
        reviewStatus.textContent = "";
      }

      const qc = r.qc || {};
      const qcHtml = [
        qcItem("Filter passed", qc.filter_passed),
        qcItem("Filter reject reason", qc.filter_reject_reason),
        qcItem("Would be good intro", qc.would_be_good_intro),
        qcItem("Is easy negative", qc.is_easy_negative),
        qcItem("Judge verdict", qc.judge_verdict),
        qcItem("Query/match jaccard", qc.query_match_jaccard),
      ].join("");

      detailBody.innerHTML = `
        <div class="field">
          <h4>Search query</h4>
          <p>${escapeHtml(pair.searchQuery)}</p>
        </div>
        <div class="qc-grid">${qcHtml}</div>
        <div class="cols">
          <div class="side">
            <h3>Seeker · ${escapeHtml(pair.userContactId || "")}</h3>
            ${profileFields(pair.userContactFile)}
          </div>
          <div class="side">
            <h3>Match · ${escapeHtml(pair.matchContactId || "")}</h3>
            ${profileFields(pair.matchContactFile)}
          </div>
        </div>
      `;
      dialog.showModal();
    }

    function currentRecords() {
      const b = BATCHES[batchSel.value];
      if (!b) return [];
      return b.records || [];
    }

    function render() {
      let list = currentRecords();
      if (statusFilter !== "all") list = list.filter((r) => r.status === statusFilter);
      if (labelFilter !== "all") list = list.filter((r) => r.label === labelFilter);
      if (sampleOnly) list = list.filter((r) => r.in_sample);
      const term = q.value.trim().toLowerCase();
      if (term) list = list.filter((r) => haystack(r).includes(term));

      const sampleCount = currentRecords().filter((r) => r.in_sample).length;
      countEl.textContent = `${list.length} of ${currentRecords().length} (${sampleCount} in review sample)`;

      const b = BATCHES[batchSel.value];
      const c = b?.manifest?.counts;
      statsEl.innerHTML = c ? [
        `<div class="stat-pill">Attempts <b>${c.attempts ?? 0}</b></div>`,
        `<div class="stat-pill">Staged <b>${c.staged ?? 0}</b></div>`,
        `<div class="stat-pill">Dropped <b>${c.dropped ?? 0}</b></div>`,
        `<div class="stat-pill">Filtered out <b>${c.filtered_out ?? 0}</b></div>`,
        `<div class="stat-pill">Judge rejected <b>${c.judge_rejected ?? 0}</b></div>`,
        `<div class="stat-pill">Generate model <b>${escapeHtml(b.manifest.generate_model || "?")}</b></div>`,
        `<div class="stat-pill">Judge model <b>${escapeHtml(b.manifest.judge_model || "?")}</b></div>`,
      ].join("") : "";

      if (!list.length) {
        grid.innerHTML = `<div class="empty">No pairs match that filter.</div>`;
        return;
      }
      grid.innerHTML = "";
      for (const r of list) {
        const pair = r.pair || {};
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "card";
        const verdict = r.human_review?.verdict;
        btn.innerHTML = `
          <div class="card-top">
            <span class="chip ${r.label}">${escapeHtml(r.label)}</span>
            <span class="chip ${r.status}">${escapeHtml(r.status)}</span>
            ${r.failure_mode ? `<span class="chip mode">${escapeHtml(r.failure_mode)}</span>` : ""}
            ${r.in_sample ? `<span class="chip sample">sample</span>` : ""}
            ${verdict ? `<span class="chip verdict-${verdict}">${verdict === "yes" ? "approved" : "rejected"}</span>` : ""}
            <span class="id">${escapeHtml((r.seed_user_id || "").slice(0, 18))}</span>
          </div>
          <p class="query">${escapeHtml(snippet(pair.searchQuery, 110))}</p>
          <p class="snippet">${escapeHtml(snippet(pair.userContactFile?.positioning, 160))}</p>
          ${r.drop_reason ? `<p class="reason">Dropped: ${escapeHtml(r.drop_reason)}</p>` : ""}
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
    document.querySelectorAll(".toggle[data-label]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".toggle[data-label]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        labelFilter = btn.dataset.label;
        render();
      });
    });
    batchSel.addEventListener("change", render);
    q.addEventListener("input", render);

    if (batchNames.length) {
      batchSel.value = batchNames[batchNames.length - 1];
      render();
    }
  </script>
</body>
</html>
"""


def _load_pair_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mark_review_sample(records: list[dict[str, Any]], pct: float, seed: int) -> None:
    """Flag a stratified sample of staged records with in_sample=True for human review."""
    staged = [r for r in records if r.get("status") == "staged"]
    for r in records:
        r["in_sample"] = False
    if not staged or pct <= 0:
        return

    target = max(1, round(len(staged) * pct / 100))
    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in staged:
        buckets[(r.get("label"), r.get("failure_mode"))].append(r)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[dict[str, Any]] = []
    bucket_keys = sorted(buckets.keys(), key=lambda k: -len(buckets[k]))
    i = 0
    while len(picked) < target and any(buckets.values()):
        key = bucket_keys[i % len(bucket_keys)]
        if buckets[key]:
            picked.append(buckets[key].pop())
        i += 1

    for r in picked:
        r["in_sample"] = True


def _load_batch(batch_dir: Path, sample_pct: float, sample_seed: int) -> dict[str, Any] | None:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    records = []
    for record in manifest.get("records", []):
        rel = record.get("staged_path") or record.get("dropped_path")
        pair = None
        human_review = None
        env_status = record.get("status")
        if rel:
            full = batch_dir.parent / rel
            payload = _load_pair_json(full)
            pair = payload.get("pair")
            human_review = payload.get("human_review")
            env_status = payload.get("status", env_status)
        records.append(
            {
                **record,
                "pair": pair,
                "rel_path": rel,
                "human_review": human_review,
                "env_status": env_status,
            }
        )

    _mark_review_sample(records, sample_pct, sample_seed)

    return {"manifest": manifest, "records": records}


def build(
    synth_dir: Path, out_path: Path, sample_pct: float = 2.0, sample_seed: int = 0
) -> dict[str, int]:
    batches: dict[str, Any] = {}
    for batch_dir in sorted(synth_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        loaded = _load_batch(batch_dir, sample_pct, sample_seed)
        if loaded is not None:
            batches[batch_dir.name] = loaded

    payload = json.dumps(batches, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")  # avoid </script> breakout
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return {name: len(b["records"]) for name, b in batches.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth-dir", type=Path, default=DEFAULT_SYNTH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=2.0,
        help="Percent of staged records per batch to flag for human review (default: 2.0)",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()
    counts = build(args.synth_dir, args.out, args.sample_pct, args.sample_seed)
    for name, n in counts.items():
        print(f"  {name}: {n} pairs")
    print(f"Wrote {args.out} with {len(counts)} batches embedded.")


if __name__ == "__main__":
    main()
