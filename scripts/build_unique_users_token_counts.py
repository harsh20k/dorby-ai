#!/usr/bin/env python3
"""Count BERT tokens per profile field and emit JSON + tabular HTML browser."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USERS = ROOT / "data" / "unique_users.json"
DEFAULT_JSON_OUT = ROOT / "data" / "unique_users_token_counts.json"
DEFAULT_HTML_OUT = ROOT / "data" / "unique_users_token_counts_browser.html"
DEFAULT_MODEL = "bert-base-uncased"

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

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Unique users — token counts</title>
<style>
  :root {
    --ink: #14201c;
    --muted: #5a6b64;
    --line: #c9d4ce;
    --paper: #f3f6f4;
    --panel: #ffffff;
    --accent: #1f6b55;
    --accent-soft: #d7ebe3;
    --header-bg: #e8f0ec;
    --row-hover: #f0f7f3;
    --shadow: 0 8px 24px rgba(20, 32, 28, 0.07);
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
      radial-gradient(1100px 520px at 8% -8%, #dff0e8 0%, transparent 55%),
      radial-gradient(800px 420px at 100% 0%, #e7eee9 0%, transparent 50%),
      var(--paper);
    line-height: 1.4;
  }

  .wrap {
    width: min(1400px, calc(100% - 1.5rem));
    margin: 0 auto;
    padding: 1.5rem 0 3rem;
  }

  header.hero {
    display: grid;
    gap: 0.65rem;
    margin-bottom: 1.25rem;
  }

  .brand {
    font-family: var(--serif);
    font-size: clamp(1.7rem, 3.5vw, 2.4rem);
    letter-spacing: -0.02em;
    margin: 0;
  }

  .lede {
    margin: 0;
    max-width: 46rem;
    color: var(--muted);
    font-size: 0.98rem;
  }

  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    align-items: end;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    box-shadow: var(--shadow);
    margin-bottom: 0.85rem;
  }

  .field {
    display: grid;
    gap: 0.2rem;
    min-width: 7.5rem;
  }

  .field.wide { flex: 1 1 200px; min-width: 180px; }

  .field label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    font-weight: 700;
  }

  .field input {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.45rem 0.55rem;
    font: inherit;
    font-size: 0.9rem;
    color: var(--ink);
    background: #fafcfb;
    width: 100%;
  }

  .field input:focus {
    outline: 2px solid var(--accent-soft);
    border-color: var(--accent);
  }

  .actions {
    display: flex;
    gap: 0.45rem;
    align-items: center;
  }

  button.reset {
    border: 1px solid var(--line);
    background: #f7faf8;
    border-radius: 8px;
    padding: 0.48rem 0.75rem;
    font: inherit;
    font-size: 0.88rem;
    cursor: pointer;
    color: var(--ink);
  }

  button.reset:hover { border-color: #9fb7ac; }

  .meta {
    color: var(--muted);
    font-size: 0.88rem;
    margin: 0 0 0.65rem;
  }

  section.summary {
    margin-bottom: 1rem;
  }

  .summary-title {
    margin: 0 0 0.25rem;
    font-family: var(--serif);
    font-size: 1.15rem;
    letter-spacing: -0.01em;
  }

  .summary-lede {
    margin: 0 0 0.55rem;
    color: var(--muted);
    font-size: 0.88rem;
  }

  .summary-scroll {
    max-height: none;
  }

  table.summary-table {
    min-width: 520px;
  }

  table.summary-table thead th {
    cursor: default;
    text-align: right;
  }

  table.summary-table thead th:first-child,
  table.summary-table tbody td:first-child {
    text-align: left;
  }

  table.summary-table tbody td.field-name {
    font-weight: 600;
    color: var(--ink);
  }

  table.summary-table tbody tr.total-row td {
    font-weight: 700;
    color: var(--accent);
    background: #f4faf7;
  }

  .table-scroll {
    overflow-x: auto;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel);
    box-shadow: var(--shadow);
  }

  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.84rem;
    min-width: 1100px;
  }

  thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--header-bg);
    border-bottom: 1px solid var(--line);
    text-align: right;
    padding: 0.55rem 0.5rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
  }

  thead th:first-child,
  tbody td:first-child { text-align: left; }

  thead th:hover { color: var(--ink); }

  thead th .sort {
    display: inline-block;
    margin-left: 0.2rem;
    opacity: 0.35;
    font-size: 0.65rem;
  }

  thead th.active .sort { opacity: 1; color: var(--accent); }

  tbody td {
    text-align: right;
    padding: 0.42rem 0.5rem;
    border-bottom: 1px solid #e4ebe7;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  tbody tr:hover td { background: var(--row-hover); }
  tbody tr:last-child td { border-bottom: 0; }

  td.id {
    font-family: var(--mono);
    font-size: 0.75rem;
    max-width: 11rem;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  button.user-link {
    appearance: none;
    border: 0;
    background: transparent;
    padding: 0;
    margin: 0;
    font: inherit;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--accent);
    text-decoration: underline;
    text-underline-offset: 2px;
    cursor: pointer;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  button.user-link:hover,
  button.user-link:focus-visible {
    color: var(--ink);
    outline: none;
  }

  td.total {
    font-weight: 700;
    color: var(--accent);
  }

  .empty {
    padding: 2.5rem 1rem;
    text-align: center;
    color: var(--muted);
  }

  dialog.detail {
    width: min(760px, calc(100% - 1.5rem));
    max-height: min(88vh, 900px);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0;
    box-shadow: 0 24px 60px rgba(20, 32, 28, 0.22);
    background: var(--panel);
    color: var(--ink);
  }

  dialog.detail::backdrop {
    background: rgba(20, 32, 28, 0.45);
    backdrop-filter: blur(2px);
  }

  .modal-head {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: start;
    padding: 1.25rem 1.35rem 1rem;
    border-bottom: 1px solid var(--line);
    position: sticky;
    top: 0;
    background: var(--panel);
    z-index: 1;
  }

  .modal-head h2 {
    margin: 0;
    font-family: var(--serif);
    font-size: 1.45rem;
    letter-spacing: -0.01em;
  }

  .modal-head .sub {
    margin: 0.35rem 0 0;
    color: var(--muted);
    font-size: 0.88rem;
    word-break: break-all;
  }

  .close {
    border: 1px solid var(--line);
    background: #f7faf8;
    border-radius: 999px;
    width: 2.2rem;
    height: 2.2rem;
    font-size: 1.2rem;
    cursor: pointer;
    color: var(--ink);
    flex: 0 0 auto;
  }

  .modal-body {
    padding: 1.1rem 1.35rem 1.5rem;
    overflow: auto;
    max-height: calc(min(88vh, 900px) - 5.5rem);
    display: grid;
    gap: 1.1rem;
  }

  .detail-card {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #fafcfb;
    padding: 0.95rem 1.05rem;
  }

  .detail-card h3 {
    margin: 0 0 0.5rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 700;
  }

  .md {
    font-size: 0.95rem;
    line-height: 1.55;
    word-break: break-word;
  }

  .md > *:first-child { margin-top: 0; }
  .md > *:last-child { margin-bottom: 0; }
  .md h1, .md h2, .md h3, .md h4 {
    font-family: var(--serif);
    letter-spacing: -0.01em;
    margin: 1rem 0 0.45rem;
    line-height: 1.25;
  }
  .md h1 { font-size: 1.25rem; }
  .md h2 { font-size: 1.12rem; }
  .md h3 { font-size: 1.02rem; }
  .md h4 { font-size: 0.95rem; }
  .md p { margin: 0.45rem 0; }
  .md ul, .md ol { margin: 0.45rem 0; padding-left: 1.25rem; }
  .md li { margin: 0.2rem 0; }
  .md code {
    font-family: var(--mono);
    font-size: 0.86em;
    background: #e8f0ec;
    padding: 0.1em 0.35em;
    border-radius: 4px;
  }
  .md pre {
    margin: 0.55rem 0;
    padding: 0.75rem 0.85rem;
    background: #e8f0ec;
    border-radius: 10px;
    overflow-x: auto;
  }
  .md pre code { background: transparent; padding: 0; }
  .md a { color: var(--accent); }
  .md blockquote {
    margin: 0.55rem 0;
    padding: 0.2rem 0 0.2rem 0.85rem;
    border-left: 3px solid var(--accent);
    color: var(--muted);
  }
  .md hr {
    border: 0;
    border-top: 1px solid var(--line);
    margin: 0.85rem 0;
  }

  .queries {
    display: grid;
    gap: 0.65rem;
  }

  .queries ol {
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.65rem;
  }

  .queries li {
    list-style: none;
    margin: 0;
    padding: 0.75rem 0.85rem;
    border-left: 3px solid var(--accent);
    background: #f4faf7;
    border-radius: 0 10px 10px 0;
  }

  .token-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .token-chips span {
    font-size: 0.75rem;
    font-variant-numeric: tabular-nums;
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 999px;
    padding: 0.2rem 0.55rem;
    font-weight: 600;
  }

  @media (max-width: 720px) {
    .wrap { width: calc(100% - 1rem); padding-top: 1rem; }
    .filters { padding: 0.75rem; }
    .modal-head, .modal-body { padding-left: 1rem; padding-right: 1rem; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">Token counts</h1>
      <p class="lede">BERT (<code>bert-base-uncased</code>) token lengths per profile field — content only, no special tokens. Click a user id for the full profile (markdown). Click column headers to sort.</p>
    </header>

    <section class="summary" aria-labelledby="summary-title">
      <h2 class="summary-title" id="summary-title">Summary across all users</h2>
      <p class="summary-lede" id="summary-lede">Min / mean / median / max over every profile (unfiltered).</p>
      <div class="table-scroll summary-scroll">
        <table class="summary-table">
          <thead>
            <tr>
              <th>Field</th>
              <th>Min</th>
              <th>Mean</th>
              <th>Median</th>
              <th>Max</th>
            </tr>
          </thead>
          <tbody id="summary-body"></tbody>
        </table>
      </div>
    </section>

    <div class="filters" id="filters">
      <div class="field wide">
        <label for="q">User id</label>
        <input id="q" type="search" placeholder="Filter by userContactId…" autocomplete="off" />
      </div>
      <div class="field">
        <label for="pairsMin">Pairs ≥</label>
        <input id="pairsMin" type="number" min="0" step="1" placeholder="0" />
      </div>
      <div class="field">
        <label for="pairsMax">Pairs ≤</label>
        <input id="pairsMax" type="number" min="0" step="1" placeholder="∞" />
      </div>
      <div class="field">
        <label for="posMin">Positioning ≥</label>
        <input id="posMin" type="number" min="0" step="1" />
      </div>
      <div class="field">
        <label for="bgMin">Background ≥</label>
        <input id="bgMin" type="number" min="0" step="1" />
      </div>
      <div class="field">
        <label for="lfMin">LookingFor ≥</label>
        <input id="lfMin" type="number" min="0" step="1" />
      </div>
      <div class="field">
        <label for="totalMin">Profile total ≥</label>
        <input id="totalMin" type="number" min="0" step="1" />
      </div>
      <div class="field">
        <label for="totalMax">Profile total ≤</label>
        <input id="totalMax" type="number" min="0" step="1" />
      </div>
      <div class="actions">
        <button class="reset" type="button" id="reset">Reset</button>
      </div>
    </div>

    <p class="meta" id="count"></p>

    <div class="table-scroll">
      <table>
        <thead>
          <tr id="head"></tr>
        </thead>
        <tbody id="body"></tbody>
      </table>
    </div>
  </div>

  <dialog class="detail" id="detail" aria-labelledby="detail-title">
    <div class="modal-head">
      <div>
        <h2 id="detail-title">User</h2>
        <p class="sub" id="detail-id"></p>
      </div>
      <button class="close" type="button" id="close" aria-label="Close">×</button>
    </div>
    <div class="modal-body" id="detail-body"></div>
  </dialog>

  <script id="users-data" type="application/json">__EMBEDDED_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("users-data").textContent);
    const USERS = DATA.records || DATA;
    const PROFILES = DATA.profiles || {};

    const COLS = [
      { key: "userContactId", label: "User id", get: (u) => u.userContactId, short: true },
      { key: "pairCount", label: "Pairs", get: (u) => u.pairCount ?? 0 },
      { key: "searchQueryCount", label: "#queries", get: (u) => u.searchQueryCount ?? 0 },
      { key: "positioning", label: "Positioning", get: (u) => u.tokens.positioning },
      { key: "background", label: "Background", get: (u) => u.tokens.background },
      { key: "lookingFor", label: "LookingFor", get: (u) => u.tokens.lookingFor },
      { key: "notes", label: "Notes", get: (u) => u.tokens.notes },
      { key: "locationAvailability", label: "Location", get: (u) => u.tokens.locationAvailability },
      { key: "introPreferences", label: "Intro", get: (u) => u.tokens.introPreferences },
      { key: "personalPreferences", label: "Personal", get: (u) => u.tokens.personalPreferences },
      { key: "meetingAndSchedulingPreferences", label: "Meeting", get: (u) => u.tokens.meetingAndSchedulingPreferences },
      { key: "searchQueries_total", label: "Queries Σ", get: (u) => u.tokens.searchQueries_total },
      { key: "total_profile_tokens", label: "Profile Σ", get: (u) => u.tokens.total_profile_tokens, total: true },
      { key: "total_seeker_tokens", label: "Seeker Σ", get: (u) => u.tokens.total_seeker_tokens, total: true },
    ];

    const PROFILE_LABELS = [
      ["positioning", "Positioning"],
      ["lookingFor", "Looking for"],
      ["locationAvailability", "Location / availability"],
      ["background", "Background"],
      ["notes", "Notes"],
      ["introPreferences", "Intro preferences"],
      ["personalPreferences", "Personal preferences"],
      ["meetingAndSchedulingPreferences", "Meeting & scheduling"],
    ];

    const SUMMARY_FIELDS = [
      { key: "pairCount", label: "Pairs", get: (u) => u.pairCount ?? 0 },
      { key: "searchQueryCount", label: "#queries", get: (u) => u.searchQueryCount ?? 0 },
      { key: "positioning", label: "Positioning", get: (u) => u.tokens?.positioning ?? 0 },
      { key: "background", label: "Background", get: (u) => u.tokens?.background ?? 0 },
      { key: "lookingFor", label: "LookingFor", get: (u) => u.tokens?.lookingFor ?? 0 },
      { key: "notes", label: "Notes", get: (u) => u.tokens?.notes ?? 0 },
      { key: "locationAvailability", label: "Location", get: (u) => u.tokens?.locationAvailability ?? 0 },
      { key: "introPreferences", label: "Intro", get: (u) => u.tokens?.introPreferences ?? 0 },
      { key: "personalPreferences", label: "Personal", get: (u) => u.tokens?.personalPreferences ?? 0 },
      { key: "meetingAndSchedulingPreferences", label: "Meeting", get: (u) => u.tokens?.meetingAndSchedulingPreferences ?? 0 },
      { key: "searchQueries_total", label: "Queries Σ", get: (u) => u.tokens?.searchQueries_total ?? 0 },
      { key: "total_profile_tokens", label: "Profile Σ", get: (u) => u.tokens?.total_profile_tokens ?? 0, total: true },
      { key: "total_seeker_tokens", label: "Seeker Σ", get: (u) => u.tokens?.total_seeker_tokens ?? 0, total: true },
    ];

    function medianOf(sorted) {
      const n = sorted.length;
      if (!n) return 0;
      const mid = Math.floor(n / 2);
      return n % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    function fieldStats(values) {
      if (!values.length) return { min: 0, mean: 0, median: 0, max: 0 };
      const sorted = values.slice().sort((a, b) => a - b);
      const sum = values.reduce((a, b) => a + b, 0);
      return {
        min: sorted[0],
        mean: sum / values.length,
        median: medianOf(sorted),
        max: sorted[sorted.length - 1],
      };
    }

    function fmtStat(n, decimals) {
      if (!Number.isFinite(n)) return "—";
      if (decimals === 0 || Number.isInteger(n)) return String(Math.round(n));
      return n.toFixed(decimals);
    }

    function renderSummary() {
      const lede = document.getElementById("summary-lede");
      const tbody = document.getElementById("summary-body");
      if (lede) {
        lede.textContent =
          `Min / mean / median / max over all ${USERS.length} profiles (unfiltered).`;
      }
      if (!tbody) return;
      tbody.innerHTML = SUMMARY_FIELDS.map((f) => {
        const s = fieldStats(USERS.map((u) => Number(f.get(u)) || 0));
        const rowCls = f.total ? ' class="total-row"' : "";
        return (
          `<tr${rowCls}>` +
          `<td class="field-name">${escapeHtml(f.label)}</td>` +
          `<td>${fmtStat(s.min, 0)}</td>` +
          `<td>${fmtStat(s.mean, 1)}</td>` +
          `<td>${fmtStat(s.median, 1)}</td>` +
          `<td>${fmtStat(s.max, 0)}</td>` +
          `</tr>`
        );
      }).join("");
    }

    const head = document.getElementById("head");
    const body = document.getElementById("body");
    const countEl = document.getElementById("count");
    const q = document.getElementById("q");
    const pairsMin = document.getElementById("pairsMin");
    const pairsMax = document.getElementById("pairsMax");
    const posMin = document.getElementById("posMin");
    const bgMin = document.getElementById("bgMin");
    const lfMin = document.getElementById("lfMin");
    const totalMin = document.getElementById("totalMin");
    const totalMax = document.getElementById("totalMax");
    const dialog = document.getElementById("detail");
    const detailTitle = document.getElementById("detail-title");
    const detailId = document.getElementById("detail-id");
    const detailBody = document.getElementById("detail-body");

    let sortKey = "total_profile_tokens";
    let sortDir = "desc";

    document.getElementById("close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (e) => {
      if (e.target === dialog) dialog.close();
    });

    function escapeHtml(s) {
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function shortId(id) {
      const s = String(id ?? "");
      if (s.length <= 14) return s;
      return s.slice(0, 6) + "…" + s.slice(-5);
    }

    function numOrNull(el) {
      const v = el.value.trim();
      if (v === "") return null;
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }

    function inlineMd(text) {
      let s = escapeHtml(text);
      s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
      s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
      s = s.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
      s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      return s;
    }

    function renderMarkdown(src) {
      const raw = String(src ?? "").replace(/\r\n/g, "\n").trim();
      if (!raw) return "<p>—</p>";

      const lines = raw.split("\n");
      const out = [];
      let i = 0;
      let inCode = false;
      let codeBuf = [];

      function flushParagraph(buf) {
        if (!buf.length) return;
        out.push("<p>" + inlineMd(buf.join(" ").trim()) + "</p>");
        buf.length = 0;
      }

      let para = [];
      let listType = null;
      let listItems = [];

      function flushList() {
        if (!listType) return;
        const tag = listType;
        out.push(
          "<" + tag + ">" +
          listItems.map((item) => "<li>" + inlineMd(item) + "</li>").join("") +
          "</" + tag + ">"
        );
        listType = null;
        listItems = [];
      }

      while (i < lines.length) {
        const line = lines[i];

        if (line.startsWith("```")) {
          flushParagraph(para);
          flushList();
          if (inCode) {
            out.push("<pre><code>" + escapeHtml(codeBuf.join("\n")) + "</code></pre>");
            codeBuf = [];
            inCode = false;
          } else {
            inCode = true;
          }
          i += 1;
          continue;
        }

        if (inCode) {
          codeBuf.push(line);
          i += 1;
          continue;
        }

        if (/^\s*$/.test(line)) {
          flushParagraph(para);
          flushList();
          i += 1;
          continue;
        }

        if (/^---+\s*$/.test(line) || /^\*\*\*+\s*$/.test(line)) {
          flushParagraph(para);
          flushList();
          out.push("<hr />");
          i += 1;
          continue;
        }

        const heading = /^(#{1,4})\s+(.+)$/.exec(line);
        if (heading) {
          flushParagraph(para);
          flushList();
          const level = heading[1].length;
          out.push("<h" + level + ">" + inlineMd(heading[2].trim()) + "</h" + level + ">");
          i += 1;
          continue;
        }

        const bq = /^>\s?(.*)$/.exec(line);
        if (bq) {
          flushParagraph(para);
          flushList();
          out.push("<blockquote><p>" + inlineMd(bq[1]) + "</p></blockquote>");
          i += 1;
          continue;
        }

        const ul = /^[-*+]\s+(.+)$/.exec(line);
        if (ul) {
          flushParagraph(para);
          if (listType && listType !== "ul") flushList();
          listType = "ul";
          listItems.push(ul[1]);
          i += 1;
          continue;
        }

        const ol = /^\d+\.\s+(.+)$/.exec(line);
        if (ol) {
          flushParagraph(para);
          if (listType && listType !== "ol") flushList();
          listType = "ol";
          listItems.push(ol[1]);
          i += 1;
          continue;
        }

        flushList();
        para.push(line.trim());
        i += 1;
      }

      if (inCode) {
        out.push("<pre><code>" + escapeHtml(codeBuf.join("\n")) + "</code></pre>");
      }
      flushParagraph(para);
      flushList();
      return out.join("") || "<p>—</p>";
    }

    function openDetail(record) {
      const id = record.userContactId;
      const profile = PROFILES[id] || {};
      const f = profile.userContactFile || {};
      const queries = Array.isArray(profile.searchQueries) ? profile.searchQueries : [];
      const tokens = record.tokens || {};

      const titleSrc = String(f.positioning || "").replace(/\s+/g, " ").trim();
      detailTitle.textContent = titleSrc
        ? (titleSrc.length > 72 ? titleSrc.slice(0, 71) + "…" : titleSrc)
        : "User";
      detailId.textContent =
        id +
        " · v" + (record.userContactFileVersion ?? profile.userContactFileVersion ?? "?") +
        " · " + (record.pairCount ?? 0) + " pairs";

      const chips = [
        ["profile Σ", tokens.total_profile_tokens],
        ["seeker Σ", tokens.total_seeker_tokens],
        ["lookingFor", tokens.lookingFor],
        ["background", tokens.background],
        ["positioning", tokens.positioning],
      ]
        .filter(([, n]) => n != null)
        .map(([label, n]) => `<span>${escapeHtml(label)} ${escapeHtml(n)}</span>`)
        .join("");

      const sections = PROFILE_LABELS.map(([key, label]) => {
        const value = f[key];
        if (value == null || String(value).trim() === "") return "";
        return (
          `<section class="detail-card"><h3>${escapeHtml(label)}</h3>` +
          `<div class="md">${renderMarkdown(value)}</div></section>`
        );
      }).join("");

      const querySection =
        `<section class="detail-card"><h3>Search queries (${queries.length})</h3>` +
        (queries.length
          ? `<div class="queries"><ol>${queries
              .map(
                (qq, idx) =>
                  `<li><strong>#${idx + 1}</strong><div class="md">${renderMarkdown(qq)}</div></li>`
              )
              .join("")}</ol></div>`
          : "<p class='md'>—</p>") +
        `</section>`;

      detailBody.innerHTML =
        `<section class="detail-card"><h3>Token summary</h3><div class="token-chips">${chips}</div></section>` +
        sections +
        querySection;

      dialog.showModal();
    }

    function renderHead() {
      head.innerHTML = COLS.map((c) => {
        const active = c.key === sortKey ? " active" : "";
        const arrow = c.key === sortKey ? (sortDir === "asc" ? "▲" : "▼") : "⇅";
        return `<th class="${active}" data-key="${c.key}" title="Sort by ${escapeHtml(c.label)}">${escapeHtml(c.label)}<span class="sort">${arrow}</span></th>`;
      }).join("");
      head.querySelectorAll("th").forEach((th) => {
        th.addEventListener("click", () => {
          const key = th.dataset.key;
          if (sortKey === key) sortDir = sortDir === "asc" ? "desc" : "asc";
          else { sortKey = key; sortDir = key === "userContactId" ? "asc" : "desc"; }
          apply();
        });
      });
    }

    function passes(u) {
      const idTerm = q.value.trim().toLowerCase();
      if (idTerm && !String(u.userContactId).toLowerCase().includes(idTerm)) return false;

      const pMin = numOrNull(pairsMin);
      const pMax = numOrNull(pairsMax);
      const pairs = u.pairCount ?? 0;
      if (pMin != null && pairs < pMin) return false;
      if (pMax != null && pairs > pMax) return false;

      const t = u.tokens || {};
      const po = numOrNull(posMin);
      const bg = numOrNull(bgMin);
      const lf = numOrNull(lfMin);
      const tMin = numOrNull(totalMin);
      const tMax = numOrNull(totalMax);
      if (po != null && (t.positioning ?? 0) < po) return false;
      if (bg != null && (t.background ?? 0) < bg) return false;
      if (lf != null && (t.lookingFor ?? 0) < lf) return false;
      if (tMin != null && (t.total_profile_tokens ?? 0) < tMin) return false;
      if (tMax != null && (t.total_profile_tokens ?? 0) > tMax) return false;
      return true;
    }

    function sorted(list) {
      const col = COLS.find((c) => c.key === sortKey) || COLS[0];
      const mul = sortDir === "asc" ? 1 : -1;
      return list.slice().sort((a, b) => {
        const av = col.get(a);
        const bv = col.get(b);
        if (typeof av === "string" || typeof bv === "string") {
          return mul * String(av).localeCompare(String(bv));
        }
        return mul * ((Number(av) || 0) - (Number(bv) || 0));
      });
    }

    function render(list) {
      countEl.textContent = `Showing ${list.length} of ${USERS.length} users · tokenizer bert-base-uncased · add_special_tokens=false · click user id for profile`;
      if (!list.length) {
        body.innerHTML = `<tr><td class="empty" colspan="${COLS.length}">No users match these filters.</td></tr>`;
        return;
      }
      body.innerHTML = list.map((u) => {
        const cells = COLS.map((c) => {
          const v = c.get(u);
          if (c.short) {
            return (
              `<td class="id">` +
              `<button type="button" class="user-link" data-id="${escapeHtml(u.userContactId)}" title="${escapeHtml(u.userContactId)}">` +
              `${escapeHtml(shortId(v))}</button></td>`
            );
          }
          const cls = c.total ? ' class="total"' : "";
          return `<td${cls}>${escapeHtml(v)}</td>`;
        }).join("");
        return `<tr>${cells}</tr>`;
      }).join("");

      body.querySelectorAll("button.user-link").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          const id = btn.dataset.id;
          const record = USERS.find((u) => u.userContactId === id);
          if (record) openDetail(record);
        });
      });
    }

    function apply() {
      renderHead();
      render(sorted(USERS.filter(passes)));
    }

    document.getElementById("reset").addEventListener("click", () => {
      [q, pairsMin, pairsMax, posMin, bgMin, lfMin, totalMin, totalMax].forEach((el) => { el.value = ""; });
      sortKey = "total_profile_tokens";
      sortDir = "desc";
      apply();
    });

    [q, pairsMin, pairsMax, posMin, bgMin, lfMin, totalMin, totalMax].forEach((el) => {
      el.addEventListener("input", apply);
    });

    renderSummary();
    apply();
  </script>
</body>
</html>
"""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_records(users: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for user in users:
        file_ = user.get("userContactFile") or {}
        field_tokens = {
            field: count_tokens(tokenizer, _as_text(file_.get(field)))
            for field in PROFILE_FIELDS
        }
        queries = user.get("searchQueries") or []
        if not isinstance(queries, list):
            queries = []
        query_tokens = [count_tokens(tokenizer, _as_text(q)) for q in queries]
        total_profile = sum(field_tokens.values())
        total_queries = sum(query_tokens)
        records.append(
            {
                "userContactId": user.get("userContactId"),
                "userContactFileVersion": user.get("userContactFileVersion"),
                "pairCount": user.get("pairCount", 0),
                "searchQueryCount": len(queries),
                "tokens": {
                    **field_tokens,
                    "searchQueries": query_tokens,
                    "searchQueries_total": total_queries,
                    "total_profile_tokens": total_profile,
                    "total_seeker_tokens": total_profile + total_queries,
                },
            }
        )
    return records


def profiles_index(users: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for user in users:
        uid = user.get("userContactId")
        if not uid:
            continue
        out[str(uid)] = {
            "userContactId": uid,
            "userContactFileVersion": user.get("userContactFileVersion"),
            "pairCount": user.get("pairCount", 0),
            "userContactFile": user.get("userContactFile") or {},
            "searchQueries": user.get("searchQueries") or [],
        }
    return out


def write_html(
    records: list[dict[str, Any]],
    users: list[dict[str, Any]],
    out_path: Path,
) -> None:
    payload_obj = {"records": records, "profiles": profiles_index(users)}
    payload = json.dumps(payload_obj, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_HTML_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Rebuild HTML from existing token JSON + unique_users (no tokenizer).",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load tokenizer from local HF cache only (default: true).",
    )
    args = parser.parse_args()

    users = json.loads(args.users.read_text(encoding="utf-8"))
    if not isinstance(users, list):
        raise SystemExit(f"Expected a JSON list in {args.users}")

    if args.html_only:
        if not args.out_json.exists():
            raise SystemExit(f"Missing {args.out_json}; run without --html-only first.")
        records = json.loads(args.out_json.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit(f"Expected a JSON list in {args.out_json}")
        write_html(records, users, args.out_html)
        print(f"Wrote {args.out_html} ({len(records)} users, html-only).")
        return

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    records = build_records(users, tokenizer)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_json} ({len(records)} users).")

    if not args.skip_html:
        write_html(records, users, args.out_html)
        print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
