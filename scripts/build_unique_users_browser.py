#!/usr/bin/env python3
"""Embed data/unique_users.json into a self-contained HTML browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "data" / "unique_users.json"
DEFAULT_OUT = ROOT / "data" / "unique_users_browser.html"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Unique users — Dorby AI</title>
<style>
  :root {
    --ink: #14201c;
    --muted: #5a6b64;
    --line: #c9d4ce;
    --paper: #f3f6f4;
    --panel: #ffffff;
    --accent: #1f6b55;
    --accent-soft: #d7ebe3;
    --shadow: 0 10px 30px rgba(20, 32, 28, 0.08);
    --radius: 14px;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    --sans: "Avenir Next", Avenir, "Century Gothic", "Gill Sans", "Trebuchet MS", sans-serif;
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

  .wrap {
    width: min(1180px, calc(100% - 2rem));
    margin: 0 auto;
    padding: 2rem 0 4rem;
  }

  header.hero {
    display: grid;
    gap: 1rem;
    margin-bottom: 1.75rem;
  }

  .brand {
    font-family: var(--serif);
    font-size: clamp(2rem, 4vw, 3rem);
    letter-spacing: -0.02em;
    margin: 0;
  }

  .lede {
    margin: 0;
    max-width: 42rem;
    color: var(--muted);
    font-size: 1.05rem;
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
  }

  .search {
    flex: 1 1 280px;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.7rem 1rem;
    box-shadow: var(--shadow);
  }

  .search input {
    border: 0;
    outline: 0;
    width: 100%;
    font: inherit;
    background: transparent;
    color: var(--ink);
  }

  .meta {
    color: var(--muted);
    font-size: 0.92rem;
    white-space: nowrap;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
  }

  .card {
    appearance: none;
    text-align: left;
    width: 100%;
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: var(--radius);
    padding: 1.1rem 1.15rem 1.2rem;
    box-shadow: var(--shadow);
    cursor: pointer;
    display: grid;
    gap: 0.7rem;
    transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
  }

  .card:hover,
  .card:focus-visible {
    transform: translateY(-2px);
    border-color: #9fb7ac;
    box-shadow: 0 14px 34px rgba(20, 32, 28, 0.12);
    outline: none;
  }

  .card-top {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: baseline;
  }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--accent);
    background: var(--accent-soft);
    border-radius: 999px;
    padding: 0.2rem 0.55rem;
  }

  .id {
    font-size: 0.75rem;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 55%;
  }

  .snippet {
    margin: 0;
    font-size: 0.98rem;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .loc, .query {
    margin: 0;
    font-size: 0.86rem;
    color: var(--muted);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .query::before {
    content: "Query · ";
    color: var(--ink);
    font-weight: 600;
  }

  .empty {
    grid-column: 1 / -1;
    padding: 2.5rem 1rem;
    text-align: center;
    color: var(--muted);
    border: 1px dashed var(--line);
    border-radius: var(--radius);
    background: rgba(255,255,255,0.55);
  }

  dialog {
    width: min(760px, calc(100% - 1.5rem));
    max-height: min(88vh, 900px);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0;
    box-shadow: 0 24px 60px rgba(20, 32, 28, 0.22);
    background: var(--panel);
    color: var(--ink);
  }

  dialog::backdrop {
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
  }

  .modal-head h2 {
    margin: 0;
    font-family: var(--serif);
    font-size: 1.55rem;
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
  }

  .modal-body {
    padding: 1.1rem 1.35rem 1.5rem;
    overflow: auto;
    max-height: calc(min(88vh, 900px) - 5.5rem);
    display: grid;
    gap: 1.1rem;
  }

  .field h3 {
    margin: 0 0 0.35rem;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 700;
  }

  .field p, .field pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font: inherit;
    font-size: 0.96rem;
  }

  .queries {
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

  .queries ol {
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.65rem;
  }

  @media (max-width: 640px) {
    .wrap { width: min(100%, calc(100% - 1.25rem)); padding-top: 1.25rem; }
    .card { padding: 1rem; }
    .modal-head, .modal-body { padding-left: 1rem; padding-right: 1rem; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">Unique users</h1>
      <p class="lede">Browse the deduped Boardy contact catalog — positioning, availability, and search queries in one place.</p>
      <div class="toolbar">
        <label class="search" for="q">
          <span aria-hidden="true">⌕</span>
          <input id="q" type="search" placeholder="Filter by positioning, looking for, query, location, or id…" autocomplete="off" />
        </label>
        <div class="meta" id="count"></div>
      </div>
    </header>

    <main class="grid" id="grid" aria-live="polite"></main>
  </div>

  <dialog id="detail" aria-labelledby="detail-title">
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
    const USERS = JSON.parse(document.getElementById("users-data").textContent);
    const grid = document.getElementById("grid");
    const countEl = document.getElementById("count");
    const q = document.getElementById("q");
    const dialog = document.getElementById("detail");
    const detailTitle = document.getElementById("detail-title");
    const detailId = document.getElementById("detail-id");
    const detailBody = document.getElementById("detail-body");
    document.getElementById("close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (e) => {
      if (e.target === dialog) dialog.close();
    });

    function text(v) {
      if (v == null) return "";
      return String(v);
    }

    function snippet(s, n = 180) {
      const t = text(s).replace(/\s+/g, " ").trim();
      if (!t) return "—";
      return t.length > n ? t.slice(0, n - 1) + "…" : t;
    }

    function haystack(u) {
      const f = u.userContactFile || {};
      return [
        u.userContactId,
        f.positioning,
        f.lookingFor,
        f.locationAvailability,
        f.background,
        f.notes,
        ...(u.searchQueries || []),
      ].map(text).join("\n").toLowerCase();
    }

    function field(label, value) {
      const v = text(value).trim();
      if (!v) return "";
      return `<section class="field"><h3>${escapeHtml(label)}</h3><p>${escapeHtml(v)}</p></section>`;
    }

    function escapeHtml(s) {
      return text(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function openDetail(u) {
      const f = u.userContactFile || {};
      const queries = Array.isArray(u.searchQueries) ? u.searchQueries : [];
      detailTitle.textContent = snippet(f.positioning, 72).replace(/…$/, "") || "User";
      detailId.textContent = `${u.userContactId} · v${u.userContactFileVersion ?? "?"} · ${u.pairCount ?? 0} pairs`;
      detailBody.innerHTML = [
        field("Positioning", f.positioning),
        field("Looking for", f.lookingFor),
        field("Location / availability", f.locationAvailability),
        field("Background", f.background),
        field("Notes", f.notes),
        field("Intro preferences", f.introPreferences),
        field("Personal preferences", f.personalPreferences),
        field("Meeting & scheduling", f.meetingAndSchedulingPreferences),
        `<section class="field"><h3>Search queries (${queries.length})</h3>` +
          (queries.length
            ? `<div class="queries"><ol>${queries.map((qq, i) =>
                `<li><strong>#${i + 1}</strong><br>${escapeHtml(qq)}</li>`
              ).join("")}</ol></div>`
            : "<p>—</p>") +
        `</section>`,
      ].join("");
      dialog.showModal();
    }

    function render(list) {
      countEl.textContent = `${list.length} of ${USERS.length}`;
      if (!list.length) {
        grid.innerHTML = `<div class="empty">No users match that filter.</div>`;
        return;
      }
      grid.innerHTML = "";
      for (const u of list) {
        const f = u.userContactFile || {};
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "card";
        btn.innerHTML = `
          <div class="card-top">
            <span class="chip">${escapeHtml(String(u.pairCount ?? 0))} pairs</span>
            <span class="id" title="${escapeHtml(u.userContactId)}">${escapeHtml(u.userContactId)}</span>
          </div>
          <p class="snippet">${escapeHtml(snippet(f.positioning, 220))}</p>
          <p class="loc">${escapeHtml(snippet(f.locationAvailability, 110))}</p>
          <p class="query">${escapeHtml(snippet((u.searchQueries || [])[0], 120))}</p>
        `;
        btn.addEventListener("click", () => openDetail(u));
        grid.appendChild(btn);
      }
    }

    function filter() {
      const term = q.value.trim().toLowerCase();
      if (!term) {
        render(USERS);
        return;
      }
      render(USERS.filter((u) => haystack(u).includes(term)));
    }

    q.addEventListener("input", filter);
    render(USERS);
  </script>
</body>
</html>
"""


def build(json_path: Path, out_path: Path) -> int:
    users = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(users, list):
        raise SystemExit(f"Expected a JSON list in {json_path}")

    payload = json.dumps(users, ensure_ascii=False, separators=(",", ":"))
    # Prevent </script> in user text from breaking the HTML script tag.
    payload = payload.replace("<", "\\u003c")
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return len(users)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    n = build(args.json, args.out)
    print(f"Wrote {args.out} with {n} users embedded.")


if __name__ == "__main__":
    main()
