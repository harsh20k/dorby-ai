#!/usr/bin/env python3
"""Build a self-contained HTML comparison page for the LLM-judge experiment.

Pulls the matched-holdout pair AUC / hard-neg / easy-neg numbers for every
embedding baseline (docs/baseline-results-holdout.json) plus all four
(model, framing) LLM-judge combinations run so far
(artifacts/llm_judge/*/metrics_holdout.json), and renders a ranked bar chart
plus a full data table — same visual language (tokens, table styling) as
scripts/build_holdout_browser.py, so this reads as one family of pages in
docs/experiment-graphs-index.md.

Usage:
    python3 scripts/build_llm_judge_browser.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "html" / "llm-judge-comparison.html"
BASELINE_JSON = ROOT / "docs" / "baseline-results-holdout.json"
LLM_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge"

# Validated default categorical palette (references/palette.md), slots 1 & 2 —
# two categories (embedding baseline vs LLM judge) validated all-pairs in
# both modes, so no secondary encoding is required beyond the direct labels
# already shown on every bar.
COLOR_BASELINE_LIGHT, COLOR_BASELINE_DARK = "#2a78d6", "#3987e5"  # slot 1: blue
COLOR_JUDGE_LIGHT, COLOR_JUDGE_DARK = "#eb6834", "#d95926"  # slot 2: orange

LABELS = {
    "tfidf": "TF-IDF (lexical)",
    "bert_frozen": "Frozen BERT",
    "voyage_nano": "Voyage-4-nano",
    "voyage_large": "Voyage-4-large (production)",
    "hybrid_tfidf_voyage": "Hybrid TF-IDF+nano",
    "hf_embedding_qwen_qwen3-embedding-8b": "Qwen3-Embedding-8B (open)",
    "hf_embedding_intfloat_e5-mistral-7b-instruct": "E5-mistral-7b-instruct (open)",
    "hf_embedding_nvidia_nv-embed-v2": "NV-Embed-v2 (open, approx.)",
    "hf_embedding_baai_bge-en-icl": "BGE-en-ICL (open)",
    "hf_embedding_zeroentropy_zembed-1-embedding": "zembed-1-embedding (open)",
    "twotower_run_001": "twotower run_001",
    "twotower_arm_a_real_only": "twotower arm_a_real_only",
}

JUDGE_RUNS = {
    "gemini-3.1-flash-lite (naive)": "openrouter_google_gemini_3_1_flash_lite_naive",
    "gemini-3.1-flash-lite (calibrated)": "openrouter_google_gemini_3_1_flash_lite_calibrated",
    "gemini-3.1-flash-lite (structured_cot)": "openrouter_google_gemini_3_1_flash_lite_structured_cot",
    "gemma-3-27b-it (Bedrock, naive)": "bedrock_google_gemma_3_27b_it_naive",
    "qwen3-32b (Bedrock, naive)": "bedrock_qwen_qwen3_32b_v1_0_naive",
}


def build_data() -> dict:
    d = json.loads(BASELINE_JSON.read_text())
    baselines = d["baselines"]

    rows = []
    for key, v in baselines.items():
        p, h = v["pair"], v["slices"]["neg_hardness"]
        rows.append(
            {
                "name": LABELS.get(key, key),
                "kind": "baseline",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": v["retrieval"]["mrr"],
            }
        )

    judge_detail = []
    for name, dirname in JUDGE_RUNS.items():
        path = LLM_JUDGE_ARTIFACTS / dirname / "metrics_holdout.json"
        if not path.exists():
            # artifacts/ is gitignored -- a run made in a different checkout
            # (e.g. the Bedrock combos, run in an earlier worktree session)
            # simply isn't on disk here. Skip rather than crash the build.
            print(f"  skip {name}: {path} not found")
            continue
        m = json.loads(path.read_text())
        p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": f"LLM judge: {name}",
                "kind": "llm_judge",
                "gets_query": False,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": None,
            }
        )
        judge_detail.append(
            {
                "name": name,
                "backend": m.get("backend", "openrouter"),
                "variant": m["prompt_variant"],
                "n": m["num_pairs_scored"],
                "pair_auc": p["roc_auc"],
                "decision_accuracy": dec["accuracy"],
                "f1": dec["f1"],
                "yes_rate": dec["yes_rate"],
                "yes_rate_pos": dec["yes_rate_on_positives"],
                "yes_rate_neg": dec["yes_rate_on_negatives"],
                "mean_confidence": dec["mean_confidence"],
                "mean_confidence_correct": dec["mean_confidence_when_correct"],
                "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
            }
        )

    rows.sort(key=lambda r: -r["pair_auc"])
    return {
        "generated_at": d["generated_at"],
        "protocol_note": d["protocol_note"],
        "rows": rows,
        "judge_detail": judge_detail,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LLM judge vs. embedding baselines — Dorby AI</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --panel: #ffffff;
    --baseline: __COLOR_BASELINE_LIGHT__;
    --judge: __COLOR_JUDGE_LIGHT__;
    --good: #0ca30c;
    --good-bg: #e3f6e3;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --panel: #161615;
      --baseline: __COLOR_BASELINE_DARK__;
      --judge: __COLOR_JUDGE_DARK__;
      --good: #0ca30c;
      --good-bg: #103010;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --panel: #161615;
    --baseline: __COLOR_BASELINE_DARK__;
    --judge: __COLOR_JUDGE_DARK__;
    --good: #0ca30c;
    --good-bg: #103010;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink);
    background: var(--page);
    line-height: 1.45;
  }
  :where(a, button, input, [tabindex]):focus-visible {
    outline: 2px solid var(--good);
    outline-offset: 2px;
    border-radius: 4px;
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }

  .wrap { width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }

  header.hero { display: grid; gap: 0.6rem; margin-bottom: 1.6rem; }
  .brand { font-size: clamp(1.6rem, 3.6vw, 2.2rem); letter-spacing: -0.02em; margin: 0; font-weight: 700; }
  .lede { margin: 0; max-width: 60rem; color: var(--ink-2); font-size: 0.95rem; }
  .lede code { background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 0.05rem 0.35rem; font-size: 0.85em; }
  .meta { margin: 0; font-size: 0.8rem; color: var(--muted); }

  .legend { display: flex; gap: 1.2rem; align-items: center; margin-top: 0.9rem; font-size: 0.85rem; color: var(--ink-2); }
  .legend .swatch { display: inline-flex; align-items: center; gap: 0.45rem; }
  .legend .dot { width: 0.7rem; height: 0.7rem; border-radius: 3px; flex: none; }

  .chart-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.4rem 1.4rem 1rem; margin-top: 1rem;
  }
  .chart-title { font-size: 0.95rem; font-weight: 700; margin: 0 0 0.15rem; }
  .chart-sub { font-size: 0.78rem; color: var(--muted); margin: 0 0 1rem; }

  .bar-row {
    display: grid; grid-template-columns: 15rem 1fr 4.2rem; align-items: center;
    gap: 0.7rem; padding: 0.28rem 0; position: relative;
  }
  .bar-row .name {
    font-size: 0.82rem; text-align: right; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; color: var(--ink-2);
  }
  .bar-row.is-judge .name { color: var(--ink); font-weight: 600; }
  .bar-track { position: relative; height: 1.15rem; background: var(--grid); border-radius: 4px; overflow: visible; }
  .bar-fill {
    position: absolute; top: 0; left: 0; height: 100%; border-radius: 4px;
    min-width: 4px;
  }
  .bar-row .value {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-size: 0.82rem; text-align: right;
  }
  .bar-row:hover .bar-track { outline: 2px solid var(--ink); outline-offset: 1px; }
  .chance-line {
    position: absolute; top: -0.2rem; bottom: -0.2rem; width: 1px;
    background: var(--muted); opacity: 0.55;
  }
  .chance-label {
    position: absolute; font-size: 0.68rem; color: var(--muted); top: -1.05rem; transform: translateX(-50%);
  }

  .section-title { margin: 1.7rem 0 0.2rem; font-size: 0.95rem; font-weight: 700; }
  .section-sub { margin: 0 0 0.4rem; font-size: 0.78rem; color: var(--muted); }

  .findings { display: grid; gap: 0.7rem; margin-top: 0.6rem; }
  .finding {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.87rem; color: var(--ink-2);
  }
  .finding b { color: var(--ink); }

  .table-scroll {
    margin-top: 0.8rem; overflow-x: auto; border: 1px solid var(--border);
    border-radius: 10px; background: var(--panel);
  }
  table { border-collapse: collapse; width: 100%; font-size: 0.84rem; }
  thead th {
    position: sticky; top: 0; background: var(--surface-1); text-align: right;
    padding: 0.6rem 0.8rem; font-weight: 700; white-space: nowrap;
    border-bottom: 1px solid var(--grid);
  }
  thead th:first-child { text-align: left; }
  tbody td, tbody th {
    padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--grid); white-space: nowrap;
  }
  tbody th { text-align: left; font-weight: 600; color: var(--ink-2); }
  tbody td {
    text-align: right; color: var(--ink);
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-size: 0.82rem;
  }
  tbody tr:last-child td, tbody tr:last-child th { border-bottom: none; }
  tbody tr:hover td, tbody tr:hover th { background: color-mix(in srgb, var(--ink) 4%, transparent); }
  tbody tr.is-judge th { color: var(--judge); }
  td.na { color: var(--muted); }

  @media (max-width: 640px) {
    .wrap { width: min(100%, calc(100% - 1.25rem)); padding-top: 1.25rem; }
    .bar-row { grid-template-columns: 8rem 1fr 3.4rem; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">LLM judge vs. embedding baselines</h1>
      <p class="lede">
        Can an LLM predict accept/decline from two full profiles alone, with
        <strong>no <code>searchQuery</code></strong> — the one thing every
        embedding baseline below gets? Matched frozen 69-pair real holdout,
        so every row is the same population. Full writeup:
        <code>docs/llm-judge-experiment.md</code>.
      </p>
      <p class="meta" id="meta-line"></p>
      <div class="legend">
        <span class="swatch"><span class="dot" style="background:var(--baseline)"></span>Embedding baseline (gets the query)</span>
        <span class="swatch"><span class="dot" style="background:var(--judge)"></span>LLM judge (no query)</span>
      </div>
    </header>

    <main>
      <div class="chart-card">
        <p class="chart-title">Pair ROC-AUC, ranked</p>
        <p class="chart-sub">Dashed line = chance (0.5). Bar length is AUC on this 0.5–1.0 scale.</p>
        <div id="chart-auc"></div>
      </div>

      <div class="chart-card">
        <p class="chart-title">Hard-negative AUC vs. easy-negative AUC</p>
        <p class="chart-sub">
          Every real negative is production's own false positive — a plausible-looking intro a human still declined —
          so hard negatives (lexically similar to the query, superficially plausible) are the only population that exists
          at serving time. Sorted by hard-neg AUC.
        </p>
        <div id="chart-hardness"></div>
      </div>

      <p class="section-title">Findings</p>
      <div class="findings" id="findings"></div>

      <p class="section-title">Full comparison table</p>
      <p class="section-sub">All metrics, matched 69-pair holdout. MRR is retrieval-only — no LLM-judge row has one (see below).</p>
      <div class="table-scroll">
        <table id="full-table"></table>
      </div>

      <p class="section-title">LLM-judge run detail</p>
      <p class="section-sub">All four (model, framing) combinations tested — decision behavior, not just the AUC summary above.</p>
      <div class="table-scroll">
        <table id="judge-table"></table>
      </div>
    </main>
  </div>

  <script id="data" type="application/json">__EMBEDDED_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("data").textContent);

    document.getElementById("meta-line").textContent =
      `${DATA.protocol_note} · baseline numbers generated ${DATA.generated_at}`;

    function fmt(x, digits = 4) {
      return x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits);
    }
    function pct(x, digits = 1) {
      return x === null || x === undefined ? "—" : (x * 100).toFixed(digits) + "%";
    }

    // ---- bar chart: Pair AUC, ranked ----
    function renderBarChart(container, rows, valueKey, opts = {}) {
      const min = opts.min ?? 0.5, max = opts.max ?? 0.7;
      const chanceAt = opts.chance;
      const el = document.getElementById(container);
      el.innerHTML = "";
      for (const r of rows) {
        const v = r[valueKey];
        if (v === null || v === undefined) continue;
        const pctWidth = Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
        const row = document.createElement("div");
        row.className = "bar-row" + (r.kind === "llm_judge" ? " is-judge" : "");
        row.innerHTML = `
          <div class="name" title="${r.name}">${r.name}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${pctWidth}%; background:${r.kind === "llm_judge" ? "var(--judge)" : "var(--baseline)"}"></div>
            ${chanceAt !== undefined ? `<div class="chance-line" style="left:${((chanceAt - min) / (max - min)) * 100}%"></div>` : ""}
          </div>
          <div class="value">${fmt(v)}</div>
        `;
        row.title = `${r.name}: ${valueKey} = ${fmt(v)}`;
        el.appendChild(row);
      }
    }

    renderBarChart("chart-auc", DATA.rows, "pair_auc", { min: 0.45, max: 0.68, chance: 0.5 });

    const hardnessRows = [...DATA.rows]
      .filter((r) => r.hard_auc !== null && r.hard_auc !== undefined)
      .sort((a, b) => b.hard_auc - a.hard_auc);
    renderBarChart("chart-hardness", hardnessRows, "hard_auc", { min: 0.35, max: 0.68, chance: 0.5 });

    // ---- findings ----
    const findingsEl = document.getElementById("findings");
    const findings = [
      `<b>gemini-3.1-flash-lite (naive)</b> is the only LLM judge that beats Voyage-4-large — Boardy's own production
       model — on pair AUC (0.6358 vs 0.6086), while never seeing the search query Voyage gets.`,
      `<b>Hard-negative AUC tells a different story than the headline ranking.</b> Both Bedrock judges
       (Gemma 3 27B, Qwen3-32B) actually edge out flash-lite here (0.6216 / 0.6224 vs 0.6466 — still behind, but the
       gap shrinks a lot). Every embedding baseline except the strongest few collapses much harder from easy to hard
       negatives than any LLM judge does.`,
      `<b>The "calibrated" prompt — telling the model production already vetted relevance and the base rate is 50/50 —
       made flash-lite worse</b> (AUC 0.6358 → 0.5901), by making it answer "yes" far less often (56.5% → 30.4%)
       rather than discriminating better.`,
      `<b>All four LLM-judge combinations land in the same 0.58–0.64 band</b> — comfortably above chance, well short of
       Qwen3-Embedding-8B's 0.6595 ceiling. None of this is deployable: a per-candidate LLM call is out of scope
       under the project's &lt;100ms serving budget, and there's no retrieval column for any LLM-judge row because
       ranking a full candidate corpus per query would need tens of thousands of calls.`,
    ];
    for (const f of findings) {
      const div = document.createElement("div");
      div.className = "finding";
      div.innerHTML = f;
      findingsEl.appendChild(div);
    }

    // ---- full comparison table ----
    const fullCols = [
      ["Model", (r) => r.name, "left"],
      ["Pair AUC", (r) => fmt(r.pair_auc)],
      ["Hard-neg AUC", (r) => fmt(r.hard_auc)],
      ["Easy-neg AUC", (r) => fmt(r.easy_auc)],
      ["Best F1", (r) => fmt(r.best_f1)],
      ["Acc @ 0.5", (r) => fmt(r.accuracy_at_half)],
      ["MRR", (r) => (r.mrr === null || r.mrr === undefined ? '<span class="na">n/a</span>' : fmt(r.mrr))],
      ["Gets query?", (r) => (r.gets_query ? "yes" : "no")],
    ];
    const fullTable = document.getElementById("full-table");
    fullTable.innerHTML = `
      <thead><tr>${fullCols.map(([label], i) => `<th style="${i===0?'text-align:left':''}">${label}</th>`).join("")}</tr></thead>
      <tbody>
        ${DATA.rows.map((r) => `
          <tr class="${r.kind === "llm_judge" ? "is-judge" : ""}">
            ${fullCols.map(([label, fn], i) => i === 0
              ? `<th>${fn(r)}</th>`
              : `<td>${fn(r)}</td>`
            ).join("")}
          </tr>
        `).join("")}
      </tbody>
    `;

    // ---- LLM-judge detail table ----
    const judgeCols = [
      ["Run", (r) => r.name, "left"],
      ["Backend", (r) => r.backend],
      ["n", (r) => r.n],
      ["Pair AUC", (r) => fmt(r.pair_auc)],
      ["Decision acc.", (r) => fmt(r.decision_accuracy)],
      ["F1", (r) => fmt(r.f1)],
      ["Yes-rate", (r) => pct(r.yes_rate)],
      ["Yes-rate (pos)", (r) => pct(r.yes_rate_pos)],
      ["Yes-rate (neg)", (r) => pct(r.yes_rate_neg)],
      ["Mean conf.", (r) => r.mean_confidence.toFixed(1)],
      ["Conf. when right", (r) => r.mean_confidence_correct.toFixed(1)],
      ["Conf. when wrong", (r) => r.mean_confidence_wrong.toFixed(1)],
    ];
    const judgeTable = document.getElementById("judge-table");
    judgeTable.innerHTML = `
      <thead><tr>${judgeCols.map(([label], i) => `<th style="${i===0?'text-align:left':''}">${label}</th>`).join("")}</tr></thead>
      <tbody>
        ${DATA.judge_detail.map((r) => `
          <tr>
            ${judgeCols.map(([label, fn], i) => i === 0
              ? `<th>${fn(r)}</th>`
              : `<td>${fn(r)}</td>`
            ).join("")}
          </tr>
        `).join("")}
      </tbody>
    `;
  </script>
</body>
</html>
"""


def main() -> int:
    data = build_data()
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", json.dumps(data))
    html = html.replace("__COLOR_BASELINE_LIGHT__", COLOR_BASELINE_LIGHT)
    html = html.replace("__COLOR_BASELINE_DARK__", COLOR_BASELINE_DARK)
    html = html.replace("__COLOR_JUDGE_LIGHT__", COLOR_JUDGE_LIGHT)
    html = html.replace("__COLOR_JUDGE_DARK__", COLOR_JUDGE_DARK)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(html)
    print(f"wrote {DEFAULT_OUT} ({len(data['rows'])} rows, {len(data['judge_detail'])} judge combos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
