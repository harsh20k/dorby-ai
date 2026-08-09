#!/usr/bin/env python3
"""Add the focused-seed evolution run (evo_focused_001) to the published
"Judge Prompt Evolution" artifact.

The artifact is a single self-contained HTML page whose per-run traces live in
`<script id="evo-data-<run>" type="application/json">` blocks. This script
appends one more such block, registers the run in the page's `RUNS` map and
run-toggle, adds its rows to the results table, and updates the surrounding
copy — rather than regenerating the page, so the previous nine runs' embedded
traces stay byte-identical.

    python scripts/add_focused_run_to_evolution_artifact.py \
        --in  <downloaded artifact>.html \
        --out <path to publish>.html

Idempotent: refuses to run twice on the same file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "evo_focused_001"
RUN_DIR = REPO_ROOT / "artifacts" / "judge_prompt_evolution_focused" / RUN_ID

# Measured this run; see docs/judge-prompt-evolution-focused-experiment.md.
SEED_AUC, SEED_ACC, SEED_F1 = 0.6474, 0.5950, 0.6197
EVO_AUC, EVO_ACC, EVO_F1 = 0.5885, 0.5900, 0.5543


def build_run_trace() -> list[dict]:
    """Reconstruct the page's per-iteration record shape from the run dir."""
    records: list[dict] = []
    for path in sorted(
        RUN_DIR.glob("iterations/*.json"),
        key=lambda p: (
            int(p.stem[:2]) if p.stem[:2].isdigit() else 0,
            1 if p.stem.endswith("s") else 0,
        ),
    ):
        raw = json.loads(path.read_text(encoding="utf-8"))
        is_seed = path.stem.endswith("_seed")
        is_summarize = path.stem.endswith("s")
        prompt = raw["prompt_after"]
        records.append(
            {
                "iteration": raw["iteration"],
                "model": None if is_seed else raw.get("optimizer_model"),
                "chars": len(prompt),
                "rationale": raw.get("rationale"),
                "prompt": prompt,
                **({"kind": "summarize"} if is_summarize else {}),
            }
        )
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    args = ap.parse_args()

    html = args.src.read_text(encoding="utf-8")
    if RUN_ID in html:
        raise SystemExit(f"{RUN_ID} is already present in {args.src} — nothing to do")

    trace = build_run_trace()
    assert trace and trace[0]["model"] is None, "first record must be the seed"

    # 1. Embed the trace next to the other runs' blocks.
    anchor = '<script id="evo-data-evo_009"'
    block = (
        f'<script id="evo-data-{RUN_ID}" type="application/json">'
        + json.dumps(trace)
        + "</script>\n"
    )
    html = html.replace(anchor, block + anchor, 1)

    # 2. Register it in the RUNS map.
    html = html.replace(
        "    evo_009: JSON.parse(document.getElementById('evo-data-evo_009').textContent),",
        "    evo_009: JSON.parse(document.getElementById('evo-data-evo_009').textContent),\n"
        f"    {RUN_ID}: JSON.parse(document.getElementById('evo-data-{RUN_ID}').textContent),",
        1,
    )

    # 3. Results table: a second seed (the focused prompt) and its evolved run.
    #    Marked `family: 'focused'` so they are excluded from the naive-family
    #    best/worst highlighting rather than competing with it.
    html = html.replace(
        "    { name: 'evo_009 (evo_008 recipe, structured_cot seed, CONTAMINATED)',"
        " auc: 0.5814, acc: 0.5400, f1: 0.4831, delta: null, contaminated: true },",
        "    { name: 'evo_009 (evo_008 recipe, structured_cot seed, CONTAMINATED)',"
        " auc: 0.5814, acc: 0.5400, f1: 0.4831, delta: null, contaminated: true },\n"
        f"    {{ name: 'Focused seed — query + trimmed fields', auc: {SEED_AUC},"
        f" acc: {SEED_ACC}, f1: {SEED_F1}, delta: null, seed: true, family: 'focused' }},\n"
        f"    {{ name: '{RUN_ID} (focused seed, 20 rounds)', auc: {EVO_AUC},"
        f" acc: {EVO_ACC}, f1: {EVO_F1}, delta: {round(EVO_AUC - SEED_AUC, 4)},"
        f" family: 'focused' }},",
        1,
    )
    html = html.replace(
        "  const nonSeedClean = RESULTS.filter(r => !r.seed && !r.contaminated);",
        "  const nonSeedClean = RESULTS.filter(r => !r.seed && !r.contaminated && !r.family);",
        1,
    )
    html = html.replace(
        "    if (r.contaminated) tr.className = 'contaminated';",
        "    if (r.contaminated) tr.className = 'contaminated';\n"
        "    if (r.family === 'focused') tr.classList.add('focused');",
        1,
    )
    html = html.replace(
        "  .results tr.contaminated td { color: var(--warn); font-style: italic; }",
        "  .results tr.contaminated td { color: var(--warn); font-style: italic; }\n"
        "  .results tr.focused td { color: var(--focused); }\n"
        "  .results tr.focused td:first-child { font-weight: 600; }",
        1,
    )

    # 4. Run toggle button.
    html = html.replace(
        '    <button class="run-btn contaminated" data-run="evo_009">'
        'evo_009 <span class="rb-auc">0.5814*</span></button>',
        '    <button class="run-btn contaminated" data-run="evo_009">'
        'evo_009 <span class="rb-auc">0.5814*</span></button>\n'
        f'    <button class="run-btn focused" data-run="{RUN_ID}">'
        f'focused_001 <span class="rb-auc">{EVO_AUC}</span></button>',
        1,
    )
    html = html.replace(
        "  .run-btn.best { color: var(--good); }",
        "  .run-btn.best { color: var(--good); }\n"
        "  .run-btn.focused { color: var(--focused); }",
        1,
    )

    # 5. A colour for the new family, defined in every theme block the page has.
    html = html.replace("    --gemini: #7a8fd4;", "    --gemini: #7a8fd4;\n    --focused: #b07cc6;", 1)
    html = html.replace(
        "    --gemini: #47539c;\n    --gemini-dim: #8891c4;\n  }",
        "    --gemini: #47539c;\n    --gemini-dim: #8891c4;\n    --focused: #7d4a93;\n  }",
        1,
    )
    html = html.replace(
        "      --gemini: #47539c;\n      --gemini-dim: #8891c4;",
        "      --gemini: #47539c;\n      --gemini-dim: #8891c4;\n      --focused: #7d4a93;",
        1,
    )

    # 6. A banner for the focused run, reusing the existing banner slots.
    html = html.replace(
        '  <div id="best-banner">',
        '  <div id="focused-banner">✓ evo_focused_001 started from the focused prompt '
        "(searchQuery + trimmed fields, 0.6474) and finished at 0.5885 — the tenth "
        "straight run to lose to its own seed, and the widest gap yet (−0.0589). "
        "Its yes/no calls held up (0.5900 vs 0.5950); what died was confidence: worth "
        "+0.0524 AUC in the seed, −0.0015 here.</div>\n"
        '  <div id="best-banner">',
        1,
    )
    html = html.replace(
        "  #best-banner.show { display: block; }",
        "  #best-banner.show { display: block; }\n"
        "  #focused-banner {\n"
        "    display: none;\n"
        "    background: color-mix(in srgb, var(--focused) 14%, transparent);\n"
        "    border: 1px solid var(--focused);\n"
        "    color: var(--focused);\n"
        "    font-family: var(--mono);\n"
        "    font-size: 12.5px;\n"
        "    padding: 12px 16px;\n"
        "    margin-bottom: 16px;\n"
        "  }\n"
        "  #focused-banner.show { display: block; }",
        1,
    )
    html = html.replace(
        "  const bestBanner = document.getElementById('best-banner');",
        "  const bestBanner = document.getElementById('best-banner');\n"
        "  const focusedBanner = document.getElementById('focused-banner');",
        1,
    )
    html = html.replace(
        "    bestBanner.classList.toggle('show', runId === 'evo_006');",
        "    bestBanner.classList.toggle('show', runId === 'evo_006');\n"
        f"    focusedBanner.classList.toggle('show', runId === '{RUN_ID}');",
        1,
    )

    # 7. Insight tile.
    html = html.replace(
        '    <div class="tile warn">\n      <div class="label">evo_009 (contaminated, structured_cot seed)</div>',
        '    <div class="tile focused-tile">\n'
        '      <div class="label">evo_focused_001 — what confidence is worth</div>\n'
        '      <div class="stat">+0.0524 <span class="unit">→ −0.0015</span></div>\n'
        '      <div class="sub">AUC the confidence score adds on top of the raw yes/no: real in the '
        "focused seed, gone after 20 rounds. The decision itself barely moved (0.5950 → 0.5900).</div>\n"
        "    </div>\n"
        '    <div class="tile warn">\n      <div class="label">evo_009 (contaminated, structured_cot seed)</div>',
        1,
    )
    html = html.replace(
        "  .tile.good .stat { color: var(--good); }",
        "  .tile.good .stat { color: var(--good); }\n"
        "  .tile.focused-tile { background: color-mix(in srgb, var(--focused) 12%, transparent); }\n"
        "  .tile.focused-tile .stat { color: var(--focused); }",
        1,
    )

    # 8. Header/footer copy.
    html = html.replace(
        "<title>Judge Prompt Evolution — evo_001 → evo_009</title>",
        "<title>Judge Prompt Evolution — evo_001 → evo_009 + focused</title>",
        1,
    )
    html = html.replace(
        "<p class=\"eyebrow\">dorby-ai · judge_prompt_evolution · nine runs</p>",
        "<p class=\"eyebrow\">dorby-ai · judge_prompt_evolution · ten runs</p>",
        1,
    )
    html = html.replace(
        "      <div>evo_009: evo_008's recipe, structured_cot seed (contaminated)</div>",
        "      <div>evo_009: evo_008's recipe, structured_cot seed (contaminated)</div>\n"
        "      <div>evo_focused_001: focused seed (query + trimmed fields), gemini optimizer</div>",
        1,
    )
    html = html.replace(
        "    <span>All nine runs' full traces pushed to LangSmith Hub · judge-prompt-evolution</span>",
        "    <span>All ten runs' full traces pushed to LangSmith Hub · judge-prompt-evolution</span>",
        1,
    )
    html = html.replace(
        "    <span>See docs/judge-prompt-evolution-experiment.md for the full writeup</span>",
        "    <span>docs/judge-prompt-evolution-experiment.md · "
        "docs/judge-prompt-evolution-focused-experiment.md</span>",
        1,
    )
    html = html.replace(
        "      checks ran on the direct Gemini API (OpenRouter credits exhausted) — same model, same numbers are\n"
        "      still comparable.</p>",
        "      checks ran on the direct Gemini API (OpenRouter credits exhausted) — same model, same numbers are\n"
        "      still comparable. <strong style=\"color:var(--focused)\">evo_focused_001</strong> is a tenth run on a\n"
        "      different seed entirely — the <em>focused</em> judge prompt, which is given the searchQuery and only\n"
        "      six profile fields, and which the optimizer was also shown (so it never writes rules about fields the\n"
        "      judge cannot see). Its examples came from all 200 pairs by choice, since that prompt's downstream job\n"
        "      is labeling new synthetic pairs rather than scoring these 200; its seed was re-scored through the same\n"
        "      code path (0.6474, vs. 0.6451 published) so the −0.0589 gap is measured, not inherited.</p>",
        1,
    )

    _check_inline_js(html)

    args.dst.write_text(html, encoding="utf-8")
    print(f"wrote {args.dst} ({len(html):,} chars, {len(trace)} iteration records added)")
    return 0


def _check_inline_js(html: str) -> None:
    """Parse the page's own <script> block before writing.

    Learned the hard way: every string replacement above can land correctly and
    still produce a page that renders nothing, because one stray brace makes the
    whole IIFE a SyntaxError and the browser silently skips it. Checking that
    the substrings are *present* does not catch that — the script has to parse.
    """
    import re
    import shutil
    import subprocess
    import tempfile

    blocks = re.findall(r"<script>\n(\(function \(\).*?)</script>", html, re.S)
    if not blocks:
        raise SystemExit("could not locate the page's inline script block to verify")

    node = shutil.which("node")
    if node is None:
        print("WARNING: node not found — inline JS was NOT syntax-checked")
        return

    for i, code in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
            fh.write(code)
            path = fh.name
        result = subprocess.run([node, "--check", path], capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"inline script #{i} does not parse:\n{result.stderr}")
    print(f"inline JS parses ({len(blocks)} block(s) checked with node --check)")


if __name__ == "__main__":
    raise SystemExit(main())
