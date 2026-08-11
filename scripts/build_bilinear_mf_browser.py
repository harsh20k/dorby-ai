#!/usr/bin/env python3
"""Build a self-contained HTML report for the `bilinear_mf/` experiment.

Reads the four `artifacts/bilinear_mf/*/results.json` runs and emits one page
with no network requests. See `docs/bilinear-mf-experiment.md` for the writeup.

    python3 scripts/build_bilinear_mf_browser.py
    open docs/html/bilinear-mf-results.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ART = REPO_ROOT / "artifacts" / "bilinear_mf"

LSA_RUNS = [
    ("Voyage-4-large (production)", "mf_voyage_001_lsa"),
    ("TF-IDF (lexical)", "mf_tfidf_001_lsa"),
]
BILINEAR_RUNS = [
    ("Voyage-4-large", "mf_voyage_002"),
    ("TF-IDF", "mf_tfidf_002"),
]

CSS = """
:root{--bg:#fbfbfa;--fg:#1c1b19;--mut:#6b6862;--line:#e3e0da;--card:#fff;
--good:#2f7d5d;--bad:#b3452f;--accent:#3a5f8a}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#e9e7e3;--mut:#9d9890;
--line:#2e2c33;--card:#1e1d23;--good:#5fbb90;--bad:#e08a71;--accent:#8fb3dd}}
:root[data-theme=dark]{--bg:#16151a;--fg:#e9e7e3;--mut:#9d9890;--line:#2e2c33;
--card:#1e1d23;--good:#5fbb90;--bad:#e08a71;--accent:#8fb3dd}
:root[data-theme=light]{--bg:#fbfbfa;--fg:#1c1b19;--mut:#6b6862;--line:#e3e0da;
--card:#fff;--good:#2f7d5d;--bad:#b3452f;--accent:#3a5f8a}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem 5rem;
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.65rem;margin:0 0 .4rem;letter-spacing:-.02em}
h2{font-size:1.15rem;margin:2.6rem 0 .6rem;letter-spacing:-.01em}
h3{font-size:.95rem;margin:1.6rem 0 .5rem;color:var(--mut);font-weight:600}
p{margin:.5rem 0;max-width:72ch}
.sub{color:var(--mut);margin-bottom:1.5rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1rem 1.15rem;margin:1rem 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:640px}
th,td{padding:.45rem .6rem;text-align:right;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
tr.base td{color:var(--mut)}
tr.pick td{background:color-mix(in srgb,var(--accent) 12%,transparent);font-weight:600}
tr.peak td{background:color-mix(in srgb,var(--good) 12%,transparent)}
.up{color:var(--good)}.down{color:var(--bad)}
.note{border-left:3px solid var(--accent);padding:.15rem 0 .15rem .9rem;
color:var(--mut);margin:1rem 0;max-width:70ch}
.warn{border-left-color:var(--bad)}
.bars{margin:.6rem 0}
.bar{display:flex;align-items:center;gap:.6rem;margin:.3rem 0;font-size:13px}
.bar .lbl{width:190px;color:var(--mut);flex:none}
.bar .track{flex:1;height:14px;background:var(--line);border-radius:7px;position:relative;min-width:120px}
.bar .fill{height:100%;border-radius:7px;background:var(--accent)}
.bar .val{width:58px;text-align:right;flex:none;font-variant-numeric:tabular-nums}
.legend{color:var(--mut);font-size:12.5px;margin-top:.5rem}
code{background:var(--line);padding:.08em .35em;border-radius:4px;font-size:.9em}
"""


def esc(x) -> str:
    return html.escape(str(x))


def f4(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def delta(v, base) -> str:
    if v is None or base is None:
        return ""
    d = v - base
    cls = "up" if d > 0 else ("down" if d < 0 else "")
    return f'<span class="{cls}">{d:+.4f}</span>'


def lsa_table(run: dict) -> str:
    lsa = run["lsa"]
    nr = lsa["no_reduction"]
    rows = []

    def row(label, entry, cls=""):
        a, t = entry["all"], entry.get("train", {})
        rows.append(
            f'<tr class="{cls}"><td>{esc(label)}</td>'
            f'<td>{f4(entry.get("explained_variance_ratio"))}</td>'
            f'<td>{f4(t.get("pair_auc"))}</td>'
            f'<td><b>{f4(a["pair_auc"])}</b></td>'
            f'<td><b>{f4(a["hard_neg_auc"])}</b></td>'
            f'<td>{f4(a["easy_neg_auc"])}</td>'
            f'<td>{f4(a["mrr"])}</td>'
            f'<td>{a["recall@1"]:.2f}</td><td>{a["recall@10"]:.2f}</td></tr>'
        )

    row(f'none (d={nr["dim"]})', {**nr, "explained_variance_ratio": None}, "base")

    ranks = run["lsa"]["ranks"]
    best_train = max(ranks.items(), key=lambda kv: kv[1]["train"]["pair_auc"])[0]
    best_all = max(ranks.items(), key=lambda kv: kv[1]["all"]["pair_auc"])[0]
    for k, v in ranks.items():
        cls = "pick" if k == best_train else ("peak" if k == best_all else "")
        row(f"k = {k}", v, cls)

    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>rank</th><th>evr</th><th>train AUC</th><th>all-200 AUC</th>"
        "<th>hard-neg</th><th>easy-neg</th><th>MRR</th><th>R@1</th><th>R@10</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        f'<div class="legend">Blue = chosen by train-split AUC (the only leak-free rule). '
        f"Green = best all-200 AUC (k={esc(best_all)}), which is selection on the eval set "
        f"and therefore a hypothesis, not a result.</div>"
    )


def bilinear_table(runs: list[tuple[str, dict]]) -> str:
    fields = [
        ("selected (d, rank, wd)", lambda b: (
            f'{b["selected"]["reduce_dim"]}, {b["selected"]["rank"]}, '
            f'{b["selected"]["weight_decay"]}')),
        ("inner CV AUC (what selection saw)", lambda b: f4(b["selected"]["inner_cv_auc"])),
        ("all-200 CV pair AUC", lambda b: f"<b>{f4(b['cv_all200']['bilinear']['pair_auc'])}</b>"),
        ("cosine, same space", lambda b: f4(b["cv_all200"]["cosine_same_space"]["pair_auc"])),
        ("Δ vs cosine", lambda b: delta(
            b["cv_all200"]["bilinear"]["pair_auc"],
            b["cv_all200"]["cosine_same_space"]["pair_auc"])),
        ("hard-neg AUC", lambda b: (
            f'{f4(b["cv_all200"]["bilinear"]["hard_neg_auc"])} '
            f'<span class="legend">(cos {f4(b["cv_all200"]["cosine_same_space"]["hard_neg_auc"])})</span>')),
        ("easy-neg AUC", lambda b: (
            f'{f4(b["cv_all200"]["bilinear"]["easy_neg_auc"])} '
            f'<span class="legend">(cos {f4(b["cv_all200"]["cosine_same_space"]["easy_neg_auc"])})</span>')),
        ("MRR", lambda b: (
            f'{f4(b["cv_all200"]["bilinear"]["mrr"])} '
            f'<span class="legend">(cos {f4(b["cv_all200"]["cosine_same_space"]["mrr"])})</span>')),
        ("fold AUC std", lambda b: f4(b["cv_all200"]["fold_auc_std"])),
        ("null mean / p95 / max", lambda b: (
            f'{f4(b["permutation_null"]["mean"])} / {f4(b["permutation_null"]["p95"])} '
            f'/ {f4(b["permutation_null"]["max"])}')),
        ("p-value vs null", lambda b: f4(b["permutation_null"]["p_value"])),
        ("69-pair holdout AUC", lambda b: (
            f'{f4(b["holdout"]["bilinear"]["pair_auc"])} '
            f'<span class="legend">(cos {f4(b["holdout"]["cosine_same_space"]["pair_auc"])})</span>')),
    ]
    head = "".join(f"<th>{esc(n)}</th>" for n, _ in runs)
    body = "".join(
        f"<tr><td>{esc(label)}</td>"
        + "".join(f"<td>{fn(run['bilinear'])}</td>" for _, run in runs)
        + "</tr>"
        for label, fn in fields
    )
    return (
        f'<div class="scroll"><table><thead><tr><th></th>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def hardneg_bars(runs: list[tuple[str, dict]]) -> str:
    """The robust finding: hard-neg AUC lifts at every rank, on both backbones."""
    out = []
    for name, run in runs:
        lsa = run["lsa"]
        base = lsa["no_reduction"]["all"]["hard_neg_auc"]
        out.append(f"<h3>{esc(name)}</h3><div class='bars'>")
        items = [("no reduction", base)] + [
            (f"k = {k}", v["all"]["hard_neg_auc"]) for k, v in lsa["ranks"].items()
        ]
        for lbl, val in items:
            pct = max(0.0, min(1.0, (val - 0.45) / 0.25)) * 100
            color = "var(--accent)" if lbl == "no reduction" else "var(--good)"
            out.append(
                f'<div class="bar"><span class="lbl">{esc(lbl)}</span>'
                f'<span class="track"><span class="fill" style="width:{pct:.1f}%;'
                f'background:{color}"></span></span>'
                f'<span class="val">{val:.4f}</span></div>'
            )
        out.append("</div>")
    out.append(
        '<div class="legend">Scale 0.45–0.70. Blue = un-reduced backbone. '
        "Every compressed rank beats it on hard negatives — the only negative "
        "population that exists in production.</div>"
    )
    return "".join(out)


def build(out_path: Path) -> Path:
    lsa_runs = [
        (name, json.loads((ART / rid / "results.json").read_text()))
        for name, rid in LSA_RUNS
    ]
    bl_runs = [
        (name, json.loads((ART / rid / "results.json").read_text()))
        for name, rid in BILINEAR_RUNS
    ]

    sections = [
        "<h2>Arm 1 — factor the text (LSA / truncated SVD)</h2>",
        "<p>Cosine in the rank-<code>k</code> SVD space. Label-free, so nothing "
        "here can leak and there is no model to overfit.</p>",
    ]
    for name, run in lsa_runs:
        sections.append(f'<div class="card"><h3>{esc(name)}</h3>{lsa_table(run)}</div>')

    sections.append("<h2>The one robust result: hard-negative AUC</h2>")
    sections.append(f'<div class="card">{hardneg_bars(lsa_runs)}</div>')
    sections.append(
        '<div class="note">Compression lifts hard-negative discrimination on '
        "both a lexical and a neural backbone, at every rank tested — a plateau, "
        "not a lucky <code>k</code>. Plausible reading: the discarded tail "
        "dimensions carry mostly topical/surface matching, which production has "
        "already filtered on, so dropping them raises the share of what remains "
        "that is about compatibility.</div>"
    )

    sections.append("<h2>Arm 2 — factor the scoring function (low-rank bilinear)</h2>")
    sections.append(
        "<p><code>score(s,c) = s·c + (As)·(Bc)</code>. Configuration selected by "
        "inner seeker-disjoint CV on the 131 train pairs only; scored by "
        "seeker-disjoint 10-fold CV over all 200 against a 50-draw "
        "label-permutation null.</p>"
    )
    sections.append(f'<div class="card">{bilinear_table(bl_runs)}</div>')
    sections.append(
        '<div class="note warn"><b>It does not work here.</b> On the production '
        "backbone it loses 0.032 all-200 AUC and drops the holdout below chance "
        "(0.4845), while MRR collapses 0.323 → 0.098. Where it appears to win "
        "(TF-IDF, +0.044) the gain is entirely easy-negative — on hard negatives "
        "it goes backwards. Retrieval degrades on both backbones, because the "
        "residual is fit against 200 observed pairs but applied to a "
        "178-candidate ranking it never constrains.</div>"
    )
    sections.append(
        '<div class="note"><b>The instructive number.</b> Inner CV said 0.7399 on '
        "TF-IDF; the honest out-of-fold number was 0.6193. On Voyage, 0.6218 → "
        "0.5410. Selection reports the max over 64 configurations on 131 pairs, "
        "so at this data size hyperparameter selection is itself a source of "
        "overfitting large enough to invent a result — with the SVD width fixed "
        "at 128 the head regularized to numerically zero, and only widening the "
        "grid produced a nonzero head at all.</div>"
    )

    meta = lsa_runs[0][1]
    body = f"""<div class="wrap">
<h1>Matrix factorization on text</h1>
<div class="sub">LSA compression and a low-rank bilinear scorer, on all
{meta['n_pairs']} real pairs ({meta['n_candidates']}-candidate corpus,
{meta['n_seekers']} seekers). See <code>docs/bilinear-mf-experiment.md</code>.</div>
{''.join(sections)}
<h2>Anchors</h2>
<div class="card"><p>With the head disabled this package reproduces the published
frozen-cosine rows digit-for-digit — TF-IDF 0.5649 AUC / 0.1313 MRR,
Voyage-4-large 0.5726 / 0.3102 — asserted in
<code>tests/test_bilinear_mf.py</code>, which is how all three measurement bugs
in this experiment were caught.</p></div>
</div>"""

    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Matrix factorization on text — Dorby AI</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "html" / "bilinear-mf-results.html"
    )
    args = ap.parse_args()
    print(f"wrote {build(args.out)}")


if __name__ == "__main__":
    main()
