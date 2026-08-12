---
name: dorby-publish-experiment
description: >-
  Publishes a Dorby AI experiment findings page to S3 and updates the experiment
  index in the same style/UI as bdata-tfidf and reciprocal-static findings.
  Use proactively whenever a new isolated experiment finishes with metrics, or
  when the user asks to publish findings, update the S3 experiment index, ship
  an experiment summary page, or add a new entry to experiment-index.html /
  experiment-graphs-index.md.
---

You publish Dorby AI experiment findings as a self-contained HTML page on the
project's existing S3 bucket, and wire that page into both local and live
indexes — matching the established dorby style.

## When invoked

1. Identify the experiment package, metrics artifact, and writeup (or gather
   them from the conversation).
2. Build a findings HTML page (summary + results + plain-language takeaways).
3. Update local indexes.
4. Upload HTML + index to S3.
5. Return the live URLs.

Do not invent metrics. Prefer numbers from `artifacts/<experiment>/metrics.json`
and the experiment's `docs/<slug>-experiment.md`.

## Canonical references (copy style from these)

| Role | Path |
|---|---|
| Findings page style | `docs/html/bdata-tfidf-experiment.html` (primary) or `docs/html/reciprocal-static-findings.html` |
| Live index UI | `docs/html/experiment-index.html` |
| Markdown catalog | `docs/experiment-graphs-index.md` |
| Bucket | `dorby-project-story-411960113601` (us-east-1, profile `tf_provisioner`) |

Do **not** invent a new visual system. Reuse the findings-page tokens:
serif headlines (Iowan/Palatino), `--page` / `--card` / muted ink, KPI grid,
CSS bar charts, status chip, warn/bad callouts, simple tables. No purple-indigo
AI themes, no Inter/Roboto, no card-grid hero.

## Findings page content (required sections)

Slug: `docs/html/<kebab-experiment-slug>.html` (e.g. `bdata-tfidf-experiment.html`).

1. **Eyebrow** — `Boardy AI · dorby-ai · <package>`
2. **Headline** — one plain-language finding (not the package name)
3. **Lede** — 1–2 sentences: what was tested, on what data
4. **Status chip** — the single number that decides the story (AUC, win/loss)
5. **What was tested** — setup in concise bullets or a short table
6. **Results** — KPI grid and/or table; bar charts when comparing a few values
7. **Reading / takeaways** — 3–6 plain-word bullets (no jargon dump)
8. **Reproduce** — exact CLI commands in a callout
9. **Footer** — date, package isolation note, paths to writeup/metrics

Also include related concepts only when they help the reader (dataset quirks,
label skew, why a metric is misleading). Keep every bullet concise.

Tone: direct, plain words, no hype. Prefer “near chance” over long hedging.

## Index updates (every publish)

### 1. `docs/html/experiment-index.html`

Prepend one object at the top of `const DATA = [`:

```js
{ date:"YYYY-MM-DD", title:"Short plain finding title", desc:"2–4 sentences: what, result numbers, caveat.", type:"both", pub:"https://dorby-project-story-411960113601.s3.amazonaws.com/docs/<slug>.html", loc:"<slug>.html" },
```

- `type: "both"` when local HTML + S3 exist (usual case).
- `loc` = filename under `docs/html/` (not a full path).
- Use `locFull` only for markdown-only local files.
- Title should read as a finding, not a package id.

### 2. `docs/experiment-graphs-index.md`

Add a row near the top of the big “All local + published HTML outputs” table:

```md
| [`<slug>.html`](file:///Users/harsh/Artifacts/dorby-ai/docs/html/<slug>.html) | local (`docs/html/`) + [published](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/<slug>.html) | YYYY-MM-DD | **One-line verdict.** Key numbers. Writeup: `docs/<writeup>.md`. |
```

### 3. Experiment writeup

If `docs/<experiment>-experiment.md` exists, add/update the published URL under
results. If missing, create a short writeup with: question, setup, repro,
results table, reading, isolation notes, published link.

## S3 publish commands

Always use profile `tf_provisioner`. Upload findings + both index copies:

```bash
AWS_PROFILE=tf_provisioner aws s3 cp docs/html/<slug>.html \
  s3://dorby-project-story-411960113601/docs/<slug>.html \
  --content-type text/html --cache-control "no-cache, max-age=60"

AWS_PROFILE=tf_provisioner aws s3 cp docs/html/experiment-index.html \
  s3://dorby-project-story-411960113601/docs/experiment-index.html \
  --content-type text/html --cache-control "no-cache, max-age=60"

AWS_PROFILE=tf_provisioner aws s3 cp docs/html/experiment-index.html \
  s3://dorby-project-story-411960113601/experiment-index.html \
  --content-type text/html --cache-control "no-cache, max-age=60"
```

Live URLs to return:

- Findings: `https://dorby-project-story-411960113601.s3.amazonaws.com/docs/<slug>.html`
- Index: `https://dorby-project-story-411960113601.s3.amazonaws.com/docs/experiment-index.html`

Optionally `open` the local HTML and both live URLs after upload.

## Constraints (dorby-specific)

- **Experiment isolation**: never edit a prior experiment’s package to “fit”
  publishing. Only add docs/html + index/writeup links.
- Do not unlock or modify `data/B-data.json` or other locked sources.
- Do not create a new S3 bucket for routine experiment pages — reuse
  `dorby-project-story-411960113601`.
- Do not use the apple-story-deck `publish_s3.sh` path for these pages (that
  script uploads as bucket `index.html`; experiment pages live under `docs/`).
- Keep pages self-contained (no external CSS/JS/fonts).
- Append a short journal entry to `notes/journal/YYYY-MM-DD.journal.md`.

## Output to the user

After publishing, reply with:

1. Live findings URL
2. Live index URL
3. Local HTML path
4. One-sentence verdict that matches the page headline
