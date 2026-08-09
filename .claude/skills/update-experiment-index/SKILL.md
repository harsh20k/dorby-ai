---
name: update-experiment-index
description: This skill should be used when the user asks to add a new experiment/visualization to the Dorby AI experiment index, "update the experiment index", "add this to the published index", or after a new experiment's HTML/artifact has just been built and needs to be logged alongside every prior one. Covers both docs/html/experiment-index.html (the published Claude artifact) and docs/experiment-graphs-index.md (the markdown reference table it's built from).
---

# Update Experiment Index

Two files track every visualization this project has produced, and they must
both be updated together — never just one:

1. **`docs/html/experiment-index.html`** — a single-page, chronological,
   filterable timeline. Its content lives in one JS array (`const DATA = [...]`)
   near the bottom of the file. This is the file that gets published as a
   Claude artifact and is what people actually browse.
2. **`docs/experiment-graphs-index.md`** — a flat markdown table with the same
   entries, but longer per-entry write-ups (a paragraph, not a line). This is
   the plain-text source of truth the HTML page's descriptions get condensed
   from — see "Two description styles" below.

## Step 0 — find the published URL

The HTML index is normally already published. Find its current URL by
grepping the markdown table:

```bash
grep -n "Dorby AI — Experiment Index" docs/experiment-graphs-index.md
```

The `[published](https://claude.ai/code/artifact/...)` link on that row is the
URL to pass as `url` to the `Artifact` tool later. If it's genuinely never
been published, publish fresh with no `url` param and record the new URL in
both files afterward.

## Step 1 — fetch the live version first, don't trust the local file

**Other sessions edit this artifact independently of any one git worktree.**
The local `docs/html/experiment-index.html` on disk can be stale or diverged
(different wording, entries added elsewhere) compared to what's actually
published. Before editing anything:

```
WebFetch(url=<published index URL>, prompt="Return the complete raw HTML/JS source, especially the full const DATA = [...] array verbatim, plus the header stat line and footer published-page count.")
```

`claude.ai/code/artifact/{uuid}` URLs are fetchable — WebFetch uses the
account's own claude.ai login for these, so this works even though the page
is otherwise private.

Compare the fetched `DATA` array against the local file's array:
- If they match, proceed with the local file as normal.
- If they differ (common — another session may have added entries, or done a
  wording pass across many entries like the 2026-08-09 revision that
  shortened most descriptions), **treat the live version as the base**. Copy
  its exact `DATA` array (and any other changed text — header stats, footer
  count) into the local file before making your own edit, so you don't
  silently revert someone else's concurrent work. Do not use `force: true` on
  the `Artifact` call to paper over this — reconcile the content instead.

## Step 2 — add the new entry to the HTML index's DATA array

Each row:

```js
{ date:"YYYY-MM-DD", title:"...", desc:"...", type:"both"|"published"|"local"|"note", pub:"https://claude.ai/code/artifact/...", loc:"filename.html" }
```

- `type: "both"` when there's a local `docs/html/*.html` file AND it's
  published as its own artifact (the common case for a new experiment).
  `"published"` if only published (no local HTML, e.g. a deck). `"local"` if
  only a local file, no separate artifact. `"note"` for a finding with no
  HTML at all (rare — `pub`/`loc` both omitted, `desc` carries everything).
- **Order matters**: entries are grouped by consecutive matching `date` —
  the render code splits into day-groups by scanning for date changes, so
  all of one day's entries must sit adjacent in the array, newest day first.
  Within a day, put the newest/most-relevant entry first.
- If the experiment is *both* a standalone local HTML page and separately
  published as its own artifact (the usual pattern for an interactive
  browser), that's **two rows**, not one — see any existing pair in the file
  (e.g. `voyage-nano-field-sweep-heatmap.html` + its `Voyage-4-nano field/query
  sweep` published row right after it). The local row's `title` is usually the
  filename-derived name; the published row's `title` is the short human title
  with `desc: "Published version of <file> above"`.

### Two description styles — keep them different on purpose

- **HTML index `desc`**: **1–2 sentences max.** This page is a scannable
  timeline, not a report — the user has explicitly asked for this to stay
  short (2026-08-09). State the headline number and the one-line "why it
  matters," nothing else. Compare to the terse entries already in the file
  (e.g. the 2026-08-09 revision's shortened `top1_ctrl field/query sweep`
  row) rather than the older multi-sentence style some earlier rows still
  have — new entries should match the short style, not the old one.
- **`docs/experiment-graphs-index.md` row**: longer is fine and expected —
  this table's existing convention is a full paragraph per local-file row
  (method, headline finding, caveats, doc/script pointers via backticks),
  with the matching `published` row directly below it just saying
  `Published version of <file> above` (do **not** duplicate the long
  description onto the published row — that's not the convention here,
  confirmed by every existing pair in the table).

## Step 3 — verify before publishing

```bash
python3 -c "
c = open('docs/html/experiment-index.html').read()
print('div balance:', c.count('<div'), c.count('</div>'))
print('new entry present:', '<the new pub UUID or a distinctive title substring>' in c)
"
```

`stat-total` (item count) is computed automatically from `DATA.length` at
render time — never hardcode it. The footer's `(N published pages, checked
YYYY-MM-DD)` text **is** hardcoded and must be bumped by hand: count entries
with a non-null `pub` field, and update the checked-date to today.

## Step 4 — publish

```
Artifact(file_path="docs/html/experiment-index.html", url="<published index URL from Step 0>", favicon="🗂️", title="Dorby AI — Experiment Index", description="Chronological index of every visualization and experiment artifact produced for the Dorby AI / Boardy recsys project.")
```

Keep the favicon (🗂️) and title stable across updates — per the Artifact
tool's own rule, changing them makes the page look like a different page to
someone who already has the tab/link open.

## Step 5 — update the markdown table

Add the matching row(s) to `docs/experiment-graphs-index.md`'s big table
(`## All local + published HTML outputs`), in the same relative position
(the table is also newest-first, grouped by date) — see "Two description
styles" above for how much detail goes here vs. the HTML page.

## Step 6 — don't commit unless asked

Per this project's usual workflow, publishing the artifact and editing these
two files is the deliverable — do not `git add`/`commit`/`push` unless the
user separately asks for that. It's common for "update the index" and "push
everything" to be two distinct requests in this project, not one.
