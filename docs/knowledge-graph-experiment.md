# Profile → knowledge-graph decomposition experiment

A small, cheap experiment (3 LLM calls total) testing a different
representation of a profile than an embedding vector: decompose it into a
knowledge graph — typed nodes and relations, `person -[works_in]->
industry -[has_subcategory]-> subcategory`, the shape used in
relationship-recommendation / social-graph research — instead of collapsing
it into one dense vector. Not a training run or a new baseline; a
diagnostic to see whether a structured representation surfaces failure
modes that cosine similarity hides.

Published: https://dorby-project-story-411960113601.s3.amazonaws.com/docs/knowledge-graph-experiment.html
Local (rebuildable): `docs/html/knowledge-graph-experiment.html`
Script: `scripts/build_kg_experiment.py`

## Setup

One real user (`cmoini4d90eyhlq02tdewefdp`, "Adrian") appears in two real
seed pairs with opposite outcomes:

- **Accepted**: `cm5x8qsuy03wyv80ldcpw85ah` ("Danny") was the seeker; Adrian
  was the match Danny was introduced to, and accepted.
- **Declined**: Adrian was the seeker; `cmdghuwkl0jenqr01uyjiktsk` ("Kunle")
  was the match Adrian was introduced to, and declined.

Each of the three profiles was sent to `google/gemini-3.1-flash-lite` via
OpenRouter, independently, with one instruction: extract a small knowledge
graph (≤14 nodes, ≤16 edges) — role/company, industry + subcategory, needs,
location, traits, affiliations. `searchQuery` was **not** given to the
extraction call; the graph comes from profile text only.

The three graphs were then merged into one:

- **People never link directly to each other** — only through concept
  nodes they each happen to touch, found by exact-match on normalized node
  label text across the three independently-generated graphs.
- **A type-taxonomy layer** was added on top: one node per concept type
  (Industry, Location, Company, Need, Trait, Affiliation, Subcategory),
  wired to every node of that type via a `type_of` edge — not drawn from
  any single profile, the "industry has_subcategory subcategory" shape
  carried one level higher. Rendered hollow/dashed to read as structural
  scaffolding rather than a fourth person's data.

## Finding: the declined match had *more* keyword overlap than the accepted one

The graph merge surfaced a genuine bridge: **"Proptech" is the only concept
node touching all three people**, reached by different path lengths —
Adrian in 2 hops (through his own company), Danny and Kunle both in 3, via
a second bridge node, "Real Estate," that connects the accepted and
declined candidate to each other without ever connecting either straight
to Adrian. Same anchor concept, opposite real outcomes — a small, literal
instance of this project's standing finding that topical/lexical overlap
doesn't separate accept from decline (`docs/objective.md`,
`docs/possible-bugs.md` #3).

Reading the raw profile text (not just the extracted graph) explains *why*,
and it's sharper than the graph alone shows:

- Adrian's own search query for the declined match was literally
  *"seed-stage proptech fintech investor with real estate and go-to-market
  experience, clean cap table friendly."* Kunle's profile hits every one of
  those keywords.
- But Kunle's actual `lookingFor` section shows he isn't an investor at
  all — he's an "operator-investor" himself, hunting for **Real Estate
  Investors**, **Joint Venture funding partners**, and **Family Offices**
  to fund his own real-estate/infrastructure plays. He's a capital
  *seeker*, same as Adrian. Production appears to have matched two people
  chasing the same thing to each other, plausibly on keyword overlap alone.
- The accepted match (Danny) has *weaker* literal keyword overlap with
  Adrian's query, but the roles complement: Danny was searching for
  "operator-angels," a peer network of residential-real-estate founders,
  and mentoring — Adrian fits as a peer/mentor fintech-proptech operator,
  not as a check-writer.

**The declined pair had more surface keyword overlap than the accepted
one, but failed on role-fit** (both were on the capital-*seeking* side of
the table) — the same class of error this project keeps finding in
aggregate (TF-IDF pair AUC 0.592, near-identical query↔match lexical
overlap across accept/decline, hard negatives defined as
topically-similar-but-wrong), now visible in one legible, real pair rather
than only as a summary statistic. It also maps directly onto this
project's own synthetic-negative taxonomy (`wrong_side` /
`wrong_role` in `synth_pipeline/prompts/generate_neg.md`) — a real example
of the exact failure mode that taxonomy was designed to construct
synthetically.

## Why this might matter for Boardy's recommender, not just as a one-off

This is one illustrative pair, not a benchmark result — but it points at a
gap plain-vector similarity structurally can't close, and a cheap way to
close it without breaking the <100ms latency budget (`CLAUDE.md`):

1. **A relation-type compatibility filter/re-ranker, not a replacement for
   retrieval.** Keep the two-tower embedding search for fast candidate
   generation — that's the only thing that scales under the latency
   budget — but extract a few structured fields per profile *offline*, at
   profile-creation time (role: seeker vs. provider of capital /
   deal-flow / mentorship; industry; need-type). Before serving top-K
   candidates, run a lightweight graph/rule check: does the candidate's
   `offers`-side edge actually match the seeker's `seeks`-side edge? Pure
   graph traversal on precomputed structured data — no LLM call, no
   online latency cost — and it would have caught Kunle (seeks capital)
   being offered to Adrian (also seeks capital) even though their
   embeddings plausibly looked close.
2. **Sharper hard negatives for training.** The synthetic-negative
   generator already targets `wrong_side`/`wrong_role` mismatches, but
   candidate selection currently leans on TF-IDF+judge scoring — the same
   lexical-similarity mechanism this experiment shows getting fooled.
   KG-extracted role/need-type mismatches (same industry node, opposite
   need-type) would let harder negatives be *constructed* directly instead
   of hoped-for via keyword ranking, aimed squarely at the hard-negative
   AUC weak spot this project keeps hitting.
3. **Explainability.** Cosine similarity gives no reason for a match. A KG
   gives something concrete to surface: "matched because you're both in
   Proptech" vs. "matched because Danny is offering mentorship you're
   seeking" — useful for debugging why production's own false positives
   (the real negative population this whole project studies) got
   recommended in the first place.

None of this has been built or measured — it's a direction this one
experiment motivates, not a result. The natural next step, if pursued,
would be extracting role/need-type for a larger slice of real pairs and
checking whether a same-side-of-the-table rule alone recovers a
non-trivial share of the real hard-negative AUC gap, before investing in
anything heavier.

## Limitations

- **N=1.** One accepted pair and one declined pair sharing one seeker.
  Illustrative, not a claim generalized beyond this trio — see the
  artifact's own footer caveat.
- **Node-type disagreement across independent calls.** "Proptech" was
  extracted as an `industry` for Adrian and Kunle but a `subcategory` for
  Danny — the three graphs were generated independently, so the model
  didn't apply one consistent ontology. Nodes were merged by label text
  only; the type mismatch is real and left visible rather than papered
  over.
- **One call per profile, no schema validation, no human review** — a
  cheap illustrative decomposition, not a validated ontology extraction
  pipeline.

## Rerun

```bash
python scripts/build_kg_experiment.py --user-id cmoini4d90eyhlq02tdewefdp \
    --cache artifacts/kg_experiment/adrian.json   # optional: skip API calls on rerun
```

Requires `OPENROUTER_API_KEY` in `.env`. Writes
`docs/html/knowledge-graph-experiment.html` in place (self-updating —
the script reads its own prior output as the render template).
