# RRF pairing pipeline (`synth_pipeline/pairing_rrf/`)

Turns a pool of unlabeled synthetic profiles into labeled pos/neg pairs using
**two independent retrieval channels and an LLM judge**, replacing the scorer
that labeled the earlier `synth_pipeline/pairing/` batches.

## Why this exists

`pairing/` labels with the TF-IDF + Voyage-nano fusion scorer, and on its first
real batch that turned out near-circular: `select.py` ranks candidates by TF-IDF,
then a TF-IDF-heavy scorer grades that ranking. Plain query cosine predicted the
assigned label at **0.868 AUC** — most of the label was lexical overlap, and a
model trained on it would mostly learn lexical overlap.

`pairing_rrf` breaks the loop by construction: **retrieval and labeling come from
different model families.** An open-weight embedder plus BM25 propose candidates;
`google/gemini-3.1-flash-lite` decides. No scorer grades its own ranking.

## Flow

```
profiles → disjoint split → 1 query per lookingFor section (Bedrock)
        → embed both sides (Qwen3-8B, Modal A100) → .npy files → Chroma
        → dense top-10 ‖ BM25 top-10 → weighted RRF → top-5
        → judge, 1 call per pair → pos / hard-neg
```

## The design decisions, and what backs each

**Seeker sectioning, N+1 vectors.** A seeker with two `lookingFor` sections gets
three vectors: the whole profile, plus one per section carrying that section and
all other fields. On the 69-pair holdout, seeker-sectioning lifted pair AUC
0.579 → 0.596 and top-1 retrieval 27.6% → 34.5%, but cost Recall@10
(0.759 → 0.690), and no aggregation softening recovered it. Keeping the whole
vector *alongside* the sharp ones buys the precision without losing the breadth.
Candidates are never sectioned — splitting that side measured worse on every
metric (0.568 vs 0.579).

**One query per section, not N per profile.** Binds each query to a known
section, which is what lets the dense channel pair a query with its own section
vector. Makes "two asks produce two distinct queries" a property of the data
rather than a hope about the model.

**The query is never embedded.** Dense retrieval matches profile against
profile; the query enters only at BM25 and at the judge. This keeps the dense
channel from chasing the generator's own phrasing.

**Parallel channels, not retrieve-then-rerank.** A reranker can only reorder what
the embedder already found, so a lexically obvious match the embedder missed is
unrecoverable. As its own channel, BM25 contributes candidates of its own.

**Fuse by rank, weighted 2:1 toward dense.** On identical inputs over the matched
holdout, RRF and score fusion gave the same pair AUC (0.6397) but RRF ranked
better — MRR 0.4043 vs 0.3665. Solo, dense beats lexical at *retrieval* (MRR
0.4610 vs 0.2939, a 57% edge) while lexical is the better *pair scorer* (AUC
0.6474 vs 0.5793). Retrieval is the job here, so dense gets the weight. `rrf_k=60`
matches the existing hybrid runs.

**Do not tune the weights on a synthetic batch.** It has no ground truth, so
tuning fits the judge's opinions. Everything carries over from real-data
measurements or is fixed in advance.

**Judge: flash-lite, naive framing, one call, no deadband.** It scored 0.6358
pair AUC on the real holdout — beating Voyage-4-large (0.6086), Boardy's own
production model — and its **hard-negative AUC of 0.6466 is the best number in
the project**, ahead of Qwen3-8B's 0.6259. Its weak slice (easy negatives,
0.5638) never reaches it: anything arriving at the judge already survived two
retrieval channels.

Three findings from `docs/llm-judge-experiment.md` are load-bearing:

- *Naive framing only.* Telling the model the truth about the task — production
  pre-filtered for relevance, base rate is 50/50 — measurably hurt: AUC
  0.6358 → 0.5901. It did not discriminate better, it got stingier (yes-rate
  56.5% → 30.4%).
- *Stated confidence is worthless.* 88.6 when right, 88.2 when wrong, 199 of 200
  answers in the 80-100 band. Recorded for audit; nothing gates on it.
- *`max_tokens` must be set.* Unset, OpenRouter reserves credit against the
  model's ceiling and rejects affordable calls as unaffordable.

**Every verdict becomes a label.** No deadband: yes is a positive, no is a hard
negative. Calls-per-labeled-pair is exactly 1.0 — no spend on a pair that does
not become data. The trade is real: the judge decides at 0.5942 accuracy on the
hard slice, so borderline pairs carry some wrong labels.

The negatives are the point. A candidate that dense retrieval ranked top-ten,
that BM25 also surfaced, and that the judge still refused is the closest this
project has come to the real population — something production recommended and a
human declined. No constructed mismatch is in that class.

## Prompts (LangSmith Hub, no local fallback)

Hub-only by design, matching `scripts/profile_gen_prompt_hub.py`: a silent local
fallback would let a run "succeed" on un-audited text. If the pull fails, the run
fails.

| Role | Hub repo | Pin |
|---|---|---|
| query generation | `-/pair-rrf-query` | `PAIR_RRF_PROMPT_QUERY` |
| judge | `-/pair-rrf-judge` | `PAIR_RRF_PROMPT_JUDGE` |

```bash
python -m synth_pipeline.pairing_rrf.push_prompts --dry-run
python -m synth_pipeline.pairing_rrf.push_prompts --tag v1
```

## Running it

The template is `scripts/generate_rrf_dataset.py` + a preset JSON. Everything
tunable lives in the preset, and the preset that produced a batch is copied into
that batch's output, so any result traces back to its settings.

```bash
export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1

python scripts/generate_rrf_dataset.py                        # defaults
python scripts/generate_rrf_dataset.py --preset my_run.json   # tuned
python scripts/generate_rrf_dataset.py --profile-run <dir> --skip-generate
python scripts/generate_rrf_dataset.py --dry-run              # resolved plan only
```

Stages (`generate_profiles`, `pairing`, `judge`, `export`) toggle independently,
and each writes before the next starts, so a late failure never costs the
expensive artifacts earlier in the chain. `queries.json` is a resume checkpoint.

The lower-level entrypoint takes flags instead:

```bash
python -m synth_pipeline.pairing_rrf --profile-run <dir> --batch-id b1 \
    --embed-backend modal --top-k 5 --skip-judge
```

## First real run — `rrf_002` (2026-07-26)

100 profiles generated, 92 usable → 40 seekers / 52 candidates, 135 query
sections (3.4 per seeker), 163 seeker vectors.

| | |
|---|---|
| Labeled pairs | **275** (64 positive / 211 negative) |
| Positive rate | 23.3% |
| Density | 3.02 edges/node (real: 0.673) |
| Judge calls per labeled pair | 1.0 |
| Judge cost | $0.6230, measured from OpenRouter usage |
| Total run cost | ≈$1.40 including generation, Bedrock queries and Modal GPU |
| Wall clock | 5m28s (embedding 87s, judging 212s) |

**The positive rate was not what the judge experiment predicted.** Naive framing
said yes 56.5% of the time on real holdout pairs; here it said yes 23.3%. The
difference is the population — retrieval hands it the top-5 of 52 candidates from
a homogeneous synthetic pool, which is a harder and more uniform slice than the
real holdout. Worth remembering when sizing a batch: expect roughly three
negatives per positive, not a balanced split.

**Density is 4.5× real data** at 3.02 edges/node against 0.673. That is the cost
of leaving `max_pairs_per_seeker` uncapped, taken deliberately so no judged pair
is discarded. Set it to ~1 in the preset if a batch needs to match real topology.

The sharper version of that number, visible in the browser's Topology tab: the
200 real pairs form **97 disconnected components**, this batch forms **exactly
one**. Real data is a scatter of tiny isolated stars — most contacts appear in a
single pair and never touch the rest of the graph — while every synthetic contact
is reachable from every other. The batch is also strictly **bipartite** (40
seekers, 51 candidates, zero overlap) where real data has 15 contacts appearing
on both sides. Neither is wrong for a retrieval batch drawn from one pool, but a
model trained here sees a very differently shaped world.

### The duplicate-pair defect (found and fixed here)

The first run produced 675 pairs from only **275 unique `(seeker, candidate)`
keys** — a seeker's several queries kept retrieving the same candidate. Judged
independently, **25 keys came back with contradictory labels**, the same two
people marked both `pos` and `neg`.

That is defensible in the abstract (a person can suit one ask and not another)
but not in this schema: pairs carry no query identity, and `promote.py` dedups on
`(userContactId, matchContactId)`, so a consumer would arbitrarily keep one copy.
`pairing/select.py` enforces the same uniqueness invariant, which this pipeline
had dropped. `fuse.deduplicate_pairs()` now enforces it by default, keeping the
copy whose query retrieved the candidate most strongly; `--allow-duplicate-pairs`
opts out. Re-running with the fix cost **$0.00** — every verdict was already in
the judge cache.

### Name collisions are expected at this scale

The v3 name-injection fix guarantees the model *uses* its assigned name; it does
not guarantee names are unique across a batch. Drawing 92 names from a 62×52 pool
gives a 72.7% chance of at least one collision, and this batch had exactly one
(`Zainab Dubois` twice). Harmless here, but a batch of 500 would collide heavily
— dedupe the draw at generation time if uniqueness matters.

## Is it trainable? — leakage and circularity probes

Run against `rrf_002` after the fact, using the same probes that condemned the
two earlier datasets. Build the browser to see them for any batch:

```bash
python scripts/build_rrf_browser.py --batch-id rrf_002
open artifacts/pairing_rrf/rrf_002/_browser.html
```

Three tabs, one self-contained file, no network requests: **Pairs** (these
probes plus every pair's retrieval provenance and judge reasoning), **Topology**
(force-directed graph, this batch beside the 200 real pairs), and **Embeddings**
(the Qwen3 vectors projected to 3D). It degrades rather than failing — `--no-real`
or a missing `data/` drops the comparison pane, and without numpy/sklearn the
embeddings tab is hidden while the first two still build.

| Probe | `rrf_002` | What killed the earlier data |
|---|---|---|
| Candidate profile alone → label | 0.634 AUC | `run_001`: **99.2% accuracy** (real data: chance) |
| TF-IDF query↔candidate cosine → label | 0.701 AUC | `pair_test_001`: **0.868 AUC** |
| Seeker identity alone → label (no text) | 0.687 AUC | — |
| Candidate identity alone → label (no text) | 0.628 AUC | — |
| RRF score → label, pooled | 0.679 AUC | — |
| **RRF score → label, within seeker** | **0.672 AUC** | — |

**The generation-artifact leak that destroyed `run_001` is gone.** A classifier
shown only the candidate's profile gets 0.634 AUC, not 99.2% accuracy — and
0.634 is roughly what a *legitimately* good model scores on the real task
(Qwen3-8B: 0.6595), i.e. it is reading matching-relevant content, not
prompt residue. Lexical circularity is also much reduced, 0.868 → 0.701.

**The real weakness is per-node base rate.** Seeker identity alone — a
leave-one-out positive rate, no text whatsoever — predicts the label at 0.687,
*higher* than any content feature. The cause is visible in the data: **12 of 40
seekers were rejected on every candidate they were shown**, and no seeker was
accepted on all of them. Train a plain pairwise classifier on this and a large
share of what it learns is "this seeker archetype gets rejected", which
transfers to nothing.

That is a training-recipe problem, not a data defect, and the fix is already
supported: **train within a seeker.** 28 of 40 seekers carry both a positive and
a negative, yielding **249 in-batch (anchor, positive, negative) triplets** —
against 5 of 91 seekers in `run_001`'s pool, which is precisely why
`two-tower-fine-tune-plan.md` chose `ContrastiveLoss` over
`MultipleNegativesRankingLoss`. That constraint no longer binds. Comparing
candidates only against each other under the same anchor cancels the per-seeker
base rate by construction, and pair-level signal does survive that
normalisation: RRF within-seeker AUC is 0.672, essentially unchanged from the
pooled 0.679.

**The embedding geometry does see the label, weakly.** Measured on the raw
4096-d Qwen3 vectors (Embeddings tab): accepted pairs sit at mean seeker↔candidate
cosine **0.603**, declined pairs at **0.550** — a separation of **+0.053** in the
right direction. So the dense channel is not blind to what the judge decided,
which is consistent with the within-seeker RRF AUC of 0.672 measured in rank
space. Read the 3D picture with care, though: PCA to three dimensions keeps only
**18.3%** of the variance, so two points looking close on screen is weak evidence
and the cosines are the real measurement.

Two caveats that no probe fixes. 275 pairs is small next to the 131 real
training pairs it would supplement, and every label is still
`gemini-3.1-flash-lite`'s opinion at 0.5942 decision accuracy on the hard slice —
so a model trained here inherits a ceiling of imitating that judge, and its
result has to be read on the **real** holdout, never on held-out synthetic pairs.

## Gotchas found the hard way

**Pin `PROFILE_GEN_PROMPT_GENERATE` to v3+.** Unpinned, it falls back to
`LANGSMITH_PROMPT_TAG=v1` and pulls the pre-fix generate prompt, which still
expects `{ref_example_1}` — removed in v2. Every profile then dies with
`KeyError: 'ref_example_1'` and the run produces zero output. `.env` is
gitignored, so this pin does not travel between checkouts.

**Any 7-8B embedder needs `--embed-backend modal`.** It will not fit on local
MPS, and an A10G (24GB) OOMs — use A100-40GB.

**Embedding cache keys are content-hashed.** `HFEmbeddingEncoder.encode()`
returns a cached array whenever `cache_name` exists without re-checking the input
texts, which already served stale embeddings once in this repo.

**Vectors are written as `.npy` before Chroma is built.** Chroma is a layer over
the files, never the only copy — a corrupt index costs a rebuild, not another GPU
run. `artifacts/` is gitignored, which is how a previous 100-profile batch was
lost entirely; the export stage copies durable outputs to a tracked path.

## What these labels are not

A model's opinion, not real accept/decline outcomes. Batches stay in
`artifacts/pairing_rrf/<batch_id>/` and **nothing is promoted** into
`data/dataset_*.json`. Promoting them would repeat the `batch_500_001` mistake in
a new form.
