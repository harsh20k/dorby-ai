# Standalone profile generation: local Ollama + AWS Bedrock

Companion to `docs/twotower-run-001-findings.md` / `possible-bugs.md` #1 and
#4 (label-leakage-into-generated-text). Those findings motivated splitting
synthetic data generation into two independent stages instead of one:

1. **Profile generation** (this doc) — generate fictional user profiles with
   no pos/neg label attached at all, so nothing about "why this is a
   good/bad match" can leak into the profile text itself.
2. **Pairing/labeling** (not yet built) — given a pool of independently
   generated profiles, pick candidate (seeker, query, match) triples and run
   them through the existing judge (`synth_pipeline/nodes/judge.py`) to
   assign pos/neg labels. See "Next: pairing" below for the sketched design.

This doc covers stage 1: two working generator scripts (`scripts/
local_gemma_profile_gen.py` against local/remote Ollama, `scripts/
bedrock_profile_gen.py` against AWS Bedrock), a shared browser for their
output, and the live Bedrock investigation that determined which Bedrock
model is actually usable.

## Design: 3-step generation with chain-of-thought

Both scripts implement the same design:

1. **Style spec** — one LLM call samples N real profiles
   (`data/unique_users.json`) and asks, per field, for typical length,
   sentence/bullet/tag format, tone/vocabulary, and content pattern. Output:
   one paragraph per field, no specific person's details repeated.
2. **Archetype list** — a second call, different fresh sample, asks the
   model to name 6-8 recurring "whole person" types (industry × role ×
   career stage × geography combinations) it sees across the sample.
3. **Per-profile generation** — for each profile, round-robin one archetype
   from the current list, build a prompt from the style spec + archetype +
   2 random real profiles shown *only* as tone/format reference (explicitly
   told not to copy names/companies/facts), and ask the model to first write
   a `reasoning` field (who this person is, one distinctive detail, what
   they're trying to achieve) and then the 8 profile fields consistent with
   that reasoning. The CoT `reasoning` field is discarded before use
   downstream — it's there to make step 3 think before committing to facts,
   not to be stored as training data.

Steps 1 and 2 are periodically refreshed from a fresh random sample so the
whole run doesn't lock onto one archetype list forever: **archetypes every 5
profiles, style every 15 profiles**, both with fresh resampling. This
cadence was a deliberate user choice (not the reverse, and no name-blocklist
mitigation for the mode-collapse issue below — see "Known issues").

Every step-3 output is validated (`validate_profile()`): all 9 required keys
present, each a string, no field under 15 chars (except
`meetingAndSchedulingPreferences`), and a truncation heuristic (field
doesn't end on terminal punctuation). Failures retry up to `--max-retries`
with escalating temperature.

## `scripts/local_gemma_profile_gen.py` — Ollama (local + remote)

Runs against two Ollama endpoints in parallel (`127.0.0.1:11434` local,
plus a remote Tailscale-networked box) using `gemma3:4b`. JSON compliance
comes from Ollama's `format` param (a JSON schema — triggers
grammar-constrained decoding), not prompt instructions.

```bash
python scripts/local_gemma_profile_gen.py                     # run until Ctrl+C
python scripts/local_gemma_profile_gen.py --max-profiles 20    # bounded test
```

Output: `artifacts/local_gemma_synth/run_<timestamp>/{profiles/,specs/,
manifest.jsonl}` (gitignored). Crash-safe: every profile write is atomic
(temp file + `Path.replace()`), `manifest.jsonl` is append-only, and Ctrl+C
finishes in-flight generations before stopping (second Ctrl+C force-quits).

### Known Ollama/gemma quirks fought during setup

- Prompt-instruction-only JSON formatting was ignored outright by the model
  — fixed by using Ollama's `format` (JSON schema) param for real
  grammar-constrained decoding.
- `think: false` silently breaks `format` enforcement on gemma4 models
  (upstream Ollama bug, GitHub #15260) — always pass `think: true`
  explicitly for gemma4.
- `num_ctx` isn't set by default and falls back to ~4096 regardless of the
  model's real max (131,072 for both gemma4:e4b and gemma3:4b) — always set
  it explicitly.
- Default temperature (1.0) caused mid-sentence field truncation despite
  valid JSON — fixed by lowering to 0.5 (style/archetype calls) and 0.6-0.65
  (per-profile calls).
- `gemma3:4b` intermittently (~1/3 rate observed) returns `Unterminated
  string` JSON truncation on the archetype-list (array-shaped) schema
  specifically — not a context-window issue, a genuine model/schema-shape
  interaction. Mitigated with retry logic on every schema-constrained call,
  not just profile generation.
- `localhost:11434` didn't resolve/connect on the dev machine; only
  `127.0.0.1:11434` worked — a local DNS/binding quirk, not an Ollama bug.

### Bug found and fixed: refresh failure crashed the whole worker thread

A ~6h51m unattended run (`run_20260722_233138`, 2026-07-22 23:31 →
2026-07-23 06:22) produced only **51 successful profiles** (2 failed) —
roughly 7-8/hour, far below the ~45s/profile the local endpoint was
demonstrating early in the run. Root cause: `refresh_archetypes()`'s
`call_ollama_json_retry()` exhausted its 3 retries around 23:43 and raised
`RuntimeError`, which was **uncaught inside `worker()`** — this silently
killed the local endpoint's thread entirely. From that point on, only the
slow remote endpoint kept generating (confirmable in the log: `local_avg`
froze at 43.3s for the rest of the run while `remote_avg` climbed to
150-200s+ per profile, with some outliers over 600-700s), meaning ~6 of the
6h51m ran single-threaded on the slow box.

**Fix (committed):** `worker()`'s periodic refresh calls now wrap
`refresh_style()`/`refresh_archetypes()` in `try/except`, log the failure,
keep the stale spec, and reset the refresh counter so it retries next cycle
instead of hammering immediately. `initial_setup()` (which has no stale
spec to fall back to, since nothing exists yet) now retries up to 5 times
with `_retry_until_success()` before raising — this is deliberately a hard
failure, since generation can't proceed with zero style/archetype spec.

**Expected impact:** with both endpoints staying alive, back-of-envelope
throughput is local ~84/hr + remote ~24/hr ≈ **~108/hr combined**, vs. the
~7-8/hr actually observed — roughly a 14x slowdown from the crash, not a
fundamental speed limit of the two-box setup.

### Known issues, not yet fixed

- **Name/company collapse across unrelated archetypes**: profiles `000000`
  and `000040` from the same run are both named "Elias Vance," founder of
  "SynapseFlow" — despite being generated for completely different
  archetypes (B2B data-workflow automation vs. medtech neuroimaging). This
  is a more serious duplicate-content risk than the milder "Vance"-surname
  reuse noted earlier in ad hoc testing. **User explicitly declined a
  name-blocklist mitigation** ("no name-blocklist required") — this remains
  accepted-as-is pending a different fix (e.g. a larger/refreshed reference
  sample, or explicit anti-repetition instructions naming recently-used
  names/companies in the prompt).
- **Malformed archetype label propagates downstream, undetected**: one
  archetype-list refresh returned a label as a literal multi-option string
  — `"The Strategic Investor - Tech & Fintech", "Family Office Investor",
  "Venture Capital Strategist", (choose one appropriate)` — instead of
  picking one. `STEP2_SCHEMA` only validates shape (each archetype has a
  `label` and `description` string), not content quality, so this reached
  every step-3 prompt that round-robinned onto it. **Not yet fixed** —
  suggested next step: extend `validate_profile()`-style checks to step 2's
  output (max label length, reject literal `"..."`-quote or
  `(choose one`-style artifacts) before accepting an archetype refresh.

## `scripts/build_profile_browser.py` — browsing generated profiles

The existing `scripts/build_synth_browser.py` only understands labeled
pos/neg *pairs* (`artifacts/synth/<batch>/{staged,dropped}`) — it can't read
either generator's output, which is single unlabeled profiles. This is a
separate, structurally simpler browser for that shape:

```bash
python3 scripts/build_profile_browser.py --runs-dir artifacts/local_gemma_synth --out artifacts/local_gemma_synth/_browser.html
open artifacts/local_gemma_synth/_browser.html   # self-contained, no server needed
```

Filters by run, OK/failed, and (for the local script) local/remote
endpoint; search by archetype or field text; click a card for all 8 profile
fields + the discarded `reasoning` field. Same record shape
(`archetype`/`profile`/`attempts`/`success`) works for both scripts' output,
so pointing `--runs-dir` at `artifacts/bedrock_synth` once that has data
works without changes.

## AWS Bedrock: cost comparison and structured-output findings

User has AWS credits and wanted to try running generation on Bedrock
instead of/alongside the local Ollama boxes, and to compare against a
larger model. Two questions came up: what's cheapest, and does the chosen
model actually support **native JSON-schema-constrained decoding**
(Bedrock's Feb 2026 GA "Structured Outputs" feature, `outputConfig` on the
Converse API / `output_config.format` or `response_format` on InvokeModel)
— the thing that would let us drop the Ollama-style retry-on-truncation
logic entirely.

### Cost comparison (approx., 500 profiles ≈ 2.6M tokens: ~1.5M input / ~1.1M output)

| Model | $/1M in / out | Est. cost, 500 profiles |
|---|---|---|
| Nova Micro | $0.035 / $0.14 | ~$0.21 |
| Nova Lite | $0.06 / $0.24 | ~$0.35 |
| Qwen3-32B | $0.15 / $0.62 | ~$0.91 |
| Llama 3.3 70B | $0.72 / $0.72 | ~$1.87 |
| Nova Pro | $0.80 / $3.20 | ~$4.72 |
| Qwen3-235B-A22B | $0.53 / $2.66 | ~$3.72 |
| Claude Haiku 4.5 | $1.00 / $5.00 | ~$7.00 |
| Nova Premier | $2.50 / $10.00 | ~$14.75 |

At this scale every option is cheap in absolute terms (a few dollars at
most) — the real differentiator turned out to be structured-output support,
not price.

### Structured-output support: Llama and Nova are the outliers, not open-weight models in general

AWS's own docs (`docs.aws.amazon.com/bedrock/.../structured-output.html`)
say structured outputs are supported for "Claude 4.5+ models... and select
open-weight models such as Meta Llama and Amazon Nova." **Live testing
against a real Bedrock account (`us-east-1`, `AWS_PROFILE=tf_provisioner`)
contradicts this specifically for Llama and Nova**, but confirms it works
cleanly on most *other* open-weight model families:

| Model | `outputConfig.textFormat` (Converse) |
|---|---|
| `us.meta.llama3-3-70b-instruct-v1:0` | ❌ `ValidationException: doesn't support the outputConfig field` |
| `us.amazon.nova-micro-v1:0` / `-lite` / `-pro` / `-premier` (all) | ❌ same error, every variant |
| `mistral.voxtral-mini-3b-2507` | ✅ clean JSON, first try |
| `mistral.ministral-3-3b-instruct` | ✅ clean JSON, first try |
| `nvidia.nemotron-nano-9b-v2` | ✅ clean JSON, first try |
| `zai.glm-4.7-flash` | ✅ clean JSON, first try |
| `deepseek.v3.2` | ✅ clean JSON, first try |
| `google.gemma-3-27b-it` | ✅ clean JSON, first try |
| `openai.gpt-oss-120b-1:0` | ✅ works — reasoning model, emits a `reasoningContent` block before the schema-compliant `text` block; read the last content block, not the first |
| `minimax.minimax-m2.5` | ⚠️ works in principle (schema-compliant plan visible in reasoning), but is a heavy reasoning model that can exhaust a small `maxTokens` budget before emitting the final JSON — needs a larger token budget than the others |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | ✅ clean JSON, first try |

Additional findings for Llama specifically (root cause, not just the
observed error):
- Llama's native `InvokeModel` request schema
  (`docs.aws.amazon.com/.../model-parameters-meta.html`) only accepts
  `prompt`/`temperature`/`top_p`/`max_gen_len` — there is no
  `response_format` field in Llama's actual API surface on Bedrock,
  regardless of what the generic open-weight example in the structured-
  output doc page implies.
- Non-strict, non-forced `toolConfig` (auto tool choice) doesn't error, but
  the model doesn't emit a real `toolUse` content block either — it just
  prints a JSON-*looking* string inside a plain text block, unparsed and
  unguaranteed. The commonly-cited "tool-calling workaround" for Llama on
  Bedrock is **not actually reliable** — it degrades to the same class of
  fragility as prompt-based JSON on gemma3:4b.

**Conclusion:** avoid Llama and Nova model families for anything needing
guaranteed JSON on Bedrock. Every other open-weight family tested (Mistral,
NVIDIA, Zhipu, DeepSeek, Google, OpenAI) and Claude 4.5+ genuinely support
it. **Chosen: `google.gemma-3-27b-it`** — same model family as the local
`gemma3:4b` runs (just bigger, 27B vs 4B), confirmed structured-output
support, ~$1.30 for 500 profiles, and smoke-tested clean (3/3 profiles,
~25s each, no retries, 8-128K context / 8K max output — double Llama's 4K
cap).

## `scripts/bedrock_profile_gen.py` — current state

Same 3-step design and crash-resilience fix as
`local_gemma_profile_gen.py`, ported to Bedrock's Converse API. No
local/remote endpoint split needed — Bedrock is one managed API, so
parallelism is `--concurrency` worker threads instead of two named
endpoints.

```bash
export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1
python scripts/bedrock_profile_gen.py --max-profiles 20   # bounded test
```

**Status: defaults to `google.gemma-3-27b-it`, confirmed working.**
Smoke-tested 2026-07-23 (`run_20260723_093500`, `--max-profiles 3
--concurrency 2`): 3/3 succeeded, no retries, ~13s style refresh, ~10s
archetype refresh, ~22-29s per profile. Content sanity-checked — coherent,
on-archetype, no truncation. Ready for a real batch run.

## Cost tracking

Bedrock's Converse API returns real `inputTokens`/`outputTokens` usage with
every response. `bedrock_profile_gen.py` records this per call in
`manifest.jsonl` — both profile-generation calls (`"kind": "profile"`) *and*
style/archetype refresh calls (`"kind": "style_refresh"` /
`"archetypes_refresh"`, added 2026-07-23; refresh calls dump several full
reference profiles into the prompt and were previously untracked, silently
undercounting true cost).

```bash
python scripts/estimate_bedrock_cost.py artifacts/bedrock_synth/run_<timestamp> --model-id google.gemma-3-27b-it
# or override pricing directly:
python scripts/estimate_bedrock_cost.py artifacts/bedrock_synth/run_<timestamp> --price-in 0.62 --price-out 1.85
```

For live/ongoing tracking rather than a post-hoc manifest read, two AWS
resources were also set up directly against the `tf_provisioner` account
(`us-east-1`, account `411960113601`) — not via Terraform, since this repo
has no existing IaC for AWS infra:

- **CloudWatch dashboard** `dorby-bedrock-profile-gen` — token usage,
  invocation count/errors/throttles, an estimated-$/hour metric-math
  widget (Gemma 3 27B pricing), and invocation latency, all filtered to
  `ModelId=google.gemma-3-27b-it`. Built from Bedrock's built-in `AWS/
  Bedrock` CloudWatch metrics namespace — no invocation logging or extra
  setup needed, these are emitted automatically by every Bedrock call.
- **AWS Budget** `dorby-bedrock-profile-gen` — $10/month, filtered to
  `Service: Amazon Bedrock`, email alerts at 50% and 100% of threshold to
  the account owner. Budget $ figures come from Cost Explorer (~24h billing
  lag, but the authoritative actual-spend number) rather than the
  CloudWatch token-count estimate.

If the target model changes from Gemma 3 27B, both the dashboard's
`ModelId` dimension and its cost-estimate metric-math pricing constants
need updating to match.

## Pairing: `synth_pipeline/pairing/` (built)

Turns a pool of independent, unlabeled profiles into labeled pos/neg pairs.

```bash
python -m synth_pipeline.pairing \
  --profile-run artifacts/bedrock_synth/run_<ts> \
  --batch-id pair_test_001 \
  --data-dir /path/to/data          # data/ is gitignored; required from a worktree
```

Five phases, run to completion in order (selection needs the whole TF-IDF
matrix, so it can't stream per-pair):

| module | does | LLM? |
|---|---|---|
| `profiles.py` | load a run, mint `cmsynthp…` ids, drop `reasoning` | no |
| `query.py` | one `searchQuery` set per profile from `lookingFor` | **yes** (Bedrock) |
| `select.py` | rank candidates against each query, take a top band | no |
| `label.py` | score with the TF-IDF+Voyage-nano fusion, label with a deadband | no |
| `stage.py` | write batch-isolated envelopes | no |

### Why no judge

The original sketch (below, superseded) routed each triple through
`synth_pipeline/nodes/judge.py`. Two things changed that:

1. **`judge_node` puts the label in the judge's own prompt.**
   `judge.py:56-63` sends `{"label": ..., "failure_mode": ..., "pair": ...}`.
   Under the old design that was fine — the label came from elsewhere and the
   judge only had veto power, so a judge error could only ever *discard* a
   correct pair, never *create* a wrong one. Under profile-first there is no
   other label source, so reusing it verbatim would have the judge grading its
   own answer key.
2. **Deliberate scope call:** for this batch, labels come from the hybrid
   scorer instead. Cheaper, deterministic, and it sidesteps the question of
   how much to trust an unvalidated judge.

### Labeling by hybrid scorer

`baselines/hybrid_tfidf_voyage/fusion.py` is reused directly — it is the
strongest pair scorer measured on the matched 69-pair holdout:

| scorer | pair AUC | hard-neg AUC |
|---|---|---|
| **hybrid TF-IDF+nano** | **0.6397** | **0.6034** |
| voyage-4-large (prod) | 0.6086 | 0.6017 |
| tfidf alone | 0.5922 | 0.5017 |
| voyage-4-nano alone | 0.5793 | 0.5707 |

Fitted on **real train pairs only** (`build_split_bundle(include_synth=False)`),
so the frozen 69-pair holdout never touches the fit.

Labels use a **deadband**, not a single threshold: `pos` above, `neg` below,
and everything in between is written to `excluded/` unlabeled. Near-boundary
pairs are close to coin flips, and labeling them confidently would manufacture
noise exactly at the decision boundary that matters.

### Finding: real-pair thresholds do not transfer to synthetic pairs

The first `pair_test_001` run labeled **164 pos / 0 neg** — every single pair
positive. Not a crash; a genuine distribution mismatch, and the main thing the
20-profile test batch bought us.

| population | fusion score range |
|---|---|
| real fit-set threshold region | ≈ **−2.18** |
| synthetic batch (n=164) | **0.57 … 9.86** (median 3.02) |

The two distributions don't overlap **at all** — the lowest-scoring synthetic
pair sits 2.5 points above the threshold that would call a real pair positive.
Two compounding causes:

1. **Homogeneity.** Synthetic profiles come from one model, one style spec, and
   ~8 archetypes, so any two of them are far more alike than any two real
   Boardy contacts. Both the TF-IDF and Voyage cosines run high across the board.
2. **Selection bias.** `select.py` picks the top-similarity band by
   construction, so the pairs being scored are the most similar ones available.

The fusion score is a z-blend using the *fit set's* mean and standard deviation,
so a systematically shifted input distribution shifts every score with it. An
absolute threshold learned on real pairs is therefore meaningless on synthetic
pairs, and the failure is silent — it produces a confident, uniform, wrong answer.

**Fix:** `--label-mode quantile` (now the default) splits each batch by its own
score distribution — top `--pos-frac` (0.30) positive, bottom `--neg-frac` (0.30)
negative, middle 40% excluded. `--label-mode absolute` keeps the old behavior for
comparison against real data, and now logs a warning when the batch scores fall
entirely outside the real-pair range.

This changes what a label *means*, and the change is worth being explicit
about: not "good by the standard real pairs set" but "among the better/worse
matches offered to this seeker in this batch". Since every candidate was already
drawn from the top-similarity band, the resulting negatives are hard by
construction — which is the intent, but it also means the batch has no easy
negatives at all (see "Still open").

### Two related hazards found and fixed

- **Random contact ids broke re-runs.** IDs were minted with `secrets`, so the
  same profile got a new identity on every run, invalidating the query
  checkpoint and making two runs over one profile pool incomparable. Now derived
  deterministically from `sha256(source_run:profile_id)` — still valid 25-char
  `cmsynthp…` ids, since hex is a subset of the id alphabet.
- **Silent stale-embedding reuse.** `TfidfEncoder.encode()` and
  `VoyageNanoEncoder.encode()` return a cached array whenever the `cache_name`
  file exists, *without* checking that the input texts still match. A fixed
  per-batch cache key would have served embeddings for the previous run's
  queries after a regeneration. The batch cache key is now content-hashed.

**Negatives carry no `failure_mode`.** A scorer produces a number, not a
diagnosis of *which* axis failed. It's left null rather than guessed, so
per-mode analysis downstream can't be misled.

### Labels are provisional — not training data

A model trained on this batch can at best learn to imitate the scorer that
labeled it, and could not then be said to beat that scorer on data it labeled.
On hard pairs specifically, the scorer is right about 60% of the time.

Three things keep that quarantined: batches live in their own
`artifacts/pairing/<batch_id>/` namespace, **nothing is promoted** into
`data/dataset_*.json`, and `manifest.json` records the labeler and thresholds
so provenance is never ambiguous. Promoting a batch like this would repeat the
`batch_500_001` mistake in a new form.

### Batch layout

```
artifacts/pairing/<batch_id>/
  manifest.json     # config, counts, fusion params, thresholds, token usage
  profiles.json     # contactId -> {profile_id, archetype, profile}
  queries.json      # contactId -> [query, ...]
  pairs/*.json      # {label}_{seekerId}_{candidateId}.json
  excluded/*.json   # deadband rejects, with drop_reason
```

Envelopes keep the exact key set `synth_pipeline/nodes/writer.py` writes (a
test asserts this via AST, so it can't drift), which keeps the review browser
and `promote.py` available later without rework — they're just not in this
path. Filenames include the seeker id because one candidate profile now appears
in many pairs, and the old `{label}_{matchContactId}.json` scheme would collide.

### `pair_test_001` results (20 profiles, 2026-07-23)

20 Bedrock profiles (`run_20260723_212205`, 20/20 clean, $0.128) → 40 queries
($0.017) → 174 unique candidate pairs → **52 pos / 52 neg / 70 excluded**.

Scorer verified exact: `scripts/verify_pairing_scorer.py` reproduces the
documented holdout pair AUC of **0.6397** to four decimals (delta −0.0000), so
the labeler is demonstrably the scorer that was measured.

| diagnostic | value | real-data comparison |
|---|---|---|
| seekers carrying both labels | 16/20 (80%) | 9/129 (7%) |
| candidates carrying both labels | 16/20 (80%) | 9/178 (5%) |
| edges per node | 5.2 | 0.67 |
| query↔candidate token jaccard (median) | 0.022 | queries aren't substrings ✓ |
| **AUC of TF-IDF query cosine → label** | **0.868** | — |

The last row is the significant one. **Most of the label is recoverable from
plain lexical overlap** — TF-IDF query-candidate cosine alone separates the
assigned labels at 0.868 AUC, and it correlates 0.70 with the fusion score
that did the labeling. That is close to circular: `select.py` ranks candidates
by TF-IDF, then a labeler with a heavy TF-IDF component grades that ranking.

This matters because `docs/possible-bugs.md` #3 already records that plain
TF-IDF (pair AUC 0.592) beats both fine-tuned runs. A model trained on these
labels would largely be learning to reproduce lexical overlap — the one thing
already known not to need learning. It is the strongest argument for putting a
semantic judge back in the labeling path before any of this becomes training
data.

### Still open

- **Label ≈ lexical overlap** (0.868 AUC, above). Decoupling selection from
  labeling — a non-lexical selector, or a judge as labeler — is the main
  unresolved design question.
- `select.py` samples only the high-cosine band, so the negative set is
  uniformly hard. The real holdout contains both easy and hard negatives, and
  `baselines/metrics.py::slice_metrics` splits on exactly that — a 100%-hard
  synthetic set is a distribution mismatch of a different kind.
- **Topology is nothing like real data**: 80% of synthetic contacts carry both
  labels vs 5% in real data, and the batch is 5.2 edges/node vs 0.67. A
  fully-connected component also can't be carved into user-disjoint
  train/train-dev splits the way `twotower/data.py` requires — sharding the
  profile pool would be needed before this scales.
- 18 distinct archetypes across 20 profiles (archetypes refresh every 5
  profiles, so labels accumulate). Archetype coloring in the graph is therefore
  nearly one color per node, and same-archetype pairs are rare (4 of 104).
