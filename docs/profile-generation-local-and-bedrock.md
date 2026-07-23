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

## Next: pairing (not yet built)

Once profile generation is solid, turning a pool of independent profiles
into labeled pos/neg pairs needs three new pieces, reusing existing
judge infrastructure rather than rebuilding it:

1. **Query generation** — one LLM call per profile turns its `lookingFor`
   into a `searchQuery` (what they'd type when searching for an intro).
2. **Candidate picking** — for each (profile, query), select a few other
   profiles as candidates: some random (easy negatives), some
   topically-similar via TF-IDF/embedding cosine (hard negatives/positives
   — similarity alone doesn't imply good/bad, the judge decides that).
3. **Judge** — run each (seeker, query, candidate) triple through the
   existing `synth_pipeline/nodes/judge.py` (v2 schema,
   `would_be_good_intro`) to assign a positive or negative-with-
   `failure_mode` label, same as the current pipeline.

The candidate-picking step (2) is the only genuinely new design surface —
everything else reuses code that already exists in `synth_pipeline/`.
