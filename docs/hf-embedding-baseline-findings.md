# Open-weight HF embedding baselines: findings

> **⚠️ CORRECTED 2026-07-31 — the headline below does not survive a larger
> sample.** Every number on this page is computed on the 69-pair holdout (29
> positive queries). Re-scored on all 200 real pairs, **Qwen3-Embedding-8B does
> not beat Voyage-4-large**: pair AUC 0.5529 vs 0.5726, MRR 0.2045 vs 0.3102,
> R@1 0.0500 vs 0.1300, and a hard-negative AUC of 0.4680 that is below chance.
> The re-run used this same `baselines/hf_embedding` code path and reproduces
> this page's holdout numbers (0.6543 vs 0.6595, bf16 noise), so the difference
> is the population, not the protocol.
>
> The holdout is now known to carry **no ranking information among the top
> models** — Spearman −0.029 across the top 6, against +0.976 across the bottom
> 8. See [`all-200-baseline-sweep.md`](all-200-baseline-sweep.md) and
> [`baseline-results-real200.md`](baseline-results-real200.md). The material
> below is left unedited as the record of what the 69-pair evidence showed.

Plain-language summary of running free/open-source embedding models
against the frozen 69-pair real holdout, via a new generalizable Modal
harness. For full metric tables see
[`baseline-results-holdout.md`](baseline-results-holdout.md); for the
harness itself see `baselines/hf_embedding/`.

## Verdict

**Qwen3-Embedding-8B (Apache 2.0, free) is the new best model on the core
accept/decline task — the first one to beat Boardy's own production
model, Voyage-4-large, on pair ROC-AUC.** It doesn't win everywhere:
Voyage-4-large still leads on precise top-of-list retrieval (MRR), and
BGE-en-ICL turned out to be a surprisingly strong retrieval model in its
own right.

| Model | Pair ROC-AUC | Retrieval MRR | License | Notes |
|---|---|---|---|---|
| **Qwen3-Embedding-8B** | **0.6595 (best)** | 0.4040 | Apache 2.0 | Needs A100-40GB on Modal |
| Hybrid TF-IDF+nano | 0.6397 | 0.4043 | — | Previous leader |
| Voyage-4-large (prod) | 0.6086 | **0.5287 (best)** | Proprietary API | Bar this project benchmarks against |
| BGE-en-ICL | 0.5750 | 0.5157 (2nd-best) | MIT | Strong retrieval, middling classification |
| TF-IDF (lexical) | 0.5922 | 0.2475 | — | No training at all |
| Voyage-4-nano | 0.5793 | 0.4610 | Custom (not fully open) | Already in this repo |
| E5-mistral-7b-instruct | 0.5664 | 0.2244 | MIT | Underperformed expectations |
| twotower arm_a_real_only | 0.5793 | 0.3882 | — | Our fine-tune |
| twotower run_001 | 0.5784 | 0.2829 | — | Our fine-tune, overfit (see `possible-bugs.md` #4) |
| Frozen BERT | 0.4595 | 0.1371 | — | Worse than chance |
| NV-Embed-v2 | 0.5034 (near chance) | 0.1092 | CC-BY-NC-4.0 (non-commercial) | Caveat below — likely understated |
| zembed-1-embedding | 0.5052 (near chance) | 0.0643 (worst) | Apache 2.0 | Genuine result, no plumbing issue found — see below |

## What we built

`baselines/hf_embedding/` — a generic open-weight embedding baseline
parameterized by HF model id, so testing a new model is a `--model` swap
instead of a new package (mirrors `baselines/voyage_nano/`'s
encode.py/eval.py split, generalized to the standard sentence-transformers
`prompt_name` API). `models.py` holds a small per-model registry for the
quirks that differ across open-weight families (query/document prompt
name, `trust_remote_code`, Matryoshka truncation, and — as it turned out —
which loading library and which `transformers` version a model needs).
`modal_eval.py` runs models too large for local MPS on Modal, reusing the
existing `dorby-twotower-hf-cache` HF cache volume.

## Compatibility surprises found while running 5 models

Two of the five models tested didn't fit the "just load it with
sentence-transformers" assumption the harness started with — both are now
handled, and the registry/harness generalizes to the next odd model that
shows up the same way:

1. **NV-Embed-v2's custom code doesn't work with current `transformers`.**
   Its `trust_remote_code` modeling file calls a KV-cache method
   (`Cache.get_usable_length()`) that was removed from `transformers` in a
   recent internal refactor (confirmed `AttributeError` on 4.57.x). Fixed
   with a second Modal image pinned to an older `transformers==4.44.2` +
   `sentence-transformers==3.0.1` (`legacy_transformers_image` /
   `eval_remote_legacy_transformers`, selected via
   `ModelSpec.requires_legacy_transformers`).

   It also turned out to ship **no sentence-transformers prompt config at
   all** (empty `prompts` dict) — NVIDIA's intended usage prepends an
   `instruction=` string manually via a custom `.encode()` method, not
   ST's `prompt_name`. We run it symmetrically (`query_prompt_name=None`)
   as an approximation, which **understates its true performance** — its
   0.5034 (near chance) should be read as "we didn't use this model
   correctly," not "this model is bad." It's also CC-BY-NC-4.0
   (non-commercial-only), so it stays a benchmark reference regardless of
   how it scores.

2. **BGE-en-ICL isn't sentence-transformers-loadable at all.** Failed with
   `ValueError: Unrecognized processing class` — this model is designed
   for BAAI's own `FlagEmbedding` library (`FlagICLModel`), which also
   supports prepending in-context few-shot examples to the query (we ran
   it zero-shot, no examples, matching every other baseline here). Added
   `FlagICLEncoder` in `encode.py` behind the identical
   `encode(texts, *, role, batch_size, cache_name)` interface used by the
   sentence-transformers path, a `flagembedding_image` for Modal, and
   `ModelSpec.loader="flagembedding_icl"` to route to it. Its result is a
   real, fair number — and the standout: **second-best retrieval MRR of
   any model measured**, just behind Voyage-4-large.

3. **A10G (24GB) OOMs on any 7-8B model.** Weights alone in bf16 take
   ~14-16GB, leaving no headroom for activations. All four large models
   here ran on A100-40GB instead (documented in
   `docs/modal-training-guide.md`).

4. **A fourth calling convention: `encode_query()`/`.encode_document()`
   methods instead of `.encode(texts, prompt_name=...)`.**
   `zeroentropy/zembed-1-embedding` (4B, Apache 2.0, Qwen3-4B-based) uses
   the same method-based convention as `voyage-4-nano`'s existing custom
   encoder rather than sentence-transformers' generic `prompt_name` API.
   Added `ModelSpec.uses_encode_methods` to `HFEmbeddingEncoder` to
   dispatch to `model.encode_query`/`model.encode_document` when set,
   rather than writing a whole separate encoder class (unlike BGE-en-ICL,
   this only needed a one-line branch — still a sentence-transformers
   model underneath, just a different call surface). Ran cleanly on the
   default A10G (4B fits comfortably, no A100 needed).

## Reading the results

- **Qwen3-Embedding-8B winning the classification task but not the
  retrieval task** mirrors this project's existing pair-vs-retrieval
  split (see `docs/possible-bugs.md` #3): different models are good at
  different jobs here, and "best on accept/decline" isn't the same
  question as "best at finding the right candidate."
- **E5-mistral-7b-instruct underperforming** despite being a similar size
  and lineage to Qwen3-Embedding-8B is a reminder that raw parameter count
  isn't predictive here — training recipe and data seem to matter more
  than scale within the 7-8B class.
- **NV-Embed-v2's number is not a fair comparison** as run — treat it as
  "compatibility integrated, verdict pending a correct instruction-format
  implementation," not a real result to rank against the others.
- **BGE-en-ICL's strong retrieval, weaker classification** is a genuinely
  new data point (not an artifact of how we ran it) worth remembering
  alongside Voyage-4-large as a retrieval-strength option.
- **zembed-1-embedding scored near-chance on both tasks (0.5052 AUC, worst
  retrieval MRR of anything tested)** despite being a purpose-built,
  purpose-trained retrieval model on a similar Qwen3 base to the
  best-performing Qwen3-Embedding-8B. No plumbing issue was found — the
  run completed cleanly with sane (non-degenerate) score distributions, so
  this reads as a genuine mismatch between zembed-1's training domain and
  this project's networking-intro-matching task, not a harness bug. Worth
  a second look only if there's reason to think our `encode_query`/
  `encode_document` usage differs from its intended calling convention.

## What's left

- Registry already has entries for Qwen3-Embedding-4B/0.6B, GTE-Qwen2-7B,
  BGE-m3, Snowflake Arctic Embed-m, and mxbai-embed-large-v1 — none run
  yet. The 4B/0.6B Qwen3 variants are the highest-value next check: if a
  smaller size gets close to the 8B's 0.6595, that's the cheaper model to
  actually use (no A100 needed).
- NV-Embed-v2's instruction-prepending format could be implemented
  properly (build the `Instruct: ...\nQuery: ...` string NVIDIA's model
  card specifies and pass it through a custom encode path, similar to how
  `FlagICLEncoder` was added for BGE-en-ICL) if its true performance
  becomes worth knowing — currently deprioritized given the non-commercial
  license caps its practical value to this project either way.
- `gemini-embedding-001` (Google's proprietary API) was scoped but not
  built — it needs its own API-based package (like `baselines/voyage_large/`)
  plus a Google API key, neither of which exist yet in this repo.
