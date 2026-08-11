"""Query-weighted seeker encoding, extended to voyage-4-large.

`query_weighted/` ran the full alpha-blend sweep (concat / profile-only /
query-only / alpha_0.1..0.9) on frozen voyage-4-nano only. This package runs
the exact same sweep, on the exact same 200 real pairs, for
**voyage-4-large** (Boardy's actual production model) — so its query-only and
alpha-blended numbers can sit beside nano's and the two-tower fine-tunes'.

No file under `query_weighted/` is modified. `query_weighted.eval.run_all_arms`
is imported and called unchanged — it's already encoder-agnostic (only calls
`encoder.encode(texts, role=..., batch_size=..., show_progress=...)`), so this
package's only job is `encoder.py`'s thin adapter reconciling
`VoyageLargeEncoder.encode(..., input_type=...)`'s parameter name with the
`role=` keyword `run_all_arms` calls with.

Cost note: `VoyageLargeEncoder`'s disk cache is content-hash keyed
(`baselines/voyage_large/encode.py::text_cache_key`), and `profile_to_text`
output is byte-identical between `query_weighted.text.profile_only` and
`baselines.text_no_query.py::seeker_to_text` (both delegate to the same
function). Pointing this package's encoder at the existing
`artifacts/voyage_large_no_query` cache therefore makes every profile-only
seeker embedding and every candidate embedding a cache hit for free; only the
200 query-only seeker texts (~11.6k tokens) are new API calls.
"""
