"""Voyage-4-large + gemini-3.1-flash-lite pairing variant, over the same 9,659-
profile pool as ``rrf_qwen_full_001``.

Isolated per this repo's experiment-isolation rule (copy-then-edit, never
modify a previous experiment's files) — a full copy of
``synth_pipeline.pairing_rrf_qwen_judge``, edited for three changes:

* **Embedding model**: Voyage-4-large (API) instead of Qwen3-Embedding-8B
  (Modal GPU).
* **Embedding fields**: no per-section whole-profile vectors. Seeker text is
  ``positioning`` + that section's ``searchQuery`` only; candidate text is
  ``positioning`` + ``background`` + ``lookingFor`` only — the best-R@10
  configuration from the query-field sweep experiment, not the full-profile
  text every prior batch used.
* **Recall**: dense-only, one channel, top-10 per query — no BM25, no
  weighted-RRF fusion. A query's top-10 dense hits are its shortlist directly.
* **Judge**: gemini-3.1-flash-lite via the direct Google API
  (``GEMINI_API_KEY``), not OpenRouter and not Bedrock, using the "focused"
  field-trimmed prompt vendored from
  ``judge_prompt_evolution_focused/focused_prompt.py`` (measured pair AUC
  0.6451 on the all-200 real split — the best judge configuration measured in
  this project).

Query generation is skipped entirely — this batch reads ``queries.json``
read-only from ``artifacts/pairing_rrf_qwen_judge/rrf_qwen_full_001/``, the
same 9,659-profile pool, rather than spending on Bedrock again.

Labels are a model's opinion, not real accept/decline outcomes. Nothing here
is promoted into ``data/dataset_*.json``.
"""
