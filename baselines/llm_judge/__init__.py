"""LLM-judge experiment: ask an LLM directly, from two full profiles alone,
whether an intro would be a good match — no ``searchQuery``, no embeddings.

Deliberately *not* shaped like the other baselines. There is no vector space
here, so there are no retrieval metrics (ranking a 200-candidate corpus per
query would need ~40k LLM calls); this package reports pair metrics plus the
negative-hardness slice only. See docs/llm-judge-experiment.md.
"""
