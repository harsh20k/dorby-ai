"""LLM-judge variant: focused-field prompt + searchQuery, isolated experiment.

A deliberate variation on ``baselines/llm_judge`` (the ``naive`` no-query
experiment), not an edit of it — see the "ML / data-science experiments"
isolation rule in CLAUDE.md. Two things change vs. the original:

1. The seeker's ``searchQuery`` **is** given to the model (the original
   experiment's whole point was withholding it — this variant asks a
   different question: with everything Voyage gets, does the LLM do even
   better?).
2. Only a subset of profile fields are shown, not the complete profile:
   seeker = ``positioning`` + ``lookingFor``; candidate = ``positioning`` +
   ``background`` + ``lookingFor``.

Reuses ``baselines/llm_judge/judge.py`` (verdict cache, backend call
plumbing), ``real_pairs.py`` (real-pair loading) and ``metrics.py``
(decision/calibration metrics) unmodified and read-only — those are generic
infra, not part of what this experiment varies. Only ``prompt.py`` and
``eval.py`` are new. See ``docs/llm-judge-focused-prompt-experiment.md``.
"""
