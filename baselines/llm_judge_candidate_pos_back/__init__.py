"""LLM-judge variant: candidate trimmed to positioning + background, isolated experiment.

A deliberate variation on ``baselines/llm_judge_with_pos_look_pos_back_look``
(the focused-field experiment, candidate = positioning + background +
lookingFor), not an edit of it — see the "ML / data-science experiments"
isolation rule in CLAUDE.md. Seeker fields (``positioning`` + ``lookingFor``),
the searchQuery, and the system prompt text are all unchanged; only the
candidate field set changes, dropping ``lookingFor`` to isolate the
candidate-side field permutation's effect.

Reuses ``baselines/llm_judge/judge.py`` (verdict cache, backend call
plumbing), ``real_pairs.py`` (real-pair loading) and ``metrics.py``
(decision/calibration metrics) unmodified and read-only — those are generic
infra, not part of what this experiment varies. Only ``prompt.py`` and
``eval.py`` are new.
"""
