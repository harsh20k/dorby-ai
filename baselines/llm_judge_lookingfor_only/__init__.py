"""LLM-judge variant: seeker trimmed to lookingFor only, isolated experiment.

A deliberate variation on ``baselines/llm_judge_with_pos_look_pos_back_look``
(the focused-field experiment, seeker = positioning + lookingFor), not an
edit of it — see the "ML / data-science experiments" isolation rule in
CLAUDE.md. Exactly one thing changes: the seeker's field set drops to
``lookingFor`` alone, to isolate that field's individual contribution to the
focused prompt's result. Candidate fields (``positioning`` + ``background``
+ ``lookingFor``), the searchQuery, and the system prompt text are all
unchanged.

Reuses ``baselines/llm_judge/judge.py`` (verdict cache, backend call
plumbing), ``real_pairs.py`` (real-pair loading) and ``metrics.py``
(decision/calibration metrics) unmodified and read-only — those are generic
infra, not part of what this experiment varies. Only ``prompt.py`` and
``eval.py`` are new.
"""
