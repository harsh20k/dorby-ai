"""LLM-judge variant: focused prompt with `background` added to the seeker side.

Isolated variation of baselines/llm_judge_with_pos_look_pos_back_look — same
prompt text, query, and candidate fields (positioning + background +
lookingFor); the only change is the seeker also gets `background`
(positioning + lookingFor + background instead of positioning + lookingFor).
See docs/llm-judge-seeker-background-experiment.md.
"""
