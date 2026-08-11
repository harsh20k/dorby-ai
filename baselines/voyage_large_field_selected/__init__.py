"""Voyage-4-large (production model) with the same field selection as the
focused LLM-judge prompt.

Isolated variation of baselines/voyage_large — same API encoder, same
metrics, different text packing: seeker = positioning + lookingFor +
searchQuery, candidate = positioning + background + lookingFor, instead of
the complete profile every other Voyage baseline uses. Built to compare
directly against baselines/llm_judge_with_pos_look_pos_back_look, which uses
the identical field selection. See docs/voyage-field-selected-experiment.md.
"""
