"""Generic open-weight HF embedding baseline, with the focused LLM judge's
field selection (positioning + lookingFor + query for the seeker,
positioning + background + lookingFor for the candidate) instead of the
complete profile.

Isolated variation of baselines/hf_embedding — same encoder machinery
(MODEL_REGISTRY, get_encoder_class), only the text packing changes. Built
first for Qwen/Qwen3-Embedding-8B, to compare directly against
baselines/llm_judge_with_pos_look_pos_back_look and
baselines/voyage_large_field_selected / voyage_nano_field_selected, which
use the identical field selection. See
docs/qwen3-embedding-field-selected-experiment.md.
"""
