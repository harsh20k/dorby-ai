"""Prompt for the Qwen3.5-4B judge experiment.

The system prompt is ``naive`` from ``baselines/llm_judge/prompt.py`` plus one
deliberate addition: ``searchQuery`` is included. That is the one variable
being tested here versus the existing ``llm_judge`` naive/no-query result —
everything else (framing, response contract, profile serialization) is held
identical so the two numbers are comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from baselines.bert_frozen.text import profile_to_text

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "naive_query.md").read_text(encoding="utf-8").strip()


def build_user_prompt(
    search_query: str,
    user_contact_file: Mapping[str, Any],
    match_contact_file: Mapping[str, Any],
) -> str:
    a = profile_to_text(user_contact_file)
    b = profile_to_text(match_contact_file)
    return (
        "=== PERSON A'S SEARCH QUERY ===\n"
        f"{search_query}\n\n"
        "=== PERSON A ===\n"
        f"{a}\n\n"
        "=== PERSON B ===\n"
        f"{b}\n\n"
        "Would introducing Person A and Person B be a good match? "
        "Answer with the JSON object described above."
    )
