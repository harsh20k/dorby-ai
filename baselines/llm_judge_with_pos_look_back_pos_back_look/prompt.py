"""Prompt + field-trimmed text serialization: focused prompt + seeker `background`.

Deliberately identical to
``baselines/llm_judge_with_pos_look_pos_back_look/prompt.py`` in every way
except one: the seeker gets ``background`` added
(``positioning`` + ``lookingFor`` + ``background``, was
``positioning`` + ``lookingFor``). Candidate fields, searchQuery handling,
and the system prompt text are unchanged — a deliberate choice (see
docs/llm-judge-seeker-background-experiment.md) to isolate the effect of
this one field addition rather than re-tuning the prompt at the same time.

Pushed to LangSmith Prompt Hub as
``llm_judge_with_pos_look_back_pos_back_look`` (see
``scripts/push_llm_judge_seeker_background_prompt.py``).
"""

from __future__ import annotations

from typing import Any, Mapping

SEEKER_FIELDS = ("positioning", "lookingFor", "background")
CANDIDATE_FIELDS = ("positioning", "background", "lookingFor")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fields_to_text(profile: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        value = profile.get(field)
        if _nonempty(value):
            parts.append(f"{field}: {value.strip()}")
    return "\n".join(parts)


def seeker_to_text(user_contact_file: Mapping[str, Any]) -> str:
    """Seeker text = positioning + lookingFor + background, no searchQuery (added separately)."""
    return _fields_to_text(user_contact_file, SEEKER_FIELDS)


def candidate_to_text(match_contact_file: Mapping[str, Any]) -> str:
    """Candidate text = positioning + background + lookingFor only (unchanged from the focused variant)."""
    return _fields_to_text(match_contact_file, CANDIDATE_FIELDS)


PROMPT_NAME = "llm_judge_with_pos_look_back_pos_back_look"

# Identical text to baselines/llm_judge_with_pos_look_pos_back_look/prompt.py's
# SYSTEM_PROMPT — deliberately not updated to reference the seeker's new
# `background` field, per the isolation goal above.
SYSTEM_PROMPT = """
You are evaluating whether two people would be a good match for a
professional networking introduction.

You will be shown a seeker (Person A) and a candidate (Person B). Person A
has a specific search query describing who they are looking for. Decide
whether Person B is a good match for that search.

Judge the match on these specifics:
- Complementary need and supply: Person A's lookingFor should map to
  something concrete in Person B's positioning or background - not just
  shared topic or industry.
- Two-way fit: check whether Person B's own lookingFor is also compatible
  with what Person A can offer. A good intro should not be one-sided.
- Specificity over vibes: shared keywords or industry buzzwords alone do
  not make a match - look for a concrete reason each side would want to
  talk to the other, based on their actual stated preferences, not surface
  similarity.

Respond with a single JSON object and nothing else:

{
  "reasoning": "<2-4 sentences of your actual reasoning, written before you decide>",
  "match": "yes" | "no",
  "confidence": <integer 0-100, how sure you are of the "match" value>
}

"confidence" is confidence in the answer you gave, not the probability of
"yes": answering "no" with confidence 90 means you are 90% sure it is not a
good match. Use the full 0-100 range - say 55 when it is close to a
coin-flip and 95 only when it is clear-cut.
""".strip()

SYSTEM_PROMPTS: dict[str, str] = {"seeker_background": SYSTEM_PROMPT}


def build_user_prompt(
    user_contact_file: Mapping[str, Any],
    match_contact_file: Mapping[str, Any],
    search_query: str,
) -> str:
    seeker_text = seeker_to_text(user_contact_file)
    candidate_text = candidate_to_text(match_contact_file)
    query = (search_query or "").strip()
    return (
        "=== PERSON A (seeker) ===\n"
        f"{seeker_text}\n\n"
        f"Search query: {query}\n\n"
        "=== PERSON B (candidate) ===\n"
        f"{candidate_text}\n\n"
        "Would Person B be a good match for Person A's search? "
        "Answer with the JSON object described above."
    )
