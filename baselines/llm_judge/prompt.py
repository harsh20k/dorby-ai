"""Prompts for the LLM-judge experiment.

Both profiles are serialized with ``profile_to_text`` from
``baselines/bert_frozen/text.py`` — the same field-tagged packing every other
baseline uses — so the LLM sees exactly the text the embedding models see.
The one deliberate difference: **``searchQuery`` is never included**, on
either side. That is the point of the experiment.

Two framings are available:

``naive`` (default)
    The literal question asked: "would these two be a good match?" Nothing
    about production, nothing about base rates. This is what a reasonable
    person would build first, and the number it produces is the honest answer
    to "can an LLM just eyeball two profiles?"

``calibrated``
    Tells the model the truth about the label (see docs/objective.md): both
    people were *already* recommended to each other by a production matcher,
    so topical relevance is a given and the real question is the residual
    "did the humans actually connect", at a 50/50 base rate. Exists because
    ``naive`` is scored against a label it was never told about — if naive
    calls almost everything a match, that is a framing artifact, not a
    capability ceiling, and this variant separates the two.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from baselines.bert_frozen.text import profile_to_text

Variant = Literal["naive", "calibrated"]

_RESPONSE_CONTRACT = """
Respond with a single JSON object and nothing else:

{
  "reasoning": "<2-4 sentences of your actual reasoning, written before you decide>",
  "match": "yes" | "no",
  "confidence": <integer 0-100, how sure you are of the "match" value>
}

"confidence" is confidence in the answer you gave, not the probability of
"yes": answering "no" with confidence 90 means you are 90% sure it is not a
good match. Use the full 0-100 range — say 55 when it is close to a
coin-flip and 95 only when it is clear-cut.
""".strip()

_NAIVE_SYSTEM = f"""
You are evaluating whether two people would be a good match for a
professional networking introduction.

You will be shown two complete profiles, Person A and Person B. Decide
whether introducing them would be a good match — whether both sides would
find real value in the conversation.

Think about it from both directions: A good intro usually needs something
each person wants that the other can supply. A one-sided intro, where one
person clearly benefits and the other gets nothing, is not a good match.

{_RESPONSE_CONTRACT}
""".strip()

_CALIBRATED_SYSTEM = f"""
You are predicting the outcome of a professional networking introduction
that has already been recommended.

You will be shown two complete profiles, Person A (who was looking) and
Person B (who was recommended to them). Important context about how these
pairs were produced:

- An automated matching system already judged this pair topically relevant
  and recommended the introduction. Every pair you see cleared that bar, so
  "they work in related areas" is true of essentially all of them and does
  not distinguish the good from the bad.
- What you are predicting is the *human outcome*: did the two people
  actually go ahead and connect (accept), or did one of them decline and it
  never moved forward (decline)?
- Exactly half of the pairs you will see were accepted and half were
  declined. Do not assume most are good.

So look past surface topical fit for the things that actually stop a real
intro: a mismatch in seniority or stage, one side having nothing the other
wants, timing or geography problems, or stated preferences that conflict.

Answer "yes" if you predict the intro was accepted, "no" if you predict it
was declined.

{_RESPONSE_CONTRACT}
""".strip()

SYSTEM_PROMPTS: dict[str, str] = {
    "naive": _NAIVE_SYSTEM,
    "calibrated": _CALIBRATED_SYSTEM,
}


def _truncate(text: str, max_field: int | None) -> str:
    if max_field is None or len(text) <= max_field:
        return text
    return text[:max_field] + "…"


def build_user_prompt(
    user_contact_file: Mapping[str, Any],
    match_contact_file: Mapping[str, Any],
    *,
    max_field: int | None = None,
) -> str:
    """Render both profiles. ``max_field`` truncates each field; None = complete."""
    a = _truncate(profile_to_text(user_contact_file), max_field)
    b = _truncate(profile_to_text(match_contact_file), max_field)
    return (
        "=== PERSON A ===\n"
        f"{a}\n\n"
        "=== PERSON B ===\n"
        f"{b}\n\n"
        "Would introducing Person A and Person B be a good match? "
        "Answer with the JSON object described above."
    )
