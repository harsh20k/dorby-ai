"""The exact judge prompt this experiment is trying to beat, frozen here.

Byte-identical to ``baselines/llm_judge/prompt.py``'s ``_NAIVE_SYSTEM`` +
``_RESPONSE_CONTRACT`` — the prompt that scored pair AUC 0.6177 on all 200
real pairs, 0.6358 on the matched 69-pair holdout (docs/llm-judge-experiment.md).
Duplicated rather than imported per the experiment-isolation rule in
CLAUDE.md: this package mutates its own copy of the prompt every iteration,
so importing the original would either freeze mid-mutation or drift the
"official" naive prompt out from under the already-published numbers.
"""

from __future__ import annotations

RESPONSE_CONTRACT = """
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

SEED_JUDGE_PROMPT = f"""
You are evaluating whether two people would be a good match for a
professional networking introduction.

You will be shown two complete profiles, Person A and Person B. Decide
whether introducing them would be a good match — whether both sides would
find real value in the conversation.

Think about it from both directions: A good intro usually needs something
each person wants that the other can supply. A one-sided intro, where one
person clearly benefits and the other gets nothing, is not a good match.

{RESPONSE_CONTRACT}
""".strip()
