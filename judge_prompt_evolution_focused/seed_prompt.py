"""The exact judge prompt this experiment starts from, frozen here.

This is the **focused** judge prompt (searchQuery given, profile fields
trimmed), not the naive one that ``judge_prompt_evolution/seed_prompt.py``
holds. Byte-identical to
``baselines/llm_judge_with_pos_look_pos_back_look/prompt.py``'s
``SYSTEM_PROMPT`` — the prompt that scored pair AUC **0.6451** on all 200 real
pairs and **0.6530** on the matched 69-pair holdout via the direct Google API
(``docs/llm-judge-focused-prompt-experiment.md``).

Duplicated rather than imported per the experiment-isolation rule in
CLAUDE.md: this package mutates its own copy of the prompt every iteration,
so importing the original would either freeze mid-mutation or drift the
"official" focused prompt out from under the already-published numbers. The
frozen copy lives in ``focused_prompt.py`` (with source-hash provenance) and
is re-exported here so the rest of this package keeps the same import shape
as the original package.
"""

from __future__ import annotations

from judge_prompt_evolution_focused.focused_prompt import SYSTEM_PROMPT

# The output-format paragraph, split back out of the focused prompt so
# ``optimizer.repair_contract`` can re-append it verbatim whenever an
# optimizer or summarizer round edits it away (bug #6/#7 in
# docs/judge-prompt-evolution-experiment.md). Sliced from SYSTEM_PROMPT
# rather than retyped so the two cannot drift.
_CONTRACT_MARKER = "Respond with a single JSON object and nothing else:"
_marker_at = SYSTEM_PROMPT.index(_CONTRACT_MARKER)
RESPONSE_CONTRACT = SYSTEM_PROMPT[_marker_at:].strip()

SEED_JUDGE_PROMPT = SYSTEM_PROMPT
