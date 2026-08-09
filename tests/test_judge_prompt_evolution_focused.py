"""Pins for the focused judge-prompt-evolution experiment.

Two things this guards, per the experiment-isolation rule in CLAUDE.md:

1. ``judge_prompt_evolution_focused/focused_prompt.py`` is a deliberate
   verbatim copy of the focused LLM judge's prompt module. If that source is
   reachable (it lives on the ``llm-judge-experiment`` branch, checked out as
   a worktree), assert the copy still matches it byte for byte.
2. The example text shown to the optimizer contains exactly the focused field
   set plus the ``searchQuery``, and nothing else — showing the optimizer a
   field the judge never sees is the specific defect this package exists to
   avoid.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from judge_prompt_evolution_focused.focused_prompt import (
    CANDIDATE_FIELDS,
    SEEKER_FIELDS,
    SYSTEM_PROMPT,
)
from judge_prompt_evolution_focused.sampling import Example
from judge_prompt_evolution_focused.seed_prompt import (
    RESPONSE_CONTRACT,
    SEED_JUDGE_PROMPT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED = REPO_ROOT / "judge_prompt_evolution_focused" / "focused_prompt.py"
SOURCE = (
    REPO_ROOT
    / ".claude/worktrees/llm-judge-experiment"
    / "baselines/llm_judge_with_pos_look_pos_back_look/prompt.py"
)
SOURCE_SHA256 = "2c80b24cbf463d997c61ca2b98a50042489df68bf6467d2bee1cc6b054e7292d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not SOURCE.exists(), reason="llm-judge-experiment worktree not present")
def test_source_prompt_has_not_drifted() -> None:
    """The upstream focused prompt still hashes to what we vendored from."""
    assert _sha256(SOURCE) == SOURCE_SHA256


@pytest.mark.skipif(not SOURCE.exists(), reason="llm-judge-experiment worktree not present")
def test_vendored_copy_matches_source_semantically() -> None:
    """The copy adds only a provenance header; every other line is identical."""
    source_lines = SOURCE.read_text(encoding="utf-8").splitlines()
    vendored_lines = VENDORED.read_text(encoding="utf-8").splitlines()
    # The provenance block is inserted inside the module docstring; the code
    # below it must be untouched, so compare from the first import onward.
    start_src = source_lines.index("from __future__ import annotations")
    start_ven = vendored_lines.index("from __future__ import annotations")
    assert source_lines[start_src:] == vendored_lines[start_ven:]


def test_seed_is_the_focused_prompt() -> None:
    assert SEED_JUDGE_PROMPT == SYSTEM_PROMPT
    assert "Search query" not in SEED_JUDGE_PROMPT  # that's the user prompt's job
    assert "search query" in SEED_JUDGE_PROMPT


def test_response_contract_is_a_slice_of_the_seed() -> None:
    """repair_contract re-appends this verbatim, so it must be exact."""
    assert RESPONSE_CONTRACT in SEED_JUDGE_PROMPT
    assert '"match": "yes" | "no"' in RESPONSE_CONTRACT
    assert '"reasoning"' in RESPONSE_CONTRACT
    assert '"confidence"' in RESPONSE_CONTRACT


def _example() -> Example:
    pair = {
        "userContactId": "cmseeker",
        "matchContactId": "cmcandidate",
        "searchQuery": "climate-tech seed investors in Berlin",
        "userContactFile": {
            "positioning": "SEEKER_POSITIONING",
            "background": "SEEKER_BACKGROUND",
            "lookingFor": "SEEKER_LOOKINGFOR",
            "notes": "SEEKER_NOTES",
            "introPreferences": "SEEKER_INTROPREFS",
            "personalPreferences": "SEEKER_PERSONALPREFS",
            "locationAvailability": "SEEKER_LOCATION",
            "meetingAndSchedulingPreferences": "SEEKER_MEETING",
        },
        "matchContactFile": {
            "positioning": "CAND_POSITIONING",
            "background": "CAND_BACKGROUND",
            "lookingFor": "CAND_LOOKINGFOR",
            "notes": "CAND_NOTES",
            "introPreferences": "CAND_INTROPREFS",
            "personalPreferences": "CAND_PERSONALPREFS",
            "locationAvailability": "CAND_LOCATION",
            "meetingAndSchedulingPreferences": "CAND_MEETING",
        },
    }
    return Example(label="accepted", hardness="hard", pair=pair, overlap=0.5)


def test_rendered_example_shows_only_focused_fields_plus_query() -> None:
    rendered = _example().render(1)

    assert SEEKER_FIELDS == ("positioning", "lookingFor")
    assert CANDIDATE_FIELDS == ("positioning", "background", "lookingFor")

    for present in (
        "SEEKER_POSITIONING",
        "SEEKER_LOOKINGFOR",
        "CAND_POSITIONING",
        "CAND_BACKGROUND",
        "CAND_LOOKINGFOR",
        "climate-tech seed investors in Berlin",
    ):
        assert present in rendered, present

    # Seeker background is excluded on purpose — adding it was measured and
    # made results worse (docs/llm-judge-seeker-background-experiment.md).
    for absent in (
        "SEEKER_BACKGROUND",
        "SEEKER_NOTES",
        "SEEKER_INTROPREFS",
        "SEEKER_PERSONALPREFS",
        "SEEKER_LOCATION",
        "SEEKER_MEETING",
        "CAND_NOTES",
        "CAND_INTROPREFS",
        "CAND_PERSONALPREFS",
        "CAND_LOCATION",
        "CAND_MEETING",
    ):
        assert absent not in rendered, absent


def test_rendered_example_discloses_outcome_but_never_hardness() -> None:
    rendered = _example().render(1)
    assert "ACCEPTED" in rendered
    assert "this intro was accepted" in rendered
    assert "hard" not in rendered.lower().replace("hardness", "")
