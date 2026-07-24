"""Synthesize a realistic `searchQuery` for each standalone profile.

This is the only LLM call in the pairing stage. It sees exactly one profile and
no candidate, so nothing it writes can correlate with a label — no label exists
yet at this point in the pipeline. Do not add candidate or counterpart context
here: that would reintroduce, one layer up, the leak that possible-bugs #4 was about.

Templating the query straight out of `lookingFor` was the alternative and is worse:
it makes the query a literal substring of the seeker profile, so seeker-side lexical
overlap goes to ~1.0 — an artifact the real 200 pairs never show (real queries
paraphrase, mean length 209 chars).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from synth_pipeline.pairing.bedrock import call_json
from synth_pipeline.pairing.profiles import SynthProfile

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

PROMPT = """You are writing search queries for Boardy, a networking product where a \
user describes the kind of person they want to be introduced to.

Below is one user's profile. Write {n} DIFFERENT search queries this user might \
submit, each targeting a different intent visible in their "lookingFor" section.

Style reference — real Boardy queries look like this:
{style_examples}

Rules:
- Write in the user's own voice, first person or as a direct request.
- PARAPHRASE. Do not copy sentences verbatim out of the profile.
- Be specific about role, seniority, industry, stage, and geography where the \
profile implies them — that specificity is what makes a query matchable.
- Each query should stand alone and target a genuinely different need.
- Aim for roughly 150-250 characters per query.

USER PROFILE:
{profile}

Return JSON: {{"queries": [...]}} with exactly {n} strings."""


def load_style_examples(
    data_dir: Path,
    *,
    split_path: Path | None = None,
    k: int = 3,
    rng: random.Random | None = None,
) -> list[str]:
    """Sample real searchQueries as style anchors, train users only."""
    rng = rng or random.Random(42)
    users = json.loads((Path(data_dir) / "unique_users.json").read_text(encoding="utf-8"))

    train_ids: set[str] | None = None
    split_path = split_path or (Path(data_dir) / "synthetic" / "seed_split.json")
    if Path(split_path).exists():
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        train_ids = set(split.get("train_user_ids") or [])

    pool: list[str] = []
    for user in users:
        if train_ids is not None and user.get("userContactId") not in train_ids:
            continue
        for query in user.get("searchQueries") or []:
            if isinstance(query, str) and query.strip():
                pool.append(query.strip())

    if not pool:
        return []
    return rng.sample(pool, min(k, len(pool)))


def _profile_block(profile: dict[str, Any]) -> str:
    parts = []
    for key, value in profile.items():
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()}")
    return "\n\n".join(parts)


def generate_queries(
    client,
    profile: SynthProfile,
    *,
    model_id: str,
    style_examples: list[str],
    n: int = 2,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> tuple[list[str], dict[str, int]]:
    style_block = "\n".join(f"- {q}" for q in style_examples) or "- (none available)"
    prompt = PROMPT.format(
        n=n,
        style_examples=style_block,
        profile=_profile_block(profile.profile),
    )
    parsed, usage = call_json(
        client,
        model_id=model_id,
        prompt=prompt,
        schema=QUERY_SCHEMA,
        schema_name="search_queries",
        schema_description="Search queries this user might submit",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    queries = [q.strip() for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()]
    return queries[:n], usage
