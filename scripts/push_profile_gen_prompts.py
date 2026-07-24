"""Push local scripts/prompts/profile_gen/*.md files to LangSmith Prompt Hub.

Usage::

    # Requires LANGCHAIN_API_KEY (or LANGSMITH_API_KEY)
    python scripts/push_profile_gen_prompts.py --dry-run
    python scripts/push_profile_gen_prompts.py --tag v1
    python scripts/push_profile_gen_prompts.py --tag v1 --role generate_profile

Creates/updates private hub prompts (ChatPromptTemplate with a system message)
named ``profile-gen-style-refresh``, ``profile-gen-archetype-refresh``,
``profile-gen-generate`` (under LANGSMITH_PROMPT_OWNER if set).

After push, scripts/bedrock_profile_gen.py pulls these hub-only (see
scripts/profile_gen_prompt_hub.py) — set LANGSMITH_PROMPT_OWNER / LANGSMITH_PROMPT_TAG
in .env, or the explicit PROFILE_GEN_PROMPT_* env vars printed below.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from scripts.profile_gen_prompt_hub import (
    DEFAULT_HUB_REPOS,
    LOCAL_FILES,
    LOCAL_PROMPTS_DIR,
    PromptRole,
    clear_prompt_cache,
)

# Named format vars each prompt uses — kept as live {var} placeholders when escaping
# the rest of the template's literal braces for ChatPromptTemplate's f-string parser.
_FORMAT_VARS: dict[PromptRole, tuple[str, ...]] = {
    "style_refresh": ("sample_count", "sample_json", "fields_list"),
    "archetype_refresh": ("sample_count", "sample_json"),
    "generate_profile": (
        "style_guide_text",
        "archetype_label",
        "archetype_description",
    ),
}


def _roles(selected: str | None) -> list[PromptRole]:
    all_roles: list[PromptRole] = ["style_refresh", "archetype_refresh", "generate_profile"]
    if not selected:
        return all_roles
    role = selected.strip()
    if role not in all_roles:
        raise SystemExit(f"Unknown --role {role!r}; choose from {all_roles}")
    return [role]  # type: ignore[list-item]


def _fstring_escape(text: str, keep: tuple[str, ...]) -> str:
    placeholders: dict[str, str] = {}
    for i, name in enumerate(keep):
        token = f"__KEEP_VAR_{i}__"
        placeholders[token] = "{" + name + "}"
        text = text.replace("{" + name + "}", token)
    text = text.replace("{", "{{").replace("}", "}}")
    for token, original in placeholders.items():
        text = text.replace(token, original)
    return text


def push_one(role: PromptRole, *, owner: str | None, tag: str | None,
             description: str | None, dry_run: bool) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from langsmith import Client

    local_name = LOCAL_FILES[role]
    text = (LOCAL_PROMPTS_DIR / local_name).read_text(encoding="utf-8")
    template = _fstring_escape(text, keep=_FORMAT_VARS[role])
    repo = DEFAULT_HUB_REPOS[role]
    identifier = f"{owner}/{repo}" if owner else repo

    prompt = ChatPromptTemplate.from_messages([("system", template)])
    print(f"{'[dry-run] ' if dry_run else ''}push {local_name} -> {identifier}")

    if dry_run:
        return f"(dry-run) {identifier}"

    client = Client()
    kwargs: dict = {
        "object": prompt,
        "description": description
        or f"dorby-ai standalone profile generation: {role} (from {local_name})",
    }
    if tag:
        kwargs["commit_tags"] = [tag]

    url = client.push_prompt(identifier, **kwargs)
    print(f"  -> {url}")
    return url


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Push local profile-gen prompts to LangSmith Hub.")
    p.add_argument("--role", default=None, help="style_refresh | archetype_refresh | generate_profile (default: all)")
    p.add_argument("--owner", default=None, help="Hub owner/handle (default: LANGSMITH_PROMPT_OWNER)")
    p.add_argument("--tag", default=None, help="Optional commit tag (e.g. v1)")
    p.add_argument("--description", default=None, help="Prompt description override")
    p.add_argument("--dry-run", action="store_true", help="Print what would be pushed; no API calls")
    args = p.parse_args(argv)

    api_key = (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()
    if not args.dry_run and not api_key:
        print("Missing LANGCHAIN_API_KEY / LANGSMITH_API_KEY. Set it in .env or pass --dry-run.", file=sys.stderr)
        return 2

    owner = (args.owner or os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip() or None
    urls: list[str] = []
    for role in _roles(args.role):
        urls.append(push_one(role, owner=owner, tag=args.tag, description=args.description, dry_run=args.dry_run))

    clear_prompt_cache()
    print(f"Done ({len(urls)} prompt(s)).")
    if not args.dry_run and owner:
        tag = args.tag or "latest"
        print(
            "\nSet these in .env to pull from hub (or rely on LANGSMITH_PROMPT_OWNER/LANGSMITH_PROMPT_TAG):\n"
            f"  LANGSMITH_PROMPT_OWNER={owner}\n"
            f"  LANGSMITH_PROMPT_TAG={tag}\n"
            "  # or explicit:\n"
            f"  PROFILE_GEN_PROMPT_STYLE_REFRESH={owner}/profile-gen-style-refresh:{tag}\n"
            f"  PROFILE_GEN_PROMPT_ARCHETYPE_REFRESH={owner}/profile-gen-archetype-refresh:{tag}\n"
            f"  PROFILE_GEN_PROMPT_GENERATE={owner}/profile-gen-generate:{tag}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
