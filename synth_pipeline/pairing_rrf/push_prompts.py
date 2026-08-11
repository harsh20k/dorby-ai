"""Push local ``prompts/*.md`` to LangSmith Hub as the RRF-pairing prompts.

The pipeline itself is hub-only (see ``prompt_hub.py``), so this is the one
place local prompt text enters the system. Run it before any paid batch::

    python -m synth_pipeline.pairing_rrf.push_prompts --dry-run
    python -m synth_pipeline.pairing_rrf.push_prompts --tag v1
    python -m synth_pipeline.pairing_rrf.push_prompts --role pair_judge --tag v2

``pair_query`` keeps ``{profile}``, ``{section}`` and ``{style_examples}`` as
live template variables; everything else is brace-escaped so the JSON contract
in the prompt body survives the ChatPromptTemplate f-string round trip.
"""

from __future__ import annotations

import argparse
import os
import sys

from synth_pipeline.config import load_dotenv
from synth_pipeline.pairing_rrf.prompt_hub import (
    ALL_ROLES,
    DEFAULT_HUB_REPOS,
    LOCAL_FILES,
    PROMPTS_DIR,
    PromptRole,
    clear_prompt_cache,
)

load_dotenv()

# Template variables each role must keep unescaped after the f-string pass.
_KEEP_VARS: dict[PromptRole, tuple[str, ...]] = {
    "pair_query": ("profile", "section", "style_examples"),
    "pair_judge": (),
}


def _fstring_escape(text: str, keep: tuple[str, ...] = ()) -> str:
    placeholders: dict[str, str] = {}
    for i, name in enumerate(keep):
        token = f"__KEEP_VAR_{i}__"
        placeholders[token] = "{" + name + "}"
        text = text.replace("{" + name + "}", token)
    text = text.replace("{", "{{").replace("}", "}}")
    for token, original in placeholders.items():
        text = text.replace(token, original)
    return text


def _roles(selected: str | None) -> list[PromptRole]:
    if not selected:
        return list(ALL_ROLES)
    role = selected.strip()
    if role not in ALL_ROLES:
        raise SystemExit(f"Unknown --role {role!r}; choose from {list(ALL_ROLES)}")
    return [role]  # type: ignore[list-item]


def push_one(
    role: PromptRole,
    *,
    owner: str | None,
    tag: str | None,
    description: str | None,
    dry_run: bool,
) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from langsmith import Client

    local_name = LOCAL_FILES[role]
    text = (PROMPTS_DIR / local_name).read_text(encoding="utf-8")
    template = _fstring_escape(text, keep=_KEEP_VARS[role])
    repo = DEFAULT_HUB_REPOS[role]
    identifier = f"{owner}/{repo}" if owner else repo

    print(f"{'[dry-run] ' if dry_run else ''}push {local_name} → {identifier}"
          f"{f' (tag {tag})' if tag else ''}")
    if dry_run:
        missing = [v for v in _KEEP_VARS[role] if "{" + v + "}" not in template]
        if missing:
            print(f"  !! template vars missing from body: {missing}")
        print(f"  {len(text)} chars, {len(text.splitlines())} lines")
        return f"(dry-run) {identifier}"

    prompt = ChatPromptTemplate.from_messages([("system", template)])
    kwargs: dict = {
        "object": prompt,
        "description": description
        or f"dorby-ai pairing_rrf {role} (from {local_name})",
    }
    if tag:
        kwargs["commit_tags"] = [tag]
    url = Client().push_prompt(identifier, **kwargs)
    print(f"  → {url}")
    return url


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--role", default=None, help=f"one of {list(ALL_ROLES)} (default: all)")
    p.add_argument("--owner", default=None, help="hub handle (default: LANGSMITH_PROMPT_OWNER)")
    p.add_argument("--tag", default=None, help="commit tag, e.g. v1")
    p.add_argument("--description", default=None)
    p.add_argument("--dry-run", action="store_true", help="validate only, no API calls")
    args = p.parse_args(argv)

    api_key = (
        os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or ""
    ).strip()
    if not args.dry_run and not api_key:
        print("Missing LANGCHAIN_API_KEY / LANGSMITH_API_KEY.", file=sys.stderr)
        return 2

    owner = (args.owner or os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip() or None
    urls = [
        push_one(
            role,
            owner=owner,
            tag=args.tag,
            description=args.description,
            dry_run=args.dry_run,
        )
        for role in _roles(args.role)
    ]
    clear_prompt_cache()
    print(f"Done ({len(urls)} prompt(s)).")
    if not args.dry_run and owner:
        tag = args.tag or "latest"
        print(
            "\nPin these in .env:\n"
            f"  PAIR_RRF_PROMPT_QUERY={owner}/{DEFAULT_HUB_REPOS['pair_query']}:{tag}\n"
            f"  PAIR_RRF_PROMPT_JUDGE={owner}/{DEFAULT_HUB_REPOS['pair_judge']}:{tag}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
