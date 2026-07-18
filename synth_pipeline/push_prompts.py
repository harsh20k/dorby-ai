"""Push local ``prompts/*.md`` files to LangSmith Prompt Hub.

Usage::

    # Requires LANGCHAIN_API_KEY (or LANGSMITH_API_KEY)
    # Optional: LANGSMITH_PROMPT_OWNER=your-handle
    python -m synth_pipeline.push_prompts
    python -m synth_pipeline.push_prompts --dry-run
    python -m synth_pipeline.push_prompts --tag v1 --role judge

Creates/updates private hub prompts (ChatPromptTemplate with a system message)
named ``synth-generate-pos``, ``synth-generate-neg``, ``synth-judge``
(optionally under ``LANGSMITH_PROMPT_OWNER/``).

After push, set env vars, e.g.::

    SYNTH_PROMPT_GENERATE_POS=your-handle/synth-generate-pos:latest
    SYNTH_PROMPT_GENERATE_NEG=your-handle/synth-generate-neg:latest
    SYNTH_PROMPT_JUDGE=your-handle/synth-judge:latest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from synth_pipeline.config import load_dotenv
from synth_pipeline import prompts as prompt_mod
from synth_pipeline.prompts import PROMPTS_DIR, PromptRole, clear_prompt_cache

_DEFAULT_HUB_REPOS = prompt_mod._DEFAULT_HUB_REPOS
_LOCAL_FILES = prompt_mod._LOCAL_FILES

load_dotenv()

logger = logging.getLogger(__name__)


def _roles(selected: str | None) -> list[PromptRole]:
    all_roles: list[PromptRole] = ["generate_pos", "generate_neg", "judge"]
    if not selected:
        return all_roles
    role = selected.strip()
    if role not in all_roles:
        raise SystemExit(f"Unknown --role {role!r}; choose from {all_roles}")
    return [role]  # type: ignore[list-item]


def _fstring_escape(text: str, keep: tuple[str, ...] = ()) -> str:
    """Escape ``{``/``}`` for ChatPromptTemplate f-strings; keep named vars."""
    placeholders: dict[str, str] = {}
    for i, name in enumerate(keep):
        token = f"__KEEP_VAR_{i}__"
        placeholders[token] = "{" + name + "}"
        text = text.replace("{" + name + "}", token)
    text = text.replace("{", "{{").replace("}", "}}")
    for token, original in placeholders.items():
        text = text.replace(token, original)
    return text


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

    local_name = _LOCAL_FILES[role]
    text = (PROMPTS_DIR / local_name).read_text(encoding="utf-8")
    keep = ("failure_mode",) if role == "generate_neg" else ()
    template = _fstring_escape(text, keep=keep)
    repo = _DEFAULT_HUB_REPOS[role]
    identifier = f"{owner}/{repo}" if owner else repo

    prompt = ChatPromptTemplate.from_messages([("system", template)])
    print(f"{'[dry-run] ' if dry_run else ''}push {local_name} → {identifier}")

    if dry_run:
        return f"(dry-run) {identifier}"

    client = Client()
    kwargs: dict = {
        "object": prompt,
        "description": description
        or f"dorby-ai synth_pipeline {role} (from {local_name})",
    }
    if tag:
        kwargs["commit_tags"] = [tag]

    url = client.push_prompt(identifier, **kwargs)
    print(f"  → {url}")
    return url


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Push local synth prompts to LangSmith Hub.")
    p.add_argument(
        "--role",
        default=None,
        help="generate_pos | generate_neg | judge (default: all)",
    )
    p.add_argument(
        "--owner",
        default=None,
        help="Hub owner/handle (default: LANGSMITH_PROMPT_OWNER)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Optional commit tag (e.g. v1, production)",
    )
    p.add_argument(
        "--description",
        default=None,
        help="Prompt description override",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be pushed; no API calls",
    )
    args = p.parse_args(argv)

    api_key = (
        os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or ""
    ).strip()
    if not args.dry_run and not api_key:
        print(
            "Missing LANGCHAIN_API_KEY / LANGSMITH_API_KEY. "
            "Set it in .env or pass --dry-run.",
            file=sys.stderr,
        )
        return 2

    owner = (args.owner or os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip() or None
    urls: list[str] = []
    for role in _roles(args.role):
        urls.append(
            push_one(
                role,
                owner=owner,
                tag=args.tag,
                description=args.description,
                dry_run=args.dry_run,
            )
        )

    clear_prompt_cache()
    print(f"Done ({len(urls)} prompt(s)).")
    if not args.dry_run and owner:
        print(
            "\nSet these in .env to pull from hub:\n"
            f"  LANGSMITH_PROMPT_OWNER={owner}\n"
            "  # or explicit:\n"
            f"  SYNTH_PROMPT_GENERATE_POS={owner}/synth-generate-pos:latest\n"
            f"  SYNTH_PROMPT_GENERATE_NEG={owner}/synth-generate-neg:latest\n"
            f"  SYNTH_PROMPT_JUDGE={owner}/synth-judge:latest\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
