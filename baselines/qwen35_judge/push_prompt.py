"""Push the Qwen3.5-4B judge prompt (naive + searchQuery) to LangSmith Hub.

Usage::

    python -m baselines.qwen35_judge.push_prompt --dry-run
    python -m baselines.qwen35_judge.push_prompt --tag v1
"""

from __future__ import annotations

import argparse
import os
import sys

from baselines.qwen35_judge.prompt import PROMPTS_DIR

DEFAULT_HUB_REPO = "qwen35-judge-naive-query"
LOCAL_FILE = "naive_query.md"


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def push(*, owner: str | None, tag: str | None, description: str | None, dry_run: bool) -> str:
    text = (PROMPTS_DIR / LOCAL_FILE).read_text(encoding="utf-8")
    template = _escape_braces(text)
    identifier = f"{owner}/{DEFAULT_HUB_REPO}" if owner else DEFAULT_HUB_REPO

    print(f"{'[dry-run] ' if dry_run else ''}push {LOCAL_FILE} -> {identifier}"
          f"{f' (tag {tag})' if tag else ''}")
    print(f"  {len(text)} chars, {len(text.splitlines())} lines")
    if dry_run:
        return f"(dry-run) {identifier}"

    from langchain_core.prompts import ChatPromptTemplate
    from langsmith import Client

    prompt = ChatPromptTemplate.from_messages([("system", template)])
    kwargs: dict = {
        "object": prompt,
        "description": description or f"dorby-ai qwen35_judge naive+searchQuery (from {LOCAL_FILE})",
    }
    if tag:
        kwargs["commit_tags"] = [tag]
    url = Client().push_prompt(identifier, **kwargs)
    print(f"  -> {url}")
    return url


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--owner", default=None, help="hub handle (default: LANGSMITH_PROMPT_OWNER)")
    p.add_argument("--tag", default=None, help="commit tag, e.g. v1")
    p.add_argument("--description", default=None)
    p.add_argument("--dry-run", action="store_true", help="validate only, no API calls")
    args = p.parse_args(argv)

    api_key = (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()
    if not args.dry_run and not api_key:
        print("Missing LANGCHAIN_API_KEY / LANGSMITH_API_KEY.", file=sys.stderr)
        return 2

    owner = (args.owner or os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip() or None
    push(owner=owner, tag=args.tag, description=args.description, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
