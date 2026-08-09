#!/usr/bin/env python3
"""Push the llm_judge_with_pos_look_back_pos_back_look prompt to LangSmith Prompt Hub.

Usage:
    python scripts/push_llm_judge_seeker_background_prompt.py --env-file .env
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def load_env_file(env_file: Path | None) -> None:
    if env_file is None or not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", type=Path, default=None)
    args = p.parse_args()
    load_env_file(args.env_file)

    from langchain_core.prompts import ChatPromptTemplate
    from langsmith import Client

    from baselines.llm_judge_with_pos_look_back_pos_back_look.prompt import (
        PROMPT_NAME,
        SYSTEM_PROMPT,
    )

    template = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")
    owner = os.getenv("LANGSMITH_PROMPT_OWNER")
    identifier = f"{owner}/{PROMPT_NAME}" if owner else PROMPT_NAME

    prompt = ChatPromptTemplate.from_messages([("system", template)])
    client = Client()
    url = client.push_prompt(
        identifier,
        object=prompt,
        description=(
            "LLM-judge variant: same prompt/query/candidate fields as "
            "llm_judge_with_pos_look_pos_back_look, seeker also gets `background`. "
            "See docs/llm-judge-seeker-background-experiment.md"
        ),
    )
    print(f"pushed: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
