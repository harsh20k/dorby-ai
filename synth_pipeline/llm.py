"""Thin chat wrappers; LangSmith picks up traces when env is configured."""

from __future__ import annotations

import json
import re
from typing import Any

from synth_pipeline.config import PipelineConfig


def _chat_model(model: str, temperature: float):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langchain-openai is required. pip install -r requirements.txt"
        ) from exc
    return ChatOpenAI(model=model, temperature=temperature)


def complete_json(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    dry_run: bool = False,
    dry_run_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dry_run:
        if dry_run_payload is None:
            raise ValueError("dry_run_payload required when dry_run=True")
        return dry_run_payload

    llm = _chat_model(model, temperature)
    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return parse_json_object(content)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def truncate_pair_for_prompt(pair: dict[str, Any], *, max_field: int = 400) -> dict[str, Any]:
    def trunc_profile(profile: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in profile.items():
            if isinstance(v, str) and len(v) > max_field:
                out[k] = v[:max_field] + "…"
            else:
                out[k] = v
        return out

    return {
        "userContactId": pair["userContactId"],
        "matchContactId": pair["matchContactId"],
        "searchQuery": pair["searchQuery"],
        "userContactFile": trunc_profile(pair["userContactFile"]),
        "matchContactFile": trunc_profile(pair["matchContactFile"]),
    }


def load_prompt(name: str) -> str:
    from pathlib import Path

    path = Path(__file__).resolve().parent / "prompts" / name
    return path.read_text(encoding="utf-8")


def model_ids(cfg: PipelineConfig) -> dict[str, str]:
    return {
        "generate_model": cfg.generate_model,
        "judge_model": cfg.judge_model,
        "prompt_version": cfg.prompt_version,
    }
