"""Thin chat wrappers via OpenRouter (DeepSeek by default).

LangSmith picks up traces when LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY are set.
"""

from __future__ import annotations

import json
import re
from typing import Any

from synth_pipeline.config import PipelineConfig, load_dotenv

load_dotenv()


def _chat_model(
    model: str,
    temperature: float,
    *,
    api_key: str,
    base_url: str,
    max_tokens: int | None = None,
):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langchain-openai is required. pip install -r requirements.txt"
        ) from exc
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY in .env "
            "(or OPENAI_API_KEY for an OpenAI-compatible proxy)."
        )
    # Unset, this defaults to the model's absolute ceiling (e.g. 65536), and
    # OpenRouter's credit check is a preflight reservation against *that*
    # figure rather than actual usage — a short JSON-object completion can
    # get rejected as unaffordable purely because the ceiling wasn't capped.
    # See baselines/llm_judge/judge.py's DEFAULT_MAX_TOKENS for the incident.
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        default_headers={
            # Optional OpenRouter attribution headers
            "HTTP-Referer": "https://github.com/harsh20k/dorby-ai",
            "X-Title": "dorby-ai-synth",
        },
    )


def complete_json(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    dry_run: bool = False,
    dry_run_payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    cfg: PipelineConfig | None = None,
    run_metadata: dict[str, Any] | None = None,
    run_tags: list[str] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    if dry_run:
        if dry_run_payload is None:
            raise ValueError("dry_run_payload required when dry_run=True")
        return dry_run_payload

    if cfg is not None:
        api_key = api_key or cfg.api_key
        base_url = base_url or cfg.base_url
    api_key = api_key or ""
    base_url = base_url or "https://openrouter.ai/api/v1"

    llm = _chat_model(model, temperature, api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    config: dict[str, Any] = {}
    if run_metadata:
        config["metadata"] = run_metadata
    if run_tags:
        config["tags"] = run_tags
    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        config=config or None,
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


def truncate_pair_for_prompt(
    pair: dict[str, Any], *, max_field: int | None = 400
) -> dict[str, Any]:
    """Truncate profile string fields for prompt-length control.

    ``max_field=None`` disables truncation entirely — use this for the seed
    pair, which the generator is instructed to keep "essentially the same":
    real seed fields (e.g. `lookingFor`) can run past 20k characters across
    many accumulated sections, and truncating at 400 chars silently hides
    most of the field from the model, producing seeker/query inconsistency
    (see docs/possible-bugs.md #1). Few-shot examples, shown repeatedly per
    prompt, should stay truncated for prompt-length/cost control.
    """

    def trunc_profile(profile: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in profile.items():
            if max_field is not None and isinstance(v, str) and len(v) > max_field:
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
    """Load prompt text by local filename or role (hub preferred when configured)."""
    from synth_pipeline.prompts import load_prompt_text

    return load_prompt_text(name)


def model_ids(cfg: PipelineConfig) -> dict[str, str]:
    return {
        "generate_model": cfg.generate_model,
        "judge_model": cfg.judge_model,
        "prompt_version": cfg.prompt_version,
        "base_url": cfg.base_url,
    }
