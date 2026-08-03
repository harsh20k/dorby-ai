"""One optimization round: current judge prompt + a fresh example batch ->
a revised judge prompt. Calls OpenRouter directly (via ``langchain_openai``,
same client ``synth_pipeline.llm`` uses, so LangSmith tracing still works via
LANGCHAIN_TRACING_V2/LANGCHAIN_API_KEY in .env) rather than through
``synth_pipeline.llm.complete_json``, because that helper's JSON parsing is
``json.loads`` in strict mode, which rejects a literal unescaped newline
inside a string value — and the optimizer's own output (a multi-paragraph
judge prompt, wrapped in a JSON string) is exactly the shape where a model
is likely to emit one. Confirmed empirically: two separate runs both failed
with the identical parse error ("Unterminated string starting at: line 2
column 21") on all 3 retries at the model's most complex response — the
error location is structural (right after `"updated_prompt": "` opens), not
content-dependent, meaning strict-mode rejection, not transient flakiness.
``json.loads(text, strict=False)`` is the standard fix (permits control
characters in strings) and is applied locally here rather than in
``synth_pipeline/llm.py``, which is shared infra used elsewhere in the repo.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from synth_pipeline.config import DEFAULT_OPENROUTER_BASE_URL, load_dotenv

from judge_prompt_evolution.config import RunConfig
from judge_prompt_evolution.sampling import Example

load_dotenv()


def _parse_json_lenient(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1], strict=False)
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def _call_optimizer(
    *, system: str, user: str, model: str, temperature: float, max_tokens: int,
    api_key: str, base_url: str, run_metadata: dict[str, Any], run_tags: list[str],
) -> dict[str, Any]:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, base_url=base_url,
        max_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "https://github.com/harsh20k/dorby-ai",
            "X-Title": "dorby-ai-judge-prompt-evolution",
        },
    )
    msg = llm.invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config={"metadata": run_metadata, "tags": run_tags},
    )
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return _parse_json_lenient(content)


DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _call_optimizer_gemini(
    *, system: str, user: str, model: str, temperature: float, max_tokens: int, api_key: str,
) -> dict[str, Any]:
    """Direct Google Gemini API call (raw REST via urllib, no new SDK
    dependency), mirroring ``eval_evolved.py::_make_gemini_call_fn`` — added
    so the optimizer model can be gemini-3.1-flash-lite directly, both to
    dodge OpenRouter's per-call credit-reservation issue (docs/possible-bugs.md
    context: it reserves against max_tokens, not actual usage) and to keep the
    optimizer and the AUC-check reference model consistent."""
    import urllib.error
    import urllib.request

    url = f"{DEFAULT_GEMINI_BASE_URL}/models/{model}:generateContent?key={api_key}"
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"no candidates in Gemini response: {payload!r}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    return _parse_json_lenient(text)

META_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "meta_optimizer.md"
SUMMARIZER_PROMPT_PATHS = {
    "aggressive": Path(__file__).resolve().parent / "prompts" / "summarizer.md",
    "gentle": Path(__file__).resolve().parent / "prompts" / "summarizer_gentle.md",
}

REQUIRED_JUDGE_CONTRACT_MARKERS = ("reasoning", "match", "confidence")


def load_meta_system_prompt(cfg: RunConfig | None = None) -> str:
    """Pull from LangSmith Hub first (source of truth for what a run actually
    used), fall back to the local file only if the Hub is unreachable."""
    if cfg is not None and cfg.push_to_hub:
        from judge_prompt_evolution.hub import pull_prompt

        text = pull_prompt(repo=f"{cfg.hub_repo}-meta", hub_owner=cfg.hub_owner, tag="v2")
        if text:
            return text.strip()
    return META_PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_summarizer_system_prompt(cfg: RunConfig | None = None) -> str:
    variant = cfg.summarizer_variant if cfg is not None else "aggressive"
    local_path = SUMMARIZER_PROMPT_PATHS[variant]
    if cfg is not None and cfg.push_to_hub:
        from judge_prompt_evolution.hub import pull_prompt

        suffix = "-summarizer" if variant == "aggressive" else f"-summarizer-{variant}"
        text = pull_prompt(repo=f"{cfg.hub_repo}{suffix}", hub_owner=cfg.hub_owner)
        if text:
            return text.strip()
    return local_path.read_text(encoding="utf-8").strip()


def _call_with_retries(
    *, system: str, user: str, cfg: RunConfig, label: str, run_tags: list[str], run_metadata_extra: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if cfg.optimizer_backend == "gemini":
                return _call_optimizer_gemini(
                    system=system,
                    user=user,
                    model=cfg.optimizer_model,
                    temperature=cfg.optimizer_temperature,
                    max_tokens=cfg.optimizer_max_tokens,
                    api_key=os.getenv("GEMINI_API_KEY", ""),
                )
            return _call_optimizer(
                system=system,
                user=user,
                model=cfg.optimizer_model,
                temperature=cfg.optimizer_temperature,
                max_tokens=cfg.optimizer_max_tokens,
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=DEFAULT_OPENROUTER_BASE_URL,
                run_metadata={
                    "experiment": "judge_prompt_evolution",
                    "run_id": cfg.run_id,
                    "attempt": attempt,
                    **run_metadata_extra,
                },
                run_tags=run_tags,
            )
        except Exception as exc:  # noqa: BLE001 — retry any parse/API hiccup
            last_error = exc
            print(f"  {label} attempt {attempt + 1}/3 failed "
                  f"({type(exc).__name__}: {exc}) — retrying")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{label} failed 3x") from last_error


def build_user_prompt(current_prompt: str, examples: list[Example]) -> str:
    rendered_examples = "\n\n".join(ex.render(i + 1) for i, ex in enumerate(examples))
    return (
        "=== CURRENT JUDGE SYSTEM PROMPT ===\n"
        f"{current_prompt}\n\n"
        "=== LABELED EXAMPLES FOR THIS ROUND ===\n"
        f"{rendered_examples}\n\n"
        "Propose a revised judge system prompt per the instructions above. "
        "Respond with the JSON object described above and nothing else."
    )


def validate_contract(prompt_text: str) -> list[str]:
    """Light sanity check that the revised prompt kept the required output
    contract intact. Returns a list of problems (empty = fine); never raises
    — a flagged prompt is still recorded, just marked in the log."""
    problems = []
    lower = prompt_text.lower()
    for marker in REQUIRED_JUDGE_CONTRACT_MARKERS:
        if marker not in lower:
            problems.append(f"missing '{marker}' in revised prompt")
    if "json" not in lower:
        problems.append("revised prompt no longer mentions JSON output")
    return problems


def run_one_iteration(
    *,
    cfg: RunConfig,
    current_prompt: str,
    examples: list[Example],
    iteration: int,
) -> dict[str, Any]:
    system = load_meta_system_prompt(cfg)
    user = build_user_prompt(current_prompt, examples)

    response = _call_with_retries(
        system=system, user=user, cfg=cfg, label=f"iter {iteration:02d}",
        run_tags=["judge-prompt-evolution", cfg.run_id, f"iter-{iteration:02d}"],
        run_metadata_extra={"iteration": iteration, "step": "optimize"},
    )

    updated_prompt = str(response.get("updated_prompt") or "").strip()
    rationale = str(response.get("rationale") or "").strip()
    if not updated_prompt:
        raise ValueError(f"optimizer returned no 'updated_prompt' at iteration {iteration}: {response!r}")

    problems = validate_contract(updated_prompt)
    return {
        "iteration": iteration,
        "kind": "optimize",
        "prompt_before": current_prompt,
        "prompt_after": updated_prompt,
        "rationale": rationale,
        "contract_problems": problems,
        "examples": [ex.to_dict() for ex in examples],
        "optimizer_model": cfg.optimizer_model,
    }


def run_summarization_step(*, cfg: RunConfig, current_prompt: str, after_iteration: int) -> dict[str, Any]:
    """A separate, concise distillation call — not the meta-optimizer prompt.

    Distinguished from a normal round in three ways: no example batch (this
    is a pure rewrite of the current prompt, not a response to new data), a
    different system prompt whose entire job is merge/cut, and its own
    ``kind`` tag in the saved record so the browser/log can tell the two
    apart.
    """
    system = load_summarizer_system_prompt(cfg)
    user = (
        "=== CURRENT JUDGE SYSTEM PROMPT ===\n"
        f"{current_prompt}\n\n"
        "Distill this per the instructions above. Respond with the JSON object "
        "described above and nothing else."
    )
    response = _call_with_retries(
        system=system, user=user, cfg=cfg, label=f"summarize@{after_iteration:02d}",
        run_tags=["judge-prompt-evolution", cfg.run_id, f"summarize-after-{after_iteration:02d}"],
        run_metadata_extra={"after_iteration": after_iteration, "step": "summarize"},
    )

    updated_prompt = str(response.get("updated_prompt") or "").strip()
    rationale = str(response.get("rationale") or "").strip()
    if not updated_prompt:
        raise ValueError(f"summarizer returned no 'updated_prompt' after iteration {after_iteration}: {response!r}")

    problems = validate_contract(updated_prompt)
    return {
        "iteration": after_iteration,
        "kind": "summarize",
        "prompt_before": current_prompt,
        "prompt_after": updated_prompt,
        "rationale": rationale,
        "contract_problems": problems,
        "examples": [],
        "optimizer_model": cfg.optimizer_model,
    }
