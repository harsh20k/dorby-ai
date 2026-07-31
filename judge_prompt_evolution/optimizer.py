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

META_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "meta_optimizer.md"

REQUIRED_JUDGE_CONTRACT_MARKERS = ("reasoning", "match", "confidence")


def load_meta_system_prompt() -> str:
    return META_PROMPT_PATH.read_text(encoding="utf-8").strip()


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
    system = load_meta_system_prompt()
    user = build_user_prompt(current_prompt, examples)

    # The revised prompt grows every round (naturally — it's accreting
    # rubric detail), so a truncated completion near the token cap is a real
    # failure mode late in a long run, not a rare fluke. Retry with the same
    # inputs rather than losing the whole loop to one bad JSON parse.
    last_error: Exception | None = None
    response: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            response = _call_optimizer(
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
                    "iteration": iteration,
                    "attempt": attempt,
                },
                run_tags=["judge-prompt-evolution", cfg.run_id, f"iter-{iteration:02d}"],
            )
            break
        except Exception as exc:  # noqa: BLE001 — retry any parse/API hiccup
            last_error = exc
            print(f"  iter {iteration:02d} attempt {attempt + 1}/3 failed "
                  f"({type(exc).__name__}: {exc}) — retrying")
            time.sleep(2.0 * (attempt + 1))
    if response is None:
        raise RuntimeError(f"optimizer call failed 3x at iteration {iteration}") from last_error

    updated_prompt = str(response.get("updated_prompt") or "").strip()
    rationale = str(response.get("rationale") or "").strip()
    if not updated_prompt:
        raise ValueError(f"optimizer returned no 'updated_prompt' at iteration {iteration}: {response!r}")

    problems = validate_contract(updated_prompt)
    return {
        "iteration": iteration,
        "prompt_before": current_prompt,
        "prompt_after": updated_prompt,
        "rationale": rationale,
        "contract_problems": problems,
        "examples": [ex.to_dict() for ex in examples],
        "optimizer_model": cfg.optimizer_model,
    }
