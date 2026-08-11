"""Cached, concurrent gemini-3.1-flash-lite verdicts via the direct Google API.

Isolated variant of ``synth_pipeline.pairing_rrf_qwen_judge.judge``. Two
differences:

* **Backend**: direct Google Gemini API (``GEMINI_API_KEY``, raw REST via
  ``urllib``), not OpenRouter and not Bedrock — matching the call mechanics in
  ``judge_prompt_evolution_focused/eval_evolved.py::_make_gemini_call_fn``,
  duplicated here rather than imported (that module is on the
  ``llm-judge-experiment`` branch, not importable from ``main``).
* **Prompt**: the "focused" field-trimmed prompt (seeker: ``positioning`` +
  ``lookingFor``; candidate: ``positioning`` + ``background`` + ``lookingFor``;
  query included), vendored from
  ``judge_prompt_evolution_focused/focused_prompt.py`` — measured pair AUC
  0.6451, decision accuracy 0.5950 on the all-200 real split
  (``docs/llm-judge-focused-prompt-experiment.md``), the best judge
  configuration measured in this project to date.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_MAX_TOKENS = 600
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

SEEKER_FIELDS = ("positioning", "lookingFor")
CANDIDATE_FIELDS = ("positioning", "background", "lookingFor")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fields_to_text(profile: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for f in fields:
        value = profile.get(f)
        if _nonempty(value):
            parts.append(f"{f}: {value.strip()}")
    return "\n".join(parts)


SYSTEM_PROMPT = """
You are evaluating whether two people would be a good match for a
professional networking introduction.

You will be shown a seeker (Person A) and a candidate (Person B). Person A
has a specific search query describing who they are looking for. Decide
whether Person B is a good match for that search.

Judge the match on these specifics:
- Complementary need and supply: Person A's lookingFor should map to
  something concrete in Person B's positioning or background - not just
  shared topic or industry.
- Two-way fit: check whether Person B's own lookingFor is also compatible
  with what Person A can offer. A good intro should not be one-sided.
- Specificity over vibes: shared keywords or industry buzzwords alone do
  not make a match - look for a concrete reason each side would want to
  talk to the other, based on their actual stated preferences, not surface
  similarity.

Respond with a single JSON object and nothing else:

{
  "reasoning": "<2-4 sentences of your actual reasoning, written before you decide>",
  "match": "yes" | "no",
  "confidence": <integer 0-100, how sure you are of the "match" value>
}

"confidence" is confidence in the answer you gave, not the probability of
"yes": answering "no" with confidence 90 means you are 90% sure it is not a
good match. Use the full 0-100 range - say 55 when it is close to a
coin-flip and 95 only when it is clear-cut.
""".strip()


def build_user_prompt(
    seeker_profile: Mapping[str, Any],
    search_query: str,
    candidate_profile: Mapping[str, Any],
) -> str:
    seeker_text = _fields_to_text(seeker_profile, SEEKER_FIELDS)
    candidate_text = _fields_to_text(candidate_profile, CANDIDATE_FIELDS)
    query = (search_query or "").strip()
    return (
        "=== PERSON A (seeker) ===\n"
        f"{seeker_text}\n\n"
        f"Search query: {query}\n\n"
        "=== PERSON B (candidate) ===\n"
        f"{candidate_text}\n\n"
        "Would Person B be a good match for Person A's search? "
        "Answer with the JSON object described above."
    )


def prompt_hash(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:16]


def parse_verdict(raw: Mapping[str, Any]) -> dict[str, Any]:
    match = raw.get("match")
    if isinstance(match, bool):
        match = "yes" if match else "no"
    if not isinstance(match, str):
        raise ValueError(f"'match' missing or not a string: {match!r}")
    match = match.strip().lower()
    if match not in {"yes", "no"}:
        raise ValueError(f"'match' must be 'yes' or 'no', got {match!r}")

    conf = raw.get("confidence")
    if isinstance(conf, str):
        conf = conf.strip().rstrip("%")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = None
    if conf is not None and 0.0 <= conf <= 1.0:
        conf *= 100.0

    return {
        "match": match,
        "confidence": conf,
        "reasoning": str(raw.get("reasoning") or "").strip(),
    }


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


@dataclass
class Usage:
    """Gemini doesn't bill in dollars directly; token counts are real, cost is
    computed from published gemini-3.1-flash-lite pricing (see
    docs/llm-judge-experiment.md's OpenRouter-measured $/200-pair reference for
    the same model — this is the same rate, just billed via a different API)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


# gemini-3.1-flash-lite published pricing, per OpenRouter's model card (used
# consistently across this project's gemini-flash-lite runs).
PRICE_PER_1M_INPUT = 0.10
PRICE_PER_1M_OUTPUT = 0.40


def call_gemini(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[dict[str, Any], Usage]:
    """Direct REST call, no SDK dependency — matches this repo's existing
    lightweight HTTP-call pattern used for the OpenRouter judge backend."""
    url = f"{base_url}/models/{model}:generateContent?key={api_key}"
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc

    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"no candidates in Gemini response: {payload!r}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)

    usage_meta = payload.get("usageMetadata") or {}
    prompt_tokens = int(usage_meta.get("promptTokenCount") or 0)
    completion_tokens = int(usage_meta.get("candidatesTokenCount") or 0)
    cost = (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT + (
        completion_tokens / 1_000_000
    ) * PRICE_PER_1M_OUTPUT
    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost, calls=1)

    return _extract_json(text), usage


@dataclass
class JudgeResult:
    pair_key: str
    verdict: dict[str, Any]
    cached: bool = False
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


class Judge:
    """Verdict calls with an on-disk cache keyed by exact prompt text."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.usage = Usage()
        self._lock = threading.Lock()

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it or pass --env-file /path/to/.env."
            )

        self.system = SYSTEM_PROMPT
        self.prompt_ref = {"kind": "inline", "name": "focused_prompt", "source": "judge_prompt_evolution_focused"}

    def _cache_path(self, pair_key: str, phash: str) -> Path:
        safe = pair_key.replace("/", "_").replace(":", "-")
        return self.cache_dir / f"{safe}__{phash}.json"

    def judge_one(
        self,
        pair_key: str,
        seeker_profile: Mapping[str, Any],
        search_query: str,
        candidate_profile: Mapping[str, Any],
    ) -> JudgeResult:
        user = build_user_prompt(seeker_profile, search_query, candidate_profile)
        phash = prompt_hash(self.system, user)
        path = self._cache_path(pair_key, phash)

        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return JudgeResult(pair_key=pair_key, verdict=cached["verdict"], cached=True)

        last_error: str | None = None
        for attempt in range(self.max_retries):
            try:
                raw, usage = call_gemini(
                    system=self.system,
                    user=user,
                    model=self.model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                verdict = parse_verdict(raw)
                with self._lock:
                    self.usage.add(usage)
                path.write_text(
                    json.dumps(
                        {
                            "pair_key": pair_key,
                            "model": self.model,
                            "prompt_hash": phash,
                            "prompt_ref": self.prompt_ref,
                            "verdict": verdict,
                            "usage": usage.to_dict(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return JudgeResult(pair_key=pair_key, verdict=verdict, usage=usage)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))

        return JudgeResult(pair_key=pair_key, verdict={}, error=last_error)

    def judge_many(
        self,
        items: Sequence[tuple[str, Mapping[str, Any], str, Mapping[str, Any]]],
        *,
        concurrency: int = 4,
        on_result: Callable[[JudgeResult], None] | None = None,
    ) -> list[JudgeResult]:
        results: list[JudgeResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(self.judge_one, key, seeker, query, cand): key
                for key, seeker, query, cand in items
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if on_result:
                    on_result(res)
        return results
