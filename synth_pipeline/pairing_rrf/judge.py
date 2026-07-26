"""Cached, concurrent flash-lite verdicts — one call per pair, cost measured.

Model choice is settled by ``docs/llm-judge-experiment.md``: on the matched
69-pair holdout ``google/gemini-3.1-flash-lite`` under the *naive* framing scored
pair AUC 0.6358, beating Voyage-4-large (0.6086) — Boardy's own production model
— and both Bedrock judges tested (gemma-3-27b 0.5823, qwen3-32b 0.5802). More to
the point for this pipeline, its hard-negative AUC of 0.6466 is the best number
in the whole project, ahead of Qwen3-Embedding-8B's 0.6259. Its one weak slice,
easy negatives at 0.5638, never reaches us: anything arriving here already
survived two retrieval channels.

Bedrock cannot serve this model. The account exposes ``google.gemma-3-{4b,12b,27b}-it``
— Google's open-weight Gemma family — and Gemini is not among them, so this is
the single OpenRouter dependency in the pipeline.

Three findings from that experiment are load-bearing here:

1. **Naive framing only.** Telling the model the truth about the task (that a
   production matcher already filtered for relevance, that the base rate is
   50/50) measurably *hurt*: AUC 0.6358 → 0.5901. It did not discriminate
   better, it just got stingier — yes-rate fell 56.5% → 30.4%.
2. **Stated confidence is worthless.** Mean 88.5, and 88.6 when right against
   88.2 when wrong, with 199 of 200 answers inside the 80-100 band. It is
   recorded here for audit but never gates anything.
3. **``max_tokens`` must be set.** Left unset, OpenRouter reserves credit
   against the model's absolute ceiling and can reject an affordable call as
   unaffordable — the 402 that killed two other judge runs.
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

from baselines.bert_frozen.text import profile_to_text
from synth_pipeline.pairing_rrf.prompt_hub import LoadedPrompt, load_prompt

DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 600
DEFAULT_TEMPERATURE = 0.0


def build_user_prompt(
    seeker_profile: Mapping[str, Any],
    search_query: str,
    candidate_profile: Mapping[str, Any],
) -> str:
    """Render the pair. Instruction text lives in the hub prompt; this is data."""
    return (
        "=== PERSON A ===\n"
        f"{profile_to_text(seeker_profile)}\n\n"
        "=== PERSON A'S SEARCH QUERY ===\n"
        f"{search_query.strip()}\n\n"
        "=== PERSON B ===\n"
        f"{profile_to_text(candidate_profile)}\n\n"
        "Would introducing Person A and Person B be a good match? "
        "Answer with the JSON object described above."
    )


def prompt_hash(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:16]


def parse_verdict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a model response. Raises ValueError so the caller can retry."""
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
        conf = None  # recorded-only field; never worth failing a call over
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
    """Token counts and the exact cost OpenRouter billed for one call."""

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


def _usage_from_response(payload: Mapping[str, Any]) -> Usage:
    u = payload.get("usage") or {}
    return Usage(
        prompt_tokens=int(u.get("prompt_tokens") or 0),
        completion_tokens=int(u.get("completion_tokens") or 0),
        # OpenRouter reports the real billed amount, so the run's cost is
        # measured rather than inferred from a price table.
        cost_usd=float(u.get("cost") or 0.0),
        calls=1,
    )


def call_openrouter(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 120,
) -> tuple[dict[str, Any], Usage]:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    content = payload["choices"][0]["message"]["content"]
    return _extract_json(content), _usage_from_response(payload)


@dataclass
class JudgeResult:
    pair_key: str
    verdict: dict[str, Any]
    cached: bool = False
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


class Judge:
    """Verdict calls with an on-disk cache keyed by exact prompt text.

    The cache key includes a hash of the rendered system+user prompt, so editing
    the hub prompt invalidates affected entries instead of silently serving
    verdicts produced by different instructions.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.usage = Usage()
        self._lock = threading.Lock()

        loaded: LoadedPrompt = load_prompt("pair_judge")
        self.system = loaded.text
        self.prompt_ref = loaded.ref

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

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
                raw, usage = call_openrouter(
                    system=self.system,
                    user=user,
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
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
                            "prompt_ref": self.prompt_ref.to_dict(),
                            "verdict": verdict,
                            "usage": usage.to_dict(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return JudgeResult(pair_key=pair_key, verdict=verdict, usage=usage)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code in (400, 401, 402, 403):
                    break  # not transient — retrying only burns time
                time.sleep(1.5 * (attempt + 1))
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
