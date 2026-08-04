"""Cached, concurrent Qwen3-32B-on-Bedrock verdicts — one call per pair, cost measured.

Isolated variant of ``synth_pipeline.pairing_rrf.judge`` (see this package's
``__init__.py``). The sibling package's judge.py has the full writeup of why
``google/gemini-3.1-flash-lite`` via OpenRouter is the measured-best judge
(``docs/llm-judge-experiment.md``: pair AUC 0.6358, hard-negative AUC 0.6466,
best in the project). This variant trades some of that accuracy for ~3x lower
cost by calling ``qwen.qwen3-32b-v1:0`` on AWS Bedrock instead (pair AUC 0.5802,
hard-negative AUC 0.6224 — the hard-negative gap is much smaller than the
headline AUC gap suggests, ~$0.03/200 pairs vs ~$0.10/200).

The naive-framing and max-tokens findings from that experiment still apply
(framing is fixed in the hub prompt; Bedrock's Converse API has its own
token-ceiling field, set explicitly below). Bedrock call mechanics (structured
JSON-schema output first, falling back to plain-text-prompted JSON on
``ValidationException``) are reused read-only from
``baselines.llm_judge.bedrock_backend`` rather than duplicated — that module
already handles the fallback this pipeline would otherwise need to reinvent.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from baselines.bert_frozen.text import profile_to_text
from baselines.llm_judge.bedrock_backend import DEFAULT_PROFILE, DEFAULT_REGION, make_client
from synth_pipeline.pairing_rrf_qwen_judge.prompt_hub import LoadedPrompt, load_prompt

DEFAULT_MODEL = "qwen.qwen3-32b-v1:0"
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


# Bedrock's Converse API doesn't report a billed-dollar figure the way
# OpenRouter does, so cost here is computed from token counts against the
# published on-demand price (scripts/update_bedrock_dashboard.py::MODEL_PRICING,
# kept in sync there — update both if AWS reprices).
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 1.20


def _usage_from_bedrock(payload: Mapping[str, Any]) -> Usage:
    u = payload.get("usage") or {}
    prompt_tokens = int(u.get("inputTokens") or 0)
    completion_tokens = int(u.get("outputTokens") or 0)
    cost = (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT + (
        completion_tokens / 1_000_000
    ) * PRICE_PER_1M_OUTPUT
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost, calls=1)


def call_bedrock(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    client: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[dict[str, Any], Usage]:
    """One verdict call. Parsing/fallback logic is ``call_bedrock_verdict``
    (imported, not duplicated); this wrapper only adds the usage/cost capture
    that helper doesn't return, by making its own Converse call the same way."""
    from botocore.exceptions import ClientError

    from baselines.llm_judge.bedrock_backend import VERDICT_SCHEMA

    messages = [{"role": "user", "content": [{"text": user}]}]
    inference_config = {"maxTokens": max_tokens, "temperature": temperature}
    try:
        resp = client.converse(
            modelId=model,
            messages=messages,
            system=[{"text": system}],
            inferenceConfig=inference_config,
            outputConfig={
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(VERDICT_SCHEMA),
                            "name": "verdict",
                            "description": "Match verdict with confidence and reasoning.",
                        }
                    },
                }
            },
        )
        text = resp["output"]["message"]["content"][0]["text"]
        return json.loads(text), _usage_from_bedrock(resp)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code", "") != "ValidationException":
            raise

    plain_system = (
        system
        + "\n\nRespond with only a single JSON object matching this shape, "
        'no other text: {"reasoning": string, "match": "yes"|"no", "confidence": integer 0-100}'
    )
    resp = client.converse(
        modelId=model,
        messages=messages,
        system=[{"text": plain_system}],
        inferenceConfig=inference_config,
    )
    text = resp["output"]["message"]["content"][0]["text"]
    return _extract_json(text), _usage_from_bedrock(resp)


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
        aws_profile: str | None = DEFAULT_PROFILE,
        region: str = DEFAULT_REGION,
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
        self._client = make_client(profile=aws_profile, region=region)

        loaded: LoadedPrompt = load_prompt("pair_judge")
        self.system = loaded.text
        self.prompt_ref = loaded.ref

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
                raw, usage = call_bedrock(
                    system=self.system,
                    user=user,
                    model=self.model,
                    client=self._client,
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
