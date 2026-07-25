"""Cached, concurrent LLM verdict calls for the LLM-judge experiment.

Caching follows the same convention as the embedding baselines' encode caches:
a completed run is free to re-score, so metric changes can be iterated on
without re-spending API budget. The cache key includes a hash of the exact
system+user prompt text, so editing a prompt correctly invalidates every
affected entry instead of silently serving stale verdicts — the failure mode
already hit once in this repo (see the ``cache_name`` note under
"Pairing standalone profiles" in CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from synth_pipeline.llm import complete_json

DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def load_env_file(env_file: Path | None) -> None:
    """Load KEY=VALUE lines from ``env_file`` without overriding real env vars.

    ``synth_pipeline.config.load_dotenv`` resolves ``.env`` relative to its own
    package root, which is wrong when running from a git worktree (the worktree
    has no ``.env``). This takes an explicit path instead.
    """
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


def prompt_hash(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:16]


def parse_verdict(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw model JSON object into {match, confidence, reasoning}.

    Raises ValueError on anything unusable, so the caller can retry rather
    than quietly coercing a malformed answer into a real data point.
    """
    match = raw.get("match")
    if isinstance(match, bool):  # some models emit a JSON bool
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'confidence' not numeric: {raw.get('confidence')!r}") from exc
    if 0.0 <= conf <= 1.0 and not float(conf).is_integer():
        # A model that answered on a 0-1 scale despite the instruction.
        conf *= 100.0
    if not 0.0 <= conf <= 100.0:
        raise ValueError(f"'confidence' out of range 0-100: {conf!r}")

    reasoning = raw.get("reasoning")
    return {
        "match": match,
        "confidence": float(conf),
        "reasoning": reasoning if isinstance(reasoning, str) else None,
    }


def verdict_to_score(verdict: dict[str, Any]) -> float:
    """Map a verdict onto a continuous [0, 1] score for ROC-AUC.

    yes → 0.5 + conf/200, no → 0.5 - conf/200. Monotone in "how much the model
    believes this is a match", and centered so that 0.5 is exactly the model's
    own decision boundary — which makes ``pair.accuracy_at_0.5`` from
    baselines/metrics.py equal the model's raw decision accuracy.
    """
    signed = verdict["confidence"] / 200.0
    return 0.5 + signed if verdict["match"] == "yes" else 0.5 - signed


class VerdictCache:
    """Thread-safe JSON-file-backed verdict cache."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except json.JSONDecodeError:
                print(f"warning: could not parse {path}, starting a fresh cache")
                self._data = {}

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = value

    def flush(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2) + "\n")
            tmp.replace(self.path)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def _call_once(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    raw = complete_json(
        system=system,
        user=user,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        run_tags=["llm-judge-experiment"],
    )
    return parse_verdict(raw)


def judge_pair(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """One verdict, retrying transient API errors and malformed JSON."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            verdict = _call_once(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
            )
            verdict["attempts"] = attempt + 1
            return verdict
        except Exception as exc:  # noqa: BLE001 — API/parse errors both retryable
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(2.0 * (2**attempt))
    raise RuntimeError(f"all {max_attempts} attempts failed: {last_exc}") from last_exc


def judge_all(
    requests: list[tuple[str, str, str]],
    *,
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
    cache: VerdictCache,
    workers: int = 8,
    max_attempts: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Judge every ``(key, system, user)`` request. Returns (verdicts, errors).

    ``key`` is the cache key and the identity used by the caller to line
    verdicts back up with pairs. Cache hits never re-call the API.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    pending: list[tuple[str, str, str]] = []

    for key, system, user in requests:
        cached = cache.get(key)
        if cached is not None:
            verdicts[key] = cached
        else:
            pending.append((key, system, user))

    total = len(requests)
    done = len(verdicts)
    if on_progress:
        on_progress(done, total)
    if not pending:
        return verdicts, errors

    flush_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                judge_pair,
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                max_attempts=max_attempts,
            ): key
            for key, system, user in pending
        }
        for i, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                verdict = future.result()
                verdicts[key] = verdict
                cache.put(key, verdict)
            except Exception as exc:  # noqa: BLE001
                errors[key] = str(exc)
            done += 1
            if on_progress:
                on_progress(done, total)
            # Checkpoint periodically so an interrupted run keeps its progress.
            if i % 20 == 0:
                with flush_lock:
                    cache.flush()

    cache.flush()
    return verdicts, errors
