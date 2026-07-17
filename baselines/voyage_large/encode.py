"""Voyage API encoder for voyage-4-large with disk cache + rate limiting."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

# Conservative defaults under current tier (3M TPM / 2k RPM).
DEFAULT_TPM_LIMIT = 2_500_000
DEFAULT_RPM_LIMIT = 1_500
# voyage-4-large: max 1000 texts/request; ~120k tokens/request — keep small.
DEFAULT_BATCH_SIZE = 16
# Rough token estimate: ~4 chars/token (Voyage-ish); used for budgeting only.
CHARS_PER_TOKEN = 4
# Refuse absurd runs before spending API tokens.
MAX_ESTIMATED_TOKENS = 5_000_000


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def estimate_tokens(texts: Sequence[str]) -> int:
    return sum(max(1, (len(t) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN) for t in texts)


def text_cache_key(
    text: str,
    model_name: str,
    input_type: str,
    output_dimension: int,
) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"\0")
    h.update(input_type.encode())
    h.update(b"\0")
    h.update(str(output_dimension).encode())
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def require_api_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. Export your Voyage API key, e.g.\n"
            "  export VOYAGE_API_KEY=pa-...\n"
            "See https://docs.voyageai.com/docs/api-key-and-installation"
        )
    return key


class RateLimiter:
    """Sliding-window throttle for TPM and RPM."""

    def __init__(self, tpm_limit: int, rpm_limit: int) -> None:
        self.tpm_limit = max(1, tpm_limit)
        self.rpm_limit = max(1, rpm_limit)
        self._token_events: deque[tuple[float, int]] = deque()
        self._request_times: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._token_events and self._token_events[0][0] < cutoff:
            self._token_events.popleft()
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()

    def wait_for_budget(self, estimated_tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._prune(now)
            tokens_used = sum(n for _, n in self._token_events)
            reqs = len(self._request_times)

            need_sleep = 0.0
            if reqs >= self.rpm_limit:
                need_sleep = max(need_sleep, 60.0 - (now - self._request_times[0]) + 0.05)
            if tokens_used + estimated_tokens > self.tpm_limit and self._token_events:
                need_sleep = max(
                    need_sleep, 60.0 - (now - self._token_events[0][0]) + 0.05
                )
            if need_sleep <= 0:
                return
            print(f"rate limit: sleeping {need_sleep:.2f}s (tpm/rpm headroom)")
            time.sleep(need_sleep)

    def record(self, tokens: int) -> None:
        now = time.monotonic()
        self._token_events.append((now, max(0, tokens)))
        self._request_times.append(now)


class VoyageLargeEncoder:
    """API voyage-4-large with per-text disk cache, dedupe, and rate limits."""

    def __init__(
        self,
        model_name: str = "voyage-4-large",
        output_dimension: int = 1024,
        truncation: bool = True,
        cache_dir: Path | None = None,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 8,
    ) -> None:
        self.model_name = model_name
        self.output_dimension = output_dimension
        self.truncation = truncation
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/voyage_large")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.emb_dir = self.cache_dir / "emb"
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, min(batch_size, 1000))
        self.max_retries = max_retries
        self.limiter = RateLimiter(tpm_limit=tpm_limit, rpm_limit=rpm_limit)

        api_key = require_api_key()
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai package required. Install with: pip install voyageai"
            ) from exc

        self._client = voyageai.Client(api_key=api_key)
        self.total_tokens_used = 0
        self.total_api_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_path(self, key: str) -> Path:
        return self.emb_dir / f"{key}.npy"

    def _load_cached(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        if path.exists():
            return np.load(path)
        return None

    def _save_cached(self, key: str, vec: np.ndarray) -> None:
        np.save(self._cache_path(key), vec.astype(np.float32))

    def _embed_batch(
        self,
        texts: list[str],
        input_type: Literal["query", "document"],
    ) -> tuple[list[list[float]], int]:
        est = estimate_tokens(texts)
        attempt = 0
        while True:
            self.limiter.wait_for_budget(est)
            try:
                result = self._client.embed(
                    texts,
                    model=self.model_name,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dimension=self.output_dimension,
                )
                usage = getattr(result, "total_tokens", None)
                tokens = int(usage) if usage is not None else est
                self.limiter.record(tokens)
                self.total_tokens_used += tokens
                self.total_api_calls += 1
                return list(result.embeddings), tokens
            except Exception as exc:
                msg = str(exc).lower()
                is_rate = "429" in msg or "rate" in msg or "too many" in msg
                attempt += 1
                if not is_rate or attempt > self.max_retries:
                    raise
                backoff = min(60.0, (2 ** (attempt - 1)) + 0.25)
                print(f"429/rate error (attempt {attempt}/{self.max_retries}): sleep {backoff:.1f}s")
                time.sleep(backoff)

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: Literal["query", "document"],
        batch_size: int | None = None,
        show_progress: bool = True,
        label: str = "",
    ) -> np.ndarray:
        texts_list = list(texts)
        dim = self.output_dimension
        if not texts_list:
            return np.zeros((0, dim), dtype=np.float32)

        est_all = estimate_tokens(texts_list)
        if est_all > MAX_ESTIMATED_TOKENS:
            raise RuntimeError(
                f"Refusing encode ({label or input_type}): estimated ~{est_all:,} tokens "
                f"exceeds safety cap {MAX_ESTIMATED_TOKENS:,}. Shrink data or raise "
                "MAX_ESTIMATED_TOKENS deliberately."
            )

        keys = [
            text_cache_key(t, self.model_name, input_type, self.output_dimension)
            for t in texts_list
        ]
        out: list[np.ndarray | None] = [None] * len(texts_list)
        miss_order: list[str] = []
        miss_key_to_indices: dict[str, list[int]] = {}

        for i, (text, key) in enumerate(zip(texts_list, keys)):
            cached = self._load_cached(key)
            if cached is not None:
                out[i] = np.asarray(cached, dtype=np.float32).reshape(-1)
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            if key not in miss_key_to_indices:
                miss_key_to_indices[key] = []
                miss_order.append(text)
            miss_key_to_indices[key].append(i)

        # Dedupe: miss_order has unique texts (first occurrence order via key map).
        unique_texts: list[str] = []
        unique_keys: list[str] = []
        seen_keys: set[str] = set()
        for text in miss_order:
            key = text_cache_key(text, self.model_name, input_type, self.output_dimension)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_texts.append(text)
            unique_keys.append(key)

        bs = batch_size or self.batch_size
        n_unique = len(unique_texts)
        est_miss = estimate_tokens(unique_texts)
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}encode input_type={input_type}: {len(texts_list)} texts, "
            f"{self.cache_hits} cache hits so far, {n_unique} unique API texts, "
            f"~{est_miss:,} est. tokens"
        )

        if n_unique and est_miss > MAX_ESTIMATED_TOKENS:
            raise RuntimeError(
                f"Refusing API call ({label or input_type}): ~{est_miss:,} uncached tokens "
                f"exceeds safety cap {MAX_ESTIMATED_TOKENS:,}."
            )

        for start in range(0, n_unique, bs):
            chunk_texts = unique_texts[start : start + bs]
            chunk_keys = unique_keys[start : start + bs]
            if show_progress:
                print(
                    f"{prefix}API batch {start // bs + 1}/{(n_unique + bs - 1) // bs} "
                    f"({len(chunk_texts)} texts, ~{estimate_tokens(chunk_texts):,} tokens)"
                )
            embeddings, _tokens = self._embed_batch(chunk_texts, input_type=input_type)
            for key, emb in zip(chunk_keys, embeddings):
                vec = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(1, -1))[0]
                self._save_cached(key, vec)
                for idx in miss_key_to_indices[key]:
                    out[idx] = vec

        matrix = np.stack([v for v in out if v is not None], axis=0).astype(np.float32)
        if matrix.shape[0] != len(texts_list):
            raise RuntimeError("internal error: missing embeddings after encode")
        return l2_normalize(matrix).astype(np.float32)

    def stats(self) -> dict[str, int]:
        return {
            "total_tokens_used": int(self.total_tokens_used),
            "total_api_calls": int(self.total_api_calls),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
        }

    def write_usage_meta(self) -> Path:
        path = self.cache_dir / "usage.json"
        path.write_text(json.dumps(self.stats(), indent=2) + "\n")
        return path


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine for L2-normalized vectors."""
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    return np.sum(a * b, axis=-1)
