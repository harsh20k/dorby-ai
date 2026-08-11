"""One ``searchQuery`` per ``lookingFor`` section, generated on Bedrock.

The earlier pairing pipeline asked for N queries per profile in a single call and
let the model decide which needs to cover. Here the sections are enumerated first
and each gets its own call, so a query is bound to a known section — which is what
lets the dense channel pair that query with that section's embedding, and what
makes "someone juggling fundraising and hiring produces two distinct asks" a
property of the data rather than a hope about the model.

The call sees exactly one profile and no candidate, so nothing it writes can
correlate with a label — no label exists at this point. Do not add counterpart
context here; that would reintroduce, one layer up, the leak diagnosed in
``docs/possible-bugs.md`` #4.

Results are checkpointed to ``queries.json`` and reused on re-run. The checkpoint
is keyed by ``contact_id::qN``, and contact ids are derived deterministically from
``sha256(source_run:profile_id)``, so a re-run genuinely resumes rather than
silently pairing new queries with old embeddings.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from baselines.bert_frozen.text import profile_to_text
from synth_pipeline.pairing.bedrock import call_json
from synth_pipeline.pairing_rrf_qwen_judge.prompt_hub import load_prompt
from synth_pipeline.pairing_rrf_qwen_judge.sections import QueryTarget

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


@dataclass
class BedrockUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, usage: Mapping[str, Any]) -> None:
        self.input_tokens += int(usage.get("inputTokens") or 0)
        self.output_tokens += int(usage.get("outputTokens") or 0)
        self.calls += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


@dataclass
class QueryGenResult:
    queries: dict[str, str] = field(default_factory=dict)
    usage: BedrockUsage = field(default_factory=BedrockUsage)
    prompt_ref: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)


def load_checkpoint(path: Path) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.get("queries", {}).items() if isinstance(v, str)}


def save_checkpoint(path: Path, queries: Mapping[str, str], meta: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"queries": dict(queries), **dict(meta)}, indent=2), encoding="utf-8"
    )


def _generate_one(
    target: QueryTarget,
    profile: Mapping[str, Any],
    *,
    client,
    model_id: str,
    template: str,
    style_block: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Returns (key, query_or_None, usage_or_None). Errors come back as query=None."""
    prompt = template.format(
        profile=profile_to_text(profile),
        section=target.section_text,
        style_examples=style_block,
    )
    try:
        parsed, usage = call_json(
            client,
            model_id=model_id,
            prompt=prompt,
            schema=QUERY_SCHEMA,
            schema_name="search_query",
            schema_description="One search query for one stated need.",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        return target.key, None, {"error": f"{type(exc).__name__}: {exc}"}

    query = str(parsed.get("query") or "").strip()
    if not query:
        return target.key, None, {"error": "empty query"}
    return target.key, query, usage


def generate_queries(
    targets: Sequence[QueryTarget],
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    client,
    model_id: str,
    style_examples: Sequence[str],
    checkpoint_path: Path | None = None,
    max_tokens: int = 600,
    temperature: float = 0.7,
    resume: bool = True,
    concurrency: int = 1,
) -> QueryGenResult:
    """Write one query per target, resuming from the checkpoint where possible.

    Bedrock client calls run through a thread pool when ``concurrency`` > 1 —
    boto3 clients are documented thread-safe for concurrent use. This was added
    after a live 12k-query run measured ~42/min fully serial (one call at a
    time, unlike the judge phase's existing ``judge_concurrency``); the fix is
    additive and defaults to the old serial behavior (concurrency=1) so nothing
    that already called this function changes unless it opts in."""
    loaded = load_prompt("pair_query")
    template = loaded.text
    result = QueryGenResult(prompt_ref=loaded.ref.to_dict())

    if resume and checkpoint_path:
        result.queries.update(load_checkpoint(checkpoint_path))

    style_block = "\n".join(f"- {q}" for q in style_examples)
    pending = [t for t in targets if t.key not in result.queries]
    for t in pending:
        if t.contact_id not in profiles:
            result.failures.append({"key": t.key, "error": "profile not found"})
    pending = [t for t in pending if t.contact_id in profiles]

    lock = threading.Lock()
    done_count = 0

    def _save() -> None:
        save_checkpoint(
            checkpoint_path,
            result.queries,
            {"model_id": model_id, "prompt_ref": result.prompt_ref},
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _generate_one,
                t,
                profiles[t.contact_id],
                client=client,
                model_id=model_id,
                template=template,
                style_block=style_block,
                max_tokens=max_tokens,
                temperature=temperature,
            ): t
            for t in pending
        }
        for fut in as_completed(futures):
            key, query, meta = fut.result()
            with lock:
                if query is None:
                    result.failures.append({"key": key, **(meta or {})})
                else:
                    result.queries[key] = query
                    result.usage.add(meta)
                done_count += 1
                if checkpoint_path and done_count % 10 == 0:
                    _save()

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            result.queries,
            {
                "model_id": model_id,
                "prompt_ref": result.prompt_ref,
                "usage": result.usage.to_dict(),
                "failures": result.failures,
            },
        )
    return result
