"""Load the RRF-pairing prompts from LangSmith Hub — hub only, no local fallback.

Deliberately strict, matching ``scripts/profile_gen_prompt_hub.py`` rather than
``synth_pipeline/prompts/__init__.py``: the point of routing prompts through the
hub is that every labeled pair traces back to an exact, versioned, reviewable
prompt commit. A silent local fallback would defeat that — a run could "succeed"
while quietly using un-audited text. If the pull fails, the run fails loudly.

The local ``prompts/*.md`` files are the source of truth you edit and commit.
``push_prompts.py`` is the only thing that reads them; this module never does.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

PromptRole = Literal["pair_query", "pair_judge"]

LOCAL_FILES: dict[PromptRole, str] = {
    "pair_query": "pair_query.md",
    "pair_judge": "pair_judge.md",
}
DEFAULT_HUB_REPOS: dict[PromptRole, str] = {
    "pair_query": "pair-rrf-query",
    "pair_judge": "pair-rrf-judge",
}
_ENV_KEYS: dict[PromptRole, str] = {
    "pair_query": "PAIR_RRF_PROMPT_QUERY",
    "pair_judge": "PAIR_RRF_PROMPT_JUDGE",
}

ALL_ROLES: tuple[PromptRole, ...] = ("pair_query", "pair_judge")


@dataclass(frozen=True)
class PromptRef:
    """Provenance for one loaded prompt — recorded in every batch manifest."""

    role: PromptRole
    identifier: str
    commit_hash: str | None = None
    owner: str | None = None
    repo: str | None = None
    tag_or_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def version_label(self) -> str:
        name = self.repo or self.identifier
        if self.commit_hash:
            return f"hub:{name}:{self.commit_hash[:12]}"
        return f"hub:{self.identifier}"


@dataclass
class LoadedPrompt:
    text: str
    ref: PromptRef


def _api_key() -> str:
    return (
        os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or ""
    ).strip()


def resolve_prompt_identifier(role: PromptRole) -> str | None:
    """Return ``owner/repo:tag`` from env, or None when nothing is configured."""
    explicit = (os.getenv(_ENV_KEYS[role]) or "").strip()
    if explicit:
        return explicit
    owner = (os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip()
    if not owner:
        return None
    tag = (os.getenv("PAIR_RRF_PROMPT_TAG") or "latest").strip() or "latest"
    return f"{owner}/{DEFAULT_HUB_REPOS[role]}:{tag}"


def _parse_identifier(identifier: str) -> tuple[str | None, str | None, str | None]:
    tag: str | None = None
    body = identifier
    if ":" in identifier:
        body, tag = identifier.rsplit(":", 1)
        tag = tag.strip() or None
    owner: str | None = None
    repo = body
    if "/" in body:
        owner, repo = body.split("/", 1)
        owner = owner.strip() or None
        repo = repo.strip()
    return owner, repo or None, tag


def _extract_system_text(prompt: Any) -> str:
    messages = getattr(prompt, "messages", None)
    if messages:
        for msg in messages:
            inner = getattr(msg, "prompt", None)
            template = getattr(inner, "template", None) if inner is not None else None
            if template is None:
                template = getattr(msg, "template", None)
            if template:
                return str(template)
    template = getattr(prompt, "template", None)
    if isinstance(template, str) and template.strip():
        return template
    raise ValueError(
        f"Unsupported prompt object {type(prompt)!r}; expected a ChatPromptTemplate"
    )


@lru_cache(maxsize=8)
def load_prompt_cached(role: PromptRole) -> LoadedPrompt:
    identifier = resolve_prompt_identifier(role)
    if not identifier:
        raise RuntimeError(
            f"No LangSmith Hub identifier for prompt role {role!r}. Set "
            f"{_ENV_KEYS[role]} or LANGSMITH_PROMPT_OWNER in .env. This pipeline "
            "has no local-file fallback by design — see prompt_hub.py."
        )
    if not _api_key():
        raise RuntimeError(
            "LANGCHAIN_API_KEY / LANGSMITH_API_KEY is not set, so the prompt for "
            f"{role!r} ({identifier}) cannot be pulled. Refusing to run rather "
            "than fall back to un-audited local text."
        )

    from langsmith import Client

    owner_hint, repo_hint, tag = _parse_identifier(identifier)
    kwargs: dict[str, Any] = {}
    if "/" in identifier.split(":", 1)[0]:
        kwargs["dangerously_pull_public_prompt"] = True

    prompt = Client().pull_prompt(identifier, **kwargs)
    text = _extract_system_text(prompt)
    meta = getattr(prompt, "metadata", None) or {}
    commit = meta.get("lc_hub_commit_hash")
    return LoadedPrompt(
        text=text,
        ref=PromptRef(
            role=role,
            identifier=identifier,
            commit_hash=str(commit) if commit else None,
            owner=str(meta.get("lc_hub_owner") or owner_hint or "") or None,
            repo=str(meta.get("lc_hub_repo") or repo_hint or "") or None,
            tag_or_version=tag,
        ),
    )


def load_prompt(role: PromptRole, **format_vars: Any) -> LoadedPrompt:
    """Pull from hub, then ``str.format`` with ``format_vars`` if any are given.

    Hub round-trips escape literal braces as ``{{``/``}}`` (the push side
    f-string-escapes them). When no format vars are supplied we unescape so the
    JSON contract in the prompt body renders as the model should see it.
    """
    loaded = load_prompt_cached(role)
    if not format_vars:
        return LoadedPrompt(
            text=loaded.text.replace("{{", "{").replace("}}", "}"), ref=loaded.ref
        )
    return LoadedPrompt(text=loaded.text.format(**format_vars), ref=loaded.ref)


def clear_prompt_cache() -> None:
    load_prompt_cached.cache_clear()
