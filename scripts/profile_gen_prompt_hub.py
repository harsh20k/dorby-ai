"""Load standalone-profile-generation prompts from LangSmith Hub — hub only, no
local-file fallback at runtime.

Unlike ``synth_pipeline/prompts/__init__.py`` (which prefers the hub but silently
falls back to a local ``.md`` on pull failure), this loader is intentionally strict:
the point of routing these prompts through the hub is that every generated profile
can be traced back to an exact, versioned, human-reviewable prompt commit. A silent
local fallback would defeat that — a run could "succeed" while quietly using
un-audited text. If the hub pull fails, generation should fail loudly.

The local ``scripts/prompts/profile_gen/*.md`` files are still the source of truth
you edit and commit — ``push_profile_gen_prompts.py`` is the only thing that reads
them directly. This module never reads them.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PROMPTS_DIR = REPO_ROOT / "scripts" / "prompts" / "profile_gen"

PromptRole = Literal["style_refresh", "archetype_refresh", "generate_profile"]

# Local filenames (pushed FROM, never pulled at runtime) + default hub repo names.
LOCAL_FILES: dict[PromptRole, str] = {
    "style_refresh": "style_refresh.md",
    "archetype_refresh": "archetype_refresh.md",
    "generate_profile": "generate_profile.md",
}
DEFAULT_HUB_REPOS: dict[PromptRole, str] = {
    "style_refresh": "profile-gen-style-refresh",
    "archetype_refresh": "profile-gen-archetype-refresh",
    "generate_profile": "profile-gen-generate",
}
# Explicit per-role override, e.g. PROFILE_GEN_PROMPT_GENERATE=myhandle/profile-gen-generate:v1
_ENV_KEYS: dict[PromptRole, str] = {
    "style_refresh": "PROFILE_GEN_PROMPT_STYLE_REFRESH",
    "archetype_refresh": "PROFILE_GEN_PROMPT_ARCHETYPE_REFRESH",
    "generate_profile": "PROFILE_GEN_PROMPT_GENERATE",
}


@dataclass(frozen=True)
class PromptRef:
    role: PromptRole
    identifier: str
    commit_hash: str | None
    owner: str | None
    repo: str | None
    tag_or_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def version_label(self) -> str:
        if self.commit_hash:
            return f"hub:{self.repo or self.identifier}:{self.commit_hash[:12]}"
        return f"hub:{self.identifier}"


@dataclass
class LoadedPrompt:
    text: str
    ref: PromptRef


def _api_key() -> str:
    return (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()


def resolve_prompt_identifier(role: PromptRole) -> str | None:
    """Return hub id like ``owner/repo:tag`` or None if hub not configured for role."""
    explicit = (os.getenv(_ENV_KEYS[role]) or "").strip()
    if explicit:
        return explicit
    owner = (os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip()
    if not owner:
        return None
    tag = (os.getenv("LANGSMITH_PROMPT_TAG") or "latest").strip() or "latest"
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
    raise ValueError(f"Unsupported prompt object type {type(prompt)!r} pulled from hub")


def _pull_hub(role: PromptRole, identifier: str) -> LoadedPrompt:
    from langsmith import Client

    owner_hint, repo_hint, tag = _parse_identifier(identifier)
    client = Client()
    kwargs: dict[str, Any] = {}
    if "/" in identifier.split(":", 1)[0]:
        kwargs["dangerously_pull_public_prompt"] = True
    prompt = client.pull_prompt(identifier, **kwargs)
    text = _extract_system_text(prompt)
    meta = getattr(prompt, "metadata", None) or {}
    commit = meta.get("lc_hub_commit_hash")
    owner = meta.get("lc_hub_owner") or owner_hint
    repo = meta.get("lc_hub_repo") or repo_hint
    return LoadedPrompt(
        text=text,
        ref=PromptRef(
            role=role,
            identifier=identifier,
            commit_hash=str(commit) if commit else None,
            owner=str(owner) if owner else None,
            repo=str(repo) if repo else None,
            tag_or_version=tag,
        ),
    )


@lru_cache(maxsize=8)
def load_prompt_cached(role: PromptRole) -> LoadedPrompt:
    """Pull from LangSmith Hub. Raises RuntimeError if hub is not configured/reachable
    — there is deliberately no local-file fallback, see module docstring."""
    identifier = resolve_prompt_identifier(role)
    if not identifier:
        raise RuntimeError(
            f"No hub identifier configured for prompt role {role!r}. Set "
            f"{_ENV_KEYS[role]}=<owner>/{DEFAULT_HUB_REPOS[role]}:<tag> or "
            "LANGSMITH_PROMPT_OWNER (+ optional LANGSMITH_PROMPT_TAG). "
            "This loader is hub-only by design; run push_profile_gen_prompts.py first."
        )
    if not _api_key():
        raise RuntimeError(
            f"Hub identifier {identifier!r} set for {role!r} but no "
            "LANGCHAIN_API_KEY/LANGSMITH_API_KEY is present."
        )
    try:
        loaded = _pull_hub(role, identifier)
    except Exception as exc:  # noqa: BLE001 — re-raise with context, no fallback
        raise RuntimeError(
            f"Hub pull failed for prompt role {role!r} ({identifier!r}): {exc}"
        ) from exc
    return loaded


def load_prompt(role: PromptRole, **format_vars: Any) -> LoadedPrompt:
    """Load + render a hub prompt. format_vars are applied with str.format(), which
    also un-escapes the hub's literal ``{{``/``}}`` braces back to ``{``/``}``."""
    loaded = load_prompt_cached(role)
    text = loaded.text.format(**format_vars)
    return LoadedPrompt(text=text, ref=loaded.ref)


def clear_prompt_cache() -> None:
    load_prompt_cached.cache_clear()
