"""Load generate/judge prompts from LangSmith Hub with local .md fallback.

Prefer hub when identifiers are configured and LANGCHAIN_API_KEY (or
LANGSMITH_API_KEY) is present. On pull failure, log and fall back to
``synth_pipeline/prompts/*.md``.

Hub pulls attach ``lc_hub_*`` metadata on the template; we surface those as
``prompt_refs`` for staging envelopes and LangSmith run metadata.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from synth_pipeline.config import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent

PromptRole = Literal["generate_pos", "generate_neg", "judge"]

# Local filenames + default hub repo names (under LANGSMITH_PROMPT_OWNER).
_LOCAL_FILES: dict[PromptRole, str] = {
    "generate_pos": "generate_pos.md",
    "generate_neg": "generate_neg.md",
    "judge": "judge.md",
}
_DEFAULT_HUB_REPOS: dict[PromptRole, str] = {
    "generate_pos": "synth-generate-pos",
    "generate_neg": "synth-generate-neg",
    "judge": "synth-judge",
}
_ENV_KEYS: dict[PromptRole, str] = {
    "generate_pos": "SYNTH_PROMPT_GENERATE_POS",
    "generate_neg": "SYNTH_PROMPT_GENERATE_NEG",
    "judge": "SYNTH_PROMPT_JUDGE",
}


@dataclass(frozen=True)
class PromptRef:
    """Provenance for one loaded prompt."""

    role: PromptRole
    source: Literal["hub", "local"]
    identifier: str | None = None
    commit_hash: str | None = None
    owner: str | None = None
    repo: str | None = None
    local_file: str | None = None
    tag_or_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def version_label(self) -> str:
        if self.source == "hub" and self.commit_hash:
            short = self.commit_hash[:12]
            name = self.repo or self.identifier or self.role
            return f"hub:{name}:{short}"
        if self.source == "hub" and self.identifier:
            return f"hub:{self.identifier}"
        return f"local:{self.local_file or self.role}"


@dataclass
class LoadedPrompt:
    text: str
    ref: PromptRef
    # Optional pulled template (for invoke/metadata); None when local fallback.
    template: Any | None = field(default=None, repr=False)


def _api_key() -> str:
    return (
        os.getenv("LANGCHAIN_API_KEY")
        or os.getenv("LANGSMITH_API_KEY")
        or ""
    ).strip()


def hub_enabled() -> bool:
    """True when an API key is set and at least one prompt identifier resolves."""
    if not _api_key():
        return False
    return any(resolve_prompt_identifier(role) for role in _LOCAL_FILES)


def resolve_prompt_identifier(role: PromptRole) -> str | None:
    """Return hub id like ``owner/repo:tag`` or None if hub not configured for role."""
    explicit = (os.getenv(_ENV_KEYS[role]) or "").strip()
    if explicit:
        return explicit
    owner = (os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip()
    if not owner:
        return None
    tag = (os.getenv("LANGSMITH_PROMPT_TAG") or "latest").strip() or "latest"
    return f"{owner}/{_DEFAULT_HUB_REPOS[role]}:{tag}"


def _parse_identifier(identifier: str) -> tuple[str | None, str | None, str | None]:
    """Split ``owner/repo:tag`` → (owner, repo, tag_or_version)."""
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
    """Best-effort: system template string from ChatPromptTemplate / PromptTemplate."""
    # ChatPromptTemplate
    messages = getattr(prompt, "messages", None)
    if messages:
        for msg in messages:
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            # SystemMessagePromptTemplate
            inner = getattr(msg, "prompt", None)
            template = getattr(inner, "template", None) if inner is not None else None
            if template is None:
                template = getattr(msg, "template", None)
            msg_class = type(msg).__name__.lower()
            if template and (
                role in ("system", "System")
                or "system" in msg_class
                or len(messages) == 1
            ):
                return str(template)
        # Fallback: first message with a template
        for msg in messages:
            inner = getattr(msg, "prompt", None)
            template = getattr(inner, "template", None) if inner is not None else None
            if template:
                return str(template)

    template = getattr(prompt, "template", None)
    if isinstance(template, str) and template.strip():
        return template

    raise ValueError(
        f"Unsupported prompt object type {type(prompt)!r}; "
        "expected ChatPromptTemplate or PromptTemplate with a system template"
    )


def _load_local(role: PromptRole) -> LoadedPrompt:
    filename = _LOCAL_FILES[role]
    path = PROMPTS_DIR / filename
    text = path.read_text(encoding="utf-8")
    return LoadedPrompt(
        text=text,
        ref=PromptRef(
            role=role,
            source="local",
            local_file=filename,
        ),
    )


def _pull_hub(role: PromptRole, identifier: str) -> LoadedPrompt:
    from langsmith import Client

    owner_hint, repo_hint, tag = _parse_identifier(identifier)
    client = Client()
    # Owner/name pulls may be public or cross-org; allow when slash present.
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
            source="hub",
            identifier=identifier,
            commit_hash=str(commit) if commit else None,
            owner=str(owner) if owner else None,
            repo=str(repo) if repo else None,
            tag_or_version=tag,
        ),
        template=prompt,
    )


@lru_cache(maxsize=16)
def load_prompt_cached(role: PromptRole) -> LoadedPrompt:
    """Load prompt for role (hub preferred, local fallback). Cached per process."""
    identifier = resolve_prompt_identifier(role)
    if identifier and _api_key():
        try:
            loaded = _pull_hub(role, identifier)
            logger.info(
                "Loaded prompt %s from LangSmith Hub (%s commit=%s)",
                role,
                identifier,
                (loaded.ref.commit_hash or "?")[:12],
            )
            return loaded
        except Exception as exc:  # noqa: BLE001 — fallback is intentional
            logger.warning(
                "Hub pull failed for %s (%s): %s — using local %s",
                role,
                identifier,
                exc,
                _LOCAL_FILES[role],
            )
    elif identifier and not _api_key():
        logger.warning(
            "Prompt identifier set for %s (%s) but no LANGCHAIN_API_KEY/"
            "LANGSMITH_API_KEY — using local %s",
            role,
            identifier,
            _LOCAL_FILES[role],
        )
    else:
        logger.debug("No hub identifier for %s — using local file", role)

    return _load_local(role)


def _render_text(
    text: str,
    *,
    format_vars: dict[str, Any] | None,
    from_hub: bool,
) -> str:
    """Apply format vars; unescape hub f-string literals (``{{`` → ``{``)."""
    if format_vars:
        return text.format(**format_vars)
    if from_hub:
        return text.replace("{{", "{").replace("}}", "}")
    return text


def load_prompt(
    role: PromptRole,
    *,
    format_vars: dict[str, Any] | None = None,
) -> LoadedPrompt:
    """Load prompt text (+ ref). Optionally ``str.format`` with format_vars."""
    loaded = load_prompt_cached(role)
    text = _render_text(
        loaded.text,
        format_vars=format_vars,
        from_hub=loaded.ref.source == "hub",
    )
    if text == loaded.text:
        return loaded
    return LoadedPrompt(text=text, ref=loaded.ref, template=loaded.template)


def load_prompt_text(name: str) -> str:
    """Back-compat shim for ``llm.load_prompt('generate_pos.md')``."""
    stem = name.removesuffix(".md")
    if stem not in _LOCAL_FILES:
        path = PROMPTS_DIR / name
        return path.read_text(encoding="utf-8")
    return load_prompt(stem).text  # type: ignore[arg-type]


def prompt_version_label(refs: dict[str, PromptRef | dict[str, Any]]) -> str:
    """Compact version string for metadata / manifest."""
    parts: list[str] = []
    for key in sorted(refs):
        ref = refs[key]
        if isinstance(ref, PromptRef):
            parts.append(f"{key}={ref.version_label}")
        elif isinstance(ref, dict):
            source = ref.get("source", "?")
            commit = ref.get("commit_hash")
            if source == "hub" and commit:
                parts.append(f"{key}=hub:{(ref.get('repo') or key)}:{str(commit)[:12]}")
            elif source == "hub":
                parts.append(f"{key}=hub:{ref.get('identifier') or key}")
            else:
                parts.append(f"{key}=local:{ref.get('local_file') or key}")
    return ";".join(parts) if parts else "local:v1"


def clear_prompt_cache() -> None:
    load_prompt_cached.cache_clear()


def hub_run_metadata(ref: PromptRef) -> dict[str, Any]:
    """Metadata dict to attach on LangSmith runs for prompt version visibility."""
    meta: dict[str, Any] = {
        "prompt_role": ref.role,
        "prompt_source": ref.source,
        "prompt_version": ref.version_label,
    }
    if ref.identifier:
        meta["prompt_identifier"] = ref.identifier
    if ref.commit_hash:
        meta["lc_hub_commit_hash"] = ref.commit_hash
    if ref.owner:
        meta["lc_hub_owner"] = ref.owner
    if ref.repo:
        meta["lc_hub_repo"] = ref.repo
    return meta
