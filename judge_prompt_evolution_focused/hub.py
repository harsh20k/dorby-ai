"""LangSmith Prompt Hub I/O for this experiment.

Every prompt this experiment produces or uses is pushed to, and — for the two
*fixed instruction* prompts — loaded back from, the Hub:

- The meta-optimizer instructions: local source of truth is
  ``prompts/meta_optimizer.md``; ``push_meta_prompt`` pushes it as
  ``<hub_repo>-meta``. Loaded at call time via ``pull_prompt`` in
  ``optimizer.py``, local file as fallback only if the Hub is unreachable.
- The summarizer instructions: local source of truth is
  ``prompts/summarizer.md``; ``push_summarizer_prompt`` pushes it as
  ``<hub_repo>-summarizer``. Same pull-first loading.
- The seed judge prompt (naive or structured_cot) and every iteration's
  *evolved* judge prompt: saved locally by ``run.py`` (always, regardless of
  Hub availability) and pushed here as one commit per iteration under
  ``<hub_repo>``, tagged ``<run_id>`` and ``iter-NN`` — so the whole 20-round
  evolution is a browsable commit history in LangSmith, not just files on
  disk. These are per-run artifacts, not reloaded at call time (the run that
  produced them already has them in memory).

Hub I/O is best-effort throughout: a missing/invalid API key or a failed pull
logs a warning and falls back to the local file, since the experiment's
value (the local evolution trail) must not depend on Hub availability.
"""

from __future__ import annotations

import os
from typing import Any


def _api_key() -> str:
    return (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()


def _escape_braces(text: str) -> str:
    """JSON examples in these prompts contain literal {..} — escape so
    ChatPromptTemplate doesn't treat them as template variables."""
    return text.replace("{", "{{").replace("}", "}}")


def _identifier(hub_owner: str | None, repo: str) -> str:
    return f"{hub_owner}/{repo}" if hub_owner else repo


def push_prompt(
    *,
    text: str,
    repo: str,
    hub_owner: str | None,
    description: str,
    commit_tags: list[str] | None = None,
) -> str | None:
    """Push one system-prompt string as a new commit. Returns the Hub URL, or
    None if the push was skipped/failed (never raises — best-effort)."""
    if not _api_key():
        print(f"[hub] LANGCHAIN_API_KEY/LANGSMITH_API_KEY not set — skipping push of {repo!r}")
        return None
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langsmith import Client

        template = ChatPromptTemplate.from_messages([("system", _escape_braces(text))])
        identifier = _identifier(hub_owner, repo)
        kwargs: dict[str, Any] = {"object": template, "description": description}
        if commit_tags:
            kwargs["commit_tags"] = commit_tags
        url = Client().push_prompt(identifier, **kwargs)
        print(f"[hub] pushed {repo!r} -> {url}")
        return url
    except Exception as exc:  # noqa: BLE001 — never let Hub issues kill the run
        print(f"[hub] push of {repo!r} failed ({type(exc).__name__}: {exc}) — continuing locally")
        return None


def _extract_system_text(prompt: Any) -> str:
    messages = getattr(prompt, "messages", None)
    if messages:
        for msg in messages:
            inner = getattr(msg, "prompt", None)
            template = getattr(inner, "template", None) if inner is not None else None
            if template is None:
                template = getattr(msg, "template", None)
            if template:
                return str(template).replace("{{", "{").replace("}}", "}")
    template = getattr(prompt, "template", None)
    if isinstance(template, str) and template.strip():
        return template
    raise ValueError(f"Unsupported prompt object {type(prompt)!r}; expected a ChatPromptTemplate")


def pull_prompt(*, repo: str, hub_owner: str | None, tag: str | None = None) -> str | None:
    """Pull the latest (or ``tag``-pinned) commit of a named prompt.
    Returns None on any failure — best-effort, never raises."""
    if not _api_key():
        return None
    try:
        from langsmith import Client

        identifier = _identifier(hub_owner, repo)
        if tag:
            identifier = f"{identifier}:{tag}"
        prompt = Client().pull_prompt(identifier)
        return _extract_system_text(prompt)
    except Exception as exc:  # noqa: BLE001 — best-effort, fall back to local
        print(f"[hub] pull of {repo!r} failed ({type(exc).__name__}: {exc}) — using local file")
        return None


def push_meta_prompt(*, hub_owner: str | None, repo: str) -> str | None:
    from judge_prompt_evolution_focused.config import REPO_ROOT

    text = (REPO_ROOT / "judge_prompt_evolution" / "prompts" / "meta_optimizer.md").read_text(
        encoding="utf-8"
    )
    return push_prompt(
        text=text,
        repo=f"{repo}-meta",
        hub_owner=hub_owner,
        description="judge_prompt_evolution: fixed instructions given to the optimizer LLM each round "
        "(v2: dropped hard/easy-negative framing, pushed the optimizer toward generalized rubric "
        "revision instead of incremental example-specific rules)",
        commit_tags=["v2"],
    )


_SUMMARIZER_FILES = {"aggressive": "summarizer.md", "gentle": "summarizer_gentle.md"}
_SUMMARIZER_DESCRIPTIONS = {
    "aggressive": "judge_prompt_evolution: distillation instructions used every N rounds (evo_004) "
    "to merge/cut the accumulated rubric, explicitly pushing toward shortness — separate from the "
    "meta-optimizer prompt, no example batch, just compression",
    "gentle": "judge_prompt_evolution: distillation instructions (evo_006+) — same clarify/generalize "
    "goal as the aggressive variant, but explicitly told length is not the objective, only merge "
    "genuinely repetitive wording; a result close to its starting size is fine",
}


def push_summarizer_prompt(*, hub_owner: str | None, repo: str, variant: str = "aggressive") -> str | None:
    from judge_prompt_evolution_focused.config import REPO_ROOT

    filename = _SUMMARIZER_FILES[variant]
    text = (REPO_ROOT / "judge_prompt_evolution" / "prompts" / filename).read_text(encoding="utf-8")
    suffix = "-summarizer" if variant == "aggressive" else f"-summarizer-{variant}"
    return push_prompt(
        text=text,
        repo=f"{repo}{suffix}",
        hub_owner=hub_owner,
        description=_SUMMARIZER_DESCRIPTIONS[variant],
    )


def push_seed_prompt(
    *, hub_owner: str | None, repo: str,
    text: str | None = None, description: str | None = None, run_id: str | None = None,
) -> str | None:
    """Push iteration 0. Defaults to the naive prompt for backward compat;
    pass ``text``/``description`` explicitly for a different seed source
    (e.g. structured_cot)."""
    if text is None:
        from judge_prompt_evolution_focused.seed_prompt import SEED_JUDGE_PROMPT

        text = SEED_JUDGE_PROMPT
        description = description or (
            "judge_prompt_evolution: iteration 0 (unmodified naive judge prompt, "
            "AUC 0.6177 on all-200 real pairs)"
        )
    tags = ["iter-00", "seed"]
    if run_id:
        tags.append(f"{run_id}--seed")
    return push_prompt(
        text=text,
        repo=repo,
        hub_owner=hub_owner,
        description=description or "judge_prompt_evolution: iteration 0",
        commit_tags=tags,
    )


def push_iteration_prompt(
    *, text: str, run_id: str, iteration: int, hub_owner: str | None, repo: str, kind: str = "optimize"
) -> str | None:
    # A LangSmith tag can only ever point at one commit, so reusing the bare
    # run_id (or a bare "iter-NN") as a tag across 20 commits 409s from the
    # second push onward. Each iteration gets one tag unique to itself; the
    # commit history (traversable via the Hub UI) is what carries the
    # run/iteration provenance, backed up by the description string too.
    # A "summarize" step shares its iteration number with the preceding
    # "optimize" step, so it needs a distinct suffix to avoid the same
    # tag-collision problem.
    suffix = "s" if kind == "summarize" else ""
    return push_prompt(
        text=text,
        repo=repo,
        hub_owner=hub_owner,
        description=f"judge_prompt_evolution run={run_id} iteration={iteration} kind={kind}",
        commit_tags=[f"{run_id}--iter-{iteration:02d}{suffix}"],
    )
