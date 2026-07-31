"""LangSmith Prompt Hub I/O for this experiment.

Every prompt this experiment produces or uses is kept **both** locally and
in the Hub:

- The fixed meta-optimizer instructions: local source of truth is
  ``prompts/meta_optimizer.md``; ``push_meta_prompt`` pushes it as a single
  named prompt (``<hub_repo>-meta``).
- Each iteration's *evolved judge prompt*: saved locally by ``run.py``
  (always, regardless of Hub availability) and pushed here as one commit per
  iteration under ``<hub_repo>``, tagged ``<run_id>`` and ``iter-NN`` — so the
  whole 20-round evolution is a browsable commit history in LangSmith, not
  just files on disk.

Hub pushes are best-effort: a missing/invalid API key logs a warning and the
run continues on local files alone, since the experiment's value (the local
evolution trail) must not depend on Hub availability.
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


def push_meta_prompt(*, hub_owner: str | None, repo: str) -> str | None:
    from judge_prompt_evolution.config import REPO_ROOT

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


def push_seed_prompt(
    *, hub_owner: str | None, repo: str,
    text: str | None = None, description: str | None = None, run_id: str | None = None,
) -> str | None:
    """Push iteration 0. Defaults to the naive prompt for backward compat;
    pass ``text``/``description`` explicitly for a different seed source
    (e.g. structured_cot)."""
    if text is None:
        from judge_prompt_evolution.seed_prompt import SEED_JUDGE_PROMPT

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
    *, text: str, run_id: str, iteration: int, hub_owner: str | None, repo: str
) -> str | None:
    # A LangSmith tag can only ever point at one commit, so reusing the bare
    # run_id (or a bare "iter-NN") as a tag across 20 commits 409s from the
    # second push onward. Each iteration gets one tag unique to itself; the
    # commit history (traversable via the Hub UI) is what carries the
    # run/iteration provenance, backed up by the description string too.
    return push_prompt(
        text=text,
        repo=repo,
        hub_owner=hub_owner,
        description=f"judge_prompt_evolution run={run_id} iteration={iteration}",
        commit_tags=[f"{run_id}--iter-{iteration:02d}"],
    )
