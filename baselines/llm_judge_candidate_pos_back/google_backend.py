"""Direct Google Generative Language API backend (no OpenRouter proxy).

A same-prompt, same-model, different-transport check against the OpenRouter
run: OpenRouter forwards ``google/gemini-3.1-flash-lite`` to Google's own
API, so this calls Google directly (``GEMINI_API_KEY``) to sanity-check that
OpenRouter passed the request through faithfully rather than silently
altering it (routing to a different backend, adding hidden system text,
etc.). Plain REST via ``requests`` (already a project dependency), not the
``google-generativeai``/``google-genai`` SDKs — neither is installed and a
single-endpoint JSON POST doesn't need one, matching how
``bedrock_backend.py`` calls boto3's Converse API directly rather than
wrapping it further.
"""

from __future__ import annotations

import json
from typing import Any

import requests

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def openrouter_to_google_model(model: str) -> str:
    """``google/gemini-3.1-flash-lite`` (OpenRouter id) -> ``gemini-3.1-flash-lite``."""
    return model.split("/", 1)[1] if "/" in model else model


def call_google_verdict(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """One call to Gemini's generateContent endpoint, forcing JSON output."""
    model_id = openrouter_to_google_model(model)
    url = f"{base_url}/models/{model_id}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"no candidates in Google API response: {data!r}")
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        finish_reason = candidates[0].get("finishReason")
        raise ValueError(
            f"no content parts in Google API response (finishReason={finish_reason!r}): {data!r}"
        )
    text = parts[0].get("text", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Google API did not return valid JSON: {text!r}") from exc
