"""AWS Bedrock call path for the LLM-judge experiment.

Mirrors ``scripts/bedrock_profile_gen.py::call_bedrock`` (Converse API,
structured JSON-schema output where supported) rather than importing it — that
script lives under ``scripts/`` and isn't an importable package, and
``synth_pipeline/pairing/bedrock.py`` already made the same call for the same
reason (see its docstring / CLAUDE.md's "Pairing standalone profiles" note).

That script's docs/profile-generation-local-and-bedrock.md found Llama 3.3 70B
and every Nova variant reject Bedrock's native structured-output enforcement
outright (``ValidationException``). Rather than hardcode another exclusion
list, ``call_bedrock_verdict`` tries structured output first and falls back to
a plain-JSON-in-the-prompt request (parsed the same way the OpenRouter path
already parses free-text JSON) on that specific error, so an untested model
degrades gracefully instead of failing outright.
"""

from __future__ import annotations

import json
from typing import Any

from synth_pipeline.llm import parse_json_object

DEFAULT_REGION = "us-east-1"
DEFAULT_PROFILE = "tf_provisioner"

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "match": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["reasoning", "match", "confidence"],
    "additionalProperties": False,
}


def make_client(*, profile: str | None, region: str):
    import boto3

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("bedrock-runtime", region_name=region)


def _structured_output_unsupported(exc: Exception) -> bool:
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code", "")
    return code in {"ValidationException"}


def call_bedrock_verdict(
    client: Any,
    *,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """One Converse call. Returns the parsed JSON object (unvalidated)."""
    messages = [{"role": "user", "content": [{"text": user}]}]
    system_blocks = [{"text": system}]
    inference_config = {"maxTokens": max_tokens, "temperature": temperature}

    try:
        resp = client.converse(
            modelId=model_id,
            messages=messages,
            system=system_blocks,
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
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001 — narrowed below, re-raised otherwise
        if not _structured_output_unsupported(exc):
            raise

    # Fallback: this model rejects structured-output enforcement. Ask for
    # JSON in plain text instead, same as the OpenRouter path, and parse
    # leniently (handles code fences / surrounding prose).
    plain_system = (
        system
        + "\n\nRespond with only a single JSON object matching this shape, "
        'no other text: {"reasoning": string, "match": "yes"|"no", "confidence": integer 0-100}'
    )
    resp = client.converse(
        modelId=model_id,
        messages=messages,
        system=[{"text": plain_system}],
        inferenceConfig=inference_config,
    )
    text = resp["output"]["message"]["content"][0]["text"]
    return parse_json_object(text)
