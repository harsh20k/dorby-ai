"""Minimal Bedrock Converse client with native JSON-schema structured output.

Deliberately duplicated from `scripts/bedrock_profile_gen.py::call_bedrock` rather
than imported: `scripts/` is not an importable package, and a package importing
from it would need a sys.path hack. The call is small and the two copies have
different concerns (the generator's version carries runner-specific retry/logging).

Model support caveat, from live testing recorded in
docs/profile-generation-local-and-bedrock.md: Llama 3.3 70B and every Nova variant
reject `outputConfig` outright. Gemma 3 27B, Mistral, DeepSeek, and Claude 4.5+ work.
"""

from __future__ import annotations

import json
import time
from typing import Any

DEFAULT_MODEL_ID = "google.gemma-3-27b-it"
DEFAULT_REGION = "us-east-1"


def make_client(region: str = DEFAULT_REGION):
    import boto3

    return boto3.client("bedrock-runtime", region_name=region)


def call_json(
    client,
    *,
    model_id: str,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    schema_description: str = "",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Return (parsed_json, usage). Retries on throttling and unparseable output."""
    from botocore.exceptions import ClientError

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
                outputConfig={
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                "schema": json.dumps(schema),
                                "name": schema_name,
                                "description": schema_description,
                            }
                        },
                    }
                },
            )
        except ClientError as e:
            last_err = e
            time.sleep(min(2**attempt, 20))
            continue

        usage = resp.get("usage", {}) or {}
        text = resp["output"]["message"]["content"][-1].get("text", "")
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as e:
            last_err = e

    raise RuntimeError(f"bedrock call failed after {max_retries} attempts: {last_err}")
