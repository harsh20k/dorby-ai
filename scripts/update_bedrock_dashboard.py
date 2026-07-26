"""(Re)generate the `dorby-bedrock-profile-gen` CloudWatch dashboard.

This dashboard was originally created directly against the AWS console/CLI
for a single model (google.gemma-3-27b-it) — see "Cost tracking" in
docs/profile-generation-local-and-bedrock.md, which notes there is no IaC for
AWS infra in this repo. As more Bedrock models get used (the LLM-judge
experiment in baselines/llm_judge/ added a second, qwen.qwen3-32b-v1:0), a
one-off manual edit for each new model doesn't scale — this script exists so
adding model #3 is a `MODEL_PRICING` dict entry and a rerun, not another
from-scratch console edit.

Token usage, invocation count/errors/throttles, and latency widgets use
CloudWatch `SEARCH()` expressions scoped to the `AWS/Bedrock` namespace's
`ModelId` dimension, so they automatically pick up any model this AWS
account calls via Bedrock — no dashboard edit needed when a new model is
tried. Only the cost-estimate widget is a manual registry: CloudWatch has no
built-in per-model $/token pricing, so an accurate estimate needs
`MODEL_PRICING` kept in sync with real pricing (docs/llm-judge-experiment.md
and profile-generation-local-and-bedrock.md's own manual cost-tracking note
mention drift here as a known reality, not a bug).

Usage:
    python scripts/update_bedrock_dashboard.py --dry-run   # print, don't push
    python scripts/update_bedrock_dashboard.py             # push to CloudWatch
"""

from __future__ import annotations

import argparse
import json

DASHBOARD_NAME = "dorby-bedrock-profile-gen"
REGION = "us-east-1"

# model_id -> (price_in_per_million_tokens, price_out_per_million_tokens, display_label)
# Source: AWS Bedrock pricing page (us-east-1, on-demand), current as of the
# commit that last touched this file. Keep in sync when re-pricing or adding
# a model, same discipline as scripts/estimate_bedrock_cost.py's --price-in/out.
MODEL_PRICING: dict[str, tuple[float, float, str]] = {
    "google.gemma-3-27b-it": (0.23, 0.38, "Gemma 3 27B"),
    "qwen.qwen3-32b-v1:0": (0.15, 1.20, "Qwen3-32B"),
}


def _search_expr(metric_name: str, stat: str, period: int) -> str:
    return f"SEARCH('{{AWS/Bedrock,ModelId}} MetricName=\"{metric_name}\"', '{stat}', {period})"


def _text_widget(x: int, y: int, w: int, h: int, markdown: str) -> dict:
    return {"type": "text", "x": x, "y": y, "width": w, "height": h, "properties": {"markdown": markdown}}


def _metric_widget(x: int, y: int, w: int, h: int, title: str, metrics: list, *, period: int = 300, view: str = "timeSeries") -> dict:
    return {
        "type": "metric",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "properties": {"title": title, "region": REGION, "view": view, "period": period, "metrics": metrics},
    }


def build_dashboard(model_pricing: dict[str, tuple[float, float, str]]) -> dict:
    models_label = ", ".join(label for _, _, label in model_pricing.values()) or "no priced models yet"
    widgets = [
        _text_widget(
            0, 0, 24, 1,
            f"# Bedrock usage — dorby-ai (all models; priced: {models_label})",
        ),
        _metric_widget(
            0, 1, 12, 6,
            "Token usage by model (per period)",
            [
                [{"expression": _search_expr("InputTokenCount", "Sum", 300), "id": "in_by_model", "label": "Input tokens"}],
                [{"expression": _search_expr("OutputTokenCount", "Sum", 300), "id": "out_by_model", "label": "Output tokens"}],
            ],
        ),
        _metric_widget(
            12, 1, 12, 6,
            "Total tokens, all models combined (per period)",
            [
                [{"expression": f"SUM({_search_expr('InputTokenCount', 'Sum', 300)})", "id": "total_in", "label": "Total input tokens"}],
                [{"expression": f"SUM({_search_expr('OutputTokenCount', 'Sum', 300)})", "id": "total_out", "label": "Total output tokens"}],
            ],
        ),
        _metric_widget(
            0, 7, 12, 6,
            "Invocations / errors / throttles by model",
            [
                [{"expression": _search_expr("Invocations", "Sum", 300), "id": "inv_by_model", "label": "Invocations"}],
                [{"expression": _search_expr("InvocationThrottles", "Sum", 300), "id": "throttle_by_model", "label": "Throttles"}],
                [{"expression": _search_expr("InvocationClientErrors", "Sum", 300), "id": "cerr_by_model", "label": "Client errors"}],
                [{"expression": _search_expr("InvocationServerErrors", "Sum", 300), "id": "serr_by_model", "label": "Server errors"}],
            ],
        ),
        _metric_widget(
            12, 7, 12, 6,
            "Invocation latency by model (ms)",
            [
                [{"expression": _search_expr("InvocationLatency", "Average", 300), "id": "lat_avg_by_model", "label": "Avg latency"}],
                [{"expression": _search_expr("InvocationLatency", "p99", 300), "id": "lat_p99_by_model", "label": "p99 latency"}],
            ],
        ),
    ]

    if model_pricing:
        cost_metrics = []
        for i, (model_id, (price_in, price_out, label)) in enumerate(model_pricing.items()):
            in_id, out_id, cost_id = f"m{i}in", f"m{i}out", f"e{i}cost"
            cost_metrics.append(
                ["AWS/Bedrock", "InputTokenCount", "ModelId", model_id, {"stat": "Sum", "id": in_id, "visible": False}]
            )
            cost_metrics.append(
                ["AWS/Bedrock", "OutputTokenCount", "ModelId", model_id, {"stat": "Sum", "id": out_id, "visible": False}]
            )
            cost_metrics.append(
                [
                    {
                        "expression": f"({in_id}/1000000*{price_in}) + ({out_id}/1000000*{price_out})",
                        "id": cost_id,
                        "label": f"Est. $/hr — {label}",
                    }
                ]
            )
        widgets.append(
            _metric_widget(
                0, 13, 24, 6,
                "Estimated cost per hour, per priced model (see MODEL_PRICING in scripts/update_bedrock_dashboard.py)",
                cost_metrics,
                period=3600,
            )
        )
    else:
        widgets.append(
            _text_widget(
                0, 13, 24, 2,
                "_No models in MODEL_PRICING yet — add an entry to see cost estimates._",
            )
        )

    return {"widgets": widgets}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aws-profile", default="tf_provisioner")
    p.add_argument("--dashboard-name", default=DASHBOARD_NAME)
    p.add_argument("--dry-run", action="store_true", help="Print the dashboard JSON, don't push it.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    body = build_dashboard(MODEL_PRICING)

    if args.dry_run:
        print(json.dumps(body, indent=2))
        return 0

    import boto3

    session = boto3.Session(profile_name=args.aws_profile)
    client = session.client("cloudwatch", region_name=REGION)
    resp = client.put_dashboard(DashboardName=args.dashboard_name, DashboardBody=json.dumps(body))
    messages = resp.get("DashboardValidationMessages", [])
    if messages:
        print("validation messages:")
        for m in messages:
            print(f"  [{m.get('Severity')}] {m.get('Message')} ({m.get('DataPath')})")
    print(f"updated dashboard '{args.dashboard_name}' in {REGION} with {len(body['widgets'])} widgets, "
          f"{len(MODEL_PRICING)} priced model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
