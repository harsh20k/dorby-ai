#!/usr/bin/env python3
"""Compute actual $ cost of a bedrock_profile_gen.py run from its manifest.jsonl.

Bedrock's Converse API returns real input/output token counts with every
response; bedrock_profile_gen.py records them for both profile-generation
calls and style/archetype refresh calls (kind: "profile" / "style_refresh" /
"archetypes_refresh"). This sums them and applies a per-1M-token price.

Usage:
    python scripts/estimate_bedrock_cost.py artifacts/bedrock_synth/run_20260723_093500
    python scripts/estimate_bedrock_cost.py <run_dir> --price-in 0.62 --price-out 1.85
"""
import argparse
import json
from pathlib import Path

# $ per 1M tokens, as confirmed live against Bedrock (see
# docs/profile-generation-local-and-bedrock.md for the structured-output
# support table this is paired with). Update if AWS repricing occurs.
KNOWN_PRICING = {
    "google.gemma-3-27b-it": (0.50, 0.50),
    "deepseek.v3.2": (0.62, 1.85),
    "zai.glm-4.7-flash": (0.07, 0.40),
    "nvidia.nemotron-nano-9b-v2": (0.06, 0.24),  # output price unconfirmed, placeholder
    "mistral.voxtral-mini-3b-2507": (0.04, 0.10),  # placeholder, unconfirmed
    "mistral.ministral-3-3b-instruct": (0.04, 0.10),  # placeholder, unconfirmed
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.00, 5.00),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model-id", default=None, help="Look up price from KNOWN_PRICING; overridden by --price-in/--price-out if given")
    ap.add_argument("--price-in", type=float, default=None, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="$ per 1M output tokens")
    args = ap.parse_args()

    manifest_path = args.run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.jsonl found under {args.run_dir}")

    totals = {"profile": {"in": 0, "out": 0, "n": 0}, "refresh": {"in": 0, "out": 0, "n": 0}}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("kind", "profile")
        bucket = "profile" if kind == "profile" else "refresh"
        totals[bucket]["in"] += rec.get("input_tokens", 0)
        totals[bucket]["out"] += rec.get("output_tokens", 0)
        totals[bucket]["n"] += 1

    total_in = totals["profile"]["in"] + totals["refresh"]["in"]
    total_out = totals["profile"]["out"] + totals["refresh"]["out"]

    print(f"Run: {args.run_dir}")
    print(f"  profile generation calls: {totals['profile']['n']:>6}  in={totals['profile']['in']:>9}  out={totals['profile']['out']:>9}")
    print(f"  style/archetype refreshes: {totals['refresh']['n']:>5}  in={totals['refresh']['in']:>9}  out={totals['refresh']['out']:>9}")
    print(f"  TOTAL                                  in={total_in:>9}  out={total_out:>9}")

    price_in, price_out = args.price_in, args.price_out
    if price_in is None or price_out is None:
        if args.model_id and args.model_id in KNOWN_PRICING:
            looked_up = KNOWN_PRICING[args.model_id]
            price_in = price_in if price_in is not None else looked_up[0]
            price_out = price_out if price_out is not None else looked_up[1]
        else:
            print("\nNo --price-in/--price-out given and --model-id not in KNOWN_PRICING "
                  f"({', '.join(KNOWN_PRICING)}) -- token totals printed above, cost not computed.")
            return

    cost = total_in / 1e6 * price_in + total_out / 1e6 * price_out
    print(f"\nPricing: ${price_in}/1M in, ${price_out}/1M out")
    print(f"Total cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
