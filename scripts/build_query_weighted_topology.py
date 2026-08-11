"""Build the query-weighted topology HTML artifact by inlining graph_data.json
into docs/html/query-weighted-topology.html's template placeholder.

Not part of any experiment package — a one-off publishing script, like
scripts/build_real_pairs_graph.py.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path("artifacts/twotower_query_weighted/graph_data.json")
TEMPLATE_PATH = Path("docs/html/_query_weighted_topology_template.html")
OUT_PATH = Path("docs/html/query-weighted-topology.html")


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    template = TEMPLATE_PATH.read_text()
    out = template.replace("__GRAPH_DATA__", json.dumps(data))
    OUT_PATH.write_text(out)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
