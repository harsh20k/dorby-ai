"""Decompose a user's profile + one accepted/one declined real intro into
knowledge graphs (one LLM call per profile), merge them on shared concept
labels, add a type-taxonomy layer, and render a self-contained HTML graph
browser. See docs/knowledge-graph-experiment.md for the finding this
produced and how to read the output.

Usage:
    python scripts/build_kg_experiment.py --user-id cmoini4d90eyhlq02tdewefdp
    python scripts/build_kg_experiment.py --user-id <id> --cache artifacts/kg_experiment/cmoini4d9.json
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from synth_pipeline.config import REPO_ROOT, load_dotenv

MODEL = "google/gemini-3.1-flash-lite"

EXTRACT_PROMPT = """You are decomposing a professional networking profile into a knowledge graph, in the style used in social-network / relationship-recommendation research (entities + typed relations, e.g. Person -[works_in]-> Industry -[has_subcategory]-> Subcategory; Person -[seeks]-> NeedType; Person -[located_in]-> Location).

Given the profile text below, extract a small knowledge graph (max 14 nodes, max 16 edges) capturing: the person's role/company, industry + subcategories, what they are looking for (needs), their location, and any notable affiliations or traits. Use short node labels (2-5 words max).

Return ONLY valid JSON, no markdown fences, in this exact shape:
{
  "nodes": [{"id": "n1", "label": "Adrian", "type": "person"}, ...],
  "edges": [{"source": "n1", "target": "n2", "relation": "works_in"}, ...]
}

Node "type" must be one of: person, company, industry, subcategory, need, location, trait, affiliation.
The person node's id must be "root".

PROFILE:
{PROFILE_TEXT}
"""

TYPE_LABELS = {
    "company": "Company", "industry": "Industry", "subcategory": "Subcategory",
    "need": "Need", "location": "Location", "trait": "Trait", "affiliation": "Affiliation",
}


def profile_to_text(profile: dict) -> str:
    parts = []
    for key in ["positioning", "background", "lookingFor", "locationAvailability", "introPreferences", "notes"]:
        value = profile.get(key)
        if value:
            parts.append(f"### {key}\n{value}")
    return "\n\n".join(parts)


def find_pairs(data_dir: Path, user_id: str) -> tuple[dict, dict]:
    """Return (accepted_pair, declined_pair) touching user_id, from the real seed pairs."""
    pos = json.loads((data_dir / "dataset_positive.json").read_text())
    neg = json.loads((data_dir / "dataset_negative.json").read_text())
    accepted = next(p for p in pos if p["userContactId"] == user_id or p["matchContactId"] == user_id)
    declined = next(p for p in neg if p["userContactId"] == user_id or p["matchContactId"] == user_id)
    return accepted, declined


def other_side(pair: dict, user_id: str) -> tuple[str, dict]:
    """Return (contact_id, profile) of whichever side of the pair isn't user_id."""
    if pair["userContactId"] == user_id:
        return pair["matchContactId"], pair["matchContactFile"]
    return pair["userContactId"], pair["userContactFile"]


def call_extract(api_key: str, profile_text: str, label: str) -> dict:
    prompt = EXTRACT_PROMPT.replace("{PROFILE_TEXT}", profile_text[:6000])
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    content = result["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"model returned non-JSON for {label}: {content[:300]}")


def norm(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def merge_graphs(graphs: dict, roles: dict) -> dict:
    merged_nodes: dict[str, dict] = {}
    id_map: dict[tuple[str, str], str] = {}
    counter = [0]

    def new_key() -> str:
        counter[0] += 1
        return f"m{counter[0]}"

    merged_edges = []
    for gkey, g in graphs.items():
        for n in g["nodes"]:
            is_root = n["id"] == "root"
            if is_root:
                key = new_key()
                merged_nodes[key] = {
                    "key": key, "label": n["label"], "type": "person", "is_root": True,
                    "graphs": [gkey], "role": roles[gkey]["role"], "contact_id": roles[gkey]["contact_id"],
                }
                id_map[(gkey, n["id"])] = key
                continue
            nlabel = norm(n["label"])
            existing = next((k for k, r in merged_nodes.items() if not r["is_root"] and r.get("norm_label") == nlabel), None)
            if existing:
                merged_nodes[existing]["graphs"].append(gkey)
                id_map[(gkey, n["id"])] = existing
            else:
                key = new_key()
                merged_nodes[key] = {
                    "key": key, "label": n["label"], "norm_label": nlabel, "type": n["type"],
                    "is_root": False, "graphs": [gkey],
                }
                id_map[(gkey, n["id"])] = key
        for e in g["edges"]:
            merged_edges.append({
                "source": id_map[(gkey, e["source"])], "target": id_map[(gkey, e["target"])],
                "relation": e["relation"], "graph": gkey,
            })

    edge_dedup: dict[tuple, dict] = {}
    for e in merged_edges:
        k = (e["source"], e["target"], e["relation"])
        edge_dedup.setdefault(k, {"source": e["source"], "target": e["target"], "relation": e["relation"], "graphs": []})
        edge_dedup[k]["graphs"].append(e["graph"])

    final_nodes = list(merged_nodes.values())
    for n in final_nodes:
        n["shared"] = len(set(n["graphs"])) > 1
    final_edges = list(edge_dedup.values())

    # type-taxonomy layer: one meta node per concept type, wired to every
    # instance of that type (not drawn from any single profile)
    concept_nodes = [n for n in final_nodes if not n["is_root"]]
    for ntype, mlabel in TYPE_LABELS.items():
        members = [n for n in concept_nodes if n["type"] == ntype]
        if not members:
            continue
        mkey = new_key()
        final_nodes.append({"key": mkey, "label": mlabel, "type": ntype, "is_root": False, "is_meta": True, "graphs": [], "shared": False})
        for m in members:
            final_edges.append({"source": mkey, "target": m["key"], "relation": "type_of", "graphs": [], "is_meta": True})

    return {"roles": roles, "nodes": final_nodes, "edges": final_edges}


def render_html(merged: dict, template_path: Path, output_path: Path) -> None:
    html = template_path.read_text()
    data_json = json.dumps(merged)
    html2 = re.sub(r"const DATA = \{.*?\};\n", "const DATA = " + data_json + ";\n", html, count=1, flags=re.DOTALL)
    if html2 == html:
        raise RuntimeError("template's `const DATA = {...};` placeholder not found/replaced")
    output_path.write_text(html2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="contact id to center the graph on")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--cache", default=None, help="path to cache/reuse raw per-profile KG extractions (skips API calls if present)")
    parser.add_argument("--output", default=str(REPO_ROOT / "docs" / "html" / "knowledge-graph-experiment.html"))
    parser.add_argument("--template", default=None, help="HTML shell to inject data into (defaults to --output, i.e. self-updating in place)")
    args = parser.parse_args()
    template_path = Path(args.template) if args.template else Path(args.output)

    load_dotenv()
    import os
    api_key = os.environ["OPENROUTER_API_KEY"]

    data_dir = Path(args.data_dir)
    accepted_pair, declined_pair = find_pairs(data_dir, args.user_id)
    user_profile = (accepted_pair["userContactFile"] if accepted_pair["userContactId"] == args.user_id else accepted_pair["matchContactFile"])
    good_id, good_profile = other_side(accepted_pair, args.user_id)
    bad_id, bad_profile = other_side(declined_pair, args.user_id)

    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        graphs = json.loads(cache_path.read_text())["graphs"]
    else:
        graphs = {
            "user": call_extract(api_key, profile_to_text(user_profile), "user"),
            "good": call_extract(api_key, profile_to_text(good_profile), "good"),
            "bad": call_extract(api_key, profile_to_text(bad_profile), "bad"),
        }
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"user_id": args.user_id, "good_connection_id": good_id, "bad_connection_id": bad_id, "graphs": graphs}, indent=2))

    roles = {
        "user": {"role": "self", "contact_id": args.user_id},
        "good": {"role": "accepted", "contact_id": good_id},
        "bad": {"role": "declined", "contact_id": bad_id},
    }
    merged = merge_graphs(graphs, roles)
    render_html(merged, template_path, Path(args.output))
    print(f"wrote {args.output} ({len(merged['nodes'])} nodes, {len(merged['edges'])} edges)")


if __name__ == "__main__":
    main()
