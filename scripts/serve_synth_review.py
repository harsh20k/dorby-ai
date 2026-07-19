#!/usr/bin/env python3
"""Serve artifacts/synth/_browser.html and persist human_review verdicts.

The static browser (scripts/build_synth_browser.py) can't write files on its
own, so approve/reject clicks POST to /api/review here, and this process
writes human_review straight into the staged pair's JSON file.

Usage:
    python3 scripts/serve_synth_review.py
    # then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import datetime
import http.server
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "artifacts" / "synth"
BROWSER_HTML = SYNTH_DIR / "_browser.html"


def _resolve_pair_path(rel_path: str) -> Path:
    full = (SYNTH_DIR / rel_path).resolve()
    if SYNTH_DIR.resolve() not in full.parents:
        raise ValueError(f"path escapes synth dir: {rel_path}")
    if not full.is_file():
        raise FileNotFoundError(full)
    return full


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html", "/_browser.html"):
            if not BROWSER_HTML.exists():
                self.send_error(404, "Run scripts/build_synth_browser.py first")
                return
            body = BROWSER_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/review":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            rel_path = payload["rel_path"]
            verdict = payload["verdict"]
            if verdict not in ("yes", "no"):
                raise ValueError(f"bad verdict: {verdict}")
            notes = payload.get("notes")

            pair_path = _resolve_pair_path(rel_path)
            env = json.loads(pair_path.read_text(encoding="utf-8"))
            env["human_review"] = {
                "verdict": verdict,
                "reviewer": "human",
                "reviewed_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "notes": notes,
            }
            pair_path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")

            self._send_json(200, {"ok": True})
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer(("localhost", args.port), Handler)
    print(f"Serving {BROWSER_HTML} at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
