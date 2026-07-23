"""Continuous synthetic profile generation against local/remote Ollama gemma3:4b endpoints.

Runs until interrupted (Ctrl+C). No profile-count ceiling by default.

Design (see docs/ discussion in chat history, not yet written up as a doc):
  - Step 1 (field style spec) and Step 2 (archetype list) are periodically
    refreshed from a fresh random sample of real profiles, always on the
    LOCAL endpoint (fastest, and keeps the remote endpoint free for
    uninterrupted generation throughput).
  - Step 3 (one profile per call) runs continuously on both endpoints in
    parallel, picking up whatever archetype/style snapshot is current.
  - Every call is validated (valid JSON, all required keys, no obvious
    mid-sentence truncation) and auto-retried up to --max-retries times.
  - Every generated profile (success or exhausted-retries failure) is
    written to disk immediately so a crash loses at most one in-flight call.

Usage:
    python scripts/local_gemma_profile_gen.py
    python scripts/local_gemma_profile_gen.py --max-profiles 20   # bounded test run
"""
import argparse
import itertools
import json
import os
import random
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

PROFILE_FIELDS = [
    "positioning", "background", "lookingFor", "notes",
    "locationAvailability", "introPreferences", "personalPreferences",
    "meetingAndSchedulingPreferences",
]

MODEL = "gemma3:4b"
REMOTE_ENDPOINT = "http://100.118.153.100:11434/api/generate"
LOCAL_ENDPOINT = "http://127.0.0.1:11434/api/generate"

STEP1_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in PROFILE_FIELDS},
    "required": PROFILE_FIELDS,
}
STEP2_SCHEMA = {
    "type": "object",
    "properties": {
        "archetypes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
                "required": ["label", "description"],
            },
        }
    },
    "required": ["archetypes"],
}
STEP3_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "3-4 sentences: who this person is, their specific role, one distinctive "
                           "career detail, what they're trying to achieve now, and why the story is "
                           "coherent. Do not reuse facts from the reference examples.",
        },
        **{f: {"type": "string"} for f in PROFILE_FIELDS},
    },
    "required": ["reasoning"] + PROFILE_FIELDS,
}


@traceable(name="ollama_call", run_type="llm")
def call_ollama(prompt: str, schema: dict, endpoint: str, num_ctx: int = 24576, temperature: float = 0.6):
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"num_ctx": num_ctx, "temperature": temperature},
    }).encode()
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1500) as resp:
        result = json.loads(resp.read())
    result["_elapsed_s"] = time.time() - t0
    return result


def validate_profile(obj: dict) -> list:
    problems = []
    required = ["reasoning"] + PROFILE_FIELDS
    for k in required:
        if k not in obj:
            problems.append(f"missing key: {k}")
            continue
        v = obj[k]
        if not isinstance(v, str):
            problems.append(f"{k}: not a string")
            continue
        if k != "meetingAndSchedulingPreferences" and len(v.strip()) < 15:
            problems.append(f"{k}: suspiciously short ({len(v.strip())} chars)")
        stripped = v.rstrip()
        if stripped and len(stripped) > 30 and stripped[-1] not in ".!?\"'”*_)":
            problems.append(f"{k}: possible truncation (ends with '...{stripped[-25:]}')")
    return problems


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def call_ollama_json_retry(prompt: str, schema: dict, endpoint: str, temperature: float,
                            max_retries: int, log_fn):
    """Call ollama and parse response as JSON, retrying on invalid/truncated JSON.

    Known failure mode: gemma3:4b intermittently truncates mid-string under
    grammar-constrained decoding (observed ~1/3 of the time on array-shaped
    schemas). Not fatal, just needs a retry.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        r = call_ollama(prompt, schema, endpoint, temperature=temperature)
        try:
            parsed = json.loads(r["response"])
            return parsed, r
        except json.JSONDecodeError as e:
            last_err = e
            log_fn(f"  retry {attempt}/{max_retries}: invalid JSON ({e}), elapsed={r['_elapsed_s']:.1f}s")
    raise RuntimeError(f"gave up after {max_retries} attempts: {last_err}")


class Runner:
    def __init__(self, out_dir: Path, full_pool: list, sample_size: int,
                 archetype_refresh_every: int, style_refresh_every: int,
                 max_retries: int, max_profiles):
        self.out_dir = out_dir
        self.full_pool = full_pool
        self.sample_size = sample_size
        self.archetype_refresh_every = archetype_refresh_every
        self.style_refresh_every = style_refresh_every
        self.max_retries = max_retries
        self.max_profiles = max_profiles

        self.lock = threading.Lock()
        self.style_spec = None
        self.style_version = 0
        self.archetypes = None
        self.archetypes_version = 0
        self.last_archetype_refresh_count = 0
        self.last_style_refresh_count = 0

        self.next_id = itertools.count()
        self.completed_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.stats = {"local": {"n": 0, "total_s": 0.0}, "remote": {"n": 0, "total_s": 0.0}}

        self.stop_event = threading.Event()
        self.manifest_path = out_dir / "manifest.jsonl"
        self.manifest_lock = threading.Lock()

        (out_dir / "profiles").mkdir(parents=True, exist_ok=True)
        (out_dir / "specs").mkdir(parents=True, exist_ok=True)

    # ---------- Step 1 / Step 2 refresh (always on local endpoint) ----------

    def _fresh_sample(self):
        return random.sample(self.full_pool, self.sample_size)

    def refresh_style(self):
        sample = self._fresh_sample()
        prompt = f"""Here are {len(sample)} real user profiles from a professional networking product, as JSON objects.

{json.dumps(sample, indent=2)}

For each of these fields — {', '.join(PROFILE_FIELDS)} — describe, in one paragraph:
1. typical length
2. whether it's written as sentences, dashes/bullets, or short tags
3. common tone/vocabulary
4. what kind of content usually appears

Do not repeat or summarize any specific person's actual details — only describe the pattern across all profiles."""
        spec, r = call_ollama_json_retry(prompt, STEP1_SCHEMA, LOCAL_ENDPOINT, temperature=0.5,
                                          max_retries=self.max_retries, log_fn=self._log)
        with self.lock:
            self.style_version += 1
            self.style_spec = spec
            self.last_style_refresh_count = self.completed_count
            v = self.style_version
        atomic_write_json(self.out_dir / "specs" / f"style_v{v}.json", spec)
        self._log(f"[style refresh v{v}] elapsed={r['_elapsed_s']:.1f}s sample_size={len(sample)}")

    def refresh_archetypes(self):
        sample = self._fresh_sample()
        prompt = f"""Here are {len(sample)} real user profiles from a professional networking product, as JSON objects.

{json.dumps(sample, indent=2)}

Looking at these profiles as whole people (not field by field), identify 6-8 recurring types you see — combinations of industry, role, career stage, and geography that naturally occur together in this population. For each type, give a short label (5-8 words) and a one-sentence description of what distinguishes it."""
        parsed, r = call_ollama_json_retry(prompt, STEP2_SCHEMA, LOCAL_ENDPOINT, temperature=0.5,
                                            max_retries=self.max_retries, log_fn=self._log)
        archetypes = parsed["archetypes"]
        with self.lock:
            self.archetypes_version += 1
            self.archetypes = archetypes
            self.last_archetype_refresh_count = self.completed_count
            v = self.archetypes_version
        atomic_write_json(self.out_dir / "specs" / f"archetypes_v{v}.json", archetypes)
        self._log(f"[archetype refresh v{v}] elapsed={r['_elapsed_s']:.1f}s sample_size={len(sample)} n_archetypes={len(archetypes)}")

    def initial_setup(self):
        self._log("Building initial style spec + archetypes (local endpoint)...")
        self.refresh_style()
        self.refresh_archetypes()

    # ---------- Step 3 generation ----------

    def _current_snapshot(self):
        with self.lock:
            return self.style_spec, self.archetypes

    def generate_one(self, profile_id: int, endpoint_label: str, endpoint_url: str):
        style_spec, archetypes = self._current_snapshot()
        archetype = archetypes[profile_id % len(archetypes)]
        style_guide_text = "\n".join(f"- {f}: {style_spec[f]}" for f in PROFILE_FIELDS)
        ref_examples = random.sample(self.full_pool, 2)

        prompt = f"""You are generating ONE fictional user profile for a synthetic test dataset for a professional networking product. This profile must describe a completely made-up person — not a real, identifiable individual.

Field style guide (learned from real profiles):
{style_guide_text}

Person type to write for: {archetype['label']} — {archetype['description']}

Below are 2 real profiles shown ONLY as style/format reference. Do not copy their names, companies, industries, or any specific facts. Only match their tone and structure.

REAL EXAMPLE 1:
{json.dumps(ref_examples[0], indent=2)}

REAL EXAMPLE 2:
{json.dumps(ref_examples[1], indent=2)}

First fill in "reasoning": think through who this fictional person is, without reusing any fact from the examples above.
Then, using that reasoning, fill in the 8 profile fields consistently with it and with the style guide."""

        attempts = []
        parsed = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = call_ollama(prompt, STEP3_SCHEMA, endpoint_url, num_ctx=16384,
                                 temperature=0.6 + 0.05 * (attempt - 1))
            except Exception as e:
                attempts.append({"attempt": attempt, "error": f"request failed: {e}"})
                continue
            raw = r.get("response", "")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                attempts.append({"attempt": attempt, "elapsed_s": r["_elapsed_s"], "error": f"invalid JSON: {e}"})
                continue
            problems = validate_profile(parsed)
            attempts.append({"attempt": attempt, "elapsed_s": r["_elapsed_s"],
                              "out_tokens": r.get("eval_count"), "problems": problems})
            if not problems:
                return {"id": profile_id, "endpoint": endpoint_label, "archetype": archetype["label"],
                        "style_version": self.style_version, "archetypes_version": self.archetypes_version,
                        "profile": parsed, "attempts": attempts, "success": True}
        return {"id": profile_id, "endpoint": endpoint_label, "archetype": archetype["label"],
                "style_version": self.style_version, "archetypes_version": self.archetypes_version,
                "profile": parsed, "attempts": attempts, "success": False}

    # ---------- bookkeeping ----------

    def _log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _save_result(self, result: dict):
        pid = result["id"]
        status = "ok" if result["success"] else "FAILED"
        fname = f"{pid:06d}_{status}.json"
        atomic_write_json(self.out_dir / "profiles" / fname, result)
        with self.manifest_lock:
            with open(self.manifest_path, "a") as f:
                f.write(json.dumps({
                    "id": pid, "endpoint": result["endpoint"], "archetype": result["archetype"],
                    "success": result["success"], "n_attempts": len(result["attempts"]),
                    "elapsed_s": sum(a.get("elapsed_s", 0) for a in result["attempts"]),
                    "style_version": result["style_version"], "archetypes_version": result["archetypes_version"],
                    "ts": time.time(),
                }) + "\n")

    def _progress_line(self, result: dict):
        ep = result["endpoint"]
        last_elapsed = result["attempts"][-1].get("elapsed_s", 0)
        with self.lock:
            l, r = self.stats["local"], self.stats["remote"]
            total = self.completed_count
            local_avg = l["total_s"] / l["n"] if l["n"] else 0
            remote_avg = r["total_s"] / r["n"] if r["n"] else 0
        status = "OK" if result["success"] else "FAILED"
        self._log(
            f"#{result['id']:>4} [{status}] endpoint={ep:<6} attempts={len(result['attempts'])} "
            f"elapsed={last_elapsed:.1f}s archetype='{result['archetype'][:35]}' | "
            f"total={total} ok={self.success_count} fail={self.fail_count} | "
            f"local_avg={local_avg:.1f}s remote_avg={remote_avg:.1f}s"
        )

    # ---------- worker loop ----------

    def worker(self, endpoint_label: str, endpoint_url: str):
        is_local = endpoint_label == "local"
        while not self.stop_event.is_set():
            if self.max_profiles is not None and self.completed_count >= self.max_profiles:
                return

            if is_local:
                with self.lock:
                    need_archetype = (self.completed_count - self.last_archetype_refresh_count) >= self.archetype_refresh_every
                    need_style = (self.completed_count - self.last_style_refresh_count) >= self.style_refresh_every
                if need_style:
                    self.refresh_style()
                    continue
                if need_archetype:
                    self.refresh_archetypes()
                    continue

            pid = next(self.next_id)
            if self.max_profiles is not None and pid >= self.max_profiles:
                return
            try:
                result = self.generate_one(pid, endpoint_label, endpoint_url)
            except Exception as e:
                self._log(f"#{pid} unhandled error on {endpoint_label}: {e}")
                continue

            self._save_result(result)
            with self.lock:
                self.completed_count += 1
                if result["success"]:
                    self.success_count += 1
                else:
                    self.fail_count += 1
                total_s = sum(a.get("elapsed_s", 0) for a in result["attempts"])
                self.stats[endpoint_label]["n"] += 1
                self.stats[endpoint_label]["total_s"] += total_s
            self._progress_line(result)

    def run(self):
        self.initial_setup()
        threads = [
            threading.Thread(target=self.worker, args=("local", LOCAL_ENDPOINT), daemon=True),
            threading.Thread(target=self.worker, args=("remote", REMOTE_ENDPOINT), daemon=True),
        ]

        def handle_sigint(signum, frame):
            if self.stop_event.is_set():
                self._log("Force exit.")
                sys.exit(1)
            self._log("Interrupt received — finishing in-flight generations, then stopping. Press Ctrl+C again to force quit.")
            self.stop_event.set()

        signal.signal(signal.SIGINT, handle_sigint)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self._log(
            f"=== STOPPED. total={self.completed_count} success={self.success_count} "
            f"failed={self.fail_count} style_versions={self.style_version} "
            f"archetype_versions={self.archetypes_version} ==="
        )
        self._log(f"Output dir: {self.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="Output directory (default: artifacts/local_gemma_synth/run_<timestamp>)")
    ap.add_argument("--sample-size", type=int, default=8, help="Real profiles sampled per style/archetype refresh")
    ap.add_argument("--archetype-refresh-every", type=int, default=5)
    ap.add_argument("--style-refresh-every", type=int, default=15)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-profiles", type=int, default=None, help="Stop after N profiles (default: unlimited, run until Ctrl+C)")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "artifacts" / "local_gemma_synth" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_pool_raw = json.load(open(Path(args.data_dir) / "unique_users.json"))
    full_pool = [{k: p["userContactFile"].get(k, "") for k in PROFILE_FIELDS} for p in full_pool_raw]

    runner = Runner(
        out_dir=out_dir,
        full_pool=full_pool,
        sample_size=args.sample_size,
        archetype_refresh_every=args.archetype_refresh_every,
        style_refresh_every=args.style_refresh_every,
        max_retries=args.max_retries,
        max_profiles=args.max_profiles,
    )
    runner.run()


if __name__ == "__main__":
    main()
