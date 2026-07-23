"""Continuous synthetic profile generation against AWS Bedrock (Llama 3.3 70B by default).

Same 3-step design as scripts/local_gemma_profile_gen.py (style spec + archetype
list, periodically refreshed, feeding continuous per-profile generation), but:
  - Uses Bedrock's Converse API structured outputs (outputConfig.textFormat,
    GA Feb 2026) for native JSON-schema-constrained decoding, instead of
    Ollama's format param + retry-on-truncation.
  - No local/remote endpoint split — Bedrock is one managed API, so
    parallelism is just N worker threads hitting the same client with
    --concurrency.
  - Refresh failures (style/archetype) no longer crash a worker thread: they
    log, keep the stale spec, and retry next cycle (see docs/possible-bugs.md
    lesson from the local_gemma_profile_gen.py run that lost its local
    thread to an uncaught RuntimeError).

Requires AWS credentials (env vars, ~/.aws/config profile, or SSO login) with
bedrock:InvokeModel / Converse access, and model access enabled for the
target model in the Bedrock console.

Usage:
    python scripts/bedrock_profile_gen.py --max-profiles 20   # bounded test run
    python scripts/bedrock_profile_gen.py                     # run until Ctrl+C
"""
import argparse
import itertools
import json
import signal
import sys
import threading
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langsmith import traceable

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

PROFILE_FIELDS = [
    "positioning", "background", "lookingFor", "notes",
    "locationAvailability", "introPreferences", "personalPreferences",
    "meetingAndSchedulingPreferences",
]

DEFAULT_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"  # geo cross-region inference ID (required in us-east-1)
DEFAULT_REGION = "us-east-1"

STEP1_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in PROFILE_FIELDS},
    "required": PROFILE_FIELDS,
    "additionalProperties": False,
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
                "additionalProperties": False,
            },
        }
    },
    "required": ["archetypes"],
    "additionalProperties": False,
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
    "additionalProperties": False,
}


@traceable(name="bedrock_call", run_type="llm")
def call_bedrock(client, model_id: str, prompt: str, schema: dict, schema_name: str,
                  schema_description: str, max_tokens: int, temperature: float):
    t0 = time.time()
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
    elapsed = time.time() - t0
    text = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    return {"response": text, "_elapsed_s": elapsed, "_usage": usage}


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


def call_bedrock_json_retry(client, model_id: str, prompt: str, schema: dict, schema_name: str,
                             schema_description: str, temperature: float, max_tokens: int,
                             max_retries: int, log_fn):
    """Call Bedrock and parse response as JSON, retrying on API errors (e.g. throttling)
    or (unexpectedly, since output is schema-constrained) invalid JSON."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = call_bedrock(client, model_id, prompt, schema, schema_name, schema_description,
                              max_tokens, temperature)
        except ClientError as e:
            last_err = e
            wait = min(2 ** attempt, 20)
            log_fn(f"  retry {attempt}/{max_retries}: Bedrock error ({e}), backing off {wait}s")
            time.sleep(wait)
            continue
        try:
            parsed = json.loads(r["response"])
            return parsed, r
        except json.JSONDecodeError as e:
            last_err = e
            log_fn(f"  retry {attempt}/{max_retries}: invalid JSON ({e}), elapsed={r['_elapsed_s']:.1f}s")
    raise RuntimeError(f"gave up after {max_retries} attempts: {last_err}")


class Runner:
    def __init__(self, out_dir: Path, full_pool: list, client, model_id: str, sample_size: int,
                 archetype_refresh_every: int, style_refresh_every: int,
                 max_retries: int, max_tokens: int, temperature: float, max_profiles):
        self.out_dir = out_dir
        self.full_pool = full_pool
        self.client = client
        self.model_id = model_id
        self.sample_size = sample_size
        self.archetype_refresh_every = archetype_refresh_every
        self.style_refresh_every = style_refresh_every
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_profiles = max_profiles

        self.lock = threading.Lock()
        self.style_spec = None
        self.style_version = 0
        self.archetypes = None
        self.archetypes_version = 0
        self.last_archetype_refresh_count = 0
        self.last_style_refresh_count = 0
        self.refresh_in_progress = False

        self.next_id = itertools.count()
        self.completed_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.total_usage = {"input_tokens": 0, "output_tokens": 0}

        self.stop_event = threading.Event()
        self.manifest_path = out_dir / "manifest.jsonl"
        self.manifest_lock = threading.Lock()

        (out_dir / "profiles").mkdir(parents=True, exist_ok=True)
        (out_dir / "specs").mkdir(parents=True, exist_ok=True)

    def _fresh_sample(self):
        import random
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
        spec, r = call_bedrock_json_retry(
            self.client, self.model_id, prompt, STEP1_SCHEMA, "style_spec",
            "Per-field style guide learned from real profiles", temperature=0.5,
            max_tokens=self.max_tokens, max_retries=self.max_retries, log_fn=self._log)
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
        parsed, r = call_bedrock_json_retry(
            self.client, self.model_id, prompt, STEP2_SCHEMA, "archetype_list",
            "Recurring persona archetypes found across the sampled profiles", temperature=0.5,
            max_tokens=self.max_tokens, max_retries=self.max_retries, log_fn=self._log)
        archetypes = parsed["archetypes"]
        with self.lock:
            self.archetypes_version += 1
            self.archetypes = archetypes
            self.last_archetype_refresh_count = self.completed_count
            v = self.archetypes_version
        atomic_write_json(self.out_dir / "specs" / f"archetypes_v{v}.json", archetypes)
        self._log(f"[archetype refresh v{v}] elapsed={r['_elapsed_s']:.1f}s sample_size={len(sample)} n_archetypes={len(archetypes)}")

    def initial_setup(self):
        self._log(f"Building initial style spec + archetypes (model={self.model_id})...")
        self._retry_until_success(self.refresh_style, "style")
        self._retry_until_success(self.refresh_archetypes, "archetypes")

    def _retry_until_success(self, fn, name: str, max_attempts: int = 5):
        """Used only for initial_setup, where there's no stale spec to fall back to."""
        for attempt in range(1, max_attempts + 1):
            try:
                fn()
                return
            except Exception as e:
                self._log(f"[{name} initial refresh] attempt {attempt}/{max_attempts} failed: {e}")
        raise RuntimeError(f"could not build initial {name} spec after {max_attempts} attempts")

    def _current_snapshot(self):
        with self.lock:
            return self.style_spec, self.archetypes

    def generate_one(self, profile_id: int):
        import random
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
                r = call_bedrock(self.client, self.model_id, prompt, STEP3_SCHEMA, "profile",
                                  "One fictional user profile", self.max_tokens,
                                  temperature=self.temperature + 0.05 * (attempt - 1))
            except ClientError as e:
                attempts.append({"attempt": attempt, "error": f"bedrock error: {e}"})
                time.sleep(min(2 ** attempt, 20))
                continue
            raw = r.get("response", "")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                attempts.append({"attempt": attempt, "elapsed_s": r["_elapsed_s"], "error": f"invalid JSON: {e}"})
                continue
            problems = validate_profile(parsed)
            usage = r.get("_usage", {})
            attempts.append({"attempt": attempt, "elapsed_s": r["_elapsed_s"],
                              "input_tokens": usage.get("inputTokens"),
                              "output_tokens": usage.get("outputTokens"),
                              "problems": problems})
            if not problems:
                return {"id": profile_id, "archetype": archetype["label"],
                        "style_version": self.style_version, "archetypes_version": self.archetypes_version,
                        "profile": parsed, "attempts": attempts, "success": True}
        return {"id": profile_id, "archetype": archetype["label"],
                "style_version": self.style_version, "archetypes_version": self.archetypes_version,
                "profile": parsed, "attempts": attempts, "success": False}

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
                    "id": pid, "archetype": result["archetype"],
                    "success": result["success"], "n_attempts": len(result["attempts"]),
                    "elapsed_s": sum(a.get("elapsed_s", 0) for a in result["attempts"]),
                    "input_tokens": sum(a.get("input_tokens") or 0 for a in result["attempts"]),
                    "output_tokens": sum(a.get("output_tokens") or 0 for a in result["attempts"]),
                    "style_version": result["style_version"], "archetypes_version": result["archetypes_version"],
                    "ts": time.time(),
                }) + "\n")

    def _progress_line(self, result: dict):
        last = result["attempts"][-1]
        with self.lock:
            total = self.completed_count
            in_tok = self.total_usage["input_tokens"]
            out_tok = self.total_usage["output_tokens"]
        status = "OK" if result["success"] else "FAILED"
        self._log(
            f"#{result['id']:>4} [{status}] attempts={len(result['attempts'])} "
            f"elapsed={last.get('elapsed_s', 0):.1f}s archetype='{result['archetype'][:35]}' | "
            f"total={total} ok={self.success_count} fail={self.fail_count} | "
            f"tokens_in={in_tok} tokens_out={out_tok}"
        )

    def worker(self, worker_id: int):
        while not self.stop_event.is_set():
            if self.max_profiles is not None and self.completed_count >= self.max_profiles:
                return

            with self.lock:
                need_archetype = (self.completed_count - self.last_archetype_refresh_count) >= self.archetype_refresh_every
                need_style = (self.completed_count - self.last_style_refresh_count) >= self.style_refresh_every
                should_refresh = (need_style or need_archetype) and not self.refresh_in_progress
                if should_refresh:
                    self.refresh_in_progress = True
            if should_refresh:
                try:
                    if need_style:
                        try:
                            self.refresh_style()
                        except Exception as e:
                            self._log(f"[style refresh] failed, keeping stale spec (v{self.style_version}): {e}")
                            with self.lock:
                                self.last_style_refresh_count = self.completed_count
                    elif need_archetype:
                        try:
                            self.refresh_archetypes()
                        except Exception as e:
                            self._log(f"[archetype refresh] failed, keeping stale archetypes (v{self.archetypes_version}): {e}")
                            with self.lock:
                                self.last_archetype_refresh_count = self.completed_count
                finally:
                    with self.lock:
                        self.refresh_in_progress = False
                continue

            pid = next(self.next_id)
            if self.max_profiles is not None and pid >= self.max_profiles:
                return
            try:
                result = self.generate_one(pid)
            except Exception as e:
                self._log(f"#{pid} unhandled error on worker {worker_id}: {e}")
                continue

            self._save_result(result)
            with self.lock:
                self.completed_count += 1
                if result["success"]:
                    self.success_count += 1
                else:
                    self.fail_count += 1
                for a in result["attempts"]:
                    self.total_usage["input_tokens"] += a.get("input_tokens") or 0
                    self.total_usage["output_tokens"] += a.get("output_tokens") or 0
            self._progress_line(result)

    def run(self, concurrency: int):
        self.initial_setup()
        threads = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(concurrency)]

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
            f"archetype_versions={self.archetypes_version} "
            f"tokens_in={self.total_usage['input_tokens']} tokens_out={self.total_usage['output_tokens']} ==="
        )
        self._log(f"Output dir: {self.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="Output directory (default: artifacts/bedrock_synth/run_<timestamp>)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Bedrock model/inference-profile ID")
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--concurrency", type=int, default=4, help="Parallel worker threads hitting Bedrock")
    ap.add_argument("--max-tokens", type=int, default=4000, help="Llama 3.3 70B's max output is 4K")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--sample-size", type=int, default=8, help="Real profiles sampled per style/archetype refresh")
    ap.add_argument("--archetype-refresh-every", type=int, default=5)
    ap.add_argument("--style-refresh-every", type=int, default=15)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-profiles", type=int, default=None, help="Stop after N profiles (default: unlimited, run until Ctrl+C)")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "artifacts" / "bedrock_synth" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_pool_raw = json.load(open(Path(args.data_dir) / "unique_users.json"))
    full_pool = [{k: p["userContactFile"].get(k, "") for k in PROFILE_FIELDS} for p in full_pool_raw]

    client = boto3.client("bedrock-runtime", region_name=args.region)

    runner = Runner(
        out_dir=out_dir,
        full_pool=full_pool,
        client=client,
        model_id=args.model_id,
        sample_size=args.sample_size,
        archetype_refresh_every=args.archetype_refresh_every,
        style_refresh_every=args.style_refresh_every,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_profiles=args.max_profiles,
    )
    runner.run(concurrency=args.concurrency)


if __name__ == "__main__":
    main()
