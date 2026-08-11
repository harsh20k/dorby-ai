"""Modal GPU entrypoint: Qwen3.5-4B as a self-hosted LLM judge on the 200 real pairs.

Same evaluation shape as ``baselines/llm_judge`` (prompted, not fine-tuned —
that's a later step) but the model runs locally on a Modal GPU from Hugging
Face weights instead of through an OpenRouter/Bedrock API call. Prompt is
pulled from LangSmith Hub (``-/qwen35-judge-naive-query:v1``, pushed by
``push_prompt.py``) so the exact prompt behind these numbers is a named,
inspectable commit — no local prompt string is trusted at run time.

Usage::

    modal run baselines/qwen35_judge/modal_eval.py --split all
    modal run baselines/qwen35_judge/modal_eval.py --split holdout
    modal volume get dorby-qwen35-judge-eval <run-id> ./artifacts/qwen35_judge
"""

from __future__ import annotations

import json

import modal

APP_NAME = "dorby-qwen35-judge-eval"
RESULTS_VOLUME = "dorby-qwen35-judge-eval"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # reuse the existing HF model cache
MODEL_ID = "Qwen/Qwen3.5-4B"
HUB_PROMPT_ID = "qwen35-judge-naive-query"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "accelerate>=0.30",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "langsmith",
        "langchain-core",
    )
    # Qwen3.5 is new enough (Feb/Mar 2026) that no pinned transformers release
    # recognizes the `qwen3_5` model_type yet — install from source.
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir(
        "baselines/qwen35_judge/prompts", remote_path="/root/baselines/qwen35_judge/prompts"
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache/huggingface": hf_cache, "/results": results},
    secrets=[modal.Secret.from_dotenv()],
    timeout=60 * 60,
)
def run(*, split: str, run_id: str, max_new_tokens: int = 500) -> dict:
    import re

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from baselines.bert_frozen.text import profile_to_text
    from baselines.llm_judge.judge import parse_verdict, verdict_to_score
    from baselines.llm_judge.metrics import calibration_buckets, decision_metrics, intent_pair_auc
    from baselines.llm_judge.real_pairs import load_real_pairs, pair_id
    from baselines.metrics import neg_hardness_slice_metrics, pair_metrics
    from baselines.qwen35_judge.prompt import build_user_prompt

    from langsmith import Client

    print(f"pulling prompt {HUB_PROMPT_ID}:v1 from LangSmith Hub")
    hub_prompt = Client().pull_prompt(f"{HUB_PROMPT_ID}:v1")
    # ChatPromptTemplate with brace-escaped literal content and no template
    # vars -> format() with no kwargs recovers the exact source text.
    system_prompt = hub_prompt.format_messages()[0].content

    print(f"loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    from pathlib import Path

    data_dir = Path("/root/data")
    positives, negatives = load_real_pairs(data_dir, split=split)
    print(f"loaded {len(positives)} positives, {len(negatives)} negatives ({split})")

    def judge_one(record: dict) -> dict:
        user_prompt = build_user_prompt(
            record.get("searchQuery", ""),
            record["userContactFile"],
            record["matchContactFile"],
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        match = re.search(r"\{.*\}", gen, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in model output: {gen!r}")
        raw = json.loads(match.group(0))
        return parse_verdict(raw)

    def judge_all(records: list[dict], label: str) -> list[dict]:
        verdicts = []
        for i, r in enumerate(records, start=1):
            try:
                verdicts.append(judge_one(r))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{label} {i}/{len(records)}] FAILED: {exc}")
                verdicts.append(None)
            if i % 10 == 0 or i == len(records):
                print(f"  [{label}] judged {i}/{len(records)}")
        return verdicts

    pos_verdicts_raw = judge_all(positives, "pos")
    neg_verdicts_raw = judge_all(negatives, "neg")

    pos_kept = [(r, v) for r, v in zip(positives, pos_verdicts_raw) if v is not None]
    neg_kept = [(r, v) for r, v in zip(negatives, neg_verdicts_raw) if v is not None]
    failed = (len(positives) - len(pos_kept)) + (len(negatives) - len(neg_kept))

    pos_records = [r for r, _ in pos_kept]
    neg_records = [r for r, _ in neg_kept]
    pos_verdicts = [v for _, v in pos_kept]
    neg_verdicts = [v for _, v in neg_kept]

    pos_scores = np.array([verdict_to_score(v) for v in pos_verdicts], dtype=np.float64)
    neg_scores = np.array([verdict_to_score(v) for v in neg_verdicts], dtype=np.float64)

    neg_seeker_texts = [profile_to_text(r["userContactFile"]) for r in neg_records]
    neg_cand_texts = [profile_to_text(r["matchContactFile"]) for r in neg_records]

    metrics = {
        "model_name": MODEL_ID,
        "experiment": "qwen35_judge",
        "prompt_variant": "naive_query",
        "uses_search_query": True,
        "backend": "modal_hf",
        "split": split,
        "num_pairs_requested": len(positives) + len(negatives),
        "num_pairs_scored": len(pos_verdicts) + len(neg_verdicts),
        "num_failed": failed,
        "decision": decision_metrics(pos_verdicts, neg_verdicts),
        "calibration": calibration_buckets(pos_verdicts, neg_verdicts),
        "pair": pair_metrics(pos_scores, neg_scores),
        "slices": {
            "intent": intent_pair_auc(pos_records, neg_records, pos_scores, neg_scores),
            "neg_hardness": neg_hardness_slice_metrics(
                neg_scores, neg_seeker_texts, neg_cand_texts, pos_scores=pos_scores
            ),
        },
    }

    out_dir = f"/results/{run_id}"
    import os

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    results.commit()

    print(json.dumps(metrics["pair"], indent=2))
    return metrics


@app.local_entrypoint()
def main(split: str = "all", run_id: str = "qwen35_4b_naive_query"):
    metrics = run.remote(split=split, run_id=run_id)
    print(f"\n=== Qwen3.5-4B judge ({split}) ===")
    print(f"pairs scored: {metrics['num_pairs_scored']}/{metrics['num_pairs_requested']}")
    print(f"pair ROC-AUC: {metrics['pair']['roc_auc']:.4f}")
    print(f"decision accuracy: {metrics['decision']['accuracy']:.4f}")
