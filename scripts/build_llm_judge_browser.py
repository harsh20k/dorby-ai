#!/usr/bin/env python3
"""Build a self-contained HTML comparison page for the LLM-judge experiment.

Pulls the matched-holdout pair AUC / hard-neg / easy-neg numbers for every
embedding baseline (docs/baseline-results-holdout.json) plus all four
(model, framing) LLM-judge combinations run so far
(artifacts/llm_judge/*/metrics_holdout.json), and renders a ranked bar chart
plus a full data table — same visual language (tokens, table styling) as
scripts/build_holdout_browser.py, so this reads as one family of pages in
docs/experiment-graphs-index.md.

Usage:
    python3 scripts/build_llm_judge_browser.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "html" / "llm-judge-comparison.html"
BASELINE_JSON = ROOT / "docs" / "baseline-results-holdout.json"
LLM_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge"
FOCUSED_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge_with_pos_look_pos_back_look"
SEEKER_BG_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge_with_pos_look_back_pos_back_look"
LOOKINGFOR_ONLY_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge_lookingfor_only"
POSITIONING_ONLY_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge_positioning_only"
# 200-pair only — no holdout run for this sweep, see
# docs/llm-judge-candidate-field-permutation-experiment.md.
CAND_POS_BACK_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge_candidate_pos_back"
CAND_POS_LOOK_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge_candidate_pos_look"
CAND_BACK_LOOK_JUDGE_ARTIFACTS = ROOT / "artifacts" / "llm_judge_candidate_back_look"

# --- 200-pair (all real pairs) section ---
ALL_BASELINE_JSON = ROOT / "docs" / "baseline-results-all.json"
VOYAGE_LARGE_FIELD_SELECTED_ALL = ROOT / "artifacts" / "voyage_large_field_selected" / "metrics.json"
VOYAGE_NANO_FIELD_SELECTED_ALL = (
    ROOT / "artifacts" / "voyage_nano_field_selected_modal" / "real_all" / "metrics.json"
)
QWEN3_FIELD_SELECTED_ALL = (
    ROOT / "artifacts" / "hf_embedding_field_selected_modal"
    / "qwen_qwen3-embedding-8b_all" / "metrics.json"
)

# --- 69-pair holdout section: same field-selected embedding runs, holdout split ---
VOYAGE_LARGE_FIELD_SELECTED_HOLDOUT = (
    ROOT / "artifacts" / "voyage_large_field_selected_holdout" / "metrics.json"
)
VOYAGE_NANO_FIELD_SELECTED_HOLDOUT = (
    ROOT / "artifacts" / "voyage_nano_field_selected_modal" / "real_holdout" / "metrics.json"
)
QWEN3_FIELD_SELECTED_HOLDOUT = (
    ROOT / "artifacts" / "hf_embedding_field_selected_modal"
    / "qwen_qwen3-embedding-8b_holdout" / "metrics.json"
)

# Validated default categorical palette (references/palette.md), slots 1 & 2 —
# two categories (embedding baseline vs LLM judge) validated all-pairs in
# both modes, so no secondary encoding is required beyond the direct labels
# already shown on every bar.
COLOR_BASELINE_LIGHT, COLOR_BASELINE_DARK = "#2a78d6", "#3987e5"  # slot 1: blue
COLOR_JUDGE_LIGHT, COLOR_JUDGE_DARK = "#eb6834", "#d95926"  # slot 2: orange

LABELS = {
    "tfidf": "TF-IDF (lexical)",
    "bert_frozen": "Frozen BERT",
    "voyage_nano": "Voyage-4-nano",
    "voyage_large": "Voyage-4-large (production)",
    "hybrid_tfidf_voyage": "Hybrid TF-IDF+nano",
    "hf_embedding_qwen_qwen3-embedding-8b": "Qwen3-Embedding-8B (open)",
    "hf_embedding_intfloat_e5-mistral-7b-instruct": "E5-mistral-7b-instruct (open)",
    "hf_embedding_nvidia_nv-embed-v2": "NV-Embed-v2 (open, approx.)",
    "hf_embedding_baai_bge-en-icl": "BGE-en-ICL (open)",
    "hf_embedding_zeroentropy_zembed-1-embedding": "zembed-1-embedding (open)",
    "twotower_run_001": "twotower run_001",
    "twotower_arm_a_real_only": "twotower arm_a_real_only",
}

JUDGE_RUNS = {
    "gemini-3.1-flash-lite (naive)": "openrouter_google_gemini_3_1_flash_lite_naive",
    "gemini-3.1-flash-lite (calibrated)": "openrouter_google_gemini_3_1_flash_lite_calibrated",
    "gemma-3-27b-it (Bedrock, naive)": "bedrock_google_gemma_3_27b_it_naive",
    "qwen3-32b (Bedrock, naive)": "bedrock_qwen_qwen3_32b_v1_0_naive",
}

# Separate artifacts root (baselines/llm_judge_with_pos_look_pos_back_look) —
# an isolated variant that, unlike everything in JUDGE_RUNS above, gives the
# model the searchQuery and only a trimmed subset of profile fields. See
# docs/llm-judge-focused-prompt-experiment.md.
FOCUSED_JUDGE_RUNS = {
    "gemini-3.1-flash-lite (focused, w/ query, OpenRouter)": "openrouter_google_gemini_3_1_flash_lite",
    "gemini-3.1-flash-lite (focused, w/ query, direct Google API)": "google_google_gemini_3_1_flash_lite",
}

# Same as FOCUSED_JUDGE_RUNS but the seeker also gets `background` — separate
# artifacts root (baselines/llm_judge_with_pos_look_back_pos_back_look),
# direct Google API only. See docs/llm-judge-seeker-background-experiment.md.
SEEKER_BG_JUDGE_RUNS = {
    "gemini-3.1-flash-lite (focused + seeker background, direct Google API)": "google_google_gemini_3_1_flash_lite",
}

# Seeker field-isolation: same focused prompt/candidate fields/query, but the
# seeker is trimmed to exactly one field instead of positioning + lookingFor,
# to isolate each field's individual contribution. Separate artifacts roots
# (baselines/llm_judge_lookingfor_only, baselines/llm_judge_positioning_only),
# direct Google API only. See docs/llm-judge-seeker-field-isolation-experiment.md.
LOOKINGFOR_ONLY_JUDGE_RUNS = {
    "gemini-3.1-flash-lite (focused, seeker=lookingFor only, direct Google API)": "google_google_gemini_3_1_flash_lite",
}
POSITIONING_ONLY_JUDGE_RUNS = {
    "gemini-3.1-flash-lite (focused, seeker=positioning only, direct Google API)": "google_google_gemini_3_1_flash_lite",
}

# Candidate-side field permutations: seeker fixed at positioning + lookingFor
# (same as the focused prompt), candidate dropped to each of the three
# possible two-field subsets of positioning/background/lookingFor. 200-pair
# only, direct Google API. See
# docs/llm-judge-candidate-field-permutation-experiment.md.
CAND_POS_BACK_JUDGE_RUNS = {
    "gemini-3.1-flash-lite (focused, candidate=positioning+background, direct Google API)": "google_google_gemini_3_1_flash_lite",
}
CAND_POS_LOOK_JUDGE_RUNS = {
    "gemini-3.1-flash-lite (focused, candidate=positioning+lookingFor, direct Google API)": "google_google_gemini_3_1_flash_lite",
}
CAND_BACK_LOOK_JUDGE_RUNS = {
    "gemini-3.1-flash-lite (focused, candidate=background+lookingFor, direct Google API)": "google_google_gemini_3_1_flash_lite",
}

# Field sets shown per row on hover, so "gets_query yes/no" isn't the only
# thing distinguishing rows that otherwise look identical on the bar chart —
# see the tooltip-table rendering in HTML_TEMPLATE's <script>.
FULL_PROFILE_FIELDS = [
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
]
FOCUSED_SEEKER_FIELDS = ["positioning", "lookingFor"]
FOCUSED_CANDIDATE_FIELDS = ["positioning", "background", "lookingFor"]
SEEKER_BG_FIELDS = ["positioning", "lookingFor", "background"]
LOOKINGFOR_ONLY_FIELDS = ["lookingFor"]
POSITIONING_ONLY_FIELDS = ["positioning"]
CAND_POS_BACK_FIELDS = ["positioning", "background"]
CAND_POS_LOOK_FIELDS = ["positioning", "lookingFor"]
CAND_BACK_LOOK_FIELDS = ["background", "lookingFor"]


def build_data() -> dict:
    d = json.loads(BASELINE_JSON.read_text())
    baselines = d["baselines"]

    rows = []
    for key, v in baselines.items():
        p, h = v["pair"], v["slices"]["neg_hardness"]
        rows.append(
            {
                "name": LABELS.get(key, key),
                "kind": "baseline",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": v["retrieval"]["mrr"],
                "seeker_fields": FULL_PROFILE_FIELDS,
                "candidate_fields": FULL_PROFILE_FIELDS,
            }
        )

    for name, path in FIELD_SELECTED_HOLDOUT.items():
        m = json.loads(path.read_text())
        p, h = m["pair"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": name,
                "kind": "field_selected",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": m["retrieval"]["mrr"],
                "seeker_fields": FOCUSED_SEEKER_FIELDS,
                "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
            }
        )

    judge_detail = []
    for name, dirname in JUDGE_RUNS.items():
        path = LLM_JUDGE_ARTIFACTS / dirname / "metrics_holdout.json"
        m = json.loads(path.read_text())
        p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": f"LLM judge: {name}",
                "kind": "llm_judge",
                "gets_query": False,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": None,
                "seeker_fields": FULL_PROFILE_FIELDS,
                "candidate_fields": FULL_PROFILE_FIELDS,
            }
        )
        judge_detail.append(
            {
                "name": name,
                "backend": m.get("backend", "openrouter"),
                "variant": m["prompt_variant"],
                "n": m["num_pairs_scored"],
                "pair_auc": p["roc_auc"],
                "decision_accuracy": dec["accuracy"],
                "f1": dec["f1"],
                "yes_rate": dec["yes_rate"],
                "yes_rate_pos": dec["yes_rate_on_positives"],
                "yes_rate_neg": dec["yes_rate_on_negatives"],
                "mean_confidence": dec["mean_confidence"],
                "mean_confidence_correct": dec["mean_confidence_when_correct"],
                "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
            }
        )

    for name, dirname in FOCUSED_JUDGE_RUNS.items():
        path = FOCUSED_JUDGE_ARTIFACTS / dirname / "metrics_holdout.json"
        m = json.loads(path.read_text())
        p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": f"LLM judge: {name}",
                "kind": "llm_judge",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": None,
                "seeker_fields": FOCUSED_SEEKER_FIELDS,
                "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
            }
        )
        judge_detail.append(
            {
                "name": name,
                "backend": m.get("backend", "openrouter"),
                "variant": m["prompt_variant"],
                "n": m["num_pairs_scored"],
                "pair_auc": p["roc_auc"],
                "decision_accuracy": dec["accuracy"],
                "f1": dec["f1"],
                "yes_rate": dec["yes_rate"],
                "yes_rate_pos": dec["yes_rate_on_positives"],
                "yes_rate_neg": dec["yes_rate_on_negatives"],
                "mean_confidence": dec["mean_confidence"],
                "mean_confidence_correct": dec["mean_confidence_when_correct"],
                "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
            }
        )

    for name, dirname in SEEKER_BG_JUDGE_RUNS.items():
        path = SEEKER_BG_JUDGE_ARTIFACTS / dirname / "metrics_holdout.json"
        m = json.loads(path.read_text())
        p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": f"LLM judge: {name}",
                "kind": "llm_judge",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": None,
                "seeker_fields": SEEKER_BG_FIELDS,
                "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
            }
        )
        judge_detail.append(
            {
                "name": name,
                "backend": m.get("backend", "google"),
                "variant": m["prompt_variant"],
                "n": m["num_pairs_scored"],
                "pair_auc": p["roc_auc"],
                "decision_accuracy": dec["accuracy"],
                "f1": dec["f1"],
                "yes_rate": dec["yes_rate"],
                "yes_rate_pos": dec["yes_rate_on_positives"],
                "yes_rate_neg": dec["yes_rate_on_negatives"],
                "mean_confidence": dec["mean_confidence"],
                "mean_confidence_correct": dec["mean_confidence_when_correct"],
                "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
            }
        )

    for artifacts_root, run_map, seeker_fields in (
        (LOOKINGFOR_ONLY_JUDGE_ARTIFACTS, LOOKINGFOR_ONLY_JUDGE_RUNS, LOOKINGFOR_ONLY_FIELDS),
        (POSITIONING_ONLY_JUDGE_ARTIFACTS, POSITIONING_ONLY_JUDGE_RUNS, POSITIONING_ONLY_FIELDS),
    ):
        for name, dirname in run_map.items():
            path = artifacts_root / dirname / "metrics_holdout.json"
            m = json.loads(path.read_text())
            p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
            rows.append(
                {
                    "name": f"LLM judge: {name}",
                    "kind": "llm_judge",
                    "gets_query": True,
                    "pair_auc": p["roc_auc"],
                    "accuracy_at_half": p["accuracy_at_0.5"],
                    "best_f1": p["best_f1"],
                    "hard_auc": h["hard"]["pair_auc"],
                    "easy_auc": h["easy"]["pair_auc"],
                    "mrr": None,
                    "seeker_fields": seeker_fields,
                    "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
                }
            )
            judge_detail.append(
                {
                    "name": name,
                    "backend": m.get("backend", "google"),
                    "variant": m["prompt_variant"],
                    "n": m["num_pairs_scored"],
                    "pair_auc": p["roc_auc"],
                    "decision_accuracy": dec["accuracy"],
                    "f1": dec["f1"],
                    "yes_rate": dec["yes_rate"],
                    "yes_rate_pos": dec["yes_rate_on_positives"],
                    "yes_rate_neg": dec["yes_rate_on_negatives"],
                    "mean_confidence": dec["mean_confidence"],
                    "mean_confidence_correct": dec["mean_confidence_when_correct"],
                    "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                    "hard_auc": h["hard"]["pair_auc"],
                    "easy_auc": h["easy"]["pair_auc"],
                }
            )

    rows.sort(key=lambda r: -r["pair_auc"])
    return {
        "generated_at": d["generated_at"],
        "protocol_note": d["protocol_note"],
        "rows": rows,
        "judge_detail": judge_detail,
    }


# Judge runs that have a metrics_all.json (200-pair) alongside their
# metrics_holdout.json — all four do, since every llm_judge run scores both
# splits from the same cached verdicts.
ALL_JUDGE_RUNS = JUDGE_RUNS
ALL_FOCUSED_JUDGE_RUNS = FOCUSED_JUDGE_RUNS
ALL_SEEKER_BG_JUDGE_RUNS = SEEKER_BG_JUDGE_RUNS
ALL_LOOKINGFOR_ONLY_JUDGE_RUNS = LOOKINGFOR_ONLY_JUDGE_RUNS
ALL_POSITIONING_ONLY_JUDGE_RUNS = POSITIONING_ONLY_JUDGE_RUNS

FIELD_SELECTED_ALL = {
    "Voyage-4-large (field-selected: positioning+lookingFor+query / positioning+background+lookingFor)": VOYAGE_LARGE_FIELD_SELECTED_ALL,
    "Voyage-4-nano (field-selected, same fields)": VOYAGE_NANO_FIELD_SELECTED_ALL,
    "Qwen3-Embedding-8B (field-selected, same fields)": QWEN3_FIELD_SELECTED_ALL,
}

FIELD_SELECTED_HOLDOUT = {
    "Voyage-4-large (field-selected: positioning+lookingFor+query / positioning+background+lookingFor)": VOYAGE_LARGE_FIELD_SELECTED_HOLDOUT,
    "Voyage-4-nano (field-selected, same fields)": VOYAGE_NANO_FIELD_SELECTED_HOLDOUT,
    "Qwen3-Embedding-8B (field-selected, same fields)": QWEN3_FIELD_SELECTED_HOLDOUT,
}


def build_data_all() -> dict:
    """Same shape as build_data(), but scored on all 200 real pairs, not the
    69-pair holdout — includes two rows build_data() does not: Voyage-4-large
    and Voyage-4-nano re-run with the identical field selection + searchQuery
    the focused LLM judge uses (baselines/voyage_*_field_selected), so those
    two rows are directly comparable to the focused-judge rows on both text
    packing and population. See docs/voyage-field-selected-experiment.md.
    """
    d = json.loads(ALL_BASELINE_JSON.read_text())
    baselines = d["baselines"]

    rows = []
    for key, v in baselines.items():
        p, h = v["pair"], v["slices"]["neg_hardness"]
        rows.append(
            {
                "name": LABELS.get(key, key),
                "kind": "baseline",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": v["retrieval"]["mrr"],
                "seeker_fields": FULL_PROFILE_FIELDS,
                "candidate_fields": FULL_PROFILE_FIELDS,
            }
        )

    for name, path in FIELD_SELECTED_ALL.items():
        m = json.loads(path.read_text())
        p, h = m["pair"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": name,
                "kind": "field_selected",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": m["retrieval"]["mrr"],
                "seeker_fields": FOCUSED_SEEKER_FIELDS,
                "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
            }
        )

    judge_detail = []
    for name, dirname in ALL_JUDGE_RUNS.items():
        path = LLM_JUDGE_ARTIFACTS / dirname / "metrics_all.json"
        m = json.loads(path.read_text())
        p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": f"LLM judge: {name}",
                "kind": "llm_judge",
                "gets_query": False,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": None,
                "seeker_fields": FULL_PROFILE_FIELDS,
                "candidate_fields": FULL_PROFILE_FIELDS,
            }
        )
        judge_detail.append(
            {
                "name": name,
                "backend": m.get("backend", "openrouter"),
                "variant": m["prompt_variant"],
                "n": m["num_pairs_scored"],
                "pair_auc": p["roc_auc"],
                "decision_accuracy": dec["accuracy"],
                "f1": dec["f1"],
                "yes_rate": dec["yes_rate"],
                "yes_rate_pos": dec["yes_rate_on_positives"],
                "yes_rate_neg": dec["yes_rate_on_negatives"],
                "mean_confidence": dec["mean_confidence"],
                "mean_confidence_correct": dec["mean_confidence_when_correct"],
                "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
            }
        )

    for name, dirname in ALL_FOCUSED_JUDGE_RUNS.items():
        path = FOCUSED_JUDGE_ARTIFACTS / dirname / "metrics_all.json"
        m = json.loads(path.read_text())
        p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": f"LLM judge: {name}",
                "kind": "llm_judge",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": None,
                "seeker_fields": FOCUSED_SEEKER_FIELDS,
                "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
            }
        )
        judge_detail.append(
            {
                "name": name,
                "backend": m.get("backend", "openrouter"),
                "variant": m["prompt_variant"],
                "n": m["num_pairs_scored"],
                "pair_auc": p["roc_auc"],
                "decision_accuracy": dec["accuracy"],
                "f1": dec["f1"],
                "yes_rate": dec["yes_rate"],
                "yes_rate_pos": dec["yes_rate_on_positives"],
                "yes_rate_neg": dec["yes_rate_on_negatives"],
                "mean_confidence": dec["mean_confidence"],
                "mean_confidence_correct": dec["mean_confidence_when_correct"],
                "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
            }
        )

    for name, dirname in ALL_SEEKER_BG_JUDGE_RUNS.items():
        path = SEEKER_BG_JUDGE_ARTIFACTS / dirname / "metrics_all.json"
        m = json.loads(path.read_text())
        p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
        rows.append(
            {
                "name": f"LLM judge: {name}",
                "kind": "llm_judge",
                "gets_query": True,
                "pair_auc": p["roc_auc"],
                "accuracy_at_half": p["accuracy_at_0.5"],
                "best_f1": p["best_f1"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
                "mrr": None,
                "seeker_fields": SEEKER_BG_FIELDS,
                "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
            }
        )
        judge_detail.append(
            {
                "name": name,
                "backend": m.get("backend", "google"),
                "variant": m["prompt_variant"],
                "n": m["num_pairs_scored"],
                "pair_auc": p["roc_auc"],
                "decision_accuracy": dec["accuracy"],
                "f1": dec["f1"],
                "yes_rate": dec["yes_rate"],
                "yes_rate_pos": dec["yes_rate_on_positives"],
                "yes_rate_neg": dec["yes_rate_on_negatives"],
                "mean_confidence": dec["mean_confidence"],
                "mean_confidence_correct": dec["mean_confidence_when_correct"],
                "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                "hard_auc": h["hard"]["pair_auc"],
                "easy_auc": h["easy"]["pair_auc"],
            }
        )

    for artifacts_root, run_map, seeker_fields in (
        (LOOKINGFOR_ONLY_JUDGE_ARTIFACTS, ALL_LOOKINGFOR_ONLY_JUDGE_RUNS, LOOKINGFOR_ONLY_FIELDS),
        (POSITIONING_ONLY_JUDGE_ARTIFACTS, ALL_POSITIONING_ONLY_JUDGE_RUNS, POSITIONING_ONLY_FIELDS),
    ):
        for name, dirname in run_map.items():
            path = artifacts_root / dirname / "metrics_all.json"
            m = json.loads(path.read_text())
            p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
            rows.append(
                {
                    "name": f"LLM judge: {name}",
                    "kind": "llm_judge",
                    "gets_query": True,
                    "pair_auc": p["roc_auc"],
                    "accuracy_at_half": p["accuracy_at_0.5"],
                    "best_f1": p["best_f1"],
                    "hard_auc": h["hard"]["pair_auc"],
                    "easy_auc": h["easy"]["pair_auc"],
                    "mrr": None,
                    "seeker_fields": seeker_fields,
                    "candidate_fields": FOCUSED_CANDIDATE_FIELDS,
                }
            )
            judge_detail.append(
                {
                    "name": name,
                    "backend": m.get("backend", "google"),
                    "variant": m["prompt_variant"],
                    "n": m["num_pairs_scored"],
                    "pair_auc": p["roc_auc"],
                    "decision_accuracy": dec["accuracy"],
                    "f1": dec["f1"],
                    "yes_rate": dec["yes_rate"],
                    "yes_rate_pos": dec["yes_rate_on_positives"],
                    "yes_rate_neg": dec["yes_rate_on_negatives"],
                    "mean_confidence": dec["mean_confidence"],
                    "mean_confidence_correct": dec["mean_confidence_when_correct"],
                    "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                    "hard_auc": h["hard"]["pair_auc"],
                    "easy_auc": h["easy"]["pair_auc"],
                }
            )


    # Candidate-side field permutations — 200-pair only, no holdout run.
    for artifacts_root, run_map, candidate_fields in (
        (CAND_POS_BACK_JUDGE_ARTIFACTS, CAND_POS_BACK_JUDGE_RUNS, CAND_POS_BACK_FIELDS),
        (CAND_POS_LOOK_JUDGE_ARTIFACTS, CAND_POS_LOOK_JUDGE_RUNS, CAND_POS_LOOK_FIELDS),
        (CAND_BACK_LOOK_JUDGE_ARTIFACTS, CAND_BACK_LOOK_JUDGE_RUNS, CAND_BACK_LOOK_FIELDS),
    ):
        for name, dirname in run_map.items():
            path = artifacts_root / dirname / "metrics_all.json"
            m = json.loads(path.read_text())
            p, dec, h = m["pair"], m["decision"], m["slices"]["neg_hardness"]
            rows.append(
                {
                    "name": f"LLM judge: {name}",
                    "kind": "llm_judge",
                    "gets_query": True,
                    "pair_auc": p["roc_auc"],
                    "accuracy_at_half": p["accuracy_at_0.5"],
                    "best_f1": p["best_f1"],
                    "hard_auc": h["hard"]["pair_auc"],
                    "easy_auc": h["easy"]["pair_auc"],
                    "mrr": None,
                    "seeker_fields": FOCUSED_SEEKER_FIELDS,
                    "candidate_fields": candidate_fields,
                }
            )
            judge_detail.append(
                {
                    "name": name,
                    "backend": m.get("backend", "google"),
                    "variant": m["prompt_variant"],
                    "n": m["num_pairs_scored"],
                    "pair_auc": p["roc_auc"],
                    "decision_accuracy": dec["accuracy"],
                    "f1": dec["f1"],
                    "yes_rate": dec["yes_rate"],
                    "yes_rate_pos": dec["yes_rate_on_positives"],
                    "yes_rate_neg": dec["yes_rate_on_negatives"],
                    "mean_confidence": dec["mean_confidence"],
                    "mean_confidence_correct": dec["mean_confidence_when_correct"],
                    "mean_confidence_wrong": dec["mean_confidence_when_wrong"],
                    "hard_auc": h["hard"]["pair_auc"],
                    "easy_auc": h["easy"]["pair_auc"],
                }
            )
    rows.sort(key=lambda r: -r["pair_auc"])
    return {
        "generated_at": d["generated_at"],
        "protocol_note": d["protocol_note"],
        "rows": rows,
        "judge_detail": judge_detail,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LLM judge vs. embedding baselines — Dorby AI</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --panel: #ffffff;
    --baseline: __COLOR_BASELINE_LIGHT__;
    --judge: __COLOR_JUDGE_LIGHT__;
    --good: #0ca30c;
    --good-bg: #e3f6e3;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --panel: #161615;
      --baseline: __COLOR_BASELINE_DARK__;
      --judge: __COLOR_JUDGE_DARK__;
      --good: #0ca30c;
      --good-bg: #103010;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --panel: #161615;
    --baseline: __COLOR_BASELINE_DARK__;
    --judge: __COLOR_JUDGE_DARK__;
    --good: #0ca30c;
    --good-bg: #103010;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink);
    background: var(--page);
    line-height: 1.45;
  }
  :where(a, button, input, [tabindex]):focus-visible {
    outline: 2px solid var(--good);
    outline-offset: 2px;
    border-radius: 4px;
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }

  .wrap { width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }

  header.hero { display: grid; gap: 0.6rem; margin-bottom: 1.6rem; }
  .brand { font-size: clamp(1.6rem, 3.6vw, 2.2rem); letter-spacing: -0.02em; margin: 0; font-weight: 700; display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; }
  .n-badge {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.02em; text-transform: uppercase;
    color: var(--good); background: var(--good-bg); border-radius: 999px; padding: 0.25rem 0.6rem;
    vertical-align: middle;
  }
  .lede { margin: 0; max-width: 60rem; color: var(--ink-2); font-size: 0.95rem; }
  .lede code { background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 0.05rem 0.35rem; font-size: 0.85em; }
  .meta { margin: 0; font-size: 0.8rem; color: var(--muted); }

  .legend { display: flex; gap: 1.2rem; align-items: center; margin-top: 0.9rem; font-size: 0.85rem; color: var(--ink-2); }
  .legend .swatch { display: inline-flex; align-items: center; gap: 0.45rem; }
  .legend .dot { width: 0.7rem; height: 0.7rem; border-radius: 3px; flex: none; }

  .chart-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.4rem 1.4rem 1rem; margin-top: 1rem;
  }
  .chart-title { font-size: 0.95rem; font-weight: 700; margin: 0 0 0.15rem; }
  .chart-sub { font-size: 0.78rem; color: var(--muted); margin: 0 0 1rem; }

  .bar-row {
    display: grid; grid-template-columns: 15rem 1fr 4.2rem; align-items: center;
    gap: 0.7rem; padding: 0.28rem 0; position: relative;
  }
  .bar-row .name {
    font-size: 0.82rem; text-align: right; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; color: var(--ink-2);
  }
  .bar-row.is-judge .name { color: var(--ink); font-weight: 600; }
  .bar-track { position: relative; height: 1.15rem; background: var(--grid); border-radius: 4px; overflow: visible; }
  .bar-fill {
    position: absolute; top: 0; left: 0; height: 100%; border-radius: 4px;
    min-width: 4px;
  }
  .bar-row .value {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-size: 0.82rem; text-align: right;
  }
  .bar-row:hover .bar-track { outline: 2px solid var(--ink); outline-offset: 1px; }
  .chance-line {
    position: absolute; top: -0.2rem; bottom: -0.2rem; width: 1px;
    background: var(--muted); opacity: 0.55;
  }
  .chance-label {
    position: absolute; font-size: 0.68rem; color: var(--muted); top: -1.05rem; transform: translateX(-50%);
  }

  .fields-tooltip {
    position: fixed; z-index: 50; pointer-events: none; opacity: 0; visibility: hidden;
    transform: translateY(4px); transition: opacity 0.1s ease, transform 0.1s ease;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18); padding: 0.6rem 0.7rem;
    max-width: min(26rem, 90vw);
  }
  .fields-tooltip.is-visible { opacity: 1; visibility: visible; transform: translateY(0); }
  .fields-tooltip .tt-name { font-size: 0.78rem; font-weight: 700; margin: 0 0 0.4rem; color: var(--ink); }
  .fields-tooltip table { border-collapse: collapse; width: 100%; table-layout: fixed; }
  .fields-tooltip th {
    text-align: left; font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.02em; color: var(--muted); padding: 0 0.5rem 0.3rem 0; white-space: normal;
  }
  .fields-tooltip td {
    text-align: left; vertical-align: top; font-size: 0.75rem; color: var(--ink-2);
    padding: 0 0.5rem 0 0; white-space: normal; font-family: inherit; font-variant-numeric: normal;
  }
  .fields-tooltip .tt-field {
    display: inline-block; background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.02rem 0.35rem; margin: 0 0.2rem 0.2rem 0; font-size: 0.72rem;
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  }
  .fields-tooltip .tt-check { font-size: 0.95rem; }
  .fields-tooltip .tt-check.yes { color: var(--good); }
  .fields-tooltip .tt-check.no { color: var(--muted); }

  .section-title { margin: 1.7rem 0 0.2rem; font-size: 0.95rem; font-weight: 700; }
  .section-divider { border: none; border-top: 1px solid var(--border); margin: 2.6rem 0 1.6rem; }
  .brand-sub {
    font-size: clamp(1.2rem, 2.6vw, 1.5rem); letter-spacing: -0.02em; margin: 0 0 0.6rem;
    font-weight: 700; display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap;
  }
  .fieldsel-tag {
    font-size: 0.72rem; font-weight: 700; color: var(--judge); background: color-mix(in srgb, var(--judge) 14%, transparent);
    border-radius: 4px; padding: 0.05rem 0.35rem;
  }
  .section-sub { margin: 0 0 0.4rem; font-size: 0.78rem; color: var(--muted); }

  .findings { display: grid; gap: 0.7rem; margin-top: 0.6rem; }
  .finding {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.87rem; color: var(--ink-2);
  }
  .finding b { color: var(--ink); }

  .table-scroll {
    margin-top: 0.8rem; overflow-x: auto; border: 1px solid var(--border);
    border-radius: 10px; background: var(--panel);
  }
  table { border-collapse: collapse; width: 100%; font-size: 0.84rem; }
  thead th {
    position: sticky; top: 0; background: var(--surface-1); text-align: right;
    padding: 0.6rem 0.8rem; font-weight: 700; white-space: nowrap;
    border-bottom: 1px solid var(--grid);
  }
  thead th:first-child { text-align: left; }
  tbody td, tbody th {
    padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--grid); white-space: nowrap;
  }
  tbody th { text-align: left; font-weight: 600; color: var(--ink-2); }
  tbody td {
    text-align: right; color: var(--ink);
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-size: 0.82rem;
  }
  tbody tr:last-child td, tbody tr:last-child th { border-bottom: none; }
  tbody tr:hover td, tbody tr:hover th { background: color-mix(in srgb, var(--ink) 4%, transparent); }
  tbody tr.is-judge th { color: var(--judge); }
  td.na { color: var(--muted); }

  @media (max-width: 640px) {
    .wrap { width: min(100%, calc(100% - 1.25rem)); padding-top: 1.25rem; }
    .bar-row { grid-template-columns: 8rem 1fr 3.4rem; }
  }
</style>
</head>
<body>
  <div class="fields-tooltip" id="fields-tooltip" role="tooltip"></div>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">LLM judge vs. embedding baselines <span class="n-badge">n = 69 (holdout)</span></h1>
      <p class="lede">
        Can an LLM predict accept/decline from two full profiles alone, with
        <strong>no <code>searchQuery</code></strong> — the one thing every
        embedding baseline below gets? <strong>Every chart and table on this
        page is scored on the frozen 69-pair real holdout split</strong>
        (<code>data/synthetic/seed_split.json</code>'s <code>eval_pair_ids</code>),
        not the full 200-pair real dataset — this is the smaller, harder,
        user-disjoint slice used for the project's one-time final comparisons,
        matching <code>docs/baseline-results-holdout.md</code>. 200-pair
        numbers exist per-experiment in <code>docs/llm-judge-experiment.md</code>
        and <code>docs/llm-judge-focused-prompt-experiment.md</code> but are not
        plotted here, to keep every row on this page the same population.
      </p>
      <p class="meta" id="meta-line"></p>
      <div class="legend">
        <span class="swatch"><span class="dot" style="background:var(--baseline)"></span>Embedding baseline (gets the query)</span>
        <span class="swatch"><span class="dot" style="background:var(--judge)"></span>LLM judge (see "Gets query?" column — most withhold it, one variant doesn't)</span>
      </div>
    </header>

    <main>
      <div class="chart-card">
        <p class="chart-title">Pair ROC-AUC, ranked <span class="n-badge">n = 69 (holdout)</span></p>
        <p class="chart-sub">Dashed line = chance (0.5). Bar length is AUC on this 0.5–1.0 scale.</p>
        <div id="chart-auc"></div>
      </div>

      <div class="chart-card">
        <p class="chart-title">Hard-negative AUC vs. easy-negative AUC <span class="n-badge">n = 69 (holdout)</span></p>
        <p class="chart-sub">
          Every real negative is production's own false positive — a plausible-looking intro a human still declined —
          so hard negatives (lexically similar to the query, superficially plausible) are the only population that exists
          at serving time. Sorted by hard-neg AUC.
        </p>
        <div id="chart-hardness"></div>
      </div>

      <p class="section-title">Findings</p>
      <div class="findings" id="findings"></div>

      <p class="section-title">Full comparison table <span class="n-badge">n = 69 (holdout)</span></p>
      <p class="section-sub">All metrics, matched 69-pair holdout — 12 embedding-baseline rows + 9 LLM-judge (model, framing, backend) rows. MRR is retrieval-only — no LLM-judge row has one (see below).</p>
      <div class="table-scroll">
        <table id="full-table"></table>
      </div>

      <p class="section-title">LLM-judge run detail <span class="n-badge">n = 69 (holdout)</span></p>
      <p class="section-sub">All nine (model, framing, backend) combinations tested — decision behavior, not just the AUC summary above.</p>
      <div class="table-scroll">
        <table id="judge-table"></table>
      </div>

      <hr class="section-divider" />

      <h2 class="brand-sub">All 200 real pairs <span class="n-badge">n = 200 (all)</span></h2>
      <p class="lede">
        Same comparison, scored on the full 200-pair real dataset (train + holdout, no synthetic
        pairs) instead of the frozen 69-pair holdout above — a larger, more stable sample, but not
        the population <code>docs/baseline-results-holdout.md</code>'s one-time final numbers use.
        Adds two rows the holdout section doesn't have: <b>Voyage-4-large and Voyage-4-nano
        re-run with the exact same field selection and searchQuery as the focused LLM judge</b>
        (<code>positioning</code> + <code>lookingFor</code> + query for the seeker,
        <code>positioning</code> + <code>background</code> + <code>lookingFor</code> for the
        candidate — see <code>baselines/voyage_large_field_selected/</code> and
        <code>baselines/voyage_nano_field_selected/</code>), so the LLM-judge and embedding
        approaches can be compared on identical inputs, not just identical pairs.
      </p>

      <div class="chart-card">
        <p class="chart-title">Pair ROC-AUC, ranked <span class="n-badge">n = 200 (all)</span></p>
        <p class="chart-sub">Dashed line = chance (0.5). Bar length is AUC on this 0.5–1.0 scale.</p>
        <div id="chart-auc-all"></div>
      </div>

      <div class="chart-card">
        <p class="chart-title">Hard-negative AUC vs. easy-negative AUC <span class="n-badge">n = 200 (all)</span></p>
        <p class="chart-sub">Sorted by hard-neg AUC. Same rationale as the holdout chart above.</p>
        <div id="chart-hardness-all"></div>
      </div>

      <p class="section-title">Full comparison table <span class="n-badge">n = 200 (all)</span></p>
      <p class="section-sub">
        Baseline rows (full-profile text) from <code>docs/baseline-results-all.json</code>; the two
        <span class="fieldsel-tag">field-selected</span> Voyage rows use the focused judge's exact
        field selection instead. MRR is retrieval-only — no LLM-judge row has one.
      </p>
      <div class="table-scroll">
        <table id="full-table-all"></table>
      </div>

      <p class="section-title">LLM-judge run detail <span class="n-badge">n = 200 (all)</span></p>
      <p class="section-sub">All twelve (model, framing, backend) combinations, scored on all 200 real pairs — includes three candidate-side field-permutation runs not scored on the holdout.</p>
      <div class="table-scroll">
        <table id="judge-table-all"></table>
      </div>
    </main>
  </div>

  <script id="data" type="application/json">__EMBEDDED_JSON__</script>
  <script id="data-all" type="application/json">__EMBEDDED_JSON_ALL__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("data").textContent);

    document.getElementById("meta-line").textContent =
      `${DATA.protocol_note} · baseline numbers generated ${DATA.generated_at}`;

    function fmt(x, digits = 4) {
      return x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits);
    }
    function pct(x, digits = 1) {
      return x === null || x === undefined ? "—" : (x * 100).toFixed(digits) + "%";
    }

    // ---- shared hover tooltip: seeker fields / candidate fields / query yes-no ----
    const tooltipEl = document.getElementById("fields-tooltip");

    function fieldChips(fields) {
      if (!fields || !fields.length) return '<span class="tt-field">(none)</span>';
      return fields.map((f) => `<span class="tt-field">${f}</span>`).join("");
    }

    function showFieldsTooltip(row, evt) {
      const r = row._data;
      tooltipEl.innerHTML = `
        <p class="tt-name">${r.name}</p>
        <table>
          <thead>
            <tr><th>Seeker fields</th><th>Candidate fields</th><th>Query</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>${fieldChips(r.seeker_fields)}</td>
              <td>${fieldChips(r.candidate_fields)}</td>
              <td><span class="tt-check ${r.gets_query ? "yes" : "no"}">${r.gets_query ? "☑" : "☐"}</span></td>
            </tr>
          </tbody>
        </table>
      `;
      tooltipEl.classList.add("is-visible");
      positionFieldsTooltip(evt);
    }

    function positionFieldsTooltip(evt) {
      const pad = 14;
      const ttRect = tooltipEl.getBoundingClientRect();
      let x = evt.clientX + pad;
      let y = evt.clientY + pad;
      if (x + ttRect.width > window.innerWidth - pad) x = evt.clientX - ttRect.width - pad;
      if (y + ttRect.height > window.innerHeight - pad) y = evt.clientY - ttRect.height - pad;
      tooltipEl.style.left = `${Math.max(pad, x)}px`;
      tooltipEl.style.top = `${Math.max(pad, y)}px`;
    }

    function hideFieldsTooltip() {
      tooltipEl.classList.remove("is-visible");
    }

    // ---- bar chart: Pair AUC, ranked ----
    function renderBarChart(container, rows, valueKey, opts = {}) {
      const min = opts.min ?? 0.5, max = opts.max ?? 0.7;
      const chanceAt = opts.chance;
      const el = document.getElementById(container);
      el.innerHTML = "";
      for (const r of rows) {
        const v = r[valueKey];
        if (v === null || v === undefined) continue;
        const pctWidth = Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
        const row = document.createElement("div");
        row.className = "bar-row" + (r.kind === "llm_judge" ? " is-judge" : "");
        row.innerHTML = `
          <div class="name">${r.name}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${pctWidth}%; background:${r.kind === "llm_judge" ? "var(--judge)" : "var(--baseline)"}"></div>
            ${chanceAt !== undefined ? `<div class="chance-line" style="left:${((chanceAt - min) / (max - min)) * 100}%"></div>` : ""}
          </div>
          <div class="value">${fmt(v)}</div>
        `;
        row._data = r;
        row.addEventListener("mouseenter", (evt) => showFieldsTooltip(row, evt));
        row.addEventListener("mousemove", positionFieldsTooltip);
        row.addEventListener("mouseleave", hideFieldsTooltip);
        el.appendChild(row);
      }
    }

    renderBarChart("chart-auc", DATA.rows, "pair_auc", { min: 0.45, max: 0.68, chance: 0.5 });

    const hardnessRows = [...DATA.rows]
      .filter((r) => r.hard_auc !== null && r.hard_auc !== undefined)
      .sort((a, b) => b.hard_auc - a.hard_auc);
    renderBarChart("chart-hardness", hardnessRows, "hard_auc", { min: 0.35, max: 0.68, chance: 0.5 });

    // ---- findings ----
    const findingsEl = document.getElementById("findings");
    const findings = [
      `<b>gemini-3.1-flash-lite (naive)</b> is the only LLM judge that beats Voyage-4-large — Boardy's own production
       model — on pair AUC (0.6358 vs 0.6086), while never seeing the search query Voyage gets.`,
      `<b>Hard-negative AUC tells a different story than the headline ranking.</b> Both Bedrock judges
       (Gemma 3 27B, Qwen3-32B) actually edge out flash-lite here (0.6216 / 0.6224 vs 0.6466 — still behind, but the
       gap shrinks a lot). Every embedding baseline except the strongest few collapses much harder from easy to hard
       negatives than any LLM judge does.`,
      `<b>The "calibrated" prompt — telling the model production already vetted relevance and the base rate is 50/50 —
       made flash-lite worse</b> (AUC 0.6358 → 0.5901), by making it answer "yes" far less often (56.5% → 30.4%)
       rather than discriminating better.`,
      `<b>All four LLM-judge combinations land in the same 0.58–0.64 band</b> — comfortably above chance, well short of
       Qwen3-Embedding-8B's 0.6595 ceiling. None of this is deployable: a per-candidate LLM call is out of scope
       under the project's &lt;100ms serving budget, and there's no retrieval column for any LLM-judge row because
       ranking a full candidate corpus per query would need tens of thousands of calls.`,
      `<b>Giving flash-lite the search query back — while trimming each profile to only positioning/lookingFor
       (seeker) and positioning/background/lookingFor (candidate)</b> — via OpenRouter landed at pair AUC 0.6203,
       between the naive no-query judge (0.6358) and Voyage-4-large (0.6086). The <b>same prompt called through
       Google's own API directly</b> (not OpenRouter) scored pair AUC 0.6530 — 2nd-best of anything in this project —
       and hard-negative AUC 0.6784, the best of any config tested. See docs/llm-judge-focused-prompt-experiment.md
       for the full OpenRouter-vs-direct-API gap this surfaced.`,
      `<b>Adding the seeker's own <code>background</code> field to that same prompt made it worse</b>, consistently:
       pair AUC 0.6530→0.6302 on the holdout and 0.6451→0.6220 on all 200 pairs, with hard-neg AUC also dropping
       (0.6784→0.6690) — small but the same direction both times. See docs/llm-judge-seeker-background-experiment.md.`,
      `<b>The same field selection on Qwen3-Embedding-8B (not an LLM judge — the strongest embedding model in this
       project) sets a new project-wide high: pair AUC 0.6862</b>, ahead of its own full-profile score (0.6595) and
       every LLM judge. But its hard-negative AUC is only 0.5689 (vs. 0.8552 easy-neg) — the same lexical-overlap
       pattern every embedding baseline shows, unlike the LLM judge's 0.6784. Best <i>overall</i> AUC and best
       <i>hard-negative</i> AUC are different models. See docs/qwen3-embedding-field-selected-experiment.md.`,
      `<b>Dropping the focused prompt's seeker down to <code>lookingFor</code> alone beats the two-field version on
       the holdout</b> — pair AUC 0.6530→0.6698 and hard-negative AUC 0.6784→0.7164, the best hard-negative AUC of
       any config in this project. <code>positioning</code> alone is much weaker (holdout pair AUC 0.5914, hard-neg
       0.5621) — close to chance-plus, and closer to an embedding baseline's usual easy/hard pattern than to the LLM
       judge's. <code>lookingFor</code>, not <code>positioning</code>, is what the focused prompt's advantage runs
       on. See docs/llm-judge-seeker-field-isolation-experiment.md.`,
      `<b>On the candidate side, no two-field subset beats the full three-field candidate</b> (positioning+background+
       lookingFor, 0.6451) — every permutation drops pair AUC, but not evenly. Dropping <code>positioning</code>
       hurts least (<code>background+lookingFor</code>: 0.6356, hard-neg 0.6449, closest to the full config).
       Dropping <code>background</code> hurts hard-negative discrimination the most — <code>positioning+lookingFor</code>
       is the only candidate config where hard-neg AUC (0.5792) falls <i>below</i> easy-neg AUC (0.7082), the pattern
       every embedding baseline shows and the opposite of what makes this judge distinctive elsewhere. 200-pair only
       (no holdout run for this sweep). See docs/llm-judge-candidate-field-permutation-experiment.md.`,
    ];
    for (const f of findings) {
      const div = document.createElement("div");
      div.className = "finding";
      div.innerHTML = f;
      findingsEl.appendChild(div);
    }

    // ---- full comparison table ----
    const fullCols = [
      ["Model", (r) => r.name + (r.kind === "field_selected" ? ' <span class="fieldsel-tag">field-selected</span>' : ""), "left"],
      ["Pair AUC", (r) => fmt(r.pair_auc)],
      ["Hard-neg AUC", (r) => fmt(r.hard_auc)],
      ["Easy-neg AUC", (r) => fmt(r.easy_auc)],
      ["Best F1", (r) => fmt(r.best_f1)],
      ["Acc @ 0.5", (r) => fmt(r.accuracy_at_half)],
      ["MRR", (r) => (r.mrr === null || r.mrr === undefined ? '<span class="na">n/a</span>' : fmt(r.mrr))],
      ["Gets query?", (r) => (r.gets_query ? "yes" : "no")],
    ];
    function renderFullTable(elId, rows) {
      const el = document.getElementById(elId);
      el.innerHTML = `
        <thead><tr>${fullCols.map(([label], i) => `<th style="${i===0?'text-align:left':''}">${label}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map((r) => `
            <tr class="${r.kind === "llm_judge" ? "is-judge" : ""}">
              ${fullCols.map(([label, fn], i) => i === 0
                ? `<th>${fn(r)}</th>`
                : `<td>${fn(r)}</td>`
              ).join("")}
            </tr>
          `).join("")}
        </tbody>
      `;
    }
    renderFullTable("full-table", DATA.rows);

    // ---- LLM-judge detail table ----
    const judgeCols = [
      ["Run", (r) => r.name, "left"],
      ["Backend", (r) => r.backend],
      ["n", (r) => r.n],
      ["Pair AUC", (r) => fmt(r.pair_auc)],
      ["Decision acc.", (r) => fmt(r.decision_accuracy)],
      ["F1", (r) => fmt(r.f1)],
      ["Yes-rate", (r) => pct(r.yes_rate)],
      ["Yes-rate (pos)", (r) => pct(r.yes_rate_pos)],
      ["Yes-rate (neg)", (r) => pct(r.yes_rate_neg)],
      ["Mean conf.", (r) => r.mean_confidence.toFixed(1)],
      ["Conf. when right", (r) => r.mean_confidence_correct.toFixed(1)],
      ["Conf. when wrong", (r) => r.mean_confidence_wrong.toFixed(1)],
    ];
    function renderJudgeTable(elId, rows) {
      const el = document.getElementById(elId);
      el.innerHTML = `
        <thead><tr>${judgeCols.map(([label], i) => `<th style="${i===0?'text-align:left':''}">${label}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map((r) => `
            <tr>
              ${judgeCols.map(([label, fn], i) => i === 0
                ? `<th>${fn(r)}</th>`
                : `<td>${fn(r)}</td>`
              ).join("")}
            </tr>
          `).join("")}
        </tbody>
      `;
    }
    renderJudgeTable("judge-table", DATA.judge_detail);

    // ---- n=200 (all real pairs) section ----
    const DATA_ALL = JSON.parse(document.getElementById("data-all").textContent);

    renderBarChart("chart-auc-all", DATA_ALL.rows, "pair_auc", { min: 0.45, max: 0.68, chance: 0.5 });
    const hardnessRowsAll = [...DATA_ALL.rows]
      .filter((r) => r.hard_auc !== null && r.hard_auc !== undefined)
      .sort((a, b) => b.hard_auc - a.hard_auc);
    renderBarChart("chart-hardness-all", hardnessRowsAll, "hard_auc", { min: 0.35, max: 0.68, chance: 0.5 });
    renderFullTable("full-table-all", DATA_ALL.rows);
    renderJudgeTable("judge-table-all", DATA_ALL.judge_detail);
  </script>
</body>
</html>
"""


def main() -> int:
    data = build_data()
    data_all = build_data_all()
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", json.dumps(data))
    html = html.replace("__EMBEDDED_JSON_ALL__", json.dumps(data_all))
    html = html.replace("__COLOR_BASELINE_LIGHT__", COLOR_BASELINE_LIGHT)
    html = html.replace("__COLOR_BASELINE_DARK__", COLOR_BASELINE_DARK)
    html = html.replace("__COLOR_JUDGE_LIGHT__", COLOR_JUDGE_LIGHT)
    html = html.replace("__COLOR_JUDGE_DARK__", COLOR_JUDGE_DARK)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(html)
    print(f"wrote {DEFAULT_OUT} ({len(data['rows'])} rows, {len(data['judge_detail'])} judge combos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
