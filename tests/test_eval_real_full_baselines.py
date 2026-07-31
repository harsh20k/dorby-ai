"""Pin the baseline all-200 eval against the baselines it claims to reproduce.

Two things must hold for a number from ``eval_real_full.baseline_eval`` to be
comparable to a row in ``docs/baseline-results-holdout.md``:

1. the candidate corpus is built identically (it defines the retrieval task), and
2. restricted to the holdout, the whole pipeline reproduces the published
   metrics exactly — not approximately.

(2) is the real gate: it exercises text serialization, corpus construction,
encoder wiring and all four metric calls end to end, against a number already
committed to the docs. TF-IDF is the vehicle because it needs no model download
and no GPU, so this stays a cheap test.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Every baseline package that carries its own copy of this function.
CORPUS_FN_SOURCES = (
    "baselines/tfidf/eval.py",
    "baselines/hf_embedding/eval.py",
    "baselines/bert_frozen/eval.py",
    "baselines/voyage_nano/eval.py",
)


def _fn_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            # Drop the docstring so wording differences between packages don't
            # register as behavioural drift.
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    raise AssertionError(f"{name} not found in {path}")


@pytest.mark.parametrize("rel", CORPUS_FN_SOURCES)
def test_candidate_corpus_matches_every_baseline(rel: str) -> None:
    """The copy must agree with all four baselines, or retrieval isn't comparable."""
    original = REPO / rel
    if not original.is_file():
        pytest.skip(f"{rel} not present")
    mine = _fn_ast(REPO / "eval_real_full" / "baseline_eval.py", "build_candidate_corpus")
    assert mine == _fn_ast(original, "build_candidate_corpus"), (
        f"build_candidate_corpus has drifted from {rel} — the all-200 numbers "
        "would no longer be comparable to that baseline's published row."
    )


def test_holdout_subset_reproduces_published_tfidf_metrics() -> None:
    """End-to-end gate: holdout must match artifacts/tfidf_holdout/metrics.json."""
    published_path = REPO / "artifacts" / "tfidf_holdout" / "metrics.json"
    if not published_path.is_file() or not (REPO / "data" / "dataset_positive.json").is_file():
        pytest.skip("data/ or the published tfidf holdout artifact is absent")

    from eval_real_full.baseline_eval import run_baseline_eval

    published = json.loads(published_path.read_text())
    metrics = run_baseline_eval(
        kind="tfidf",
        data_dir=REPO / "data",
        split_path=REPO / "data" / "synthetic" / "seed_split.json",
        label="tfidf",
        subsets=("holdout",),
        cache_dir=Path("/tmp/test_eval_real_full_tfidf"),
    )["subsets"]["holdout"]

    assert metrics["n_candidates"] == 65
    for key in ("roc_auc", "average_precision", "best_f1"):
        assert metrics["pair"][key] == pytest.approx(published["pair"][key], abs=1e-12)
    for key in ("mrr", "recall@1", "recall@10", "ndcg@10"):
        assert metrics["retrieval"][key] == pytest.approx(
            published["retrieval"][key], abs=1e-12
        )
    assert metrics["slices"]["neg_hardness"]["hard"]["pair_auc"] == pytest.approx(
        published["slices"]["neg_hardness"]["hard"]["pair_auc"], abs=1e-12
    )
