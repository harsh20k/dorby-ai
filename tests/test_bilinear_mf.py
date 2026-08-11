"""Guards for `bilinear_mf/`.

Two classes of thing are pinned here:

* the **duplicate** — ``build_candidate_corpus`` is copied from
  ``eval_real_full``, so it is compared by AST, per the isolation rule.
* the **anchors** — with the head disabled this package must reproduce the
  published frozen-cosine rows digit-for-digit. Both real bugs found while
  building it (a TF-IDF fit set that silently shifted every IDF weight, and
  retrieval metrics computed in the un-reduced space) were caught by exactly
  this check, which is why it is an assertion and not a comment.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from bilinear_mf import evaluate, features as feat_mod
from bilinear_mf.model import SvdReducer, cosine_pair_scores, train_bilinear

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SPLIT_PATH = DATA_DIR / "synthetic" / "seed_split.json"

# docs/baseline-results-real200.md, "All 200 pairs — corpus 178 candidates".
PUBLISHED_ALL200 = {
    "tfidf": {"pair_auc": 0.5649, "mrr": 0.1313, "recall@1": 0.05},
    "voyage_large": {"pair_auc": 0.5726, "mrr": 0.3102, "recall@1": 0.13},
}

pytestmark = pytest.mark.skipif(
    not SPLIT_PATH.exists(), reason="data/ is gitignored; needs the real checkout"
)


def _normalized_ast(fn) -> str:
    """Source AST with docstring stripped, so only behavior is compared."""
    tree = ast.parse(inspect.getsource(fn).strip())
    body = tree.body[0].body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return ast.dump(tree)


def test_candidate_corpus_matches_eval_real_full():
    """The deliberate copy must stay numerically identical to its original."""
    from eval_real_full import baseline_eval

    assert _normalized_ast(feat_mod.build_candidate_corpus) == _normalized_ast(
        baseline_eval.build_candidate_corpus
    )


@pytest.fixture(scope="module")
def tfidf_features():
    return feat_mod.build_features(
        data_dir=DATA_DIR, split_path=SPLIT_PATH, backbone="tfidf"
    )


def test_population_shape(tfidf_features):
    f = tfidf_features
    assert len(f.labels) == 200
    assert int(f.labels.sum()) == 100
    assert len(f.corpus_ids) == 178
    assert len(set(f.seeker_ids)) == 129
    assert sum(1 for s in f.subsets if s == "train") == 131
    assert sum(1 for s in f.subsets if s == "holdout") == 69


def test_subset_corpus_reproduces_full_corpus(tfidf_features):
    """The no-re-encoding shortcut must agree with the real corpus builder."""
    f = tfidf_features
    ids, emb = evaluate.subset_corpus(f, np.arange(len(f.labels)))
    assert ids == f.corpus_ids
    assert np.allclose(emb, f.corpus_emb, atol=1e-6)


def test_tfidf_cosine_reproduces_published_row(tfidf_features):
    f = tfidf_features
    m = evaluate.cosine_baseline_metrics(
        f, f.seeker_emb, f.cand_emb, np.arange(len(f.labels))
    )
    want = PUBLISHED_ALL200["tfidf"]
    assert m["pair"]["roc_auc"] == pytest.approx(want["pair_auc"], abs=5e-4)
    assert m["retrieval"]["mrr"] == pytest.approx(want["mrr"], abs=5e-4)
    assert m["retrieval"]["recall@1"] == pytest.approx(want["recall@1"], abs=5e-4)


@pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"), reason="needs VOYAGE_API_KEY (cache is free)"
)
def test_voyage_cosine_reproduces_published_row():
    f = feat_mod.build_features(
        data_dir=DATA_DIR, split_path=SPLIT_PATH, backbone="voyage_large"
    )
    m = evaluate.cosine_baseline_metrics(
        f, f.seeker_emb, f.cand_emb, np.arange(len(f.labels))
    )
    want = PUBLISHED_ALL200["voyage_large"]
    assert m["pair"]["roc_auc"] == pytest.approx(want["pair_auc"], abs=5e-4)
    assert m["retrieval"]["mrr"] == pytest.approx(want["mrr"], abs=5e-4)


def test_reduced_space_changes_retrieval(tfidf_features):
    """Regression: retrieval once ranked in the backbone space regardless of `k`.

    Every rank reported an identical MRR, which read as 'compression is
    harmless' when it actually meant the metric was ignoring the model.
    """
    f = tfidf_features
    idx = np.arange(len(f.labels))
    stacked = np.vstack([f.seeker_emb, f.cand_emb, f.corpus_emb])
    mrrs = set()
    for k in (16, 64):
        r = SvdReducer(n_components=k, seed=0).fit(stacked)
        m = evaluate.cosine_baseline_metrics(
            f, r.transform(f.seeker_emb), r.transform(f.cand_emb), idx
        )
        mrrs.add(round(m["retrieval"]["mrr"], 6))
    assert len(mrrs) == 2, "reduced-space retrieval must differ between ranks"


def test_folds_are_seeker_disjoint(tfidf_features):
    f = tfidf_features
    idx = np.arange(len(f.labels))
    folds = evaluate.seeker_folds(f.seeker_ids, idx, n_folds=10)
    assert sum(len(x) for x in folds) == len(idx)
    seen: set[str] = set()
    for fold in folds:
        seekers = {f.seeker_ids[int(i)] for i in fold}
        assert not (seekers & seen), "a seeker appeared in two folds"
        seen |= seekers


def test_bilinear_reduces_to_cosine_at_zero_init():
    """With no residual the head *is* frozen cosine — the baseline it starts from."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=(40, 16)).astype(np.float32)
    c = rng.normal(size=(40, 16)).astype(np.float32)
    y = (rng.random(40) > 0.5).astype(np.int64)
    model = train_bilinear(
        s, c, y, rank=4, weight_decay=0.0, lr=0.0, steps=1, init_scale=0.0, seed=0
    )
    assert np.allclose(model.pair_scores(s, c), cosine_pair_scores(s, c), atol=1e-6)


def test_score_matrix_agrees_with_pair_scores():
    """Ranking and pair scoring must be the same function, or the two metric
    families would describe different models."""
    rng = np.random.default_rng(1)
    s = rng.normal(size=(12, 8)).astype(np.float32)
    c = rng.normal(size=(12, 8)).astype(np.float32)
    y = (rng.random(12) > 0.5).astype(np.int64)
    model = train_bilinear(
        s, c, y, rank=3, weight_decay=1e-3, lr=1e-2, steps=5, init_scale=1e-2, seed=0
    )
    matrix = model.score_matrix(s, c)
    assert np.allclose(np.diag(matrix), model.pair_scores(s, c), atol=1e-5)
