"""Every knob for the sectioned MoE, in one place.

Sizing rationale differs from ``moe_reranker/config.py`` in one important way:
that model was fit on 111 *pairs*, this one on ~2,300 *(pair, section) rows*, so
the experts can afford to be actual two-layer networks rather than a linear map
with a kink. See ``model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Free, local, no GPU. Reproduces the repo's lexical channel exactly.
ENCODER_TFIDF = "tfidf"
#: Requires embeddings produced by ``modal_encode.py`` (a paid Modal run).
ENCODER_QWEN3 = "qwen3"


@dataclass
class SectionedConfig:
    # ---- data ----
    data_dir: Path = Path("data")
    split_path: Path = Path("data/synthetic/seed_split.json")
    #: Promoted `cmsynth*` pairs stay quarantined — see
    #: data/archive/batch_500_001_quarantined/README.md.
    include_synth: bool = False

    # ---- sections ----
    #: Sections per seeker are median 5 with a tail to 116. The tail is a handful
    #: of seekers whose lookingFor is effectively a document; letting them
    #: contribute 116 rows each would let a few people dominate training.
    max_sections: int = 8
    #: A "section" shorter than this is a heading with no content — drop it.
    min_section_chars: int = 40

    # ---- embeddings ----
    encoder: str = ENCODER_TFIDF
    #: Where ``modal_encode.py`` writes / where the qwen3 encoder reads.
    embedding_dir: Path = Path("artifacts/moe_sectioned/embeddings")
    #: Reduce embeddings to this many dims **before** any learned layer, by PCA
    #: fitted on training rows only. 0 disables it.
    #:
    #: This is not a nicety. Without it the learned projections *are* the model:
    #: a 20,000-d TF-IDF vector gives interaction_proj 20000x32 = 640k parameters
    #: and gate_proj 20000x16 = 320k, i.e. ~960k parameters fit on 708 rows;
    #: Qwen3's 4,096 dims still give ~197k. Runs sec_001 (tfidf) and sec_002
    #: (qwen3) were both done at emb_pca_dims=0 and their arm rankings reversed
    #: completely between the two encoders — which is what pure overfitting looks
    #: like from the outside. At 48 dims the same projections cost ~2.3k
    #: parameters total, which 708 rows can actually support.
    emb_pca_dims: int = 48
    #: Interaction block width. The elementwise product of two (reduced)
    #: embeddings is projected to this many dims by a learned layer.
    interaction_dims: int = 32
    #: Gate input width — the section embedding projected down, nothing else.
    gate_dims: int = 16

    # ---- architecture ----
    n_experts: int = 4
    expert_hidden: int = 24
    expert_out: int = 16
    #: Gate temperature. 0.05 won the section-aggregation sweep; that sweep also
    #: found tau -> 0 was *worse*, so sharper is not automatically better.
    tau: float = 0.05
    expert_dropout: float = 0.2

    # ---- pooling ----
    #: "attention" learns the pooling weights; "softmax" freezes them at the rule
    #: the aggregation sweep already measured as best. The second is the control
    #: arm, and the comparison is the point of the experiment.
    pooling: str = "attention"
    #: Temperature for the frozen-softmax pooling control.
    pool_tau: float = 0.05

    # ---- gate regularization (the two terms pull against each other) ----
    sharpen_weight: float = 0.05
    balance_weight: float = 0.10

    # ---- optimization ----
    epochs: int = 60
    lr: float = 3e-3
    weight_decay: float = 1e-3
    batch_pairs: int = 32
    seed: int = 42

    # ---- evaluation ----
    folds: int = 5
    #: The one-shot holdout is never touched unless this is set explicitly *and*
    #: the cross-validated result has already cleared the bar below.
    run_holdout: bool = False
    #: Plain logistic regression on ~105 real pairs. Nothing has beaten it yet.
    holdout_bar_auc: float = 0.6398

    # ---- artifacts ----
    output_dir: Path = Path("artifacts/moe_sectioned")
