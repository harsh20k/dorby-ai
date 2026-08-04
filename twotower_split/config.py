"""Config for the genuinely-split two-tower experiment: separate LoRA
adapters for the query side and the candidate side, instead of one shared
model reading different input text.

Hyperparameters match `top1_ctrl` where the concept transfers (LoRA rank 8 /
alpha 16 / dropout 0.05 on q/k/v/o_proj, lr 2e-4, batch 6, 5 epochs,
truncate_dim 1024, max_seq_length 4096) — the point of comparison is the
architecture change, not a hyperparameter change. Query-side rows are the
same `query_only` anchor text as `twotower_query_only/` (already proven the
strongest seeker representation among the ones tested); candidate-side text
is unchanged `candidate_to_text`.
"""

from __future__ import annotations

from pathlib import Path

from twotower.config import TrainConfig

MODEL_NAME = "voyageai/voyage-4-nano"


def build_config(
    *, run_id: str, epochs: int = 5, train_batch_size: int = 6, output_dir: Path | None = None
) -> TrainConfig:
    return TrainConfig(
        model_name=MODEL_NAME,
        trust_remote_code=True,
        truncate_dim=1024,
        max_seq_length=4096,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        expected_layers_per_target=12,
        epochs=epochs,
        learning_rate=2e-4,
        train_batch_size=train_batch_size,
        eval_batch_size=train_batch_size,
        seed=42,
        query_prompt="Represent the query for retrieving supporting documents: ",
        document_prompt="Represent the document for retrieval: ",
        output_dir=output_dir or (Path("artifacts/twotower_split") / run_id),
        primary_metric="recall@1",
    )
