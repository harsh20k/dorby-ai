"""Config for the field-gate experiment: one shared tower, seeker side split
into pieces (query + lookingFor + positioning), combined by a small learned
gate instead of one fixed alpha or one flat concatenated string.

Same LoRA/training hyperparameters as `top1_ctrl` — the point of comparison
is the input structure + gate, not a hyperparameter change.
"""

from __future__ import annotations

from pathlib import Path

from twotower.config import TrainConfig

MODEL_NAME = "voyageai/voyage-4-nano"
PIECE_KEYS = ("query", "lookingFor", "positioning")


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
        output_dir=output_dir or (Path("artifacts/twotower_field_gate") / run_id),
        primary_metric="recall@1",
    )
