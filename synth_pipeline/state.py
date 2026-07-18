"""LangGraph state for one synthetic pair attempt."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

Label = Literal["pos", "neg"]
Status = Literal[
    "pending",
    "filtered",
    "judged",
    "staged",
    "approved",
    "dropped",
]


class PairState(TypedDict):
    """State for a single generate → filter → judge → stage run."""

    label: Label
    failure_mode: NotRequired[str | None]
    seed_user_id: str
    seed_pair_id: str
    seed_pair: dict[str, Any]
    few_shot_pairs: list[dict[str, Any]]
    batch_id: str
    split_hash: str
    pair: NotRequired[dict[str, Any] | None]
    metadata: dict[str, Any]
    qc: dict[str, Any]
    status: Status
    retry_count: int
    drop_reason: NotRequired[str | None]
