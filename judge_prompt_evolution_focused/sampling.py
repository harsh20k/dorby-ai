"""Deterministic, hardness-aware sampling of real-pair examples for each
optimization round. Reads real pairs read-only via
``baselines.llm_judge.real_pairs`` — never writes to ``data/``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from baselines.bert_frozen.text import profile_to_text
from baselines.llm_judge.real_pairs import load_real_pairs

from judge_prompt_evolution_focused.config import RunConfig
from judge_prompt_evolution_focused.focused_prompt import (
    candidate_to_text,
    seeker_to_text,
)

Outcome = Literal["accepted", "declined"]


@dataclass
class Example:
    label: Outcome
    hardness: str | None  # "hard" | "easy" | None (positives have no hardness)
    pair: dict[str, Any]
    overlap: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "userContactId": self.pair["userContactId"],
            "matchContactId": self.pair["matchContactId"],
            "label": self.label,
            "hardness": self.hardness,
            "overlap": self.overlap,
        }

    def render(self, index: int) -> str:
        # Hardness is used internally to stratify sampling (sampling.py), but
        # deliberately not disclosed here — the optimizer only ever sees
        # "accepted" / "declined", never a hard/easy label.
        #
        # Differs from judge_prompt_evolution/sampling.py: the optimizer sees
        # exactly what the focused judge will see at eval time — the trimmed
        # field set (seeker: positioning + lookingFor; candidate: positioning
        # + background + lookingFor) plus the searchQuery — not the complete
        # profiles. Writing rules about fields the judge is never shown was
        # the mismatch this experiment exists to avoid.
        a = seeker_to_text(self.pair["userContactFile"])
        b = candidate_to_text(self.pair["matchContactFile"])
        query = (self.pair.get("searchQuery") or "").strip()
        return (
            f"--- Example {index}: {self.label.upper()} ---\n"
            f"=== PERSON A (seeker) ===\n{a}\n\n"
            f"Search query: {query}\n\n"
            f"=== PERSON B (candidate) ===\n{b}\n\n"
            f"Ground truth: this intro was {self.label}."
        )


def _jaccard(text_a: str, text_b: str) -> float:
    sa, sb = set(text_a.lower().split()), set(text_b.lower().split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class ExampleBank:
    """Holds the train-split pool, pre-scored for hardness, sampled without
    replacement across a whole run (recycles with a fresh shuffle only if a
    run asks for more draws than the pool has, which is logged loudly)."""

    def __init__(self, cfg: RunConfig) -> None:
        pos, neg = load_real_pairs(cfg.data_dir, split=cfg.split)
        self._rng = random.Random(cfg.seed)

        # Hardness is still measured on the *complete* profile text, not the
        # trimmed one the optimizer sees. That keeps "hard negative" meaning
        # the same thing it means in baselines/metrics.py's neg_hardness slice
        # and in every earlier evo run, so batch composition stays comparable.
        overlaps = [
            _jaccard(
                profile_to_text(p["userContactFile"]),
                profile_to_text(p["matchContactFile"]),
            )
            for p in neg
        ]
        median = sorted(overlaps)[len(overlaps) // 2] if overlaps else 0.0

        self._positives = [Example("accepted", None, p, None) for p in pos]
        self._hard_negatives = [
            Example("declined", "hard", p, o) for p, o in zip(neg, overlaps) if o >= median
        ]
        self._easy_negatives = [
            Example("declined", "easy", p, o) for p, o in zip(neg, overlaps) if o < median
        ]

        self._rng.shuffle(self._positives)
        self._rng.shuffle(self._hard_negatives)
        self._rng.shuffle(self._easy_negatives)

        self._queues = {
            "positive": list(self._positives),
            "hard": list(self._hard_negatives),
            "easy": list(self._easy_negatives),
        }
        self._pools = {
            "positive": self._positives,
            "hard": self._hard_negatives,
            "easy": self._easy_negatives,
        }

    def _draw(self, pool_name: str, n: int) -> list[Example]:
        queue = self._queues[pool_name]
        drawn: list[Example] = []
        for _ in range(n):
            if not queue:
                print(
                    f"[sampling] {pool_name} pool exhausted — reshuffling and recycling "
                    "(examples will repeat across iterations)"
                )
                queue = list(self._pools[pool_name])
                self._rng.shuffle(queue)
                self._queues[pool_name] = queue
            drawn.append(queue.pop())
        return drawn

    def draw_batch(self, cfg: RunConfig) -> list[Example]:
        batch = (
            self._draw("positive", cfg.n_positive_examples)
            + self._draw("hard", cfg.n_hard_negative_examples)
            + self._draw("easy", cfg.n_easy_negative_examples)
        )
        self._rng.shuffle(batch)
        return batch
