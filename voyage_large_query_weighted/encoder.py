"""Adapts VoyageLargeEncoder's ``input_type=`` interface to the ``role=``
interface ``query_weighted.eval.run_all_arms`` calls its encoder with.

``VoyageNanoEncoder`` (what ``query_weighted/`` was built against) and
``VoyageLargeEncoder`` (this package's target) both wrap the same Voyage
asymmetric-embedding convention but were written with different keyword names
for the same concept — ``role`` vs. ``input_type``. Rather than edit either
baseline encoder (both are shared, published-results code), this is a
~10-line pass-through wrapper.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from baselines.voyage_large.encode import VoyageLargeEncoder


class VoyageLargeRoleAdapter:
    """Exposes VoyageLargeEncoder as ``encode(texts, role=..., ...)``."""

    def __init__(self, encoder: VoyageLargeEncoder) -> None:
        self._encoder = encoder
        self.model_name = encoder.model_name
        self.truncate_dim = encoder.output_dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        role: str,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self._encoder.encode(
            texts, input_type=role, batch_size=batch_size, show_progress=show_progress
        )
