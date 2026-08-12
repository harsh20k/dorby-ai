"""Eval-only: rescore ask_offer_001 with offer text = positioning + background.

Does not retrain. The offer tower was trained on all fields except lookingFor;
this package only changes what text is fed at eval time.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
