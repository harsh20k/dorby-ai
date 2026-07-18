"""Synthetic contact ID helpers (25-char `cm…`)."""

from __future__ import annotations

import secrets
import string

from synth_pipeline.config import ID_LENGTH, ID_PREFIX

_ALPHABET = string.ascii_lowercase + string.digits


def new_contact_id(*, prefix: str = "cmsynth") -> str:
    """Return a 25-char id starting with `cm`, unique-looking for synth rows."""
    body_prefix = prefix if prefix.startswith(ID_PREFIX) else f"{ID_PREFIX}{prefix}"
    if len(body_prefix) >= ID_LENGTH:
        raise ValueError(f"prefix too long for {ID_LENGTH}-char ids: {body_prefix!r}")
    n = ID_LENGTH - len(body_prefix)
    return body_prefix + "".join(secrets.choice(_ALPHABET) for _ in range(n))


def is_valid_contact_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != ID_LENGTH:
        return False
    if not value.startswith(ID_PREFIX):
        return False
    return all(c in _ALPHABET for c in value)
