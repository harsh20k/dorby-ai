"""Identity key for B-data unique contacts (positioning hash + fallback)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_unique_contacts_B_data.py"
_SPEC = importlib.util.spec_from_file_location("build_unique_contacts_B_data", _MOD_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
build_unique_contacts = _MOD.build_unique_contacts
identity_key = _MOD.identity_key


def test_identity_prefers_positioning():
    key = identity_key({"positioning": "  Alice founder  ", "background": "other"})
    assert key is not None
    field, digest = key
    assert field == "positioning"
    assert len(digest) == 64
    assert identity_key({"positioning": "Alice founder"})[1] == digest


def test_identity_falls_back_when_positioning_empty():
    a = identity_key({"positioning": "  ", "background": "ex-faang"})
    b = identity_key({"background": "ex-faang"})
    assert a is not None and b is not None
    assert a[0] == "background"
    assert a == b


def test_identity_none_when_all_empty():
    assert identity_key({}) is None
    assert identity_key({"positioning": " ", "notes": "ignored"}) is None


def test_build_merges_seeker_and_match_on_same_positioning():
    rows = [
        {
            "contactId": "cm1",
            "query": "investors",
            "contactFile": {"positioning": "Pat founder", "background": "short"},
            "matches": [
                {
                    "status": "ACCEPT",
                    "matchType": "SMS",
                    "contactFile": {
                        "positioning": "Pat founder",
                        "background": "much longer background text",
                    },
                },
                {
                    "status": "PENDING",
                    "matchType": "BACKGROUND_MATCH",
                    "contactFile": {"lookingFor": "only lookingFor here"},
                },
            ],
        }
    ]
    contacts, stats = build_unique_contacts(rows)
    assert stats["n_dropped_all_empty"] == 0
    assert stats["n_unique_contacts"] == 2
    by_field = {c["identityField"]: c for c in contacts}
    pat = by_field["positioning"]
    assert pat["role"] == "both"
    assert pat["contactIds"] == ["cm1"]
    assert "much longer background text" in pat["contactFile"]["background"]
    assert pat["matchStatuses"]["ACCEPT"] == 1
    look = by_field["lookingFor"]
    assert look["role"] == "candidate"
    assert "contactIds" not in look
