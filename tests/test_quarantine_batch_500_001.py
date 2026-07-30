"""Guards on the quarantined `batch_500_001` synthetic pairs.

The 460 promoted `cmsynth*` pairs are known-harmful (`docs/possible-bugs.md` #4:
a candidate-profile-only classifier hits 99.2% accuracy on them because the
generator leaked the label into the profile text). They are **not deleted** from
`data/dataset_*.json`, because removing them would retroactively change what
`twotower` `run_001` trained on and make its published numbers unreproducible.

Quarantine is therefore enforced two ways, and both are pinned here:

1. an archived copy exists and still matches the live dataset files, so the
   archive cannot silently drift out of date;
2. every loader path defaults to excluding them, so opting in is explicit.

See `data/archive/batch_500_001_quarantined/README.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARCHIVE = DATA_DIR / "archive" / "batch_500_001_quarantined"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "dataset_positive.json").exists(),
    reason="data/ is gitignored; only runs in a checkout with the real dataset",
)


def _is_synth(pair: dict) -> bool:
    return pair["userContactId"].startswith("cmsynth") or pair[
        "matchContactId"
    ].startswith("cmsynth")


@pytest.mark.parametrize(("label", "expected"), [("positive", 220), ("negative", 240)])
def test_archive_matches_live_dataset(label: str, expected: int) -> None:
    """The archived copy is exactly the synthetic subset of the live file."""
    live = json.loads((DATA_DIR / f"dataset_{label}.json").read_text())
    archived = json.loads((ARCHIVE / f"dataset_{label}_quarantined.json").read_text())

    live_synth = [p for p in live if _is_synth(p)]
    assert len(archived) == expected
    assert len(live_synth) == expected

    def key(p: dict) -> tuple[str, str]:
        return (p["userContactId"], p["matchContactId"])

    assert sorted(map(key, archived)) == sorted(map(key, live_synth))


def test_provenance_recorded() -> None:
    prov = json.loads((ARCHIVE / "provenance.json").read_text())
    assert prov["batch_id"] == "batch_500_001"
    for label in ("positive", "negative"):
        entry = prov["files"][label]
        assert len(entry["source_sha256"]) == 64
        assert len(entry["quarantined_sha256"]) == 64


def test_train_config_quarantines_by_default() -> None:
    """Training must not pull in the harmful pairs unless asked."""
    from twotower.config import TrainConfig

    assert TrainConfig().include_synth is False


def test_cli_requires_explicit_opt_in() -> None:
    """`--include-synth` is the only way to get them back on the CLI path."""
    from twotower.train import parse_args

    assert parse_args([]).include_synth is False
    assert parse_args(["--include-synth"]).include_synth is True
    # --real-only still wins if both are passed.
    args = parse_args(["--include-synth", "--real-only"])
    assert (args.include_synth and not args.real_only) is False


def test_split_bundle_excludes_them_by_default() -> None:
    from twotower.data import build_split_bundle

    bundle = build_split_bundle(DATA_DIR, DATA_DIR / "synthetic" / "seed_split.json")
    for lp in list(bundle.train) + list(bundle.train_dev) + list(bundle.holdout):
        assert not _is_synth(lp.pair), "quarantined synth pair leaked into a split"
