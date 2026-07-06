"""Manifest loading: roundtrip, blank lines, missing file."""
from __future__ import annotations

import pytest

from voice_suite.dataset import load_manifest


def test_load_manifest_roundtrip(tmp_path, make_utt):
    utts = [make_utt(1), make_utt(2)]
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(u.model_dump_json() for u in utts) + "\n\n",
                 encoding="utf-8")
    assert load_manifest(p) == utts


def test_load_manifest_missing_file_mentions_ingest(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        load_manifest(tmp_path / "nope.jsonl")
