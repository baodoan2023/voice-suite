"""Content hashing and the back-transcription cache (no models involved)."""
from __future__ import annotations

import pytest

from voice_suite.scoring.backtranscribe import (
    append_bt_cache, audio_sha256, load_bt_cache,
)


def test_sha_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"RIFF" + b"\x01" * 64)
    b.write_bytes(b"RIFF" + b"\x02" * 64)
    assert audio_sha256(a) == audio_sha256(a)
    assert audio_sha256(a) != audio_sha256(b)


def test_cache_roundtrip_and_last_write_wins(tmp_path):
    cache_file = tmp_path / "bt.jsonl"
    assert load_bt_cache(cache_file) == {}
    append_bt_cache(cache_file, "sha1", "hello there")
    append_bt_cache(cache_file, "sha2", "second")
    append_bt_cache(cache_file, "sha1", "revised")
    cache = load_bt_cache(cache_file)
    assert cache == {"sha1": "revised", "sha2": "second"}


def test_append_creates_parent_dirs(tmp_path):
    cache_file = tmp_path / "deep" / "nested" / "bt.jsonl"
    append_bt_cache(cache_file, "sha1", "text")
    assert load_bt_cache(cache_file) == {"sha1": "text"}


def test_get_transcriber_missing_dependency_raises(monkeypatch):
    import sys

    from voice_suite.scoring.backtranscribe import get_transcriber

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
        get_transcriber()
