"""Shared test fixtures: factories for ground-truth and result objects."""
from __future__ import annotations

import pytest

from voice_suite.protocol import StageTimings, Utterance, VoiceResult


@pytest.fixture
def make_utt():
    def _make(i: int = 1, **overrides):
        base = dict(
            id=f"u{i:03d}",
            audio_path=f"data/utterances/u{i:03d}.wav",
            src_lang="vi",
            dst_lang="en",
            ref_transcript=f"xin chào số {i}",
            ref_translation=f"hello number {i}",
        )
        base.update(overrides)
        return Utterance(**base)
    return _make


@pytest.fixture
def make_result():
    def _make(i: int = 1, **overrides):
        base = dict(
            asr_text=f"xin chào số {i}",
            mt_text=f"hello number {i}",
            audio_path=None,
            timings=StageTimings(asr_ms=100, mt_ms=50, tts_ms=80),
            error=None,
        )
        base.update(overrides)
        return VoiceResult(**base)
    return _make
