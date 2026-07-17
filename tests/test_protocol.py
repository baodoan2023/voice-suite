"""Schema invariants: frozen models, defaults, JSON roundtrip, duck-typing."""
from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from voice_suite.protocol import (
    JudgeScore, RawRecord, ScoredRecord, StageTimings, Utterance,
    VoiceImpl, VoiceResult,
)


def test_utterance_is_frozen(make_utt):
    utt = make_utt()
    with pytest.raises(pydantic.ValidationError):
        utt.id = "changed"


def test_voice_result_defaults():
    r = VoiceResult()
    assert r.asr_text == ""
    assert r.mt_text == ""
    assert r.audio_path is None
    assert r.timings == StageTimings()
    assert r.error is None


def test_raw_record_json_roundtrip(make_utt, make_result):
    rec = RawRecord(impl="m2v_phowhisper", utt_id=make_utt().id, result=make_result())
    line = rec.model_dump_json()
    assert RawRecord.model_validate_json(line) == rec


def test_scored_record_roundtrip():
    rec = ScoredRecord(
        impl="a", utt_id="u001", wer=0.25,
        judge=JudgeScore(mt_adequacy=0.9, mt_fluency=0.8, e2e_adequacy=0.7,
                         verdict="good", rationale="close"))
    assert ScoredRecord.model_validate_json(rec.model_dump_json()) == rec


def test_voice_impl_is_runtime_checkable():
    class Fake:
        name = "fake"
        def setup(self) -> None: ...
        def translate_batch(self, utts, out_dir: Path):
            return []
    assert isinstance(Fake(), VoiceImpl)
