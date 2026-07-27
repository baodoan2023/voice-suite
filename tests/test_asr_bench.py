"""ASR-only benchmark runner: wav reading, incremental runs, config errors."""
from __future__ import annotations

import wave
from pathlib import Path

import pytest

from voice_suite.asr_bench import (_read_wave_mono16, get_cloud_transcriber,
                                   load_asr_results, parse_gemini_response,
                                   results_path, run_asr, sherpa_model_dir)
from voice_suite.protocol import Utterance


def _utt(uid: str, wav: Path) -> Utterance:
    return Utterance(id=uid, audio_path=str(wav), src_lang="vi", dst_lang="en",
                     ref_transcript="xin chào", ref_translation="hello")


def _write_wav(path: Path, n: int = 160) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x10" * n)
    return path


def test_read_wave_mono16(tmp_path):
    samples, rate = _read_wave_mono16(_write_wav(tmp_path / "a.wav", n=8))
    assert rate == 16000
    assert len(samples) == 8
    assert all(-1.0 <= s <= 1.0 for s in samples)


def test_read_wave_rejects_stereo(tmp_path):
    p = tmp_path / "st.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x10\x00\x10" * 4)
    with pytest.raises(ValueError, match="mono"):
        _read_wave_mono16(p)


def test_run_asr_is_incremental(tmp_path):
    wav = _write_wav(tmp_path / "a.wav")
    utts = [_utt("u1", wav), _utt("u2", wav)]
    out = tmp_path / "asr.jsonl"
    calls: list[str] = []

    def fake_transcribe(path: Path) -> str:
        calls.append(str(path))
        return "xin chào"

    assert run_asr(utts, fake_transcribe, out) == 2
    assert run_asr(utts, fake_transcribe, out) == 0  # all cached
    assert len(calls) == 2
    rows = load_asr_results(out)
    assert set(rows) == {"u1", "u2"}
    assert rows["u1"]["asr_text"] == "xin chào"
    assert rows["u1"]["asr_ms"] >= 0


def test_sherpa_model_dir_missing_config(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        sherpa_model_dir(tmp_path / "nope.toml")


def test_run_asr_skips_failed_and_retries_on_rerun(tmp_path):
    wav = _write_wav(tmp_path / "a.wav")
    utts = [_utt("ok", wav), _utt("boom", wav)]
    out = tmp_path / "asr.jsonl"
    fail_once = {"armed": True}

    def flaky(path: Path) -> str:
        # "boom" is second in order; fail its first attempt only.
        if fail_once["armed"] and len(load_asr_results(out)) == 1:
            fail_once["armed"] = False
            raise RuntimeError("cloud 500")
        return "xin chào"

    messages: list[str] = []
    assert run_asr(utts, flaky, out, progress=messages.append) == 1
    assert any("FAILED" in m for m in messages)
    assert set(load_asr_results(out)) == {"ok"}
    assert run_asr(utts, flaky, out) == 1  # rerun retries only the failure
    assert set(load_asr_results(out)) == {"ok", "boom"}


def test_results_path_per_engine():
    assert results_path("sherpa").name == "asr_sherpa.jsonl"
    assert results_path("openai").name == "asr_openai.jsonl"


def test_get_cloud_transcriber_unknown_engine():
    with pytest.raises(ValueError, match="unknown cloud engine"):
        get_cloud_transcriber("sherpa")


def test_parse_gemini_response():
    body = {"candidates": [{"content": {"parts": [
        {"text": "xin "}, {"text": "chào"}]}}]}
    assert parse_gemini_response(body) == "xin chào"


def test_parse_gemini_response_malformed():
    with pytest.raises(RuntimeError, match="unexpected gemini"):
        parse_gemini_response({"candidates": []})
    with pytest.raises(RuntimeError, match="empty gemini"):
        parse_gemini_response(
            {"candidates": [{"content": {"parts": [{"text": " "}]}}]})
