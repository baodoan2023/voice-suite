"""Impl wrapper against the fake eval_batch: contract, ordering, failures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from impls.m2v_phowhisper.adapter import M2vBatchImpl

FAKE = Path(__file__).parent / "fake_eval_batch.py"


def _impl(*extra: str) -> M2vBatchImpl:
    return M2vBatchImpl(cmd=[sys.executable, str(FAKE), *extra])


def test_batch_results_in_input_order(tmp_path, make_utt):
    impl = _impl()
    impl.setup()
    utts = [make_utt(2), make_utt(1)]  # deliberately unsorted
    results = impl.translate_batch(utts, tmp_path)
    assert [r.asr_text for r in results] == ["asr u002", "asr u001"]
    assert results[0].mt_text == "mt u002"
    assert results[0].timings.asr_ms == 10
    assert results[0].error is None
    assert all(Path(r.audio_path).exists() for r in results)


def test_missing_id_becomes_error_result(tmp_path, make_utt):
    impl = _impl("--skip-id", "u001")
    impl.setup()
    results = impl.translate_batch([make_utt(1), make_utt(2)], tmp_path)
    assert results[0].error == "missing from eval_batch output"
    assert results[1].error is None


def test_nonzero_exit_raises_with_stderr(tmp_path, make_utt):
    impl = _impl("--fail")
    impl.setup()
    with pytest.raises(RuntimeError, match="boom: model not found"):
        impl.translate_batch([make_utt(1)], tmp_path)


def test_setup_without_config_shows_template(tmp_path):
    impl = M2vBatchImpl(config_path=tmp_path / "local.toml")
    with pytest.raises(RuntimeError, match="missing impl config"):
        impl.setup()
    with pytest.raises(RuntimeError, match="eval_batch.exe"):
        impl.setup()  # template with example paths is embedded in the error


def test_setup_flags_nonexistent_paths(tmp_path):
    cfg = tmp_path / "local.toml"
    cfg.write_text(
        'exe = "does/not/exist.exe"\n'
        'whisper_model = "x"\nmt_dir = "y"\n'
        'tts_onnx_dir = "z"\ntts_voice_style = "w"\n', encoding="utf-8")
    impl = M2vBatchImpl(config_path=cfg)
    with pytest.raises(RuntimeError, match="exe does not exist"):
        impl.setup()


def test_setup_flags_missing_keys(tmp_path):
    cfg = tmp_path / "local.toml"
    cfg.write_text('exe = "x"\n', encoding="utf-8")
    impl = M2vBatchImpl(config_path=cfg)
    with pytest.raises(RuntimeError, match="missing keys"):
        impl.setup()


def test_setup_builds_command_from_config(tmp_path):
    # Create real files for config paths
    exe = tmp_path / "eval_batch.exe"
    whisper_model = tmp_path / "whisper_model.bin"
    mt_dir = tmp_path / "mt_dir"
    tts_onnx_dir = tmp_path / "tts_onnx_dir"
    tts_voice_style = tmp_path / "tts_voice_style.json"

    exe.touch()
    whisper_model.touch()
    mt_dir.mkdir()
    tts_onnx_dir.mkdir()
    tts_voice_style.touch()

    # Write valid local.toml with all required keys and extra_args
    cfg = tmp_path / "local.toml"
    cfg.write_text(
        f'exe = "{exe.as_posix()}"\n'
        f'whisper_model = "{whisper_model.as_posix()}"\n'
        f'mt_dir = "{mt_dir.as_posix()}"\n'
        f'tts_onnx_dir = "{tts_onnx_dir.as_posix()}"\n'
        f'tts_voice_style = "{tts_voice_style.as_posix()}"\n'
        'extra_args = ["--beam-size", "5"]\n',
        encoding="utf-8")

    impl = M2vBatchImpl(config_path=cfg)
    impl.setup()

    # Verify command construction with correct order
    # Paths in TOML use forward slashes (as_posix()), so expected_cmd uses them too
    expected_cmd = [
        exe.as_posix(),
        "--whisper-model", whisper_model.as_posix(),
        "--mt-dir", mt_dir.as_posix(),
        "--tts-onnx-dir", tts_onnx_dir.as_posix(),
        "--tts-voice-style", tts_voice_style.as_posix(),
        "--src", "vi",
        "--dst", "en",
        "--beam-size", "5",
    ]
    assert impl._cmd == expected_cmd
