"""Runner: single-batch dispatch, idempotent cache, count-mismatch guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from voice_suite.protocol import VoiceResult
from voice_suite.runner import load_raw_records, run_impl


class SpyImpl:
    """Records every translate_batch call; echoes one result per utterance."""

    name = "spy"

    def __init__(self, short: bool = False):
        self.calls: list[list[str]] = []
        self.setup_calls = 0
        self.short = short

    def setup(self) -> None:
        self.setup_calls += 1

    def translate_batch(self, utts, out_dir: Path):
        self.calls.append([u.id for u in utts])
        results = [VoiceResult(asr_text=f"asr {u.id}") for u in utts]
        return results[:-1] if self.short else results


def test_run_records_all_and_skips_cached(tmp_path, make_utt):
    utts = [make_utt(1), make_utt(2), make_utt(3)]
    raw = tmp_path / "raw.jsonl"
    impl = SpyImpl()

    assert run_impl(impl, utts[:2], raw, tmp_path / "audio") == 2
    assert impl.calls == [["u001", "u002"]]

    # Second run over the full set: only the new utterance reaches the impl.
    assert run_impl(impl, utts, raw, tmp_path / "audio") == 1
    assert impl.calls[-1] == ["u003"]
    recs = load_raw_records(raw)
    assert [r.utt_id for r in recs] == ["u001", "u002", "u003"]
    assert recs[0].result.asr_text == "asr u001"
    assert recs[0].impl == "spy"


def test_fully_cached_run_skips_setup(tmp_path, make_utt):
    utts = [make_utt(1)]
    raw = tmp_path / "raw.jsonl"
    impl = SpyImpl()
    run_impl(impl, utts, raw, tmp_path / "audio")
    assert run_impl(impl, utts, raw, tmp_path / "audio") == 0
    assert impl.setup_calls == 1


def test_audio_out_dir_is_per_impl(tmp_path, make_utt):
    run_impl(SpyImpl(), [make_utt(1)], tmp_path / "raw.jsonl", tmp_path / "audio")
    assert (tmp_path / "audio" / "spy").is_dir()


def test_result_count_mismatch_raises(tmp_path, make_utt):
    with pytest.raises(RuntimeError, match="returned 1 results for 2"):
        run_impl(SpyImpl(short=True), [make_utt(1), make_utt(2)],
                 tmp_path / "raw.jsonl", tmp_path / "audio")


def test_load_raw_records_missing_file(tmp_path):
    assert load_raw_records(tmp_path / "none.jsonl") == []
