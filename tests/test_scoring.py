"""Score orchestration: cache keys, idempotence, phases, error semantics."""
from __future__ import annotations

from pathlib import Path

from voice_suite.protocol import RawRecord
from voice_suite.scoring import load_scored_records, score_all, score_key

GOOD = {"mt_adequacy": 0.9, "mt_fluency": 0.8, "e2e_adequacy": 0.7,
        "verdict": "good", "rationale": "r"}


def _judge_ok(prompt: str) -> dict:
    return dict(GOOD)


def test_score_key_sensitive_to_content_and_judge(make_result):
    r1 = make_result()
    r2 = make_result(asr_text="khác hẳn")
    base = score_key("a", "u001", r1, "sha", "cli:m")
    assert base == score_key("a", "u001", r1, "sha", "cli:m")
    assert base != score_key("a", "u001", r2, "sha", "cli:m")
    assert base != score_key("a", "u001", r1, "sha2", "cli:m")
    assert base != score_key("a", "u001", r1, "sha", "api:m")
    assert base != score_key("b", "u001", r1, "sha", "cli:m")


def test_score_all_writes_and_is_idempotent(tmp_path, make_utt, make_result):
    utts = [make_utt(1), make_utt(2)]
    raw = [RawRecord(impl="a", utt_id=u.id,
                     result=make_result(asr_text=u.ref_transcript,
                                        mt_text=u.ref_translation))
           for u in utts]
    scored = tmp_path / "scored.jsonl"
    calls = []

    def judge(prompt):
        calls.append(prompt)
        return dict(GOOD)

    score_all(raw, utts, scored, call_judge=judge, judge_id="t:m")
    score_all(raw, utts, scored, call_judge=judge, judge_id="t:m")
    recs = load_scored_records(scored)
    assert len(recs) == 2
    assert len(calls) == 2  # second call fully cache-hit
    rec1 = next(r for r in recs if r.utt_id == "u001")
    assert rec1.judge.mt_adequacy == 0.9
    assert rec1.wer == 0.0
    assert rec1.asr_text == "xin chào số 1"
    assert rec1.cache_key


def test_unknown_utt_id_skipped(tmp_path, make_utt, make_result):
    raw = [RawRecord(impl="a", utt_id="ghost", result=make_result())]
    score_all(raw, [make_utt(1)], tmp_path / "s.jsonl",
              call_judge=_judge_ok, judge_id="t:m")
    assert load_scored_records(tmp_path / "s.jsonl") == []


def test_pipeline_error_scores_zero_without_judge_call(tmp_path, make_utt, make_result):
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(asr_text="", mt_text="",
                                        error="asr crashed"))]
    called = []
    score_all(raw, [utt], tmp_path / "s.jsonl",
              call_judge=lambda p: called.append(p) or dict(GOOD),
              judge_id="t:m")
    rec = load_scored_records(tmp_path / "s.jsonl")[0]
    assert called == []
    assert rec.judge.verdict == "pipeline-error"
    assert rec.judge.e2e_adequacy == 0.0
    assert rec.wer == 1.0
    assert rec.run_error == "asr crashed"


def test_no_judge_path_marks_skipped(tmp_path, make_utt, make_result):
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(asr_text=utt.ref_transcript))]
    score_all(raw, [utt], tmp_path / "s.jsonl", call_judge=None,
              judge_id="no-judge")
    rec = load_scored_records(tmp_path / "s.jsonl")[0]
    assert rec.judge.verdict == "skipped"
    assert rec.wer == 0.0


def test_back_transcription_cached_by_content(tmp_path, make_utt, make_result):
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF-fake-audio")
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(audio_path=str(wav)))]
    bt_calls = []

    def transcribe(path: Path) -> str:
        bt_calls.append(path)
        return "back text"

    cache_file = tmp_path / "bt.jsonl"
    score_all(raw, [utt], tmp_path / "s1.jsonl", call_judge=_judge_ok,
              judge_id="t:m", transcribe=transcribe, bt_cache_path=cache_file)
    # New judge_id → new score cache key, but bt cache hits (same audio bytes).
    score_all(raw, [utt], tmp_path / "s2.jsonl", call_judge=_judge_ok,
              judge_id="t:m2", transcribe=transcribe, bt_cache_path=cache_file)
    assert len(bt_calls) == 1
    rec = load_scored_records(tmp_path / "s2.jsonl")[0]
    assert rec.back_transcript == "back text"
    assert rec.judge.e2e_adequacy == 0.7


def test_bt_failure_forces_e2e_zero_keeps_mt(tmp_path, make_utt, make_result):
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF-fake-audio")
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(audio_path=str(wav)))]

    def transcribe(path: Path) -> str:
        raise RuntimeError("decode failed")

    score_all(raw, [utt], tmp_path / "s.jsonl", call_judge=_judge_ok,
              judge_id="t:m", transcribe=transcribe,
              bt_cache_path=tmp_path / "bt.jsonl")
    rec = load_scored_records(tmp_path / "s.jsonl")[0]
    assert rec.back_error == "decode failed"
    assert rec.judge.e2e_adequacy == 0.0
    assert rec.judge.mt_adequacy == 0.9


def test_parallel_workers_write_all_records(tmp_path, make_utt, make_result):
    utts = [make_utt(i) for i in range(1, 9)]
    raw = [RawRecord(impl="a", utt_id=u.id, result=make_result(i))
           for i, u in enumerate(utts, 1)]
    scored = tmp_path / "s.jsonl"
    score_all(raw, utts, scored, call_judge=_judge_ok, judge_id="t:m",
              workers=4)
    recs = load_scored_records(scored)
    assert len(recs) == 8
    assert {r.utt_id for r in recs} == {u.id for u in utts}
