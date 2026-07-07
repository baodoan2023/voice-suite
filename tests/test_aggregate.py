"""Leaderboard math: means, exclusions, percentiles, ranking, worst-k."""
from __future__ import annotations

from voice_suite.aggregate import _pct, aggregate, render_report, worst_utterances
from voice_suite.protocol import JudgeScore, ScoredRecord, StageTimings


def _rec(impl="a", i=1, wer=0.2, e2e=0.8, judge_error=None, run_error=None,
         asr_ms=100):
    return ScoredRecord(
        impl=impl, utt_id=f"u{i:03d}", cache_key=f"k{impl}{i}", wer=wer,
        asr_text=f"asr {i}", mt_text=f"mt {i}", back_transcript=f"bt {i}",
        judge=JudgeScore(mt_adequacy=e2e, mt_fluency=e2e, e2e_adequacy=e2e,
                         verdict="v", rationale="r", error=judge_error),
        timings=StageTimings(asr_ms=asr_ms, mt_ms=50, tts_ms=80),
        run_error=run_error,
    )


def test_aggregate_means_and_error_counts():
    recs = [_rec(i=1, wer=0.0, e2e=1.0),
            _rec(i=2, wer=0.4, e2e=0.5),
            _rec(i=3, wer=1.0, e2e=0.0, run_error="crashed"),
            _rec(i=4, judge_error="api down")]
    rows = aggregate(recs)
    assert len(rows) == 1
    row = rows[0]
    assert row["scored"] == 4
    assert row["errors"] == 1
    assert row["judge_errors"] == 1
    # Judge means over the 3 non-judge-errored records: (1.0 + 0.5 + 0.0) / 3.
    assert abs(row["e2e_adequacy"] - 0.5) < 1e-9
    # WER mean over ALL records.
    assert abs(row["mean_wer"] - (0.0 + 0.4 + 1.0 + 0.2) / 4) < 1e-9


def test_ranking_by_e2e_then_wer():
    recs = [_rec(impl="low", i=1, e2e=0.2),
            _rec(impl="high", i=1, e2e=0.9),
            _rec(impl="tie_worse_wer", i=1, e2e=0.5, wer=0.5),
            _rec(impl="tie_better_wer", i=1, e2e=0.5, wer=0.1)]
    rows = aggregate(recs)
    assert [r["impl"] for r in rows] == ["high", "tie_better_wer",
                                         "tie_worse_wer", "low"]


def test_latency_percentiles_exclude_run_errors():
    recs = [_rec(i=i, asr_ms=i * 10) for i in range(1, 11)]
    recs.append(_rec(i=99, asr_ms=99999, run_error="crashed"))
    row = aggregate(recs)[0]
    assert row["asr_ms_p50"] == 60
    assert row["asr_ms_p95"] == 100
    assert row["mt_ms_p50"] == 50


def test_pct_empty_is_zero():
    assert _pct([], 0.5) == 0


def test_worst_utterances_sorted_and_capped(make_utt):
    utts = {f"u{i:03d}": make_utt(i) for i in range(1, 6)}
    recs = [_rec(i=i, e2e=i / 10) for i in range(1, 6)]
    worst = worst_utterances(recs, utts, k=3)
    assert [w["utt_id"] for w in worst] == ["u001", "u002", "u003"]
    assert worst[0]["ref_transcript"] == utts["u001"].ref_transcript
    assert worst[0]["mt_text"] == "mt 1"


def test_render_report_contains_table_and_appendix(make_utt):
    recs = [_rec(i=1)]
    rows = aggregate(recs)
    worst = worst_utterances(recs, {"u001": make_utt(1)})
    text = render_report(rows, worst)
    assert "| 1 | a |" in text
    assert "Best implementation: `a`" in text
    assert "Worst utterances" in text
    assert "ref (vi):" in text


def test_render_report_empty_is_valid():
    assert "voice-suite Leaderboard" in render_report([], [])
