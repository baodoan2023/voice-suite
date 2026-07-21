"""Leaderboard math: means, exclusions, percentiles, ranking, worst-k."""
from __future__ import annotations

from voice_suite.aggregate import (
    _error_rates,
    _impl_legend,
    _latency_ratios,
    _metric_deltas,
    _pct,
    _shared_worst_utterances,
    aggregate,
    render_analysis,
    render_metrics_glossary,
    render_report,
    worst_utterances,
)
from voice_suite.protocol import JudgeScore, ScoredRecord, StageTimings


def _rec(impl="a", i=1, wer=0.2, e2e=0.8, judge_error=None, run_error=None,
         asr_ms=100, mt_ms=50, tts_ms=80):
    return ScoredRecord(
        impl=impl, utt_id=f"u{i:03d}", cache_key=f"k{impl}{i}", wer=wer,
        asr_text=f"asr {i}", mt_text=f"mt {i}", back_transcript=f"bt {i}",
        judge=JudgeScore(mt_adequacy=e2e, mt_fluency=e2e, e2e_adequacy=e2e,
                         verdict="v", rationale="r", error=judge_error),
        timings=StageTimings(asr_ms=asr_ms, mt_ms=mt_ms, tts_ms=tts_ms),
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


def test_render_report_judge_errors_advisory():
    recs = [_rec(i=1, judge_error="api down"), _rec(i=2)]
    rows = aggregate(recs)
    text = render_report(rows, [])
    assert "judge call(s) failed and are excluded from judge means" in text


def test_render_report_shows_utterance_error(make_utt):
    recs = [_rec(i=1, run_error="asr crashed")]
    utts = {"u001": make_utt(1)}
    rows = aggregate(recs)
    worst = worst_utterances(recs, utts)
    text = render_report(rows, worst)
    assert "- error: asr crashed" in text


def test_worst_utterances_manifest_row_missing():
    recs = [_rec(i=1)]
    worst = worst_utterances(recs, {})
    assert worst[0]["ref_transcript"] == "(manifest row missing)"


def test_render_report_includes_metric_glossary():
    text = render_report([], [])
    assert "## Metric definitions" in text
    assert "WER" in text
    assert "e2e_adequacy" in text


def test_metric_deltas_reports_leader_and_gap():
    recs = [_rec(impl="a", i=1, wer=0.1, e2e=0.9),
            _rec(impl="b", i=1, wer=0.3, e2e=0.5)]
    rows = aggregate(recs)
    text = "\n".join(_metric_deltas(rows))
    assert "`a` leads at 0.900 vs `b` at 0.500 (0.400 gap)" in text
    assert "WER" in text


def test_metric_deltas_omits_ties():
    recs = [_rec(impl="a", i=1, wer=0.2, e2e=0.8),
            _rec(impl="b", i=1, wer=0.2, e2e=0.8)]
    rows = aggregate(recs)
    assert _metric_deltas(rows) == []


def test_latency_ratios_reports_fastest_vs_slowest():
    recs = [_rec(impl="fast", i=1, asr_ms=100),
            _rec(impl="slow", i=1, asr_ms=400)]
    rows = aggregate(recs)
    text = "\n".join(_latency_ratios(rows))
    assert "`fast` is 4.0x faster than `slow` (100 vs 400 ms)" in text


def test_latency_ratios_omits_ties():
    recs = [_rec(impl="a", i=1, asr_ms=100),
            _rec(impl="b", i=1, asr_ms=100)]
    rows = aggregate(recs)
    assert _latency_ratios(rows) == []


def test_error_rates_per_impl():
    recs = [_rec(impl="a", i=1), _rec(impl="a", i=2, run_error="boom"),
            _rec(impl="b", i=1)]
    rows = aggregate(recs)
    text = "\n".join(_error_rates(rows))
    assert "`a`: 1/2 (50.0%)" in text
    assert "`b`: 0/1 (0.0%)" in text


def test_shared_worst_utterances_flags_overlap(make_utt):
    recs = [_rec(impl="a", i=1, e2e=0.1), _rec(impl="b", i=1, e2e=0.2),
            _rec(impl="a", i=2, e2e=0.9)]
    utts = {"u001": make_utt(1), "u002": make_utt(2)}
    worst = worst_utterances(recs, utts, k=10)
    text = "\n".join(_shared_worst_utterances(worst))
    assert "u001" in text
    assert "u002" not in text


def test_shared_worst_utterances_empty_when_no_overlap(make_utt):
    recs = [_rec(impl="a", i=1, e2e=0.1), _rec(impl="b", i=2, e2e=0.2)]
    utts = {"u001": make_utt(1), "u002": make_utt(2)}
    worst = worst_utterances(recs, utts, k=10)
    assert _shared_worst_utterances(worst) == []


def test_render_analysis_omitted_for_single_impl():
    recs = [_rec(i=1)]
    rows = aggregate(recs)
    assert render_analysis(rows, []) == []


def test_render_report_analysis_section_present_for_multi_impl():
    recs = [_rec(impl="a", i=1, wer=0.1, e2e=0.9),
            _rec(impl="b", i=1, wer=0.3, e2e=0.5)]
    rows = aggregate(recs)
    text = render_report(rows, [])
    assert "## Analysis" in text
    assert "errors" in text


def test_render_report_leads_with_e2e_then_wer():
    rows = aggregate([_rec(i=1)])
    text = render_report(rows, [])
    assert "| Rank | Impl | n | e2e_adequacy↑ | WER↓ |" in text


def test_render_report_note_below_table():
    rows = aggregate([_rec(i=1)])
    text = render_report(rows, [])
    assert "Ranked by e2e_adequacy (desc), then WER (asc).\n\n|" in text
    assert "**Note:** Judge means exclude judge-errored records; " \
           "pipeline errors count as zeros." in text


def test_impl_legend_known_impl():
    rows = aggregate([_rec(impl="m2v_phowhisper", i=1)])
    legend = _impl_legend(rows)
    assert legend == ["- `m2v_phowhisper` = PhoWhisper (ASR) + Marian ONNX "
                      "(MT) + StyleTTS2 (TTS)"]


def test_impl_legend_omits_unknown_impl():
    rows = aggregate([_rec(impl="a", i=1)])
    assert _impl_legend(rows) == []


def test_metrics_glossary_defines_n_and_expands_abbreviations():
    text = "\n".join(render_metrics_glossary())
    assert "| `n` |" in text
    assert "automatic speech recognition" in text
    assert "machine translation" in text
    assert "milliseconds" in text
    assert "50th/95th percentile" in text
    assert "100%" in text


def test_worst_utterances_heading_notes_pooled_across_impls(make_utt):
    recs = [_rec(i=1)]
    rows = aggregate(recs)
    worst = worst_utterances(recs, {"u001": make_utt(1)})
    text = render_report(rows, worst)
    assert "## Worst utterances (lowest e2e_adequacy, pooled across all impls)" in text
