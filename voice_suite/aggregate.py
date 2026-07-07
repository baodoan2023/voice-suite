"""Aggregate scored records into a per-impl leaderboard and markdown report."""
from __future__ import annotations

from voice_suite.protocol import ScoredRecord, Utterance

WORST_K = 10


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(sorted_vals: list[int], q: float) -> int:
    """Nearest-rank percentile on a pre-sorted list (matches the app's pct())."""
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]


def aggregate(records: list[ScoredRecord]) -> list[dict]:
    """Group by impl; compute means, latency percentiles, error counts.

    Judge means average records whose judge call succeeded (judge.error is
    None) — pipeline-error zeros ARE included, so crashing configs rank
    worse. Latency percentiles cover records that actually ran. Rows sort
    best-first by e2e_adequacy, then lower WER (which also ranks --no-judge
    runs, where every e2e is 0).
    """
    by_impl: dict[str, list[ScoredRecord]] = {}
    for rec in records:
        by_impl.setdefault(rec.impl, []).append(rec)

    rows: list[dict] = []
    for impl, recs in by_impl.items():
        judged = [r.judge for r in recs if r.judge.error is None]
        ran = [r for r in recs if r.run_error is None]
        lat: dict[str, int] = {}
        for stage in ("asr_ms", "mt_ms", "tts_ms"):
            vals = sorted(getattr(r.timings, stage) for r in ran)
            lat[f"{stage}_p50"] = _pct(vals, 0.50)
            lat[f"{stage}_p95"] = _pct(vals, 0.95)
        rows.append({
            "impl": impl,
            "scored": len(recs),
            "errors": sum(1 for r in recs if r.run_error is not None),
            "judge_errors": len(recs) - len(judged),
            "mean_wer": _mean([r.wer for r in recs]),
            "mt_adequacy": _mean([j.mt_adequacy for j in judged]),
            "mt_fluency": _mean([j.mt_fluency for j in judged]),
            "e2e_adequacy": _mean([j.e2e_adequacy for j in judged]),
            **lat,
        })
    rows.sort(key=lambda r: (-r["e2e_adequacy"], r["mean_wer"]))
    return rows


def worst_utterances(records: list[ScoredRecord],
                     utts_by_id: dict[str, Utterance],
                     k: int = WORST_K) -> list[dict]:
    """The k lowest-e2e (then highest-WER) utterances, with all texts."""
    ranked = sorted(records, key=lambda r: (r.judge.e2e_adequacy, -r.wer))
    out: list[dict] = []
    for rec in ranked[:k]:
        utt = utts_by_id.get(rec.utt_id)
        out.append({
            "impl": rec.impl,
            "utt_id": rec.utt_id,
            "e2e_adequacy": rec.judge.e2e_adequacy,
            "wer": rec.wer,
            "ref_transcript": utt.ref_transcript if utt else "(manifest row missing)",
            "ref_translation": utt.ref_translation if utt else "",
            "asr_text": rec.asr_text,
            "mt_text": rec.mt_text,
            "back_transcript": rec.back_transcript,
            "error": rec.run_error or rec.judge.error or rec.back_error,
        })
    return out


def render_report(rows: list[dict], worst: list[dict]) -> str:
    """Render the leaderboard + worst-utterances appendix as markdown."""
    lines = [
        "# voice-suite Leaderboard", "",
        "Ranked by e2e_adequacy (desc), then WER (asc). Judge means exclude "
        "judge-errored records; pipeline errors count as zeros.", "",
        "| Rank | Impl | n | WER↓ | mt_adequacy↑ | mt_fluency↑ | "
        "e2e_adequacy↑ | asr_ms P50/P95 | mt_ms P50/P95 | tts_ms P50/P95 | "
        "errors |",
        "|------|------|---|------|--------------|-------------|"
        "---------------|----------------|---------------|----------------|"
        "--------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['impl']} | {r['scored']} | {r['mean_wer']:.3f} | "
            f"{r['mt_adequacy']:.3f} | {r['mt_fluency']:.3f} | "
            f"{r['e2e_adequacy']:.3f} | "
            f"{r['asr_ms_p50']}/{r['asr_ms_p95']} | "
            f"{r['mt_ms_p50']}/{r['mt_ms_p95']} | "
            f"{r['tts_ms_p50']}/{r['tts_ms_p95']} | {r['errors']} |"
        )
    for r in rows:
        if r["judge_errors"]:
            lines.append(f"\n{r['impl']}: {r['judge_errors']} judge call(s) "
                         f"failed and are excluded from judge means.")
    if rows:
        lines += ["", f"**Best implementation: `{rows[0]['impl']}`**"]
    if worst:
        lines += ["", "## Worst utterances (lowest e2e_adequacy)", ""]
        for w in worst:
            lines += [
                f"### {w['utt_id']} — {w['impl']} "
                f"(e2e {w['e2e_adequacy']:.2f}, WER {w['wer']:.2f})",
                f"- ref (vi): {w['ref_transcript']}",
                f"- asr (vi): {w['asr_text']}",
                f"- mt (en): {w['mt_text']}",
                f"- audio back-transcript (en): {w['back_transcript']}",
                f"- ref translation (en): {w['ref_translation']}",
            ]
            if w["error"]:
                lines.append(f"- error: {w['error']}")
            lines.append("")
    return "\n".join(lines)
