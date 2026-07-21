"""Aggregate scored records into a per-impl leaderboard and markdown report."""
from __future__ import annotations

from voice_suite.protocol import ScoredRecord, Utterance

WORST_K = 10

# Real model stack behind each impl id, for the report legend (render_report).
_DISPLAY_NAMES = {
    "m2v_phowhisper": "PhoWhisper (ASR) + Marian ONNX (MT) + StyleTTS2 (TTS)",
    "m2v_sherpa": "sherpa-onnx (ASR) + Marian ONNX (MT) + StyleTTS2 (TTS)",
    "m2v_nemotron": "Nemotron (ASR) + Marian ONNX (MT) + StyleTTS2 (TTS)",
}


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


def _metric_deltas(rows: list[dict]) -> list[str]:
    """Per-quality-metric leader and gap across impls; omits ties."""
    metrics = [
        ("mean_wer", "WER", False),
        ("mt_adequacy", "mt_adequacy", True),
        ("mt_fluency", "mt_fluency", True),
        ("e2e_adequacy", "e2e_adequacy", True),
    ]
    lines: list[str] = []
    for key, label, higher_is_better in metrics:
        pick = max if higher_is_better else min
        antipick = min if higher_is_better else max
        best = pick(rows, key=lambda r: r[key])
        worst_row = antipick(rows, key=lambda r: r[key])
        if best["impl"] == worst_row["impl"]:
            continue
        gap = abs(best[key] - worst_row[key])
        lines.append(
            f"- **{label}**: `{best['impl']}` leads at {best[key]:.3f} vs "
            f"`{worst_row['impl']}` at {worst_row[key]:.3f} ({gap:.3f} gap)."
        )
    return lines


def _latency_ratios(rows: list[dict]) -> list[str]:
    """Fastest-vs-slowest ratio per pipeline stage; omits zero/tied stages."""
    stages = [("asr_ms_p50", "asr_ms P50"), ("mt_ms_p50", "mt_ms P50"),
              ("tts_ms_p50", "tts_ms P50")]
    lines: list[str] = []
    for key, label in stages:
        fastest = min(rows, key=lambda r: r[key])
        slowest = max(rows, key=lambda r: r[key])
        if fastest["impl"] == slowest["impl"] or fastest[key] == 0:
            continue
        ratio = slowest[key] / fastest[key]
        lines.append(
            f"- **{label}**: `{fastest['impl']}` is {ratio:.1f}x faster than "
            f"`{slowest['impl']}` ({fastest[key]} vs {slowest[key]} ms)."
        )
    return lines


def _error_rates(rows: list[dict]) -> list[str]:
    """Pipeline-error rate per impl."""
    lines = []
    for r in rows:
        rate = r["errors"] / r["scored"] * 100
        lines.append(f"- **errors**: `{r['impl']}`: {r['errors']}/{r['scored']} ({rate:.1f}%).")
    return lines


def _impl_legend(rows: list[dict]) -> list[str]:
    """One line per impl naming its real model stack, when known."""
    return [f"- `{r['impl']}` = {_DISPLAY_NAMES[r['impl']]}"
            for r in rows if r["impl"] in _DISPLAY_NAMES]


def _shared_worst_utterances(worst: list[dict]) -> list[str]:
    """Utterance ids that landed in more than one impl's worst-k list."""
    by_utt: dict[str, list[str]] = {}
    for w in worst:
        by_utt.setdefault(w["utt_id"], []).append(w["impl"])
    shared = sorted(uid for uid, impls in by_utt.items() if len(impls) > 1)
    if not shared:
        return []
    return [f"- **shared difficulty**: {len(shared)} utterance(s) challenged "
            f"more than one impl (not impl-specific): {', '.join(shared)}."]


def render_analysis(rows: list[dict], worst: list[dict]) -> list[str]:
    """Mechanical cross-impl comparison, derived purely from already-computed
    aggregate/worst data — no extra judging or LLM calls."""
    if len(rows) < 2:
        return []
    body = (_metric_deltas(rows) + _latency_ratios(rows) +
            _error_rates(rows) + _shared_worst_utterances(worst))
    if not body:
        return []
    return ["## Analysis", ""] + body + [""]


def render_metrics_glossary() -> list[str]:
    """Static glossary of every leaderboard column. Full depth: docs/metrics.md."""
    return [
        "## Metric definitions", "",
        "ASR (automatic speech recognition), MT (machine translation), "
        "ms (milliseconds), P50/P95 (50th/95th percentile). Rows marked "
        "(LLM judge) are scored by Claude (a large language model) "
        "reading the text and rating it 0.0-1.0, rather than a fixed "
        "formula like WER; run with `score --no-judge` and they're "
        "always 0.0.", "",
        "| Metric | Meaning | Direction |",
        "|---|---|---|",
        "| `n` | Number of evaluated utterances | — |",
        "| `WER` | ASR transcript vs. reference transcript word error rate: "
        "edits ÷ reference word count, capped at 1.0 (1.0 = 100%, i.e. "
        "fully wrong) | lower is better |",
        "| `mt_adequacy` | Does the MT output preserve the reference "
        "translation's meaning? (LLM judge) | higher is better |",
        "| `mt_fluency` | Is the MT output natural, grammatical English? "
        "(LLM judge) | higher is better |",
        "| `e2e_adequacy` | Does the synthesized output audio (re-heard via "
        "back-transcription) still convey the reference meaning? (LLM judge) "
        "| higher is better |",
        "| `asr_ms` / `mt_ms` / `tts_ms` | Per-utterance wall-clock time per "
        "pipeline stage, P50/P95 across non-crashed runs | lower is faster |",
        "| `errors` | Utterances where the impl itself crashed/failed to "
        "produce output | lower is better |",
        "",
        "Pipeline-error and judge-error records count as `0.0` in the judge-"
        "score means (crashing configs rank worse, not excluded). Full "
        "depth: `docs/metrics.md`.",
        "",
    ]


def render_report(rows: list[dict], worst: list[dict]) -> str:
    """Render the leaderboard + worst-utterances appendix as markdown."""
    lines = [
        "# voice-suite Leaderboard", "",
        "Ranked by e2e_adequacy (desc), then WER (asc).", "",
        "| Rank | Impl | n | e2e_adequacy↑ | WER↓ | mt_adequacy↑ | "
        "mt_fluency↑ | asr_ms P50/P95 | mt_ms P50/P95 | tts_ms P50/P95 | "
        "errors |",
        "|------|------|---|---------------|------|--------------|"
        "-------------|----------------|---------------|----------------|"
        "--------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['impl']} | {r['scored']} | "
            f"{r['e2e_adequacy']:.3f} | {r['mean_wer']:.3f} | "
            f"{r['mt_adequacy']:.3f} | {r['mt_fluency']:.3f} | "
            f"{r['asr_ms_p50']}/{r['asr_ms_p95']} | "
            f"{r['mt_ms_p50']}/{r['mt_ms_p95']} | "
            f"{r['tts_ms_p50']}/{r['tts_ms_p95']} | {r['errors']} |"
        )
    lines += ["", "**Note:** Judge means exclude judge-errored records; "
                  "pipeline errors count as zeros.", *_impl_legend(rows)]
    for r in rows:
        if r["judge_errors"]:
            lines.append(f"\n{r['impl']}: {r['judge_errors']} judge call(s) "
                         f"failed and are excluded from judge means.")
    if rows:
        lines += ["", f"**Best implementation: `{rows[0]['impl']}`**"]
    lines += ["", *render_metrics_glossary()]
    lines += render_analysis(rows, worst)
    if worst:
        lines += ["", "## Worst utterances (lowest e2e_adequacy, pooled "
                      "across all impls)", ""]
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
