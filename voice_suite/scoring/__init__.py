"""Scoring orchestration: WER + back-transcription + judge → ScoredRecords.

Two phases per score run:
1. Back-transcription, sequential (one local faster-whisper instance; the
   content-hash cache makes re-runs cheap).
2. Judge calls, optionally parallel (network-bound), with lock-guarded
   JSONL writes.
"""
from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from voice_suite.protocol import (
    JudgeScore, RawRecord, ScoredRecord, Utterance, VoiceResult,
)
from voice_suite.scoring.backtranscribe import (
    TranscribeFn, append_bt_cache, audio_sha256, load_bt_cache,
)
from voice_suite.scoring.judge import JUDGE_VERSION, JudgeFn, score_utterance
from voice_suite.scoring.wer import score_wer

_SEQUENTIAL = 1


def score_key(impl: str, utt_id: str, result: VoiceResult, audio_sha: str,
              judge_id: str = "") -> str:
    """Stable cache key over impl, utterance, output content, judge identity.

    Changing the judge backend/model (judge_id), the rubric (JUDGE_VERSION),
    or any output (texts, audio bytes) re-scores only what changed.
    """
    payload = "\x1f".join([impl, utt_id, result.asr_text, result.mt_text,
                           audio_sha, JUDGE_VERSION, judge_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_scored_records(path: Path) -> list[ScoredRecord]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        ScoredRecord.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _back_transcripts(
    pending: dict[str, tuple[RawRecord, Utterance, str]],
    transcribe: TranscribeFn | None,
    bt_cache_path: Path | None,
) -> dict[str, tuple[str, str | None]]:
    """Phase 1: cache_key → (back_transcript, back_error). Sequential."""
    cache = load_bt_cache(bt_cache_path) if bt_cache_path else {}
    out: dict[str, tuple[str, str | None]] = {}
    for key, (rec, _utt, sha) in pending.items():
        if transcribe is None:
            out[key] = ("", "back-transcription skipped")
        elif not sha:
            out[key] = ("", rec.result.error or "no output audio")
        elif sha in cache:
            out[key] = (cache[sha], None)
        else:
            try:
                text = transcribe(Path(rec.result.audio_path))
                cache[sha] = text
                if bt_cache_path:
                    append_bt_cache(bt_cache_path, sha, text)
                out[key] = (text, None)
            except Exception as exc:
                out[key] = ("", str(exc))
    return out


def _score_one(key: str, rec: RawRecord, utt: Utterance, back_transcript: str,
               back_error: str | None,
               call_judge: JudgeFn | None) -> ScoredRecord:
    """WER + judge for one record.

    Pipeline failures score deterministic zeros WITHOUT a judge call and
    stay in the aggregate means (a config that crashes must rank worse).
    """
    wer = score_wer(utt.ref_transcript, rec.result.asr_text)
    if rec.result.error is not None:
        judge = JudgeScore(verdict="pipeline-error")
    elif call_judge is None:
        judge = JudgeScore(verdict="skipped")
    else:
        judge = score_utterance(utt, rec.result, back_transcript,
                                call_judge=call_judge)
        if back_error is not None and judge.error is None:
            # Output audio unusable: e2e is deterministically 0; MT scores stand.
            judge = judge.model_copy(update={"e2e_adequacy": 0.0})
    return ScoredRecord(
        impl=rec.impl,
        utt_id=rec.utt_id,
        cache_key=key,
        wer=wer,
        asr_text=rec.result.asr_text,
        mt_text=rec.result.mt_text,
        back_transcript=back_transcript,
        back_error=back_error,
        judge=judge,
        timings=rec.result.timings,
        run_error=rec.result.error,
    )


def score_all(
    raw: list[RawRecord],
    utts: list[Utterance],
    out_path: Path,
    call_judge: JudgeFn | None = None,
    workers: int = 1,
    judge_id: str = "",
    transcribe: TranscribeFn | None = None,
    bt_cache_path: Path | None = None,
) -> None:
    """Score every raw record not already scored; append to out_path.

    Parameters
    ----------
    raw:           Raw pipeline outputs to score.
    utts:          Ground-truth utterances (the sampled subset to score).
    out_path:      JSONL file to append ScoredRecords to.
    call_judge:    LLM judge callable; None = --no-judge (skipped verdicts).
    workers:       Parallel judge workers (>1 uses a ThreadPoolExecutor with
                   a lock around file writes).
    judge_id:      "<backend>:<model>" or "no-judge"; part of the cache key.
    transcribe:    Back-transcription callable; None skips phase 1.
    bt_cache_path: JSONL content-hash cache for back-transcripts.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_id = {u.id: u for u in utts}
    done = {s.cache_key for s in load_scored_records(out_path) if s.cache_key}

    pending: dict[str, tuple[RawRecord, Utterance, str]] = {}
    for rec in raw:
        utt = by_id.get(rec.utt_id)
        if utt is None:
            continue  # record for an utterance outside this sample
        audio = rec.result.audio_path
        sha = audio_sha256(Path(audio)) if audio and Path(audio).exists() else ""
        key = score_key(rec.impl, rec.utt_id, rec.result, sha, judge_id)
        if key in done or key in pending:
            continue
        pending[key] = (rec, utt, sha)

    if not pending:
        return

    bts = _back_transcripts(pending, transcribe, bt_cache_path)

    lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as fh:

        def _write(scored: ScoredRecord) -> None:
            with lock:
                fh.write(scored.model_dump_json() + "\n")
                fh.flush()

        if workers <= _SEQUENTIAL:
            for key, (rec, utt, _sha) in pending.items():
                bt, bt_err = bts[key]
                _write(_score_one(key, rec, utt, bt, bt_err, call_judge))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_score_one, key, rec, utt,
                                    bts[key][0], bts[key][1], call_judge)
                    for key, (rec, utt, _sha) in pending.items()
                ]
                for fut in as_completed(futures):
                    _write(fut.result())
