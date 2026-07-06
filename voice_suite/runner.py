"""Orchestrate impl × utterance runs with an idempotent JSONL cache.

Key difference from a per-item runner: the impl is an external process with
a multi-GB model load, so uncached utterances are handed over in ONE
``translate_batch`` call per impl.
"""
from __future__ import annotations

from pathlib import Path

from voice_suite.protocol import RawRecord, Utterance, VoiceImpl


def load_raw_records(path: Path) -> list[RawRecord]:
    """Load all RawRecords from a JSONL file; returns empty list if absent."""
    p = Path(path)
    if not p.exists():
        return []
    return [
        RawRecord.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_impl(impl: VoiceImpl, utts: list[Utterance], out_path: Path,
             audio_out_dir: Path) -> int:
    """Run `impl` over uncached utterances; append RawRecords to out_path.

    Already-recorded (impl.name, utt_id) pairs are skipped (idempotent).
    Returns the number of newly recorded utterances. Batch-level failures
    (setup, non-zero exit) propagate to the caller; per-utterance failures
    arrive as VoiceResult.error and are recorded like successes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r.impl, r.utt_id) for r in load_raw_records(out_path)}
    todo = [u for u in utts if (impl.name, u.id) not in done]
    if not todo:
        return 0

    impl.setup()
    out_dir = Path(audio_out_dir) / impl.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = impl.translate_batch(todo, out_dir)
    if len(results) != len(todo):
        raise RuntimeError(
            f"{impl.name}.translate_batch returned {len(results)} results "
            f"for {len(todo)} utterances")

    with out_path.open("a", encoding="utf-8") as fh:
        for utt, result in zip(todo, results):
            rec = RawRecord(impl=impl.name, utt_id=utt.id, result=result)
            fh.write(rec.model_dump_json() + "\n")
    return len(todo)
