"""Pydantic schemas and the VoiceImpl adapter protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class StageTimings(BaseModel):
    """Per-stage compute time in milliseconds for one utterance."""
    model_config = ConfigDict(frozen=True)

    asr_ms: int = 0
    mt_ms: int = 0
    tts_ms: int = 0


class Utterance(BaseModel):
    """One ground-truth row: source audio plus reference texts."""
    model_config = ConfigDict(frozen=True)

    id: str
    audio_path: str          # input WAV (16 kHz mono, source-language speech)
    src_lang: str            # e.g. "vi"
    dst_lang: str            # e.g. "en"
    ref_transcript: str      # what was said (source language)
    ref_translation: str     # reference translation (target language)


class VoiceResult(BaseModel):
    """Output produced by an implementation for a single utterance."""
    model_config = ConfigDict(frozen=True)

    asr_text: str = ""
    mt_text: str = ""
    audio_path: str | None = None    # translated-speech WAV; None on error
    timings: StageTimings = Field(default_factory=StageTimings)
    error: str | None = None


class RawRecord(BaseModel):
    """Raw (unscored) record linking an implementation run to an utterance."""
    model_config = ConfigDict(frozen=True)

    impl: str
    utt_id: str
    result: VoiceResult


class JudgeScore(BaseModel):
    """Translation-quality scores produced by the LLM judge."""
    model_config = ConfigDict(frozen=True)

    mt_adequacy: float = 0.0
    mt_fluency: float = 0.0
    e2e_adequacy: float = 0.0
    verdict: str = ""
    rationale: str = ""
    error: str | None = None


class ScoredRecord(BaseModel):
    """Fully scored record: WER + back-transcription + judge + latency.

    Carries asr_text/mt_text copies so scored.jsonl is independently
    readable (the report's worst-utterances appendix needs the texts).
    """
    model_config = ConfigDict(frozen=True)

    impl: str
    utt_id: str
    cache_key: str = ""
    wer: float
    asr_text: str = ""
    mt_text: str = ""
    back_transcript: str = ""
    back_error: str | None = None
    judge: JudgeScore
    timings: StageTimings = Field(default_factory=StageTimings)
    run_error: str | None = None


@runtime_checkable
class VoiceImpl(Protocol):
    """Contract every implementation adapter must satisfy.

    An impl is a *pipeline configuration*. It is batch-invoked: the runner
    filters out cached utterances and hands the remainder over in ONE call,
    because real impls are external processes with multi-GB model loads.
    """
    name: str

    def setup(self) -> None: ...

    def translate_batch(self, utts: list[Utterance],
                        out_dir: Path) -> list[VoiceResult]: ...
