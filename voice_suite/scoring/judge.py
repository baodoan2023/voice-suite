"""Claude LLM-judge for translation quality (one call per utterance)."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from typing import Callable

from voice_suite.protocol import JudgeScore, Utterance, VoiceResult

JUDGE_VERSION = "v1"  # bump to invalidate cached judge scores (part of the score cache key)
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
DEFAULT_CLI_BIN = "claude"
_CLI_TIMEOUT_S = 180

JudgeFn = Callable[[str], dict]

_RUBRIC = """\
You are a strict, neutral judge of speech-translation quality. A system \
transcribed Vietnamese speech (ASR), translated it to English (MT), and \
synthesized the translation as English speech (TTS). The system's output \
audio was transcribed back to text by a strong independent ASR.
Score three values, each between 0.0 and 1.0:
- mt_adequacy: does MT OUTPUT convey the meaning of the reference \
translation, given the Vietnamese source? 1.0 = same meaning, 0.0 = unrelated.
- mt_fluency: is MT OUTPUT natural, grammatical English (regardless of accuracy)?
- e2e_adequacy: does OUTPUT AUDIO TRANSCRIPT still convey the reference \
meaning? Penalize meaning lost between MT OUTPUT and the audio (dropped or \
garbled words); do not penalize back-transcription spelling or punctuation \
variance.
Respond ONLY with JSON:
{"mt_adequacy": <float>, "mt_fluency": <float>, "e2e_adequacy": <float>,
 "verdict": "<short verdict>", "rationale": "<why>"}"""


def build_judge_prompt(utt: Utterance, result: VoiceResult,
                       back_transcript: str) -> str:
    """Build the evaluation prompt. Bias-blind: no impl identity in inputs."""
    return (
        f"{_RUBRIC}\n\n"
        f"VIETNAMESE SOURCE (reference transcript):\n{utt.ref_transcript}\n\n"
        f"REFERENCE ENGLISH TRANSLATION:\n{utt.ref_translation}\n\n"
        f"SYSTEM ASR (what the system heard, Vietnamese):\n{result.asr_text}\n\n"
        f"MT OUTPUT (English):\n{result.mt_text}\n\n"
        f"OUTPUT AUDIO TRANSCRIPT (English, back-transcribed):\n"
        f"{back_transcript or '(no output audio)'}\n"
    )


_anthropic_client: "anthropic.Anthropic | None" = None
_anthropic_client_lock = threading.Lock()


def _get_anthropic_client() -> "anthropic.Anthropic":
    global _anthropic_client
    if _anthropic_client is None:
        with _anthropic_client_lock:
            if _anthropic_client is None:
                import anthropic
                _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _anthropic_judge(prompt: str, model: str = DEFAULT_JUDGE_MODEL) -> dict:
    """Call the Anthropic API and return parsed JSON sub-scores."""
    client = _get_anthropic_client()
    msg = client.messages.create(
        model=model, max_tokens=1024, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
    return _loads_lenient(text)


_DECODER = json.JSONDecoder()


def _loads_lenient(text: str) -> dict:
    """Parse the first JSON object from model output, tolerating fences/prose.

    Uses ``raw_decode`` from each candidate ``{`` so trailing braces in prose
    (e.g. ``{...} done {oops``) do not corrupt a valid leading object.
    """
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = _DECODER.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            return obj
        idx = text.find("{", idx + 1)
    raise ValueError(f"no JSON object in judge output: {text!r}")


def _cli_judge(prompt: str, model: str = DEFAULT_JUDGE_MODEL,
               cli_bin: str = DEFAULT_CLI_BIN) -> dict:
    """Judge via the Claude Code CLI (`claude -p`), using its own auth.

    Uses --output-format json: stdout is an envelope whose `result` field
    holds the assistant's text, from which we parse the JSON sub-scores.
    The prompt goes over stdin so it is neither exposed in process listings
    nor capped by OS argv length limits.
    """
    exe = shutil.which(cli_bin)
    if exe is None:
        raise RuntimeError(
            f"claude CLI not found on PATH (cli_bin={cli_bin!r}). "
            f"Install Claude Code or use --judge api.")
    proc = subprocess.run(
        [exe, "-p", "--output-format", "json", "--model", model],
        input=prompt, capture_output=True, text=True, timeout=_CLI_TIMEOUT_S,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): {proc.stderr.strip()}")
    envelope = json.loads(proc.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(
            f"claude CLI error: {str(envelope.get('result', '')).strip()}")
    return _loads_lenient(envelope.get("result", ""))


def get_judge(backend: str = "cli", model: str = DEFAULT_JUDGE_MODEL) -> JudgeFn:
    """Return a JudgeFn: 'cli' (Claude Code login) or 'api' (Anthropic SDK)."""
    if backend == "cli":
        return lambda prompt: _cli_judge(prompt, model=model)
    if backend == "api":
        return lambda prompt: _anthropic_judge(prompt, model=model)
    raise ValueError(f"unknown judge backend: {backend!r} (use 'cli' or 'api')")


def score_utterance(utt: Utterance, result: VoiceResult, back_transcript: str,
                    call_judge: JudgeFn | None = None) -> JudgeScore:
    """Score one utterance. `call_judge(prompt)->dict` is injectable for testing."""
    judge = call_judge or _anthropic_judge
    prompt = build_judge_prompt(utt, result, back_transcript)
    try:
        raw = judge(prompt)
        return JudgeScore(
            mt_adequacy=float(raw["mt_adequacy"]),
            mt_fluency=float(raw["mt_fluency"]),
            e2e_adequacy=float(raw["e2e_adequacy"]),
            verdict=str(raw["verdict"]),
            rationale=str(raw["rationale"]),
        )
    except Exception as exc:
        return JudgeScore(verdict="error", error=str(exc))
