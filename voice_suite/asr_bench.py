"""ASR-only benchmark: run the sherpa zipformer directly, no MT/TTS.

Loads the same model dir the m2v_sherpa impl points at (single source of
truth: impls/m2v_sherpa/local.toml) but skips the eval-batch binary entirely,
so a 200-utterance WER run takes minutes instead of an hour and cannot be
sunk by a TTS crash.
"""
from __future__ import annotations

import array
import ctypes
import importlib.util
import json
import os
import time
import tomllib
import wave
from pathlib import Path
from typing import Callable, Iterable

from voice_suite.protocol import Utterance

ASR_RESULTS = Path("out") / "asr_sherpa.jsonl"
_SHERPA_CONFIG = Path("impls") / "m2v_sherpa" / "local.toml"

TranscribeFn = Callable[[Path], str]


def sherpa_model_dir(config_path: Path = _SHERPA_CONFIG) -> Path:
    """Resolve the zipformer model dir from the impl's local.toml."""
    if not config_path.exists():
        raise RuntimeError(
            f"missing {config_path} — create it (see local.toml.example)")
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model_dir = Path(cfg["sherpa_model_dir"])
    if not model_dir.exists():
        raise RuntimeError(f"sherpa_model_dir does not exist: {model_dir}")
    return model_dir


def _preload_onnxruntime_dll() -> None:
    """Load the pip onnxruntime DLL by absolute path before sherpa_onnx.

    On Windows, sherpa's native module resolves ``onnxruntime.dll`` by name;
    a stale copy in C:\\Windows\\System32 beats the venv on the search path
    (the venv redirector makes the base Python dir the "application dir")
    and aborts with an ORT API version mismatch. A by-path preload wins:
    later by-name lookups reuse the already-loaded module.
    """
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        return
    dll = Path(spec.origin).parent / "capi" / "onnxruntime.dll"
    if dll.exists():
        ctypes.WinDLL(str(dll))


def _read_wave_mono16(path: Path) -> tuple[list[float], int]:
    """(samples in [-1, 1], sample_rate) from a 16-bit mono PCM WAV."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"{path}: expected 16-bit mono PCM, got "
                f"{wf.getnchannels()}ch/{8 * wf.getsampwidth()}-bit")
        pcm = array.array("h")
        pcm.frombytes(wf.readframes(wf.getnframes()))
        return [s / 32768.0 for s in pcm], wf.getframerate()


def get_transcriber(model_dir: Path, num_threads: int = 4) -> TranscribeFn:
    """Vietnamese transcriber backed by sherpa-onnx (lazy heavy import)."""
    _preload_onnxruntime_dll()
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "sherpa-onnx is not installed — pip install -e '.[asr]'") from exc
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.onnx"),
        decoder=str(model_dir / "decoder.onnx"),
        joiner=str(model_dir / "joiner.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=num_threads,
    )

    def transcribe(wav_path: Path) -> str:
        samples, sample_rate = _read_wave_mono16(wav_path)
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        rec.decode_stream(stream)
        return stream.result.text.strip()

    return transcribe


def load_asr_results(path: Path = ASR_RESULTS) -> dict[str, dict]:
    """{utt_id: row} from the append-only results file (last write wins)."""
    if not Path(path).exists():
        return {}
    rows: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["utt_id"]] = row
    return rows


def run_asr(utts: Iterable[Utterance], transcribe: TranscribeFn,
            out_path: Path = ASR_RESULTS,
            progress: Callable[[str], None] = lambda _: None) -> int:
    """Transcribe utterances not already in out_path; append results.

    Returns the number of newly transcribed utterances. Append-only with
    skip-if-present makes reruns incremental and crash-safe.
    """
    done = load_asr_results(out_path)
    todo = [u for u in utts if u.id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("a", encoding="utf-8") as fh:
        for i, utt in enumerate(todo, 1):
            t0 = time.perf_counter()
            text = transcribe(Path(utt.audio_path))
            ms = int(round((time.perf_counter() - t0) * 1000))
            fh.write(json.dumps(
                {"utt_id": utt.id, "asr_text": text, "asr_ms": ms},
                ensure_ascii=False) + "\n")
            fh.flush()
            progress(f"[{i}/{len(todo)}] {utt.id} ({ms} ms)")
    return len(todo)
