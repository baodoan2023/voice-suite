"""ASR-only benchmark: local sherpa zipformer plus cloud engines, no MT/TTS.

The sherpa path loads the same model dir the m2v_sherpa impl points at
(single source of truth: impls/m2v_sherpa/local.toml) but skips the
eval-batch binary entirely, so a 200-utterance WER run takes minutes instead
of an hour and cannot be sunk by a TTS crash. Cloud engines (OpenAI, Gemini)
POST each WAV to the provider's API; keys come from environment variables.
"""
from __future__ import annotations

import array
import base64
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

ENGINES = ("sherpa", "openai", "gemini")
CLOUD_DEFAULT_MODELS = {"openai": "gpt-4o-transcribe",
                        "gemini": "gemini-3.6-flash"}

TranscribeFn = Callable[[Path], str]


def results_path(engine: str) -> Path:
    """Per-engine results file; `asr_sherpa.jsonl` stays the sherpa one."""
    return Path("out") / f"asr_{engine}.jsonl"


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


def _load_hotwords_dir(hotwords_dir: Path | None) -> str:
    """Concatenate all .txt files in a hotwords directory into one string."""
    if hotwords_dir is None or not hotwords_dir.is_dir():
        return ""
    parts: list[str] = []
    for txt in sorted(hotwords_dir.glob("*.txt")):
        parts.append(txt.read_text(encoding="utf-8"))
    return "\n".join(parts)


def hotwords_dir_from_config(config_path: Path = _SHERPA_CONFIG) -> Path | None:
    """Resolve hotwords dir from extra_args in local.toml, if present."""
    if not config_path.exists():
        return None
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    extra = cfg.get("extra_args", [])
    for i, arg in enumerate(extra):
        if arg == "--hotwords-dir" and i + 1 < len(extra):
            p = Path(extra[i + 1])
            return p if p.is_dir() else None
    return None


def get_transcriber(model_dir: Path, num_threads: int = 4,
                    hotwords_dir: Path | None = None) -> TranscribeFn:
    """Vietnamese transcriber backed by sherpa-onnx (lazy heavy import)."""
    _preload_onnxruntime_dll()
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "sherpa-onnx is not installed — pip install -e '.[asr]'") from exc

    kwargs: dict = dict(
        encoder=str(model_dir / "encoder.onnx"),
        decoder=str(model_dir / "decoder.onnx"),
        joiner=str(model_dir / "joiner.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=num_threads,
        decoding_method="modified_beam_search",
        max_active_paths=4,
    )
    hotwords_str = _load_hotwords_dir(hotwords_dir)
    if hotwords_str:
        kwargs["hotwords_score"] = 1.5

    rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)

    def transcribe(wav_path: Path) -> str:
        samples, sample_rate = _read_wave_mono16(wav_path)
        if hotwords_str:
            stream = rec.create_stream(hotwords=hotwords_str)
        else:
            stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        rec.decode_stream(stream)
        return stream.result.text.strip()

    return transcribe


def _require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise RuntimeError(f"{name} is not set — export it to use this engine")
    return val


def _retrying(call: Callable[[], str], tries: int = 3) -> str:
    """Retry transient cloud failures with exponential backoff (2s, 4s)."""
    for attempt in range(1, tries + 1):
        try:
            return call()
        except Exception:
            if attempt == tries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def get_openai_transcriber(model: str | None = None) -> TranscribeFn:
    """Cloud ASR via OpenAI's /v1/audio/transcriptions (needs OPENAI_API_KEY)."""
    import httpx  # transitive dep of anthropic; always present
    key = _require_env("OPENAI_API_KEY")
    model_id = model or CLOUD_DEFAULT_MODELS["openai"]
    client = httpx.Client(timeout=120)

    def transcribe(wav_path: Path) -> str:
        def call() -> str:
            resp = client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data={"model": model_id, "language": "vi",
                      "response_format": "json"},
                files={"file": (wav_path.name, wav_path.read_bytes(),
                                "audio/wav")})
            if resp.status_code != 200:
                raise RuntimeError(
                    f"openai HTTP {resp.status_code}: {resp.text[:300]}")
            return str(resp.json()["text"]).strip()
        return _retrying(call)

    return transcribe


_GEMINI_PROMPT = ("Transcribe this Vietnamese audio exactly as spoken. "
                  "Return only the transcript text, with no introduction, "
                  "quotes, or commentary.")


def parse_gemini_response(body: dict) -> str:
    """Extract transcript text from a generateContent response body."""
    try:
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"unexpected gemini response: {json.dumps(body)[:300]}") from exc
    if not text:
        raise RuntimeError(
            f"empty gemini transcript: {json.dumps(body)[:300]}")
    return text


def get_gemini_transcriber(model: str | None = None) -> TranscribeFn:
    """Cloud ASR via Gemini audio input (needs GEMINI_API_KEY)."""
    import httpx
    key = _require_env("GEMINI_API_KEY")
    model_id = model or CLOUD_DEFAULT_MODELS["gemini"]
    client = httpx.Client(timeout=120)
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_id}:generateContent")

    def transcribe(wav_path: Path) -> str:
        payload = {
            "contents": [{"parts": [
                {"text": _GEMINI_PROMPT},
                {"inline_data": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(wav_path.read_bytes()).decode(),
                }},
            ]}],
            "generationConfig": {"temperature": 0.0},
        }

        def call() -> str:
            resp = client.post(url, params={"key": key}, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"gemini HTTP {resp.status_code}: {resp.text[:300]}")
            return parse_gemini_response(resp.json())
        return _retrying(call)

    return transcribe


def get_cloud_transcriber(engine: str, model: str | None = None) -> TranscribeFn:
    if engine == "openai":
        return get_openai_transcriber(model)
    if engine == "gemini":
        return get_gemini_transcriber(model)
    raise ValueError(f"unknown cloud engine {engine!r}; cloud engines: "
                     f"{[e for e in ENGINES if e != 'sherpa']}")


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
    skip-if-present makes reruns incremental and crash-safe. A failing
    utterance (bad audio, cloud error after retries) is reported and
    skipped, not persisted — rerunning retries exactly the failures.
    """
    done = load_asr_results(out_path)
    todo = [u for u in utts if u.id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with Path(out_path).open("a", encoding="utf-8") as fh:
        for i, utt in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                text = transcribe(Path(utt.audio_path))
            except Exception as exc:
                progress(f"[{i}/{len(todo)}] {utt.id} FAILED: {exc}")
                continue
            ms = int(round((time.perf_counter() - t0) * 1000))
            fh.write(json.dumps(
                {"utt_id": utt.id, "asr_text": text, "asr_ms": ms},
                ensure_ascii=False) + "\n")
            fh.flush()
            ok += 1
            progress(f"[{i}/{len(todo)}] {utt.id} ({ms} ms)")
    if ok < len(todo):
        progress(f"{len(todo) - ok} utterances failed — rerun to retry them")
    return ok
