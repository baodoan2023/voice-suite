"""Back-transcription of output audio with faster-whisper, content-hash cached.

The scorer deliberately uses a LARGER model (large-v3) than the
PhoWhisper-small under test so it out-hears the pipeline. It is the slowest
scorer, so transcripts are cached by audio content hash; the cache survives
judge/model changes because audio bytes, not run identity, key it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

BT_MODEL = "large-v3"

TranscribeFn = Callable[[Path], str]


def audio_sha256(path: Path) -> str:
    """Content hash of an audio file (streamed; files can be large)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bt_cache(path: Path) -> dict[str, str]:
    """Load {audio_sha256: transcript} from a JSONL cache file."""
    p = Path(path)
    if not p.exists():
        return {}
    cache: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["sha256"]] = row["transcript"]
    return cache


def append_bt_cache(path: Path, sha: str, transcript: str) -> None:
    """Append one cache entry (append-only JSONL; readers keep the last value)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sha256": sha, "transcript": transcript}) + "\n")


def get_transcriber(model_size: str = BT_MODEL, lang: str = "en") -> TranscribeFn:
    """Return a TranscribeFn backed by faster-whisper (lazy heavy import).

    First call downloads the model from HuggingFace (~3 GB, cached under the
    HF hub cache). CPU + int8: slow but dependency-light and deterministic.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed — pip install -e '.[bt]' "
            "or use --no-judge for the WER+latency-only path") from exc
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(path: Path) -> str:
        segments, _info = model.transcribe(str(path), language=lang, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()

    return transcribe
