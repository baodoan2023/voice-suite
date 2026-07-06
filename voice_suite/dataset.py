"""Load the ground-truth manifest into Utterance objects."""
from __future__ import annotations

from pathlib import Path

from voice_suite.protocol import Utterance


def load_manifest(path: Path) -> list[Utterance]:
    """Parse manifest.jsonl; helpful error pointing at `ingest` when absent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"manifest not found: {p} — run `voice-suite ingest` first")
    return [
        Utterance.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
