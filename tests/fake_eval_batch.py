"""Fake eval_batch: deterministic canned outputs for wrapper tests.

Honors the cross-repo contract (reads --manifest, writes <out-dir>/<id>.wav
and <out-dir>/results.jsonl). Extra knobs: --skip-id omits one row from the
output; --fail simulates a setup failure (non-zero exit + stderr).
"""
from __future__ import annotations

import argparse
import json
import struct
import wave
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-id", default=None)
    ap.add_argument("--fail", action="store_true")
    args, _rest = ap.parse_known_args()  # ignore model flags etc.

    if args.fail:
        raise SystemExit("boom: model not found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line)
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            if row["id"] == args.skip_id:
                continue
            wav_path = out_dir / f"{row['id']}.wav"
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(struct.pack("<8h", *([1000] * 8)))
            fh.write(json.dumps({
                "id": row["id"],
                "asr_text": f"asr {row['id']}",
                "mt_text": f"mt {row['id']}",
                "audio_out": str(wav_path),
                "asr_ms": 10, "mt_ms": 5, "tts_ms": 7,
                "error": None,
            }) + "\n")


if __name__ == "__main__":
    main()
