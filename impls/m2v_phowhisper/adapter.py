"""Batch adapter: shells out to my-2nd-voice's eval-batch binary (PhoWhisper
ASR via whisper.cpp + Marian MT + StyleTTS2).

Same eval_batch manifest/results.jsonl contract as m2v_sherpa; only the
model config and CLI flags differ. See impls/m2v_sherpa/adapter.py for the
shared design notes.
"""
from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from voice_suite.protocol import StageTimings, Utterance, VoiceResult

_REQUIRED_KEYS = ("exe", "whisper_model", "mt_dir", "tts_venv",
                  "tts_server_script", "tts_ref_wav")

_TEMPLATE = """\
# impls/m2v_phowhisper/local.toml — machine-local paths (git-ignored)
exe = "C:/project/training_ai/my-2nd-voice/target/release/eval-batch.exe"
whisper_model = "C:/project/training_ai/my-2nd-voice/models/whisper/ggml-phowhisper-base.bin"
mt_dir = "C:/project/training_ai/my-2nd-voice/models/mt/vi-en"
tts_venv = "C:/project/training_ai/my-2nd-voice/.venv-styletts2"
tts_server_script = "C:/project/training_ai/my-2nd-voice/scripts/styletts2_server.py"
tts_ref_wav = "C:/project/training_ai/my-2nd-voice/assets/voice_ref/tsa/ref.wav"
# extra_args = ["--beam-size", "5"]
"""


class M2vPhoWhisperImpl:
    """VoiceImpl adapter around one eval-batch invocation per batch."""

    def __init__(self, name: str = "m2v_phowhisper",
                 config_path: Path | None = None,
                 cmd: list[str] | None = None):
        self.name = name
        self._config_path = config_path or Path(__file__).parent / "local.toml"
        self._cmd = cmd  # test seam: full command prefix, bypasses local.toml

    def setup(self) -> None:
        """Resolve and validate the command; fail fast with a config template."""
        if self._cmd is not None:
            return
        if not self._config_path.exists():
            raise RuntimeError(
                f"missing impl config: {self._config_path}\n"
                f"Create it with your local paths:\n\n{_TEMPLATE}")
        cfg = tomllib.loads(self._config_path.read_text(encoding="utf-8"))
        missing = [k for k in _REQUIRED_KEYS if k not in cfg]
        if missing:
            raise RuntimeError(
                f"{self._config_path} is missing keys: {missing}\n\n{_TEMPLATE}")
        for key in _REQUIRED_KEYS:
            if not Path(cfg[key]).exists():
                raise RuntimeError(
                    f"{key} does not exist: {cfg[key]} "
                    f"(from {self._config_path})")
        self._cmd = [
            str(cfg["exe"]),
            "--whisper-model", str(cfg["whisper_model"]),
            "--mt-dir", str(cfg["mt_dir"]),
            "--tts-venv", str(cfg["tts_venv"]),
            "--tts-server-script", str(cfg["tts_server_script"]),
            "--tts-ref-wav", str(cfg["tts_ref_wav"]),
            "--src", "vi", "--dst", "en",
            *[str(a) for a in cfg.get("extra_args", [])],
        ]

    def translate_batch(self, utts: list[Utterance],
                        out_dir: Path) -> list[VoiceResult]:
        """One eval-batch process for the whole batch; results in input order."""
        if self._cmd is None:
            raise RuntimeError("setup() must be called before translate_batch()")
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = out_dir / "_manifest.jsonl"
        manifest.write_text(
            "".join(u.model_dump_json() + "\n" for u in utts),
            encoding="utf-8")

        # No timeout: a 200-utterance CPU batch legitimately runs for a while.
        proc = subprocess.run(
            [*self._cmd, "--manifest", str(manifest), "--out-dir", str(out_dir)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"eval-batch exited {proc.returncode}:\n{proc.stderr.strip()}")

        by_id: dict[str, VoiceResult] = {}
        results_path = out_dir / "results.jsonl"
        if results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                by_id[row["id"]] = VoiceResult(
                    asr_text=row.get("asr_text", ""),
                    mt_text=row.get("mt_text", ""),
                    audio_path=row.get("audio_out"),
                    timings=StageTimings(asr_ms=int(round(row.get("asr_ms", 0))),
                                         mt_ms=int(round(row.get("mt_ms", 0))),
                                         tts_ms=int(round(row.get("tts_ms", 0)))),
                    error=row.get("error"),
                )
        missing = VoiceResult(error="missing from eval_batch output")
        return [by_id.get(u.id, missing) for u in utts]
