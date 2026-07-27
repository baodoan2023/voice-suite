# voice-suite

Speech-translation eval harness for [my-2nd-voice]. Benchmarks the full
pipeline — Vietnamese audio in → English audio out — on ASR accuracy (WER),
translation quality (LLM judge), end-to-end intelligibility
(back-transcription of the output audio), and per-stage latency, with a
leaderboard comparing pipeline configurations.

Design doc: `docs/superpowers/specs/2026-07-06-voice-suite-design.md`.

## Prerequisites

- **Python ≥ 3.12**, then: `pip install -e ".[dev,ingest,bt]"`
  - `dev` = pytest; `ingest` = datasets/soundfile (FLEURS download);
    `bt` = faster-whisper (back-transcription; first score run downloads
    ~3 GB `large-v3` from HuggingFace, cached).
- **my-2nd-voice** checked out at branch `feat/run-on-windows` with its
  Windows build deps (VS Build Tools, LLVM, CMake — see that repo's README).
- **Judge auth** (skip with `--no-judge`): either the Claude Code CLI logged
  in (`--judge cli`, default) or `ANTHROPIC_API_KEY` set (`--judge api`).

## Models (one-time, ~machine-local)

Model files live in the my-2nd-voice repo under `models/` (git-ignored
there). Layouts below match what `eval_batch` loads; the loaders name any
missing file in their error message.

### 1. PhoWhisper ggml — `models/whisper/ggml-phowhisper-small-tsa.bin`

If you already have this file from earlier work, reuse it. Reproducible
path: convert `vinai/PhoWhisper-small` with whisper.cpp's converter:

    hf download vinai/PhoWhisper-small --local-dir /tmp/phowhisper-small
    git clone https://github.com/ggml-org/whisper.cpp /tmp/whisper.cpp
    git clone https://github.com/openai/whisper /tmp/openai-whisper
    pip install torch transformers
    python /tmp/whisper.cpp/models/convert-h5-to-ggml.py \
        /tmp/phowhisper-small /tmp/openai-whisper models/whisper/
    # rename the produced ggml-model.bin to ggml-phowhisper-small-tsa.bin
    # (or point local.toml at whatever name you keep)

### 2. Sherpa-onnx Zipformer VI — `models/asr/sherpa-vi/`

Needs `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`, `bpe.model`,
plus a `.sha256` sidecar per ONNX/tokens file (my-2nd-voice's `SherpaAsr`
loader verifies these on load). A download script already lives in that repo
and writes the sidecars for you:

    cd ../my-2nd-voice
    python python/download_sherpa_vi.py --output-dir models/asr/sherpa-vi
    # or on Windows: .\python\download_sherpa_vi.ps1

Pulls the 68M Zipformer model from `csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20`
on HuggingFace and renames the epoch-suffixed files to the canonical names above.

### 3. Nemotron venv — `.venv-nemo/`

NVIDIA's Nemotron ASR runs as a bundled Python subprocess
(`scripts/nemotron_asr_server.py`) instead of an ONNX/ggml model file — one-time
venv setup, no separate model-download step:

    cd ../my-2nd-voice
    python -m venv .venv-nemo
    .venv-nemo\Scripts\python.exe -m pip install "nemo_toolkit[asr]"   # Windows
    # .venv-nemo/bin/python -m pip install "nemo_toolkit[asr]"          # macOS/Linux

First inference downloads `nvidia/nemotron-3.5-asr-streaming-0.6b` from
HuggingFace and caches it. CPU-only for now — expect ~38s P50 per utterance
vs. Sherpa's ~0.4s (see `impls/m2v_nemotron/local.toml.example`).

### 4. Marian vi→en ONNX — `models/mt/vi-en/`

Needs exactly: `tokenizer.json`, `generation_config.json`,
`encoder_model_quantized.onnx`, `decoder_model_quantized.onnx` (flat in the
dir — transformers.js/optimum layout). `Xenova/opus-mt-vi-en` ships these:

    hf download Xenova/opus-mt-vi-en --local-dir /tmp/opus-vi-en
    mkdir -p models/mt/vi-en
    cp /tmp/opus-vi-en/tokenizer.json /tmp/opus-vi-en/generation_config.json models/mt/vi-en/
    cp /tmp/opus-vi-en/onnx/encoder_model_quantized.onnx models/mt/vi-en/
    cp /tmp/opus-vi-en/onnx/decoder_model_quantized.onnx models/mt/vi-en/

### 5. StyleTTS2 — `.venv-styletts2/` + a reference voice wav

All three impls synthesize through the same StyleTTS2 python venv + server
script (`scripts/styletts2_server.py`), not Supertonic. One-time setup and a
voice-cloning walkthrough: `my-2nd-voice/docs/styletts2-voice-setup.md`
(`scripts/install-venv-styletts2.sh` creates the venv; record ~10s of
reference speech to a wav and point `tts_ref_wav` at it).

No silero VAD model is needed — eval mode skips VAD by design.

### 6. Build the eval binary

    cd ../my-2nd-voice
    cargo build --release --bin eval-batch
    # GPU: cargo build --release --features cuda --bin eval-batch
    #      (cuDNN DLLs next to the exe, per the app's run scripts)

## Configure the impl

Each impl under `impls/` needs its own `local.toml` (git-ignored) — copy the
example that matches the impl(s) you'll run and edit the paths:

    cp impls/m2v_phowhisper/local.toml.example impls/m2v_phowhisper/local.toml
    cp impls/m2v_sherpa/local.toml.example impls/m2v_sherpa/local.toml
    cp impls/m2v_nemotron/local.toml.example impls/m2v_nemotron/local.toml

- `m2v_phowhisper` — PhoWhisper + Marian ONNX + StyleTTS2; needs the
  PhoWhisper ggml model (see "Models" above) plus a StyleTTS2 python venv +
  server script and a reference voice wav. See
  `impls/m2v_phowhisper/local.toml.example`.
- `m2v_sherpa` — sherpa-onnx ASR + Marian MT + StyleTTS2; needs its own
  sherpa model dir, a StyleTTS2 python venv + server script, and a
  reference voice wav. See `impls/m2v_sherpa/local.toml.example`.
- `m2v_nemotron` — NVIDIA Nemotron streaming ASR + Marian MT + StyleTTS2;
  needs its own Nemotron python venv + server script, plus the same
  StyleTTS2 venv + server script and reference voice wav as the other
  impls. See `impls/m2v_nemotron/local.toml.example`.

Impls are auto-discovered from `impls/`, so all three are available as
`--impl` values once configured. Impl ids follow `m2v_{asr_model}` — named
after the ASR engine, since that's what most defines a pipeline's character
(e.g. `m2v_phowhisper`, `m2v_sherpa`, `m2v_nemotron`); name new impls the
same way.

## Quickstart

    voice-suite ingest --n 200 --seed 0        # FLEURS vi_vn ⋈ en_us → data/
    voice-suite run   --impl m2v_phowhisper --limit 20 --seed 0
    voice-suite score --limit 20 --seed 0 --workers 4     # judge via Claude CLI
    voice-suite report                          # out/report.md + stdout

`report` includes a metric glossary and a mechanical cross-impl analysis
(leader/gap per metric, latency ratios, error rates, shared-worst
utterances) whenever more than one impl has been scored. Full metric
definitions: `docs/metrics.md`.

Free offline path (WER + latency only, no judge, no back-transcription):

    voice-suite score --no-judge --limit 20 --seed 0

Use the SAME `--limit`/`--seed` on `run` and `score` so both select the same
utterances. Re-runs are incremental: raw results cache per (impl, utterance),
scores cache per content+judge, back-transcripts cache per audio hash.

## WER-only benchmark with manual review (sherpa)

Focused ASR track: sherpa zipformer only, no MT/TTS, WER as the single
metric, with human adjudication instead of an LLM judge.

    pip install -e '.[asr]'                    # sherpa-onnx
    voice-suite asr-run                        # transcribe the whole manifest (~min, CPU)
    voice-suite review                         # opens a local page: listen, see word
                                               #   diffs, tick ops that are NOT real
                                               #   errors (variants, spoken numbers)
    voice-suite wer-report                     # out/wer_report.md: raw + adjudicated WER

Cloud engines for comparison (same manifest, same WER scoring):

    export OPENAI_API_KEY=...                  # or GEMINI_API_KEY
    voice-suite asr-run --engine openai        # gpt-4o-transcribe (--model whisper-1 …)
    voice-suite asr-run --engine gemini        # gemini-2.5-flash audio transcription
    voice-suite wer-report                     # one table per engine + per-utt matrix

`asr-run` reuses `impls/m2v_sherpa/local.toml` for the sherpa model dir and
is incremental per engine (`out/asr_<engine>.jsonl`, append-only; failed
cloud calls are skipped and retried on rerun). Review decisions live in
`out/wer_decisions_<engine>.json` (`review --engine <name>`); a decision is
an accepted word op ("this sub/del/ins does not count"), so adjudicated WER
= errors the reviewer left standing ÷ reference words.

## Manual smoke (models required)

Run `eval-batch` directly against a tiny manifest, bypassing the harness:

    cd ../my-2nd-voice
    cargo run --release --bin eval-batch -- \
      --sherpa-model-dir models/asr/sherpa-vi \
      --mt-dir models/mt/vi-en \
      --tts-venv .venv-styletts2 --tts-ref-wav assets/voice_ref/tsa/ref.wav \
      --src vi --dst en \
      --manifest manifest.jsonl --out-dir out/

(Swap the ASR flags for `--whisper-model <path>` or `--nemotron-venv <path>`
to smoke-test a different impl's ASR backend instead.)

Then the real thing end-to-end: `voice-suite run --impl m2v_phowhisper --limit 2`
followed by `voice-suite score --limit 2 --no-judge` and `voice-suite report`.

## Tests

    python -m pytest        # fully offline: no models, audio, or API

## Cost & runtime notes

- Judge: one Claude call per (utterance × impl × judge-model). 200 utterances
  ≈ 200 calls per impl per judge config.
- Back-transcription: CPU faster-whisper large-v3, roughly real-time or
  slower per clip on CPU — the content cache means you pay it once per
  distinct output audio.
- eval_batch on CPU with beam 5: expect seconds per utterance; start with
  `--limit 20` before a full 200-utterance run.
