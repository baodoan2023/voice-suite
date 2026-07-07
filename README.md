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

### 2. Marian vi→en ONNX — `models/mt/vi-en/`

Needs exactly: `tokenizer.json`, `generation_config.json`,
`encoder_model_quantized.onnx`, `decoder_model_quantized.onnx` (flat in the
dir — transformers.js/optimum layout). `Xenova/opus-mt-vi-en` ships these:

    hf download Xenova/opus-mt-vi-en --local-dir /tmp/opus-vi-en
    mkdir -p models/mt/vi-en
    cp /tmp/opus-vi-en/tokenizer.json /tmp/opus-vi-en/generation_config.json models/mt/vi-en/
    cp /tmp/opus-vi-en/onnx/encoder_model_quantized.onnx models/mt/vi-en/
    cp /tmp/opus-vi-en/onnx/decoder_model_quantized.onnx models/mt/vi-en/

### 3. Supertonic TTS — `models/tts/supertonic-3/`

The download recipe exists on the app repo's `origin/feat/python-setup`
branch; read it and port the paths:

    git -C ../my-2nd-voice show origin/feat/python-setup:download_supertonic.sh

It fetches the `Supertone/supertonic-3` HF repo: the ONNX dir (`tts.json`,
four `.onnx` files, `unicode_indexer.json`) plus `voice_styles/*.json`.
Point `tts_onnx_dir` at the ONNX dir and `tts_voice_style` at an **English**
voice style JSON.

No silero VAD model is needed — eval mode skips VAD by design.

### 4. Build the eval binary

    cd ../my-2nd-voice
    cargo build --release --bin eval_batch
    # GPU: cargo build --release --features cuda --bin eval_batch
    #      (cuDNN DLLs next to the exe, per the app's run scripts)

## Configure the impl

    cp impls/m2v_default/local.toml.example impls/m2v_default/local.toml
    # edit paths to your exe + models (local.toml is git-ignored)

## Quickstart

    voice-suite ingest --n 200 --seed 0        # FLEURS vi_vn ⋈ en_us → data/
    voice-suite run   --impl m2v_default --limit 20 --seed 0
    voice-suite score --limit 20 --seed 0 --workers 4     # judge via Claude CLI
    voice-suite report                          # out/report.md + stdout

Free offline path (WER + latency only, no judge, no back-transcription):

    voice-suite score --no-judge --limit 20 --seed 0

Use the SAME `--limit`/`--seed` on `run` and `score` so both select the same
utterances. Re-runs are incremental: raw results cache per (impl, utterance),
scores cache per content+judge, back-transcripts cache per audio hash.

## Manual smoke (models required)

    cd ../my-2nd-voice
    M2V_WHISPER_MODEL=models/whisper/ggml-phowhisper-small-tsa.bin \
    M2V_MT_DIR=models/mt/vi-en \
    M2V_TTS_ONNX_DIR=models/tts/supertonic-3/onnx \
    M2V_TTS_VOICE_STYLE=models/tts/supertonic-3/voice_styles/M1.json \
    cargo test --test eval_batch_smoke -- --nocapture

Then the real thing end-to-end: `voice-suite run --impl m2v_default --limit 2`
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
