# Design: voice-suite — Speech-Translation Eval Harness for my-2nd-voice

Date: 2026-07-06
Status: DRAFT (pending user review)
Origin: brainstormed from rag-suite (`C:\project\training_ai\rag-suite`), evaluating
my-2nd-voice (`C:\project\training_ai\my-2nd-voice`), branch
**`feat/run-on-windows`** (Windows build + WASAPI/APM/denoise/beam-search work
already merged there; that branch is the base for all app-side changes).

---

## Problem statement

my-2nd-voice is a near-realtime voice translator (mic → VAD → Whisper ASR →
Marian MT → Supertonic TTS → virtual mic). Its DESIGN.md plans eval suites
(WER, MT quality, TTS intelligibility, latency) but none exist. rag-suite is a
proven benchmark harness, but its task is pinned to German-law QA-with-citations.

voice-suite is a new harness, forked from rag-suite's skeleton, whose task is
**full-pipeline speech translation**: English audio in → Vietnamese audio out,
scored on ASR accuracy, translation quality, end-to-end intelligibility, and
per-stage latency, with a leaderboard comparing pipeline configurations.

## Why fork instead of generalize (decision record)

Measured on rag-suite's 1,259-line core: ~48% is the German-law task (ingest,
articleparse, registry, textnorm, retrieval scoring, protocol types) and would
be deleted/rewritten for voice regardless of approach. Generalizing rag-suite
into a multi-domain harness (option B) adds a refactor of the *other* half on
top of that work, risks a live benchmark, and designs plugin abstractions from
only two examples. Fork now; extract a shared core if a third eval task ever
appears (rule of three). Accepted cost: runner/sampling/aggregate/judge
plumbing (~350 lines) exists twice.

## Scope (v1)

- Direction: **vi → en** only — matching how the branch is actually used and
  tuned (PhoWhisper Vietnamese ASR, `models/mt/vi-en` Marian, Vietnamese
  whisper prompt, beam search for tonal languages). en → vi is a config
  addition later; the schema supports it from day one (`src_lang`/`dst_lang`
  per utterance).
- Full pipeline audio→audio, offline/batch. Live-mic latency is out of scope
  (covered by the app's own planned latency CI).
- Runs on **Windows** (this machine). The app's eval binary must build here.
- One impl at launch (`m2v_default`); the impl mechanism supports N configs.

### Non-goals

- No VAD in eval mode (deliberate, see eval_batch). No streaming/partials.
- No TTS naturalness/MOS scoring (intelligibility via back-transcription only).
- No cloud path evaluation (ElevenLabs E5).
- No generalization of rag-suite; it stays untouched.

---

## Architecture

Two sides meeting at one file-based contract (manifest in, JSONL + WAVs out):

```
voice-suite (Python, this repo)                my-2nd-voice (Rust)
┌──────────────────────────────┐               ┌─────────────────────────┐
│ ingest: FLEURS → manifest    │               │ new bin: eval_batch     │
│ run: impl.translate_batch ───┼── manifest ──▶│  load models ONCE       │
│      per-utterance cache     │◀── JSONL ─────┼  per WAV: ASR→MT→TTS    │
│ score: WER · judge · back-   │    + WAVs     │  write WAV + JSONL line │
│        transcription · lat.  │               └─────────────────────────┘
│ report: leaderboard          │
└──────────────────────────────┘
```

### Side 1 — app-side changes (my-2nd-voice repo, branch `feat/run-on-windows`)

Windows build gating **already exists** on this branch: `screencapturekit` and
Metal whisper are macOS-gated, non-macOS gets CPU whisper-rs with an optional
**`cuda` feature** (`whisper-rs/cuda` + `ort/cuda`); `hound` is already a
dependency. The only app-side work is the new binary:

1. **`src/bin/eval_batch.rs`** (new, ~200 lines). Args: `--manifest <jsonl>`,
   `--out-dir <dir>`, `--whisper-model`, `--mt-dir`, `--tts-onnx-dir`,
   `--tts-voice-style`, `--src vi`, `--dst en`, plus quality knobs matching
   the live pipeline: `--beam-size` (default 5) and `--whisper-prompt`
   (default = the branch's Vietnamese anchor prompt). Behavior:
   - Load Whisper + Marian + Supertonic once (reuses existing
     `WhisperAsr`/`MarianMt`/`Supertonic`, the stage traits, and the branch's
     ASR episode normalization).
   - Per manifest row: read WAV (`hound`), resample to 16 kHz mono if needed
     (existing `SincResampler`), ASR → MT → TTS, write
     `<out-dir>/<id>.wav` + append JSONL line:
     `{"id", "asr_text", "mt_text", "audio_out", "asr_ms", "mt_ms", "tts_ms", "error"}`.
   - A failing utterance sets `error` and continues; the process only exits
     non-zero for setup failures (missing model/dir).
   - Marian is single-pair, so a batch is direction-homogeneous: eval_batch
     rejects manifest rows whose langs don't match `--src`/`--dst` (that row
     gets `error`, not silent mistranslation).
   - **No VAD/merge/denoise/APM**: FLEURS clips are clean single utterances;
     skipping the live-audio front-end keeps runs deterministic and measures
     AI quality (ASR/MT/TTS), not chunking or noise handling. It also removes
     the silero model from eval prerequisites. (VAD-on could be a later
     flag/impl-config.)
   - Builds with or without `--features cuda`; the impl config points at
     whichever exe was built (cuDNN DLL placement is handled by the app's
     existing run scripts / documented setup).

Note: this branch's run-demo scripts reference an F5-TTS voice-clone stack
(`--tts-voice`, `--tts-ref-text`) that `translate.rs` here does not compile —
they belong to sibling branches (`feat/integrate-sherpa-styletts2`,
`develop-sherpa-vi-asr`). v1 evaluates what **this branch builds**: Supertonic
TTS. When F5/sherpa merges, it becomes a second impl config on the leaderboard
— exactly the comparison this harness exists for.

### Side 2 — harness repo layout (this repo)

```
voice-suite/
├── data/
│   ├── utterances/*.wav      # FLEURS input audio (git-ignored)
│   └── manifest.jsonl        # compiled ground truth (git-ignored; rebuildable)
├── impls/
│   └── m2v_default/          # IMPL: wraps eval_batch with default config
├── voice_suite/
│   ├── cli.py                # ingest / run / score / report
│   ├── config.py             # paths + impl discovery      (from rag-suite)
│   ├── protocol.py           # contract types              (new)
│   ├── dataset.py            # manifest load               (adapted)
│   ├── runner.py             # run + per-utterance cache   (adapted: batch impl)
│   ├── sampling.py           # --limit/--seed              (verbatim)
│   ├── aggregate.py          # leaderboard                 (adapted metrics)
│   ├── ingest_fleurs.py      # FLEURS → manifest + WAVs    (new)
│   └── scoring/
│       ├── __init__.py       # orchestration + cache       (adapted)
│       ├── wer.py            # ASR word error rate         (new, jiwer)
│       ├── judge.py          # Claude judge                (plumbing from rag-suite,
│       │                     #                              new translation rubric)
│       └── backtranscribe.py # faster-whisper on output    (new)
├── docs/superpowers/specs/   # this document
├── out/                      # raw_results.jsonl, scored.jsonl, report.md (git-ignored)
└── tests/                    # fully offline
```

---

## The contract

Key difference from rag-suite: **batch, not per-item**. The impl is an external
process with a multi-GB model load; per-utterance invocation would reload
models N times. The runner filters out already-cached utterances and hands the
remainder to the impl in one call.

```python
class StageTimings(BaseModel):           # frozen
    asr_ms: int = 0
    mt_ms: int = 0
    tts_ms: int = 0

class Utterance(BaseModel):              # frozen — one ground-truth row
    id: str
    audio_path: str                      # input WAV (16 kHz mono, src speech)
    src_lang: str                        # "vi"
    dst_lang: str                        # "en"
    ref_transcript: str                  # what was said (Vietnamese)
    ref_translation: str                 # reference translation (English)

class VoiceResult(BaseModel):            # frozen — one utterance's output
    asr_text: str
    mt_text: str
    audio_path: str | None               # translated speech WAV, None on error
    timings: StageTimings = StageTimings()
    error: str | None = None

class RawRecord(BaseModel):              # frozen — run-step output line
    impl: str
    utt_id: str
    result: VoiceResult

@runtime_checkable
class VoiceImpl(Protocol):
    name: str
    def setup(self) -> None: ...         # verify exe + model files; fail fast
    def translate_batch(self, utts: list[Utterance],
                        out_dir: Path) -> list[VoiceResult]: ...
```

`impls/m2v_default/` writes a temp manifest for `utts`, invokes `eval_batch`
once, parses the result JSONL, and returns `VoiceResult`s in input order
(missing ids → `VoiceResult(error=...)`). Exe and model paths come from
`impls/m2v_default/local.toml` (git-ignored); `setup()` fails with a template
of that file when it's missing. An "impl" is a *pipeline configuration*; later
impls reuse the same wrapper with different flags (whisper medium, VAD on,
NLLB when the app grows it).

Discovery is rag-suite's: scan `impls/*/__init__.py` for `IMPL`, tolerate
import errors.

---

## Dataset & ingest

Source: **FLEURS** (HuggingFace `google/fleurs`), configs `vi_vn` and `en_us`,
`test` split. FLEURS sentences are n-way parallel (FLoRes heritage): joining
`vi_vn` and `en_us` rows on the FLEURS `id` field yields
(vi audio, vi transcript, en text) triplets — exactly the needed ground truth.

`voice-suite ingest --n 200 --seed 0`:
1. Download both configs via `huggingface_hub`/`datasets` (cached by HF).
2. Join on `id`; drop ids missing on either side.
3. Deterministic sample of N (default 200) with seed.
4. Write `data/utterances/<id>.wav` (16 kHz mono — FLEURS native rate,
   Vietnamese speech) and `data/manifest.jsonl` rows:
   `{"id", "audio_path", "src_lang": "vi", "dst_lang": "en", "ref_transcript", "ref_translation"}`.
   For FLEURS, `ref_transcript` = vi `transcription` (raw casing),
   `ref_translation` = en `transcription` of the same id.

`run`/`score` accept `--limit N --seed S` (sampling the manifest) exactly like
rag-suite — use matching values on both.

## CLI

```bash
voice-suite ingest [--n 200] [--seed 0]         # FLEURS → manifest + WAVs
voice-suite run   --impl all|m2v_default [--limit N --seed S]
voice-suite score [--judge cli|api] [--model <id>] [--workers N]
                  [--no-judge] [--limit N --seed S]
voice-suite report
```

Judge defaults mirror rag-suite: backend flags `cli` (Claude Code login) /
`api` (`ANTHROPIC_API_KEY`), default judge model `claude-opus-4-8`,
`--workers` thread pool with lock-guarded JSONL writes, `--no-judge` for the
free offline path (WER + latency only).

---

## Scoring

Per (impl, utterance), four signals:

| Signal | How | Cost |
|---|---|---|
| ASR quality | `wer(norm(asr_text), norm(ref_transcript))` via `jiwer` on the Vietnamese text; norm = lowercase, strip punctuation, collapse whitespace (unicode-safe — Vietnamese diacritics preserved, casing/punct folded) | free |
| MT quality | LLM judge (below): `mt_adequacy`, `mt_fluency` ∈ [0,1] on the English translation | 1 judge call |
| E2E intelligibility | back-transcribe output WAV with `faster-whisper large-v3` (CPU, local, language forced to `en`); judge scores `e2e_adequacy` from the back-transcript | slow-local + same judge call |
| Latency | pass through `asr_ms/mt_ms/tts_ms` | free |

### Judge (one call per utterance, three scores)

Plumbing copied from rag-suite (`_cli_judge`, `_anthropic_judge`,
`_loads_lenient`, injectable `JudgeFn`, worker pool). New rubric — English
prompt, bias-blind (no impl identity), judging the English translation of a
Vietnamese source:

- Inputs: `ref_transcript` (vi), `ref_translation` (en), `asr_text`,
  `mt_text`, `back_transcript` (en, from output audio).
- Output JSON: `{"mt_adequacy": f, "mt_fluency": f, "e2e_adequacy": f,
  "verdict": str, "rationale": str}`.
- `mt_*` judge `mt_text` against `ref_translation` (given the source);
  `e2e_adequacy` judges `back_transcript` against `ref_translation` — "after
  TTS, does the audio still say the right thing?"

### Back-transcription scorer

`faster-whisper large-v3` — deliberately a larger model than the
PhoWhisper-small under test, so the scorer out-hears the system. Language
forced to `en` (the TTS output language).
Results cached by output-audio content hash (it is the slowest scorer).
If the output WAV is missing/unreadable → `e2e` scored 0 with error recorded;
MT scores still computed.

### Caching

rag-suite's content-keyed model, adapted:
- Run cache: per `(impl, utt_id)` in `out/raw_results.jsonl` (idempotent re-runs).
- Score cache key: `(impl, utt_id, asr_text, mt_text, audio_sha256,
  judge_version, judge_id)` where `judge_id = <backend>:<model>` or
  `no-judge`. Changing judge/model/text re-judges only what changed.
- Back-transcription cache: `audio_sha256 → transcript` (survives judge changes).

## Report

`out/report.md` leaderboard, one row per impl:

| impl | n | WER↓ | mt_adequacy↑ | mt_fluency↑ | e2e_adequacy↑ | asr_ms P50/P95 | mt_ms P50/P95 | tts_ms P50/P95 | errors |

Plus a worst-10 utterances appendix (lowest e2e_adequacy) with texts, for
eyeballing failure modes.

---

## Error handling

| Failure | Behavior |
|---|---|
| Model/exe missing | `impl.setup()` fails fast, named path in message, before any run |
| One utterance crashes in eval_batch | JSONL line with `error`; batch continues; harness scores it 0 and reports it |
| eval_batch exits non-zero | run aborts for that impl with stderr surfaced; other impls unaffected |
| Judge call fails/unparseable | per-utterance score record gets `error`; run continues (rag-suite pattern) |
| Back-transcription fails | `e2e_adequacy=0` + error recorded; MT/WER still scored |
| Result JSONL missing an id | `VoiceResult(error="missing from eval_batch output")` |

## Testing (offline; CI needs no models, audio hardware, or API)

- **Fake eval_batch**: a Python fixture script echoing canned JSONL — exercises
  the impl wrapper, runner, caching, and order-restoration end to end.
- WER unit tests with known pairs — v1 scores Vietnamese ASR text, so cases
  cover diacritics preservation and punctuation/case folding.
- Ingest tests against fake FLEURS rows (join, drop-missing, deterministic sample).
- Judge tests via injectable `JudgeFn` mock (rag-suite trick), incl. lenient
  JSON parsing cases.
- Back-transcription mocked; audio-hash cache tested with tiny generated WAVs.
- Live tests marked `live`, deselected by default.
- App-side: unit tests for manifest parse/WAV IO; one env-gated integration
  test with a real canned WAV (requires models; manual).

## Prerequisites (documented in README at implementation)

- Rust toolchain + Windows build deps per the app README (VS Build Tools,
  LLVM, CMake); the branch already builds on Windows. Optional `--features
  cuda` build (cuDNN DLLs next to the exe, per the app's run scripts).
- Models on disk (one-time; `download_models.sh` is referenced by the app's
  scripts but absent on this branch — the implementation plan includes
  reconstructing the download steps and pinning exact sources):
  - PhoWhisper ggml (branch scripts use
    `models/whisper/ggml-phowhisper-small-tsa.bin`)
  - Marian vi→en ONNX dir (`models/mt/vi-en`): `tokenizer.json`,
    `generation_config.json`, `encoder_model_quantized.onnx`,
    `decoder_model_quantized.onnx` (optimum/transformers.js layout)
  - Supertonic ONNX dir (`tts.json`, 4 `.onnx`, `unicode_indexer.json`) +
    an English voice style JSON
  - No silero VAD model needed (eval mode skips VAD)
- Python ≥3.12, `pip install -e ".[dev]"`; extras: `jiwer`, `faster-whisper`,
  `huggingface_hub`/`datasets`, `soundfile`.
- Judge: Claude Code CLI login (default) or `ANTHROPIC_API_KEY`.

## Deliberate simplifications (v1)

- No VAD in eval mode — deterministic, measures AI quality; VAD chunking is a
  live-pipeline concern. Revisit as an impl config if chunking quality matters.
- vi→en only — schema is direction-agnostic; en→vi = second Marian model +
  second manifest slice + one impl config.
- v1 evaluates this branch's TTS (Supertonic). The F5/sherpa voice-clone stack
  from sibling branches becomes a second impl config when it merges.
- TTS scored only via intelligibility (back-transcription), not naturalness.
- Latency numbers from eval_batch are *stage compute times* on this machine,
  not the app's live end-to-end latency budget (VAD/jitter/devices excluded).
