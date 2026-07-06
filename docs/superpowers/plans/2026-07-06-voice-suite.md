# voice-suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build voice-suite, a Python eval harness that benchmarks my-2nd-voice's full speech-translation pipeline (Vietnamese audio in → English audio out) on WER, LLM-judged translation quality, end-to-end intelligibility, and per-stage latency, with a leaderboard comparing pipeline configurations.

**Architecture:** Two sides meet at one file-based contract. The harness (a fork of rag-suite's skeleton: runner/sampling/aggregate/judge plumbing) samples a FLEURS-derived manifest and hands uncached utterances to an impl in ONE `translate_batch` call; the `m2v_default` impl shells out to a new `eval_batch` binary in my-2nd-voice (branch `feat/run-on-windows`) which loads Whisper+Marian+Supertonic once and emits per-utterance JSONL + WAVs. Scoring combines free local signals (jiwer WER, faster-whisper back-transcription) with one Claude judge call per utterance; everything is content-key cached.

**Tech Stack:** Python ≥3.12, pydantic v2 (frozen models), typer, jiwer, faster-whisper (`bt` extra), datasets+soundfile (`ingest` extra), anthropic SDK / Claude Code CLI (judge). App side: Rust — clap, hound, serde_json, plus the existing `WhisperAsr`/`MarianMt`/`Supertonic` stages and `SincResampler`.

**Reference docs:** design spec at `docs/superpowers/specs/2026-07-06-voice-suite-design.md`; rag-suite source (read-only donor) at `C:\project\training_ai\rag-suite`.

## Global Constraints

- Harness repo: `C:\project\training_ai\voice-suite` (local-only git repo). All paths and commands are relative to it unless a task says otherwise.
- App repo: `C:\project\training_ai\my-2nd-voice`, branch **`feat/run-on-windows`** (Task 14 only). rag-suite stays untouched.
- Direction v1: **vi → en** only; the schema still carries `src_lang`/`dst_lang` per utterance.
- Python `requires-python = ">=3.12"`; every schema type is pydantic v2 with `model_config = ConfigDict(frozen=True)`.
- Tests are fully offline by default: no models, no audio hardware, no API. Anything live is marked `live` and deselected via pytest addopts `-m 'not live'`.
- Judge defaults: model `claude-opus-4-8`; backend `cli` (Claude Code login) default, `api` (`ANTHROPIC_API_KEY`) optional. `JUDGE_VERSION = "v1"` participates in the score cache key.
- Back-transcription pinned: faster-whisper `large-v3`, CPU, `compute_type="int8"`, `language="en"`.
- Impls are batch-invoked (`translate_batch`), never per-utterance — the Rust exe loads ~2 GB of models once per batch.
- No VAD in eval mode; `eval_batch` has no VAD/merge/denoise/APM path (deliberate, per spec).
- my-2nd-voice iron rules apply in Task 14: no `unwrap`/`expect` outside tests+`main`, no ASR/MT text in `tracing` logs (the results JSONL file is the only text sink), `cargo fmt` + `cargo clippy -- -D warnings` clean.
- Commits: conventional format (`feat:`/`test:`/`docs:`/`chore:`); never pass `--no-verify`.
- Style: files ≤800 lines, functions <50 lines, early returns, explicit error handling, no hardcoded machine paths outside git-ignored `local.toml`.

## File Structure Map

```
voice-suite/                              (repo: C:\project\training_ai\voice-suite)
├── pyproject.toml                        T1   package metadata, deps, extras, pytest config
├── .gitignore                            T1   data/, out/, local.toml, caches
├── README.md                             T15  prereqs, model downloads, quickstart
├── voice_suite/
│   ├── __init__.py                       T1   package marker
│   ├── protocol.py                       T2   pydantic types + VoiceImpl Protocol
│   ├── ingest_fleurs.py                  T3   FLEURS → manifest + WAVs (pure join/sample fns)
│   ├── dataset.py                        T3   manifest.jsonl → list[Utterance]
│   ├── config.py                         T4   path constants + discover_impls()
│   ├── runner.py                         T5   run cache + one-batch dispatch
│   ├── sampling.py                       T6   deterministic --limit/--seed subset
│   ├── aggregate.py                      T11  leaderboard rows + markdown report
│   ├── cli.py                            T12  typer app: ingest/run/score/report
│   └── scoring/
│       ├── __init__.py                   T7 (empty) → T10 orchestration + score cache
│       ├── wer.py                        T7   norm_text + score_wer (jiwer)
│       ├── judge.py                      T8   rubric, prompt, cli/api backends, lenient JSON
│       └── backtranscribe.py             T9   faster-whisper wrapper + sha256 cache
├── impls/
│   └── m2v_default/
│       ├── __init__.py                   T13  exposes IMPL
│       ├── adapter.py                    T13  M2vBatchImpl (toml config, subprocess, order restore)
│       └── local.toml.example            T13  template for the git-ignored local.toml
├── tests/
│   ├── conftest.py                       T2   make_utt / make_result factory fixtures
│   ├── fake_eval_batch.py                T13  canned-JSONL test double for the Rust exe
│   ├── test_protocol.py                  T2
│   ├── test_ingest.py                    T3
│   ├── test_dataset.py                   T3
│   ├── test_config.py                    T4
│   ├── test_runner.py                    T5
│   ├── test_sampling.py                  T6
│   ├── test_wer.py                       T7
│   ├── test_judge.py                     T8
│   ├── test_backtranscribe.py            T9
│   ├── test_scoring.py                   T10
│   ├── test_aggregate.py                 T11
│   ├── test_cli.py                       T12
│   └── test_impl_m2v.py                  T13
├── data/                                 git-ignored: utterances/*.wav, manifest.jsonl
├── out/                                  git-ignored: raw_results.jsonl, audio/, scored.jsonl,
│                                                      backtranscripts.jsonl, report.md
└── docs/superpowers/                     specs + this plan

my-2nd-voice/                             (repo: C:\project\training_ai\my-2nd-voice, branch feat/run-on-windows)
├── Cargo.toml                            T14  hound → [dependencies]; new [[bin]] + [[test]]
├── src/bin/eval_batch.rs                 T14  manifest → ASR→MT→TTS → results.jsonl + WAVs
└── tests/integration/eval_batch_smoke.rs T14  env-gated end-to-end smoke (needs models)
```

### Cross-repo contract (referenced by Tasks 13 and 14 — must match exactly)

Invocation (all flags long-form; adapter appends the last two):

```
eval_batch --whisper-model <path> --mt-dir <path> --tts-onnx-dir <path>
           --tts-voice-style <path> --src vi --dst en
           [--beam-size 5] [--whisper-prompt <str>]
           --manifest <jsonl> --out-dir <dir>
```

- Reads manifest JSONL rows `{"id", "audio_path", "src_lang", "dst_lang", …}` (extra fields ignored).
- Writes `<out-dir>/<id>.wav` (16-bit PCM mono at the TTS engine rate) per successful row.
- Writes `<out-dir>/results.jsonl`, one row per manifest row (created fresh each run):
  `{"id": str, "asr_text": str, "mt_text": str, "audio_out": str|null, "asr_ms": int, "mt_ms": int, "tts_ms": int, "error": str|null}`.
- A failing utterance sets `error` and the batch continues. Rows whose langs don't match `--src`/`--dst` get `error` (Marian is single-pair). Exit code is non-zero only for setup failures (bad manifest, missing model/dir).

### Task 1: Project scaffolding — pyproject.toml, .gitignore, package init

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `voice_suite/__init__.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: installable `voice-suite` package; pytest configured with `live` marker deselected and `pythonpath = ["."]` (later tasks import `impls.*` in tests); extras `dev` / `ingest` / `bt`. The `[project.scripts]` entry points at `voice_suite.cli:app`, which is created in Task 12 — `pip install -e` does not import it, so installing now is safe.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "voice-suite"
version = "0.1.0"
description = "Speech-translation eval harness for my-2nd-voice (vi→en, audio→audio)"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.6",
    "anthropic>=0.40",
    "typer>=0.12",
    "jiwer>=3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]
ingest = ["datasets>=2.20,<3.0", "soundfile>=0.12", "numpy>=1.26"]
bt = ["faster-whisper>=1.0"]

[project.scripts]
voice-suite = "voice_suite.cli:app"

[tool.pytest.ini_options]
markers = ["live: tests that need models, audio hardware, or a live API; deselected by default"]
addopts = "-m 'not live'"
pythonpath = ["."]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["voice_suite*"]
```

`datasets` is pinned `<3.0` because `google/fleurs` is a script-based dataset; datasets 3.x removed loading-script support.

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.egg-info/
.pytest_cache/
.venv/
venv/
data/
out/
impls/*/local.toml
```

- [ ] **Step 3: Write `voice_suite/__init__.py`**

```python
"""voice-suite: speech-translation eval harness for my-2nd-voice."""
```

- [ ] **Step 4: Install editable with dev extra**

Run: `pip install -e ".[dev]"`
Expected: ends with `Successfully installed ... voice-suite-0.1.0`

- [ ] **Step 5: Confirm the package imports**

Run: `python -c "import voice_suite; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore voice_suite/__init__.py
git commit -m "chore: scaffold voice-suite package"
```

---
### Task 2: Protocol types (voice_suite/protocol.py) + shared test fixtures (tests/conftest.py)

**Files:**
- Create: `voice_suite/protocol.py`
- Create: `tests/conftest.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: nothing.
- Produces (every later task builds on these — exact fields matter):
  - `StageTimings(asr_ms: int = 0, mt_ms: int = 0, tts_ms: int = 0)` — frozen.
  - `Utterance(id, audio_path, src_lang, dst_lang, ref_transcript, ref_translation)` — frozen, all `str`.
  - `VoiceResult(asr_text: str = "", mt_text: str = "", audio_path: str | None = None, timings: StageTimings = StageTimings(), error: str | None = None)` — frozen.
  - `RawRecord(impl: str, utt_id: str, result: VoiceResult)` — frozen.
  - `JudgeScore(mt_adequacy: float = 0.0, mt_fluency: float = 0.0, e2e_adequacy: float = 0.0, verdict: str = "", rationale: str = "", error: str | None = None)` — frozen.
  - `ScoredRecord(impl, utt_id, cache_key: str = "", wer: float, asr_text: str = "", mt_text: str = "", back_transcript: str = "", back_error: str | None = None, judge: JudgeScore, timings: StageTimings = StageTimings(), run_error: str | None = None)` — frozen.
  - `VoiceImpl` Protocol (runtime_checkable): attribute `name: str`, `setup() -> None`, `translate_batch(utts: list[Utterance], out_dir: Path) -> list[VoiceResult]`.
  - Fixtures `make_utt(i=1, **overrides) -> Utterance` and `make_result(i=1, **overrides) -> VoiceResult` available to ALL tests via conftest.

- [ ] **Step 1: Write the failing test**

`tests/test_protocol.py`:

```python
"""Schema invariants: frozen models, defaults, JSON roundtrip, duck-typing."""
from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from voice_suite.protocol import (
    JudgeScore, RawRecord, ScoredRecord, StageTimings, Utterance,
    VoiceImpl, VoiceResult,
)


def test_utterance_is_frozen(make_utt):
    utt = make_utt()
    with pytest.raises(pydantic.ValidationError):
        utt.id = "changed"


def test_voice_result_defaults():
    r = VoiceResult()
    assert r.asr_text == ""
    assert r.mt_text == ""
    assert r.audio_path is None
    assert r.timings == StageTimings()
    assert r.error is None


def test_raw_record_json_roundtrip(make_utt, make_result):
    rec = RawRecord(impl="m2v_default", utt_id=make_utt().id, result=make_result())
    line = rec.model_dump_json()
    assert RawRecord.model_validate_json(line) == rec


def test_scored_record_roundtrip():
    rec = ScoredRecord(
        impl="a", utt_id="u001", wer=0.25,
        judge=JudgeScore(mt_adequacy=0.9, mt_fluency=0.8, e2e_adequacy=0.7,
                         verdict="good", rationale="close"))
    assert ScoredRecord.model_validate_json(rec.model_dump_json()) == rec


def test_voice_impl_is_runtime_checkable():
    class Fake:
        name = "fake"
        def setup(self) -> None: ...
        def translate_batch(self, utts, out_dir: Path):
            return []
    assert isinstance(Fake(), VoiceImpl)
```

`tests/conftest.py`:

```python
"""Shared test fixtures: factories for ground-truth and result objects."""
from __future__ import annotations

import pytest

from voice_suite.protocol import StageTimings, Utterance, VoiceResult


@pytest.fixture
def make_utt():
    def _make(i: int = 1, **overrides):
        base = dict(
            id=f"u{i:03d}",
            audio_path=f"data/utterances/u{i:03d}.wav",
            src_lang="vi",
            dst_lang="en",
            ref_transcript=f"xin chào số {i}",
            ref_translation=f"hello number {i}",
        )
        base.update(overrides)
        return Utterance(**base)
    return _make


@pytest.fixture
def make_result():
    def _make(i: int = 1, **overrides):
        base = dict(
            asr_text=f"xin chào số {i}",
            mt_text=f"hello number {i}",
            audio_path=None,
            timings=StageTimings(asr_ms=100, mt_ms=50, tts_ms=80),
            error=None,
        )
        base.update(overrides)
        return VoiceResult(**base)
    return _make
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.protocol'`

- [ ] **Step 3: Write the implementation**

`voice_suite/protocol.py`:

```python
"""Pydantic schemas and the VoiceImpl adapter protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class StageTimings(BaseModel):
    """Per-stage compute time in milliseconds for one utterance."""
    model_config = ConfigDict(frozen=True)

    asr_ms: int = 0
    mt_ms: int = 0
    tts_ms: int = 0


class Utterance(BaseModel):
    """One ground-truth row: source audio plus reference texts."""
    model_config = ConfigDict(frozen=True)

    id: str
    audio_path: str          # input WAV (16 kHz mono, source-language speech)
    src_lang: str            # e.g. "vi"
    dst_lang: str            # e.g. "en"
    ref_transcript: str      # what was said (source language)
    ref_translation: str     # reference translation (target language)


class VoiceResult(BaseModel):
    """Output produced by an implementation for a single utterance."""
    model_config = ConfigDict(frozen=True)

    asr_text: str = ""
    mt_text: str = ""
    audio_path: str | None = None    # translated-speech WAV; None on error
    timings: StageTimings = Field(default_factory=StageTimings)
    error: str | None = None


class RawRecord(BaseModel):
    """Raw (unscored) record linking an implementation run to an utterance."""
    model_config = ConfigDict(frozen=True)

    impl: str
    utt_id: str
    result: VoiceResult


class JudgeScore(BaseModel):
    """Translation-quality scores produced by the LLM judge."""
    model_config = ConfigDict(frozen=True)

    mt_adequacy: float = 0.0
    mt_fluency: float = 0.0
    e2e_adequacy: float = 0.0
    verdict: str = ""
    rationale: str = ""
    error: str | None = None


class ScoredRecord(BaseModel):
    """Fully scored record: WER + back-transcription + judge + latency.

    Carries asr_text/mt_text copies so scored.jsonl is independently
    readable (the report's worst-utterances appendix needs the texts).
    """
    model_config = ConfigDict(frozen=True)

    impl: str
    utt_id: str
    cache_key: str = ""
    wer: float
    asr_text: str = ""
    mt_text: str = ""
    back_transcript: str = ""
    back_error: str | None = None
    judge: JudgeScore
    timings: StageTimings = Field(default_factory=StageTimings)
    run_error: str | None = None


@runtime_checkable
class VoiceImpl(Protocol):
    """Contract every implementation adapter must satisfy.

    An impl is a *pipeline configuration*. It is batch-invoked: the runner
    filters out cached utterances and hands the remainder over in ONE call,
    because real impls are external processes with multi-GB model loads.
    """
    name: str

    def setup(self) -> None: ...

    def translate_batch(self, utts: list[Utterance],
                        out_dir: Path) -> list[VoiceResult]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/protocol.py tests/conftest.py tests/test_protocol.py
git commit -m "feat: protocol types and VoiceImpl contract"
```

---
### Task 3: FLEURS ingest + manifest loading (voice_suite/ingest_fleurs.py, voice_suite/dataset.py)

**Files:**
- Create: `voice_suite/ingest_fleurs.py`
- Create: `voice_suite/dataset.py`
- Test: `tests/test_ingest.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `Utterance` (Task 2).
- Produces:
  - `join_ids(vi_ids: list[int], en_ids: list[int]) -> list[tuple[int, int, int]]` — `(fleurs_id, vi_index, en_index)` triplets, id-sorted, first occurrence per id.
  - `sample_triplets(triplets, n: int, seed: int) -> list[tuple[int, int, int]]` — deterministic, id-sorted.
  - `utt_id(fleurs_id: int) -> str` — `"fleurs-000123"` format.
  - `ingest(n: int, seed: int, wav_dir: Path, manifest_path: Path) -> int` — downloads FLEURS, writes WAVs + manifest, returns row count (heavy deps imported lazily; NOT unit-tested).
  - `load_manifest(path: Path) -> list[Utterance]` — raises `FileNotFoundError` mentioning `ingest` when absent.

Design note: the join operates on id *columns* (plain ints) rather than full rows so (a) it is trivially testable and (b) `ingest` never decodes audio for rows outside the sample — HF `datasets` decodes FLEURS audio lazily on row access.

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest.py`:

```python
"""Join and sampling logic against fake FLEURS id columns (offline)."""
from __future__ import annotations

from voice_suite.ingest_fleurs import join_ids, sample_triplets, utt_id


def test_join_keeps_only_ids_on_both_sides():
    trips = join_ids([1, 2, 3], [2, 3, 4])
    assert [t[0] for t in trips] == [2, 3]


def test_join_dedupes_repeated_ids_keeping_first_index():
    # FLEURS repeats ids when several speakers read one sentence.
    trips = join_ids([5, 5, 6], [6, 5])
    assert trips == [(5, 0, 1), (6, 2, 0)]


def test_join_result_is_id_sorted():
    trips = join_ids([9, 1, 4], [4, 9, 1])
    assert [t[0] for t in trips] == [1, 4, 9]


def test_sample_is_deterministic_and_id_sorted():
    trips = join_ids(list(range(100)), list(range(100)))
    a = sample_triplets(trips, 10, seed=0)
    b = sample_triplets(trips, 10, seed=0)
    c = sample_triplets(trips, 10, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 10
    assert [t[0] for t in a] == sorted(t[0] for t in a)


def test_sample_larger_than_population_returns_all():
    trips = join_ids([1, 2], [1, 2])
    assert sample_triplets(trips, 10, seed=0) == trips


def test_utt_id_zero_padded():
    assert utt_id(7) == "fleurs-000007"
```

`tests/test_dataset.py`:

```python
"""Manifest loading: roundtrip, blank lines, missing file."""
from __future__ import annotations

import pytest

from voice_suite.dataset import load_manifest


def test_load_manifest_roundtrip(tmp_path, make_utt):
    utts = [make_utt(1), make_utt(2)]
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(u.model_dump_json() for u in utts) + "\n\n",
                 encoding="utf-8")
    assert load_manifest(p) == utts


def test_load_manifest_missing_file_mentions_ingest(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        load_manifest(tmp_path / "nope.jsonl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ingest.py tests/test_dataset.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.ingest_fleurs'`

- [ ] **Step 3: Write the implementations**

`voice_suite/ingest_fleurs.py`:

```python
"""FLEURS → ground-truth manifest + input WAVs.

FLEURS sentences are n-way parallel (FLoRes heritage): joining the vi_vn and
en_us configs on the FLEURS ``id`` field yields (vi audio, vi transcript,
en text) triplets. Some ids appear several times (multiple speakers reading
one sentence); we keep the first occurrence per id on both sides.
"""
from __future__ import annotations

import random
from pathlib import Path

from voice_suite.protocol import Utterance

FLEURS_DATASET = "google/fleurs"
SRC_CONFIG = "vi_vn"
DST_CONFIG = "en_us"
SPLIT = "test"


def _first_index_per_id(ids: list[int]) -> dict[int, int]:
    """Map each FLEURS id to the index of its first occurrence."""
    out: dict[int, int] = {}
    for i, fid in enumerate(ids):
        out.setdefault(int(fid), i)
    return out


def join_ids(vi_ids: list[int], en_ids: list[int]) -> list[tuple[int, int, int]]:
    """(fleurs_id, vi_index, en_index) for ids present on both sides, id-sorted."""
    vi_first = _first_index_per_id(vi_ids)
    en_first = _first_index_per_id(en_ids)
    return [(fid, vi_first[fid], en_first[fid])
            for fid in sorted(vi_first.keys() & en_first.keys())]


def sample_triplets(triplets: list[tuple[int, int, int]], n: int,
                    seed: int) -> list[tuple[int, int, int]]:
    """Deterministic sample of n triplets, id-sorted (tuples sort by id first)."""
    if n >= len(triplets):
        return list(triplets)
    rng = random.Random(seed)
    return sorted(rng.sample(triplets, n))


def utt_id(fleurs_id: int) -> str:
    return f"fleurs-{fleurs_id:06d}"


def ingest(n: int, seed: int, wav_dir: Path, manifest_path: Path) -> int:
    """Download FLEURS, join vi⋈en on id, sample, write WAVs + manifest.

    Returns the number of manifest rows written. Only sampled rows have
    their audio decoded. Requires the ``ingest`` extra (datasets/soundfile).
    """
    import soundfile as sf              # heavy: ingest extra
    from datasets import load_dataset   # heavy: ingest extra

    vi = load_dataset(FLEURS_DATASET, SRC_CONFIG, split=SPLIT,
                      trust_remote_code=True)
    en = load_dataset(FLEURS_DATASET, DST_CONFIG, split=SPLIT,
                      trust_remote_code=True)
    triplets = sample_triplets(join_ids(vi["id"], en["id"]), n, seed)
    en_texts = en["transcription"]

    wav_dir = Path(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as fh:
        for fid, vi_i, en_i in triplets:
            row = vi[vi_i]                       # decodes audio for this row only
            audio = row["audio"]
            wav_path = wav_dir / f"{utt_id(fid)}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"],
                     subtype="PCM_16")
            utt = Utterance(
                id=utt_id(fid),
                audio_path=str(wav_path),
                src_lang="vi",
                dst_lang="en",
                ref_transcript=row["transcription"],
                ref_translation=en_texts[en_i],
            )
            fh.write(utt.model_dump_json() + "\n")
    return len(triplets)
```

`voice_suite/dataset.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ingest.py tests/test_dataset.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/ingest_fleurs.py voice_suite/dataset.py tests/test_ingest.py tests/test_dataset.py
git commit -m "feat: FLEURS ingest and manifest loading"
```

---
### Task 4: Paths + impl auto-discovery (voice_suite/config.py)

**Files:**
- Create: `voice_suite/config.py` (adapted from `rag_suite/config.py`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `VoiceImpl` (Task 2).
- Produces:
  - Path constants (all relative to CWD): `DATA_DIR`, `UTTERANCES_DIR = data/utterances`, `MANIFEST = data/manifest.jsonl`, `IMPLS_DIR = impls`, `OUT_DIR = out`, `RAW_RESULTS = out/raw_results.jsonl`, `AUDIO_OUT_DIR = out/audio`, `SCORED = out/scored.jsonl`, `BT_CACHE = out/backtranscripts.jsonl`, `REPORT = out/report.md`.
  - `discover_impls(impls_dir: Path) -> dict[str, VoiceImpl]` — scans `impls/*/__init__.py` for a module-level `IMPL`, skips import failures with a stderr note.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
"""Impl discovery against throwaway impl packages in tmp dirs."""
from __future__ import annotations

from voice_suite.config import discover_impls

GOOD_IMPL = '''
class _Impl:
    name = "dummy"
    def setup(self): ...
    def translate_batch(self, utts, out_dir): return []
IMPL = _Impl()
'''


def test_discovers_impl_exposing_IMPL(tmp_path):
    pkg = tmp_path / "impls" / "dummy"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(GOOD_IMPL, encoding="utf-8")
    found = discover_impls(tmp_path / "impls")
    assert list(found) == ["dummy"]
    assert found["dummy"].name == "dummy"


def test_skips_broken_import_and_missing_IMPL(tmp_path, capsys):
    base = tmp_path / "impls"
    (base / "broken").mkdir(parents=True)
    (base / "broken" / "__init__.py").write_text(
        "import not_a_real_module_xyz", encoding="utf-8")
    (base / "no_impl").mkdir()
    (base / "no_impl" / "__init__.py").write_text("X = 1", encoding="utf-8")
    (base / "good").mkdir()
    (base / "good" / "__init__.py").write_text(GOOD_IMPL, encoding="utf-8")
    found = discover_impls(base)
    assert list(found) == ["dummy"]
    assert "skipping impl 'broken'" in capsys.readouterr().err


def test_missing_impls_dir_returns_empty(tmp_path):
    assert discover_impls(tmp_path / "absent") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.config'`

- [ ] **Step 3: Write the implementation**

`voice_suite/config.py`:

```python
"""Paths and impl auto-discovery."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from voice_suite.protocol import VoiceImpl

# Default project paths (relative to CWD).
DATA_DIR = Path("data")
UTTERANCES_DIR = DATA_DIR / "utterances"
MANIFEST = DATA_DIR / "manifest.jsonl"
IMPLS_DIR = Path("impls")
OUT_DIR = Path("out")
RAW_RESULTS = OUT_DIR / "raw_results.jsonl"
AUDIO_OUT_DIR = OUT_DIR / "audio"
SCORED = OUT_DIR / "scored.jsonl"
BT_CACHE = OUT_DIR / "backtranscripts.jsonl"
REPORT = OUT_DIR / "report.md"


def _load_package_init(name: str, init_path: Path):
    """Load a package's __init__.py by file path and return the module."""
    spec = importlib.util.spec_from_file_location(name, init_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_impls(impls_dir: Path) -> dict[str, VoiceImpl]:
    """Find subpackages exposing an ``IMPL`` VoiceImpl instance.

    Scans each direct subdirectory of *impls_dir* for an ``__init__.py``
    that exposes a module-level ``IMPL`` attribute. Import-time failures
    (e.g. missing optional deps) skip that impl with a stderr note so one
    broken impl never hides the others.
    """
    found: dict[str, VoiceImpl] = {}
    base = Path(impls_dir)
    if not base.exists():
        return found
    # Impls use absolute imports (``from impls.m2v_default.adapter import …``),
    # so the impls dir's parent must be importable regardless of how the CLI
    # was launched (the console script has only the venv bin on sys.path).
    root = str(base.resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    for child in sorted(base.iterdir()):
        init_path = child / "__init__.py"
        if not child.is_dir() or not init_path.exists():
            continue
        try:
            module = _load_package_init(f"_impl_{child.name}", init_path)
        except ImportError as exc:
            print(f"discover_impls: skipping impl {child.name!r} "
                  f"(import failed: {exc})", file=sys.stderr)
            continue
        impl = getattr(module, "IMPL", None)
        if impl is not None:
            found[impl.name] = impl
    return found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/config.py tests/test_config.py
git commit -m "feat: path config and impl discovery"
```

---
### Task 5: Runner — batch contract + per-utterance run cache (voice_suite/runner.py)

**Files:**
- Create: `voice_suite/runner.py` (adapted from `rag_suite/runner.py` — key change: ONE `translate_batch` call over uncached utterances instead of a per-item loop)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RawRecord`, `Utterance`, `VoiceImpl`, `VoiceResult` (Task 2).
- Produces:
  - `load_raw_records(path: Path) -> list[RawRecord]` — empty list if the file is absent.
  - `run_impl(impl: VoiceImpl, utts: list[Utterance], out_path: Path, audio_out_dir: Path) -> int` — appends `RawRecord`s, returns count of newly recorded utterances; output audio goes under `audio_out_dir/<impl.name>/`. Batch-level failures propagate to the caller (CLI isolates per impl); per-utterance failures arrive as `VoiceResult.error` and are recorded like successes. Fully cached run returns 0 without calling `setup()`.

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:

```python
"""Runner: single-batch dispatch, idempotent cache, count-mismatch guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from voice_suite.protocol import VoiceResult
from voice_suite.runner import load_raw_records, run_impl


class SpyImpl:
    """Records every translate_batch call; echoes one result per utterance."""

    name = "spy"

    def __init__(self, short: bool = False):
        self.calls: list[list[str]] = []
        self.setup_calls = 0
        self.short = short

    def setup(self) -> None:
        self.setup_calls += 1

    def translate_batch(self, utts, out_dir: Path):
        self.calls.append([u.id for u in utts])
        results = [VoiceResult(asr_text=f"asr {u.id}") for u in utts]
        return results[:-1] if self.short else results


def test_run_records_all_and_skips_cached(tmp_path, make_utt):
    utts = [make_utt(1), make_utt(2), make_utt(3)]
    raw = tmp_path / "raw.jsonl"
    impl = SpyImpl()

    assert run_impl(impl, utts[:2], raw, tmp_path / "audio") == 2
    assert impl.calls == [["u001", "u002"]]

    # Second run over the full set: only the new utterance reaches the impl.
    assert run_impl(impl, utts, raw, tmp_path / "audio") == 1
    assert impl.calls[-1] == ["u003"]
    recs = load_raw_records(raw)
    assert [r.utt_id for r in recs] == ["u001", "u002", "u003"]
    assert recs[0].result.asr_text == "asr u001"
    assert recs[0].impl == "spy"


def test_fully_cached_run_skips_setup(tmp_path, make_utt):
    utts = [make_utt(1)]
    raw = tmp_path / "raw.jsonl"
    impl = SpyImpl()
    run_impl(impl, utts, raw, tmp_path / "audio")
    assert run_impl(impl, utts, raw, tmp_path / "audio") == 0
    assert impl.setup_calls == 1


def test_audio_out_dir_is_per_impl(tmp_path, make_utt):
    run_impl(SpyImpl(), [make_utt(1)], tmp_path / "raw.jsonl", tmp_path / "audio")
    assert (tmp_path / "audio" / "spy").is_dir()


def test_result_count_mismatch_raises(tmp_path, make_utt):
    with pytest.raises(RuntimeError, match="returned 1 results for 2"):
        run_impl(SpyImpl(short=True), [make_utt(1), make_utt(2)],
                 tmp_path / "raw.jsonl", tmp_path / "audio")


def test_load_raw_records_missing_file(tmp_path):
    assert load_raw_records(tmp_path / "none.jsonl") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.runner'`

- [ ] **Step 3: Write the implementation**

`voice_suite/runner.py`:

```python
"""Orchestrate impl × utterance runs with an idempotent JSONL cache.

Key difference from a per-item runner: the impl is an external process with
a multi-GB model load, so uncached utterances are handed over in ONE
``translate_batch`` call per impl.
"""
from __future__ import annotations

from pathlib import Path

from voice_suite.protocol import RawRecord, Utterance, VoiceImpl


def load_raw_records(path: Path) -> list[RawRecord]:
    """Load all RawRecords from a JSONL file; returns empty list if absent."""
    p = Path(path)
    if not p.exists():
        return []
    return [
        RawRecord.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_impl(impl: VoiceImpl, utts: list[Utterance], out_path: Path,
             audio_out_dir: Path) -> int:
    """Run `impl` over uncached utterances; append RawRecords to out_path.

    Already-recorded (impl.name, utt_id) pairs are skipped (idempotent).
    Returns the number of newly recorded utterances. Batch-level failures
    (setup, non-zero exit) propagate to the caller; per-utterance failures
    arrive as VoiceResult.error and are recorded like successes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r.impl, r.utt_id) for r in load_raw_records(out_path)}
    todo = [u for u in utts if (impl.name, u.id) not in done]
    if not todo:
        return 0

    impl.setup()
    out_dir = Path(audio_out_dir) / impl.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = impl.translate_batch(todo, out_dir)
    if len(results) != len(todo):
        raise RuntimeError(
            f"{impl.name}.translate_batch returned {len(results)} results "
            f"for {len(todo)} utterances")

    with out_path.open("a", encoding="utf-8") as fh:
        for utt, result in zip(todo, results):
            rec = RawRecord(impl=impl.name, utt_id=utt.id, result=result)
            fh.write(rec.model_dump_json() + "\n")
    return len(todo)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/runner.py tests/test_runner.py
git commit -m "feat: batch runner with idempotent run cache"
```

---
### Task 6: Deterministic sampling — --limit/--seed (voice_suite/sampling.py)

**Files:**
- Create: `voice_suite/sampling.py` (near-verbatim from `rag_suite/sampling.py`; only the item type changes)
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Utterance` (Task 2).
- Produces: `sample_items(items: list[Utterance], limit: int | None, seed: int = 0) -> list[Utterance]` — id-sorted, reproducible; `run` and `score` call it with matching flags to select the same subset independently.

- [ ] **Step 1: Write the failing test**

`tests/test_sampling.py`:

```python
"""Deterministic sampling invariants."""
from __future__ import annotations

from voice_suite.sampling import sample_items


def test_no_limit_returns_all_id_sorted(make_utt):
    items = [make_utt(3), make_utt(1), make_utt(2)]
    assert [u.id for u in sample_items(items, None)] == ["u001", "u002", "u003"]


def test_same_seed_same_subset(make_utt):
    items = [make_utt(i) for i in range(1, 51)]
    a = sample_items(items, 10, seed=7)
    b = sample_items(items, 10, seed=7)
    assert a == b
    assert len(a) == 10


def test_different_seed_different_subset(make_utt):
    items = [make_utt(i) for i in range(1, 51)]
    assert sample_items(items, 10, seed=0) != sample_items(items, 10, seed=1)


def test_limit_zero_or_negative_empty(make_utt):
    items = [make_utt(1)]
    assert sample_items(items, 0) == []
    assert sample_items(items, -5) == []


def test_limit_larger_than_population(make_utt):
    items = [make_utt(2), make_utt(1)]
    assert [u.id for u in sample_items(items, 99)] == ["u001", "u002"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.sampling'`

- [ ] **Step 3: Write the implementation**

`voice_suite/sampling.py`:

```python
"""Deterministic utterance subset selection for cost-bounded eval runs."""
from __future__ import annotations

import random

from voice_suite.protocol import Utterance


def sample_items(items: list[Utterance], limit: int | None,
                 seed: int = 0) -> list[Utterance]:
    """Return up to *limit* items, chosen reproducibly by *seed*.

    The result is always sorted by ``id`` for stable downstream ordering.
    Same ``(items, limit, seed)`` → same subset, so ``run`` and ``score``
    can independently select the *same* utterances and stay in sync.

    - ``limit`` is ``None`` or ``>= len(items)`` → all items (id-sorted).
    - ``limit <= 0`` → empty list.
    """
    ordered = sorted(items, key=lambda it: it.id)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit <= 0:
        return []
    rng = random.Random(seed)
    picked = rng.sample(ordered, limit)
    return sorted(picked, key=lambda it: it.id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/sampling.py tests/test_sampling.py
git commit -m "feat: deterministic sampling for limit/seed"
```

---
### Task 7: WER scorer — Vietnamese-safe text normalization (voice_suite/scoring/wer.py)

**Files:**
- Create: `voice_suite/scoring/__init__.py` (docstring-only placeholder; Task 10 replaces it with orchestration)
- Create: `voice_suite/scoring/wer.py`
- Test: `tests/test_wer.py`

**Interfaces:**
- Consumes: `jiwer` (core dependency, Task 1).
- Produces:
  - `norm_text(s: str) -> str` — lowercase, punctuation → space, whitespace collapsed; Vietnamese diacritics preserved (`\w` is unicode-aware).
  - `score_wer(ref: str, hyp: str) -> float` — WER on normalized text, capped at 1.0; empty ref → 0.0 iff hyp is also empty, else 1.0.

- [ ] **Step 1: Write the failing test**

`tests/test_wer.py`:

```python
"""WER normalization and scoring: diacritics, folding, edge cases."""
from __future__ import annotations

import pytest

from voice_suite.scoring.wer import norm_text, score_wer


def test_norm_preserves_vietnamese_diacritics():
    assert norm_text("Đây là tiếng Việt.") == "đây là tiếng việt"


def test_norm_folds_case_punct_whitespace():
    assert norm_text("  Xin CHÀO,   bạn!  ") == "xin chào bạn"


def test_identical_after_normalization_is_zero():
    assert score_wer("Xin chào, bạn!", "xin chào bạn") == 0.0


def test_missing_diacritics_count_as_substitution():
    # "chao" ≠ "chào": one substitution out of two reference words.
    assert score_wer("xin chào", "xin chao") == pytest.approx(0.5)


def test_empty_hypothesis_is_one():
    assert score_wer("một hai ba", "") == 1.0


def test_empty_reference():
    assert score_wer("", "") == 0.0
    assert score_wer("", "gì đó") == 1.0


def test_wer_capped_at_one():
    assert score_wer("một", "a b c d e f") == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wer.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.scoring'`

- [ ] **Step 3: Write the implementation**

`voice_suite/scoring/__init__.py` (placeholder until Task 10):

```python
"""Scoring package: WER, LLM judge, back-transcription."""
```

`voice_suite/scoring/wer.py`:

```python
"""ASR word-error-rate on normalized text (Vietnamese-safe)."""
from __future__ import annotations

import re

import jiwer

# Everything that is not a word character or whitespace counts as
# punctuation. ``\w`` is unicode-aware in Python 3: Vietnamese letters with
# diacritics (à, đ, ệ, …) are preserved; casing and punctuation are folded.
_PUNCT_RE = re.compile(r"[^\w\s]")


def norm_text(s: str) -> str:
    """Lowercase, punctuation → space, collapse whitespace."""
    s = _PUNCT_RE.sub(" ", s.lower())
    return " ".join(s.split())


def score_wer(ref: str, hyp: str) -> float:
    """WER of hyp against ref after normalization, capped at 1.0.

    The cap keeps one long hallucination from dominating a mean (insertions
    can push raw WER above 1.0). Empty ref: 0.0 if hyp is also empty, else
    1.0 (jiwer cannot divide by zero reference words).
    """
    ref_n, hyp_n = norm_text(ref), norm_text(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return min(jiwer.wer(ref_n, hyp_n), 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wer.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/scoring/__init__.py voice_suite/scoring/wer.py tests/test_wer.py
git commit -m "feat: WER scorer with unicode-safe normalization"
```

---
### Task 8: LLM judge — translation rubric + cli/api backends (voice_suite/scoring/judge.py)

**Files:**
- Create: `voice_suite/scoring/judge.py` (plumbing — `_anthropic_judge`, `_cli_judge`, `_loads_lenient`, `get_judge` — copied from `rag_suite/scoring/judge.py`; rubric and prompt builder are new)
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `JudgeScore`, `Utterance`, `VoiceResult` (Task 2).
- Produces:
  - `JUDGE_VERSION = "v1"`, `DEFAULT_JUDGE_MODEL = "claude-opus-4-8"`, `JudgeFn = Callable[[str], dict]`.
  - `build_judge_prompt(utt: Utterance, result: VoiceResult, back_transcript: str) -> str` — bias-blind (no impl identity); empty back_transcript renders as `(no output audio)`.
  - `get_judge(backend: str = "cli", model: str = DEFAULT_JUDGE_MODEL) -> JudgeFn` — `'cli'` (Claude Code) or `'api'` (Anthropic SDK); anything else raises `ValueError`.
  - `score_utterance(utt, result, back_transcript, call_judge: JudgeFn | None = None) -> JudgeScore` — judge exceptions/malformed output captured into `JudgeScore.error`, never raised.
  - `_loads_lenient(text: str) -> dict` — first JSON object in model output, tolerating fences/prose.

- [ ] **Step 1: Write the failing test**

`tests/test_judge.py`:

```python
"""Judge plumbing: prompt content, score mapping, lenient JSON, backends."""
from __future__ import annotations

import pytest

from voice_suite.scoring.judge import (
    _loads_lenient, build_judge_prompt, get_judge, score_utterance,
)

GOOD = {"mt_adequacy": 0.9, "mt_fluency": 0.8, "e2e_adequacy": 0.7,
        "verdict": "good", "rationale": "close match"}


def test_prompt_contains_all_five_blocks_and_no_impl_name(make_utt, make_result):
    utt = make_utt(ref_transcript="nguồn tiếng việt", ref_translation="english ref")
    res = make_result(asr_text="asr nghe được", mt_text="mt english out")
    prompt = build_judge_prompt(utt, res, "back transcript here")
    for needle in ("nguồn tiếng việt", "english ref", "asr nghe được",
                   "mt english out", "back transcript here"):
        assert needle in prompt
    assert "m2v" not in prompt  # bias-blind: no impl identity


def test_prompt_marks_missing_output_audio(make_utt, make_result):
    prompt = build_judge_prompt(make_utt(), make_result(), "")
    assert "(no output audio)" in prompt


def test_score_utterance_maps_fields(make_utt, make_result):
    score = score_utterance(make_utt(), make_result(), "bt",
                            call_judge=lambda prompt: dict(GOOD))
    assert score.mt_adequacy == 0.9
    assert score.mt_fluency == 0.8
    assert score.e2e_adequacy == 0.7
    assert score.verdict == "good"
    assert score.error is None


def test_score_utterance_captures_judge_exception(make_utt, make_result):
    def boom(prompt):
        raise RuntimeError("api down")
    score = score_utterance(make_utt(), make_result(), "bt", call_judge=boom)
    assert score.error == "api down"
    assert score.verdict == "error"
    assert score.mt_adequacy == 0.0


def test_score_utterance_missing_key_is_error(make_utt, make_result):
    score = score_utterance(make_utt(), make_result(), "bt",
                            call_judge=lambda p: {"mt_adequacy": 1.0})
    assert score.error is not None


def test_loads_lenient_plain_and_fenced():
    assert _loads_lenient('{"a": 1}') == {"a": 1}
    assert _loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_loads_lenient_prose_and_trailing_brace():
    assert _loads_lenient('score: {"a": 1} done {oops') == {"a": 1}


def test_loads_lenient_no_json_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        _loads_lenient("nothing here")


def test_get_judge_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown judge backend"):
        get_judge("smoke-signals")


def test_cli_judge_missing_binary_raises(monkeypatch):
    from voice_suite.scoring import judge as judge_mod
    monkeypatch.setattr(judge_mod.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="claude CLI not found"):
        judge_mod._cli_judge("prompt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_judge.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.scoring.judge'`

- [ ] **Step 3: Write the implementation**

`voice_suite/scoring/judge.py`:

```python
"""Claude LLM-judge for translation quality (one call per utterance)."""
from __future__ import annotations

import json
import shutil
import subprocess
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


def _anthropic_judge(prompt: str, model: str = DEFAULT_JUDGE_MODEL) -> dict:
    """Call the Anthropic API and return parsed JSON sub-scores."""
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=1024, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
    return json.loads(text)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_judge.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/scoring/judge.py tests/test_judge.py
git commit -m "feat: Claude judge with translation rubric"
```

---
### Task 9: Back-transcription scorer + audio content-hash cache (voice_suite/scoring/backtranscribe.py)

**Files:**
- Create: `voice_suite/scoring/backtranscribe.py`
- Test: `tests/test_backtranscribe.py`

**Interfaces:**
- Consumes: nothing project-internal; `faster-whisper` lazily (only inside `get_transcriber`).
- Produces:
  - `BT_MODEL = "large-v3"`, `TranscribeFn = Callable[[Path], str]`.
  - `audio_sha256(path: Path) -> str` — streamed content hash.
  - `load_bt_cache(path: Path) -> dict[str, str]` — `{sha256: transcript}`; last write wins.
  - `append_bt_cache(path: Path, sha: str, transcript: str) -> None`.
  - `get_transcriber(model_size: str = BT_MODEL, lang: str = "en") -> TranscribeFn` — loads faster-whisper on CPU/int8; raises `RuntimeError` naming the `bt` extra if faster-whisper is missing. First call downloads ~3 GB from HuggingFace (cached).

- [ ] **Step 1: Write the failing test**

`tests/test_backtranscribe.py`:

```python
"""Content hashing and the back-transcription cache (no models involved)."""
from __future__ import annotations

from voice_suite.scoring.backtranscribe import (
    append_bt_cache, audio_sha256, load_bt_cache,
)


def test_sha_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"RIFF" + b"\x01" * 64)
    b.write_bytes(b"RIFF" + b"\x02" * 64)
    assert audio_sha256(a) == audio_sha256(a)
    assert audio_sha256(a) != audio_sha256(b)


def test_cache_roundtrip_and_last_write_wins(tmp_path):
    cache_file = tmp_path / "bt.jsonl"
    assert load_bt_cache(cache_file) == {}
    append_bt_cache(cache_file, "sha1", "hello there")
    append_bt_cache(cache_file, "sha2", "second")
    append_bt_cache(cache_file, "sha1", "revised")
    cache = load_bt_cache(cache_file)
    assert cache == {"sha1": "revised", "sha2": "second"}


def test_append_creates_parent_dirs(tmp_path):
    cache_file = tmp_path / "deep" / "nested" / "bt.jsonl"
    append_bt_cache(cache_file, "sha1", "text")
    assert load_bt_cache(cache_file) == {"sha1": "text"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtranscribe.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.scoring.backtranscribe'`

- [ ] **Step 3: Write the implementation**

`voice_suite/scoring/backtranscribe.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backtranscribe.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/scoring/backtranscribe.py tests/test_backtranscribe.py
git commit -m "feat: back-transcription scorer with content-hash cache"
```

---
### Task 10: Score orchestration + score cache + worker pool (voice_suite/scoring/__init__.py)

**Files:**
- Modify: `voice_suite/scoring/__init__.py` (replace the Task 7 placeholder with orchestration; adapted from `rag_suite/scoring/__init__.py`)
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: Tasks 2, 7, 8, 9 — `score_wer`, `score_utterance`, `JUDGE_VERSION`, `JudgeFn`, `TranscribeFn`, `audio_sha256`, `load_bt_cache`, `append_bt_cache`, protocol types.
- Produces:
  - `score_key(impl: str, utt_id: str, result: VoiceResult, audio_sha: str, judge_id: str = "") -> str` — sha256 over `(impl, utt_id, asr_text, mt_text, audio_sha, JUDGE_VERSION, judge_id)` joined with `\x1f`.
  - `load_scored_records(path: Path) -> list[ScoredRecord]`.
  - `score_all(raw, utts, out_path, call_judge=None, workers=1, judge_id="", transcribe=None, bt_cache_path=None) -> None` — appends only not-yet-scored records.
- Error semantics (from the spec's error-handling table):
  - `result.error` set (pipeline failure) → no judge call; `JudgeScore(verdict="pipeline-error")` — zeros that COUNT in judge means; WER runs against the (empty) asr_text.
  - `call_judge is None` (--no-judge) → `JudgeScore(verdict="skipped")`.
  - Back-transcription failure → `back_error` recorded, `e2e_adequacy` forced to 0.0, `mt_*` judge scores stand.
  - Judge call failure → `judge.error` set (excluded from means at aggregate time); scoring continues.
- Phasing: back-transcription runs SEQUENTIALLY first (one local model instance, content-cached), then judge calls run in the worker pool with lock-guarded JSONL writes.

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py`:

```python
"""Score orchestration: cache keys, idempotence, phases, error semantics."""
from __future__ import annotations

from pathlib import Path

from voice_suite.protocol import RawRecord
from voice_suite.scoring import load_scored_records, score_all, score_key

GOOD = {"mt_adequacy": 0.9, "mt_fluency": 0.8, "e2e_adequacy": 0.7,
        "verdict": "good", "rationale": "r"}


def _judge_ok(prompt: str) -> dict:
    return dict(GOOD)


def test_score_key_sensitive_to_content_and_judge(make_result):
    r1 = make_result()
    r2 = make_result(asr_text="khác hẳn")
    base = score_key("a", "u001", r1, "sha", "cli:m")
    assert base == score_key("a", "u001", r1, "sha", "cli:m")
    assert base != score_key("a", "u001", r2, "sha", "cli:m")
    assert base != score_key("a", "u001", r1, "sha2", "cli:m")
    assert base != score_key("a", "u001", r1, "sha", "api:m")
    assert base != score_key("b", "u001", r1, "sha", "cli:m")


def test_score_all_writes_and_is_idempotent(tmp_path, make_utt, make_result):
    utts = [make_utt(1), make_utt(2)]
    raw = [RawRecord(impl="a", utt_id=u.id,
                     result=make_result(asr_text=u.ref_transcript,
                                        mt_text=u.ref_translation))
           for u in utts]
    scored = tmp_path / "scored.jsonl"
    calls = []

    def judge(prompt):
        calls.append(prompt)
        return dict(GOOD)

    score_all(raw, utts, scored, call_judge=judge, judge_id="t:m")
    score_all(raw, utts, scored, call_judge=judge, judge_id="t:m")
    recs = load_scored_records(scored)
    assert len(recs) == 2
    assert len(calls) == 2  # second call fully cache-hit
    rec1 = next(r for r in recs if r.utt_id == "u001")
    assert rec1.judge.mt_adequacy == 0.9
    assert rec1.wer == 0.0
    assert rec1.asr_text == "xin chào số 1"
    assert rec1.cache_key


def test_unknown_utt_id_skipped(tmp_path, make_utt, make_result):
    raw = [RawRecord(impl="a", utt_id="ghost", result=make_result())]
    score_all(raw, [make_utt(1)], tmp_path / "s.jsonl",
              call_judge=_judge_ok, judge_id="t:m")
    assert load_scored_records(tmp_path / "s.jsonl") == []


def test_pipeline_error_scores_zero_without_judge_call(tmp_path, make_utt, make_result):
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(asr_text="", mt_text="",
                                        error="asr crashed"))]
    called = []
    score_all(raw, [utt], tmp_path / "s.jsonl",
              call_judge=lambda p: called.append(p) or dict(GOOD),
              judge_id="t:m")
    rec = load_scored_records(tmp_path / "s.jsonl")[0]
    assert called == []
    assert rec.judge.verdict == "pipeline-error"
    assert rec.judge.e2e_adequacy == 0.0
    assert rec.wer == 1.0
    assert rec.run_error == "asr crashed"


def test_no_judge_path_marks_skipped(tmp_path, make_utt, make_result):
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(asr_text=utt.ref_transcript))]
    score_all(raw, [utt], tmp_path / "s.jsonl", call_judge=None,
              judge_id="no-judge")
    rec = load_scored_records(tmp_path / "s.jsonl")[0]
    assert rec.judge.verdict == "skipped"
    assert rec.wer == 0.0


def test_back_transcription_cached_by_content(tmp_path, make_utt, make_result):
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF-fake-audio")
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(audio_path=str(wav)))]
    bt_calls = []

    def transcribe(path: Path) -> str:
        bt_calls.append(path)
        return "back text"

    cache_file = tmp_path / "bt.jsonl"
    score_all(raw, [utt], tmp_path / "s1.jsonl", call_judge=_judge_ok,
              judge_id="t:m", transcribe=transcribe, bt_cache_path=cache_file)
    # New judge_id → new score cache key, but bt cache hits (same audio bytes).
    score_all(raw, [utt], tmp_path / "s2.jsonl", call_judge=_judge_ok,
              judge_id="t:m2", transcribe=transcribe, bt_cache_path=cache_file)
    assert len(bt_calls) == 1
    rec = load_scored_records(tmp_path / "s2.jsonl")[0]
    assert rec.back_transcript == "back text"
    assert rec.judge.e2e_adequacy == 0.7


def test_bt_failure_forces_e2e_zero_keeps_mt(tmp_path, make_utt, make_result):
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF-fake-audio")
    utt = make_utt(1)
    raw = [RawRecord(impl="a", utt_id=utt.id,
                     result=make_result(audio_path=str(wav)))]

    def transcribe(path: Path) -> str:
        raise RuntimeError("decode failed")

    score_all(raw, [utt], tmp_path / "s.jsonl", call_judge=_judge_ok,
              judge_id="t:m", transcribe=transcribe,
              bt_cache_path=tmp_path / "bt.jsonl")
    rec = load_scored_records(tmp_path / "s.jsonl")[0]
    assert rec.back_error == "decode failed"
    assert rec.judge.e2e_adequacy == 0.0
    assert rec.judge.mt_adequacy == 0.9


def test_parallel_workers_write_all_records(tmp_path, make_utt, make_result):
    utts = [make_utt(i) for i in range(1, 9)]
    raw = [RawRecord(impl="a", utt_id=u.id, result=make_result(i))
           for i, u in enumerate(utts, 1)]
    scored = tmp_path / "s.jsonl"
    score_all(raw, utts, scored, call_judge=_judge_ok, judge_id="t:m",
              workers=4)
    recs = load_scored_records(scored)
    assert len(recs) == 8
    assert {r.utt_id for r in recs} == {u.id for u in utts}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_all' from 'voice_suite.scoring'`

- [ ] **Step 3: Write the implementation**

Replace `voice_suite/scoring/__init__.py` entirely with:

```python
"""Scoring orchestration: WER + back-transcription + judge → ScoredRecords.

Two phases per score run:
1. Back-transcription, sequential (one local faster-whisper instance; the
   content-hash cache makes re-runs cheap).
2. Judge calls, optionally parallel (network-bound), with lock-guarded
   JSONL writes.
"""
from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from voice_suite.protocol import (
    JudgeScore, RawRecord, ScoredRecord, Utterance, VoiceResult,
)
from voice_suite.scoring.backtranscribe import (
    TranscribeFn, append_bt_cache, audio_sha256, load_bt_cache,
)
from voice_suite.scoring.judge import JUDGE_VERSION, JudgeFn, score_utterance
from voice_suite.scoring.wer import score_wer

_SEQUENTIAL = 1


def score_key(impl: str, utt_id: str, result: VoiceResult, audio_sha: str,
              judge_id: str = "") -> str:
    """Stable cache key over impl, utterance, output content, judge identity.

    Changing the judge backend/model (judge_id), the rubric (JUDGE_VERSION),
    or any output (texts, audio bytes) re-scores only what changed.
    """
    payload = "\x1f".join([impl, utt_id, result.asr_text, result.mt_text,
                           audio_sha, JUDGE_VERSION, judge_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_scored_records(path: Path) -> list[ScoredRecord]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        ScoredRecord.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _back_transcripts(
    pending: dict[str, tuple[RawRecord, Utterance, str]],
    transcribe: TranscribeFn | None,
    bt_cache_path: Path | None,
) -> dict[str, tuple[str, str | None]]:
    """Phase 1: cache_key → (back_transcript, back_error). Sequential."""
    cache = load_bt_cache(bt_cache_path) if bt_cache_path else {}
    out: dict[str, tuple[str, str | None]] = {}
    for key, (rec, _utt, sha) in pending.items():
        if transcribe is None:
            out[key] = ("", "back-transcription skipped")
        elif not sha:
            out[key] = ("", rec.result.error or "no output audio")
        elif sha in cache:
            out[key] = (cache[sha], None)
        else:
            try:
                text = transcribe(Path(rec.result.audio_path))
                cache[sha] = text
                if bt_cache_path:
                    append_bt_cache(bt_cache_path, sha, text)
                out[key] = (text, None)
            except Exception as exc:
                out[key] = ("", str(exc))
    return out


def _score_one(key: str, rec: RawRecord, utt: Utterance, back_transcript: str,
               back_error: str | None,
               call_judge: JudgeFn | None) -> ScoredRecord:
    """WER + judge for one record.

    Pipeline failures score deterministic zeros WITHOUT a judge call and
    stay in the aggregate means (a config that crashes must rank worse).
    """
    wer = score_wer(utt.ref_transcript, rec.result.asr_text)
    if rec.result.error is not None:
        judge = JudgeScore(verdict="pipeline-error")
    elif call_judge is None:
        judge = JudgeScore(verdict="skipped")
    else:
        judge = score_utterance(utt, rec.result, back_transcript,
                                call_judge=call_judge)
        if back_error is not None and judge.error is None:
            # Output audio unusable: e2e is deterministically 0; MT scores stand.
            judge = judge.model_copy(update={"e2e_adequacy": 0.0})
    return ScoredRecord(
        impl=rec.impl,
        utt_id=rec.utt_id,
        cache_key=key,
        wer=wer,
        asr_text=rec.result.asr_text,
        mt_text=rec.result.mt_text,
        back_transcript=back_transcript,
        back_error=back_error,
        judge=judge,
        timings=rec.result.timings,
        run_error=rec.result.error,
    )


def score_all(
    raw: list[RawRecord],
    utts: list[Utterance],
    out_path: Path,
    call_judge: JudgeFn | None = None,
    workers: int = 1,
    judge_id: str = "",
    transcribe: TranscribeFn | None = None,
    bt_cache_path: Path | None = None,
) -> None:
    """Score every raw record not already scored; append to out_path.

    Parameters
    ----------
    raw:           Raw pipeline outputs to score.
    utts:          Ground-truth utterances (the sampled subset to score).
    out_path:      JSONL file to append ScoredRecords to.
    call_judge:    LLM judge callable; None = --no-judge (skipped verdicts).
    workers:       Parallel judge workers (>1 uses a ThreadPoolExecutor with
                   a lock around file writes).
    judge_id:      "<backend>:<model>" or "no-judge"; part of the cache key.
    transcribe:    Back-transcription callable; None skips phase 1.
    bt_cache_path: JSONL content-hash cache for back-transcripts.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_id = {u.id: u for u in utts}
    done = {s.cache_key for s in load_scored_records(out_path) if s.cache_key}

    pending: dict[str, tuple[RawRecord, Utterance, str]] = {}
    for rec in raw:
        utt = by_id.get(rec.utt_id)
        if utt is None:
            continue  # record for an utterance outside this sample
        audio = rec.result.audio_path
        sha = audio_sha256(Path(audio)) if audio and Path(audio).exists() else ""
        key = score_key(rec.impl, rec.utt_id, rec.result, sha, judge_id)
        if key in done or key in pending:
            continue
        pending[key] = (rec, utt, sha)

    if not pending:
        return

    bts = _back_transcripts(pending, transcribe, bt_cache_path)

    lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as fh:

        def _write(scored: ScoredRecord) -> None:
            with lock:
                fh.write(scored.model_dump_json() + "\n")
                fh.flush()

        if workers <= _SEQUENTIAL:
            for key, (rec, utt, _sha) in pending.items():
                bt, bt_err = bts[key]
                _write(_score_one(key, rec, utt, bt, bt_err, call_judge))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_score_one, key, rec, utt,
                                    bts[key][0], bts[key][1], call_judge)
                    for key, (rec, utt, _sha) in pending.items()
                ]
                for fut in as_completed(futures):
                    _write(fut.result())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite (regression check)**

Run: `python -m pytest -q`
Expected: all tests pass (protocol, ingest, dataset, config, runner, sampling, wer, judge, backtranscribe, scoring)

- [ ] **Step 6: Commit**

```bash
git add voice_suite/scoring/__init__.py tests/test_scoring.py
git commit -m "feat: score orchestration with content-keyed cache"
```

---
### Task 11: Aggregate + leaderboard report with worst-10 appendix (voice_suite/aggregate.py)

**Files:**
- Create: `voice_suite/aggregate.py` (adapted from `rag_suite/aggregate.py`: voice metrics, latency percentiles, worst-k appendix)
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `ScoredRecord`, `Utterance`, `JudgeScore`, `StageTimings` (Task 2).
- Produces:
  - `aggregate(records: list[ScoredRecord]) -> list[dict]` — one row per impl with keys `impl, scored, errors, judge_errors, mean_wer, mt_adequacy, mt_fluency, e2e_adequacy, asr_ms_p50, asr_ms_p95, mt_ms_p50, mt_ms_p95, tts_ms_p50, tts_ms_p95`; sorted best-first by `e2e_adequacy` desc then `mean_wer` asc (the tiebreak also ranks `--no-judge` runs, where every e2e is 0).
  - `worst_utterances(records, utts_by_id: dict[str, Utterance], k: int = 10) -> list[dict]` — lowest e2e (then highest WER) with all texts for failure eyeballing.
  - `render_report(rows: list[dict], worst: list[dict]) -> str` — markdown per the spec's table:
    `| impl | n | WER↓ | mt_adequacy↑ | mt_fluency↑ | e2e_adequacy↑ | asr_ms P50/P95 | mt_ms P50/P95 | tts_ms P50/P95 | errors |`
- Semantics: judge means average records with `judge.error is None` (pipeline-error zeros INCLUDED — crashing configs rank worse); `errors` counts `run_error`; latency percentiles cover records that actually ran (`run_error is None`), nearest-rank method matching the app's `pct()`.

- [ ] **Step 1: Write the failing test**

`tests/test_aggregate.py`:

```python
"""Leaderboard math: means, exclusions, percentiles, ranking, worst-k."""
from __future__ import annotations

from voice_suite.aggregate import _pct, aggregate, render_report, worst_utterances
from voice_suite.protocol import JudgeScore, ScoredRecord, StageTimings


def _rec(impl="a", i=1, wer=0.2, e2e=0.8, judge_error=None, run_error=None,
         asr_ms=100):
    return ScoredRecord(
        impl=impl, utt_id=f"u{i:03d}", cache_key=f"k{impl}{i}", wer=wer,
        asr_text=f"asr {i}", mt_text=f"mt {i}", back_transcript=f"bt {i}",
        judge=JudgeScore(mt_adequacy=e2e, mt_fluency=e2e, e2e_adequacy=e2e,
                         verdict="v", rationale="r", error=judge_error),
        timings=StageTimings(asr_ms=asr_ms, mt_ms=50, tts_ms=80),
        run_error=run_error,
    )


def test_aggregate_means_and_error_counts():
    recs = [_rec(i=1, wer=0.0, e2e=1.0),
            _rec(i=2, wer=0.4, e2e=0.5),
            _rec(i=3, wer=1.0, e2e=0.0, run_error="crashed"),
            _rec(i=4, judge_error="api down")]
    rows = aggregate(recs)
    assert len(rows) == 1
    row = rows[0]
    assert row["scored"] == 4
    assert row["errors"] == 1
    assert row["judge_errors"] == 1
    # Judge means over the 3 non-judge-errored records: (1.0 + 0.5 + 0.0) / 3.
    assert abs(row["e2e_adequacy"] - 0.5) < 1e-9
    # WER mean over ALL records.
    assert abs(row["mean_wer"] - (0.0 + 0.4 + 1.0 + 0.2) / 4) < 1e-9


def test_ranking_by_e2e_then_wer():
    recs = [_rec(impl="low", i=1, e2e=0.2),
            _rec(impl="high", i=1, e2e=0.9),
            _rec(impl="tie_worse_wer", i=1, e2e=0.5, wer=0.5),
            _rec(impl="tie_better_wer", i=1, e2e=0.5, wer=0.1)]
    rows = aggregate(recs)
    assert [r["impl"] for r in rows] == ["high", "tie_better_wer",
                                         "tie_worse_wer", "low"]


def test_latency_percentiles_exclude_run_errors():
    recs = [_rec(i=i, asr_ms=i * 10) for i in range(1, 11)]
    recs.append(_rec(i=99, asr_ms=99999, run_error="crashed"))
    row = aggregate(recs)[0]
    assert row["asr_ms_p50"] == 60
    assert row["asr_ms_p95"] == 100
    assert row["mt_ms_p50"] == 50


def test_pct_empty_is_zero():
    assert _pct([], 0.5) == 0


def test_worst_utterances_sorted_and_capped(make_utt):
    utts = {f"u{i:03d}": make_utt(i) for i in range(1, 6)}
    recs = [_rec(i=i, e2e=i / 10) for i in range(1, 6)]
    worst = worst_utterances(recs, utts, k=3)
    assert [w["utt_id"] for w in worst] == ["u001", "u002", "u003"]
    assert worst[0]["ref_transcript"] == utts["u001"].ref_transcript
    assert worst[0]["mt_text"] == "mt 1"


def test_render_report_contains_table_and_appendix(make_utt):
    recs = [_rec(i=1)]
    rows = aggregate(recs)
    worst = worst_utterances(recs, {"u001": make_utt(1)})
    text = render_report(rows, worst)
    assert "| 1 | a |" in text
    assert "Best implementation: `a`" in text
    assert "Worst utterances" in text
    assert "ref (vi):" in text


def test_render_report_empty_is_valid():
    assert "voice-suite Leaderboard" in render_report([], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.aggregate'`

- [ ] **Step 3: Write the implementation**

`voice_suite/aggregate.py`:

```python
"""Aggregate scored records into a per-impl leaderboard and markdown report."""
from __future__ import annotations

from voice_suite.protocol import ScoredRecord, Utterance

WORST_K = 10


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(sorted_vals: list[int], q: float) -> int:
    """Nearest-rank percentile on a pre-sorted list (matches the app's pct())."""
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]


def aggregate(records: list[ScoredRecord]) -> list[dict]:
    """Group by impl; compute means, latency percentiles, error counts.

    Judge means average records whose judge call succeeded (judge.error is
    None) — pipeline-error zeros ARE included, so crashing configs rank
    worse. Latency percentiles cover records that actually ran. Rows sort
    best-first by e2e_adequacy, then lower WER (which also ranks --no-judge
    runs, where every e2e is 0).
    """
    by_impl: dict[str, list[ScoredRecord]] = {}
    for rec in records:
        by_impl.setdefault(rec.impl, []).append(rec)

    rows: list[dict] = []
    for impl, recs in by_impl.items():
        judged = [r.judge for r in recs if r.judge.error is None]
        ran = [r for r in recs if r.run_error is None]
        lat: dict[str, int] = {}
        for stage in ("asr_ms", "mt_ms", "tts_ms"):
            vals = sorted(getattr(r.timings, stage) for r in ran)
            lat[f"{stage}_p50"] = _pct(vals, 0.50)
            lat[f"{stage}_p95"] = _pct(vals, 0.95)
        rows.append({
            "impl": impl,
            "scored": len(recs),
            "errors": sum(1 for r in recs if r.run_error is not None),
            "judge_errors": len(recs) - len(judged),
            "mean_wer": _mean([r.wer for r in recs]),
            "mt_adequacy": _mean([j.mt_adequacy for j in judged]),
            "mt_fluency": _mean([j.mt_fluency for j in judged]),
            "e2e_adequacy": _mean([j.e2e_adequacy for j in judged]),
            **lat,
        })
    rows.sort(key=lambda r: (-r["e2e_adequacy"], r["mean_wer"]))
    return rows


def worst_utterances(records: list[ScoredRecord],
                     utts_by_id: dict[str, Utterance],
                     k: int = WORST_K) -> list[dict]:
    """The k lowest-e2e (then highest-WER) utterances, with all texts."""
    ranked = sorted(records, key=lambda r: (r.judge.e2e_adequacy, -r.wer))
    out: list[dict] = []
    for rec in ranked[:k]:
        utt = utts_by_id.get(rec.utt_id)
        out.append({
            "impl": rec.impl,
            "utt_id": rec.utt_id,
            "e2e_adequacy": rec.judge.e2e_adequacy,
            "wer": rec.wer,
            "ref_transcript": utt.ref_transcript if utt else "(manifest row missing)",
            "ref_translation": utt.ref_translation if utt else "",
            "asr_text": rec.asr_text,
            "mt_text": rec.mt_text,
            "back_transcript": rec.back_transcript,
            "error": rec.run_error or rec.judge.error or rec.back_error,
        })
    return out


def render_report(rows: list[dict], worst: list[dict]) -> str:
    """Render the leaderboard + worst-utterances appendix as markdown."""
    lines = [
        "# voice-suite Leaderboard", "",
        "Ranked by e2e_adequacy (desc), then WER (asc). Judge means exclude "
        "judge-errored records; pipeline errors count as zeros.", "",
        "| Rank | Impl | n | WER↓ | mt_adequacy↑ | mt_fluency↑ | "
        "e2e_adequacy↑ | asr_ms P50/P95 | mt_ms P50/P95 | tts_ms P50/P95 | "
        "errors |",
        "|------|------|---|------|--------------|-------------|"
        "---------------|----------------|---------------|----------------|"
        "--------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['impl']} | {r['scored']} | {r['mean_wer']:.3f} | "
            f"{r['mt_adequacy']:.3f} | {r['mt_fluency']:.3f} | "
            f"{r['e2e_adequacy']:.3f} | "
            f"{r['asr_ms_p50']}/{r['asr_ms_p95']} | "
            f"{r['mt_ms_p50']}/{r['mt_ms_p95']} | "
            f"{r['tts_ms_p50']}/{r['tts_ms_p95']} | {r['errors']} |"
        )
    for r in rows:
        if r["judge_errors"]:
            lines.append(f"\n{r['impl']}: {r['judge_errors']} judge call(s) "
                         f"failed and are excluded from judge means.")
    if rows:
        lines += ["", f"**Best implementation: `{rows[0]['impl']}`**"]
    if worst:
        lines += ["", "## Worst utterances (lowest e2e_adequacy)", ""]
        for w in worst:
            lines += [
                f"### {w['utt_id']} — {w['impl']} "
                f"(e2e {w['e2e_adequacy']:.2f}, WER {w['wer']:.2f})",
                f"- ref (vi): {w['ref_transcript']}",
                f"- asr (vi): {w['asr_text']}",
                f"- mt (en): {w['mt_text']}",
                f"- audio back-transcript (en): {w['back_transcript']}",
                f"- ref translation (en): {w['ref_translation']}",
            ]
            if w["error"]:
                lines.append(f"- error: {w['error']}")
            lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add voice_suite/aggregate.py tests/test_aggregate.py
git commit -m "feat: leaderboard aggregation and markdown report"
```

---
### Task 12: CLI — ingest / run / score / report (voice_suite/cli.py)

**Files:**
- Create: `voice_suite/cli.py` (structure from `rag_suite/cli.py`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 3–11 (`load_manifest`, `sample_items`, `discover_impls`, `run_impl`, `load_raw_records`, `score_all`, `load_scored_records`, `aggregate`, `worst_utterances`, `render_report`, `judge` module, `get_transcriber`).
- Produces: the `voice-suite` console entry point (declared in Task 1) with four commands:
  - `ingest [--n 200] [--seed 0]`
  - `run --impl <name>|all [--limit N] [--seed S]` — one impl's failure doesn't stop the others; exit 1 if any failed.
  - `score [--no-judge] [--judge cli|api] [--model ID] [--workers N] [--limit N] [--seed S]` — `--no-judge` skips judge AND back-transcription (free offline path).
  - `report` — writes `out/report.md` and echoes it.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
"""CLI smoke: run/score/report against a fake discovered impl, no network."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from voice_suite.cli import app

runner = CliRunner()

FAKE_IMPL = '''
from pathlib import Path
from voice_suite.protocol import StageTimings, VoiceResult

class _Impl:
    name = "fake"
    def setup(self): ...
    def translate_batch(self, utts, out_dir: Path):
        return [VoiceResult(asr_text=u.ref_transcript, mt_text=u.ref_translation,
                            timings=StageTimings(asr_ms=10, mt_ms=5, tts_ms=7))
                for u in utts]
IMPL = _Impl()
'''


def _write_world(root: Path, make_utt, n: int = 3):
    (root / "impls" / "fake").mkdir(parents=True)
    (root / "impls" / "fake" / "__init__.py").write_text(FAKE_IMPL,
                                                         encoding="utf-8")
    (root / "data").mkdir()
    utts = [make_utt(i) for i in range(1, n + 1)]
    (root / "data" / "manifest.jsonl").write_text(
        "".join(u.model_dump_json() + "\n" for u in utts), encoding="utf-8")
    return utts


def test_run_score_report_no_judge(tmp_path, monkeypatch, make_utt):
    _write_world(tmp_path, make_utt)
    monkeypatch.chdir(tmp_path)

    r = runner.invoke(app, ["run", "--impl", "fake"])
    assert r.exit_code == 0, r.output
    assert "ran: fake (3 new)" in r.output

    # Re-run is a cache hit: 0 new.
    r = runner.invoke(app, ["run", "--impl", "fake"])
    assert r.exit_code == 0, r.output
    assert "ran: fake (0 new)" in r.output

    r = runner.invoke(app, ["score", "--no-judge"])
    assert r.exit_code == 0, r.output
    assert "scored: 3 records" in r.output

    r = runner.invoke(app, ["report"])
    assert r.exit_code == 0, r.output
    assert "| 1 | fake |" in r.output
    assert (tmp_path / "out" / "report.md").exists()


def test_run_with_limit_samples_subset(tmp_path, monkeypatch, make_utt):
    _write_world(tmp_path, make_utt, n=5)
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["run", "--impl", "fake", "--limit", "2", "--seed", "0"])
    assert r.exit_code == 0, r.output
    assert "ran: fake (2 new)" in r.output


def test_run_unknown_impl_fails(tmp_path, monkeypatch, make_utt):
    _write_world(tmp_path, make_utt)
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["run", "--impl", "nope"])
    assert r.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'voice_suite.cli'`

- [ ] **Step 3: Write the implementation**

`voice_suite/cli.py`:

```python
"""Command-line interface: ingest / run / score / report."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import typer

from voice_suite import config
from voice_suite.aggregate import aggregate, render_report, worst_utterances
from voice_suite.dataset import load_manifest
from voice_suite.runner import load_raw_records, run_impl
from voice_suite.sampling import sample_items
from voice_suite.scoring import load_scored_records, score_all
from voice_suite.scoring import judge as judge_mod

app = typer.Typer(help="Speech-translation eval harness for my-2nd-voice.")


@app.callback()
def _setup(verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Verbose (DEBUG) logging; default shows INFO progress.")):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class JudgeBackend(str, Enum):
    """Judge backends selectable via `--judge`."""
    cli = "cli"
    api = "api"


@app.command()
def ingest(
    n: int = typer.Option(200, "--n", help="Number of utterances to sample."),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
):
    """Build data/manifest.jsonl + input WAVs from FLEURS (vi_vn ⋈ en_us)."""
    from voice_suite.ingest_fleurs import ingest as run_ingest  # heavy deps
    count = run_ingest(n, seed, config.UTTERANCES_DIR, config.MANIFEST)
    typer.echo(f"ingested {count} utterances → {config.MANIFEST}")


@app.command()
def run(
    impl: str = typer.Option(..., help="Impl name to run, or 'all'."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Only run a deterministic sample of N "
        "utterances. Use the SAME --limit/--seed on `score`."),
    seed: int = typer.Option(0, "--seed", help="Seed for --limit sampling."),
):
    """Generate raw results for one or all discovered impls."""
    utts = sample_items(load_manifest(config.MANIFEST), limit, seed)
    if limit is not None:
        typer.echo(f"sampled {len(utts)} utterances (limit={limit}, seed={seed})")
    impls = config.discover_impls(config.IMPLS_DIR)
    if not impls:
        raise typer.BadParameter("no impls discovered under impls/")
    if impl != "all" and impl not in impls:
        raise typer.BadParameter(
            f"unknown impl {impl!r}; discovered: {sorted(impls)}")
    targets = list(impls.values()) if impl == "all" else [impls[impl]]
    failures = 0
    for target in targets:
        try:
            n = run_impl(target, utts, config.RAW_RESULTS, config.AUDIO_OUT_DIR)
            typer.echo(f"ran: {target.name} ({n} new)")
        except Exception as exc:
            # One impl's setup/exit failure must not sink the other impls.
            failures += 1
            typer.echo(f"FAILED: {target.name}: {exc}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def score(
    no_judge: bool = typer.Option(
        False, "--no-judge",
        help="Skip judge AND back-transcription (free offline path: "
             "WER + latency only)."),
    judge: JudgeBackend = typer.Option(
        JudgeBackend.cli, "--judge",
        help="Judge backend: 'cli' (Claude Code login) or 'api' "
             "(ANTHROPIC_API_KEY)."),
    model: str = typer.Option(judge_mod.DEFAULT_JUDGE_MODEL, "--model",
                              help="Judge model id."),
    workers: int = typer.Option(1, "--workers", min=1,
                                help="Parallel judge workers."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Match the --limit used on `run`."),
    seed: int = typer.Option(0, "--seed", help="Match the `run` seed."),
):
    """Score raw results: WER always; back-transcription + judge unless --no-judge."""
    utts = sample_items(load_manifest(config.MANIFEST), limit, seed)
    raw = load_raw_records(config.RAW_RESULTS)
    if no_judge:
        call_judge = None
        transcribe = None
        judge_id = "no-judge"
    else:
        from voice_suite.scoring.backtranscribe import get_transcriber
        call_judge = judge_mod.get_judge(judge.value, model=model)
        transcribe = get_transcriber()
        judge_id = f"{judge.value}:{model}"
    score_all(raw, utts, config.SCORED, call_judge=call_judge,
              workers=workers, judge_id=judge_id, transcribe=transcribe,
              bt_cache_path=config.BT_CACHE)
    typer.echo(f"scored: {len(load_scored_records(config.SCORED))} records")


@app.command()
def report():
    """Render the leaderboard markdown report."""
    records = load_scored_records(config.SCORED)
    rows = aggregate(records)
    try:
        utts_by_id = {u.id: u for u in load_manifest(config.MANIFEST)}
    except FileNotFoundError:
        utts_by_id = {}
    text = render_report(rows, worst_utterances(records, utts_by_id))
    config.REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.REPORT.write_text(text, encoding="utf-8")
    typer.echo(text)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Confirm the console script works**

Run: `voice-suite --help`
Expected: usage text listing `ingest`, `run`, `score`, `report`

- [ ] **Step 6: Commit**

```bash
git add voice_suite/cli.py tests/test_cli.py
git commit -m "feat: typer CLI wiring all pipeline stages"
```

---
### Task 13: m2v_default impl wrapper + fake eval_batch test double (impls/m2v_default/, tests/fake_eval_batch.py)

**Files:**
- Create: `impls/m2v_default/adapter.py`
- Create: `impls/m2v_default/__init__.py`
- Create: `impls/m2v_default/local.toml.example`
- Create: `tests/fake_eval_batch.py`
- Test: `tests/test_impl_m2v.py`

**Interfaces:**
- Consumes: `StageTimings`, `Utterance`, `VoiceResult` (Task 2); the **cross-repo contract** (see File Structure Map) — the adapter appends `--manifest <path> --out-dir <path>` to a command prefix and parses `<out-dir>/results.jsonl`.
- Produces:
  - `M2vBatchImpl(name: str = "m2v_default", config_path: Path | None = None, cmd: list[str] | None = None)` — `cmd` is the test seam: a full command prefix that bypasses `local.toml`.
  - `IMPL` instance exposed from `impls/m2v_default/__init__.py` (found by Task 4's discovery).
  - `setup()` raises `RuntimeError` embedding a copy-paste config template when `local.toml` is missing/incomplete, and names any config path that doesn't exist on disk.
  - `translate_batch()` returns results in input order; ids missing from the exe's output become `VoiceResult(error="missing from eval_batch output")`; non-zero exit raises `RuntimeError` with stderr.

- [ ] **Step 1: Write the test double**

`tests/fake_eval_batch.py` — a Python stand-in honoring the cross-repo contract:

```python
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
```

- [ ] **Step 2: Write the failing test**

`tests/test_impl_m2v.py`:

```python
"""Impl wrapper against the fake eval_batch: contract, ordering, failures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from impls.m2v_default.adapter import M2vBatchImpl

FAKE = Path(__file__).parent / "fake_eval_batch.py"


def _impl(*extra: str) -> M2vBatchImpl:
    return M2vBatchImpl(cmd=[sys.executable, str(FAKE), *extra])


def test_batch_results_in_input_order(tmp_path, make_utt):
    impl = _impl()
    impl.setup()
    utts = [make_utt(2), make_utt(1)]  # deliberately unsorted
    results = impl.translate_batch(utts, tmp_path)
    assert [r.asr_text for r in results] == ["asr u002", "asr u001"]
    assert results[0].mt_text == "mt u002"
    assert results[0].timings.asr_ms == 10
    assert results[0].error is None
    assert all(Path(r.audio_path).exists() for r in results)


def test_missing_id_becomes_error_result(tmp_path, make_utt):
    impl = _impl("--skip-id", "u001")
    impl.setup()
    results = impl.translate_batch([make_utt(1), make_utt(2)], tmp_path)
    assert results[0].error == "missing from eval_batch output"
    assert results[1].error is None


def test_nonzero_exit_raises_with_stderr(tmp_path, make_utt):
    impl = _impl("--fail")
    impl.setup()
    with pytest.raises(RuntimeError, match="boom: model not found"):
        impl.translate_batch([make_utt(1)], tmp_path)


def test_setup_without_config_shows_template(tmp_path):
    impl = M2vBatchImpl(config_path=tmp_path / "local.toml")
    with pytest.raises(RuntimeError, match="missing impl config"):
        impl.setup()
    with pytest.raises(RuntimeError, match="eval_batch.exe"):
        impl.setup()  # template with example paths is embedded in the error


def test_setup_flags_nonexistent_paths(tmp_path):
    cfg = tmp_path / "local.toml"
    cfg.write_text(
        'exe = "does/not/exist.exe"\n'
        'whisper_model = "x"\nmt_dir = "y"\n'
        'tts_onnx_dir = "z"\ntts_voice_style = "w"\n', encoding="utf-8")
    impl = M2vBatchImpl(config_path=cfg)
    with pytest.raises(RuntimeError, match="exe does not exist"):
        impl.setup()


def test_setup_flags_missing_keys(tmp_path):
    cfg = tmp_path / "local.toml"
    cfg.write_text('exe = "x"\n', encoding="utf-8")
    impl = M2vBatchImpl(config_path=cfg)
    with pytest.raises(RuntimeError, match="missing keys"):
        impl.setup()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_impl_m2v.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'impls.m2v_default.adapter'` (pyproject's `pythonpath = ["."]` makes `impls` importable once the package exists)

- [ ] **Step 4: Write the implementation**

`impls/m2v_default/adapter.py`:

```python
"""Batch adapter: shells out to my-2nd-voice's eval_batch binary.

An "impl" is a *pipeline configuration*: exe + model paths + quality flags
come from a git-ignored ``local.toml`` next to this file. Later impls
(other models, VAD on, F5 TTS, …) reuse this class with a different config
file and/or extra_args.
"""
from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from voice_suite.protocol import StageTimings, Utterance, VoiceResult

_REQUIRED_KEYS = ("exe", "whisper_model", "mt_dir", "tts_onnx_dir",
                  "tts_voice_style")

_TEMPLATE = """\
# impls/m2v_default/local.toml — machine-local paths (git-ignored)
exe = "C:/project/training_ai/my-2nd-voice/target/release/eval_batch.exe"
whisper_model = "C:/project/training_ai/my-2nd-voice/models/whisper/ggml-phowhisper-small-tsa.bin"
mt_dir = "C:/project/training_ai/my-2nd-voice/models/mt/vi-en"
tts_onnx_dir = "C:/project/training_ai/my-2nd-voice/models/tts/supertonic/onnx"
tts_voice_style = "C:/project/training_ai/my-2nd-voice/models/tts/supertonic/voice_styles/M1.json"
# extra_args = ["--beam-size", "5"]
"""


class M2vBatchImpl:
    """VoiceImpl adapter around one eval_batch invocation per batch."""

    def __init__(self, name: str = "m2v_default",
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
            "--tts-onnx-dir", str(cfg["tts_onnx_dir"]),
            "--tts-voice-style", str(cfg["tts_voice_style"]),
            "--src", "vi", "--dst", "en",
            *[str(a) for a in cfg.get("extra_args", [])],
        ]

    def translate_batch(self, utts: list[Utterance],
                        out_dir: Path) -> list[VoiceResult]:
        """One eval_batch process for the whole batch; results in input order."""
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
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"eval_batch exited {proc.returncode}:\n{proc.stderr.strip()}")

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
                    timings=StageTimings(asr_ms=row.get("asr_ms", 0),
                                         mt_ms=row.get("mt_ms", 0),
                                         tts_ms=row.get("tts_ms", 0)),
                    error=row.get("error"),
                )
        missing = VoiceResult(error="missing from eval_batch output")
        return [by_id.get(u.id, missing) for u in utts]
```

`impls/m2v_default/__init__.py`:

```python
"""Default my-2nd-voice pipeline config: PhoWhisper-small + Marian vi-en + Supertonic."""
from impls.m2v_default.adapter import M2vBatchImpl

IMPL = M2vBatchImpl()
```

`impls/m2v_default/local.toml.example` — the `_TEMPLATE` content verbatim:

```toml
# impls/m2v_default/local.toml — machine-local paths (git-ignored)
exe = "C:/project/training_ai/my-2nd-voice/target/release/eval_batch.exe"
whisper_model = "C:/project/training_ai/my-2nd-voice/models/whisper/ggml-phowhisper-small-tsa.bin"
mt_dir = "C:/project/training_ai/my-2nd-voice/models/mt/vi-en"
tts_onnx_dir = "C:/project/training_ai/my-2nd-voice/models/tts/supertonic/onnx"
tts_voice_style = "C:/project/training_ai/my-2nd-voice/models/tts/supertonic/voice_styles/M1.json"
# extra_args = ["--beam-size", "5"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_impl_m2v.py -v`
Expected: 6 passed

- [ ] **Step 6: Run the whole suite (regression check)**

Run: `python -m pytest -q`
Expected: all tests pass. (test_cli is unaffected by the new impl: it chdirs to a tmp dir whose `impls/` contains only its own `fake` impl. Discovery of `m2v_default` from the repo root happens at real-run time, and its lazy `setup()` only fires when that impl is actually run.)

- [ ] **Step 7: Commit**

```bash
git add impls/m2v_default/ tests/fake_eval_batch.py tests/test_impl_m2v.py
git commit -m "feat: m2v_default batch impl wrapper"
```

---
### Task 14: eval_batch.rs batch driver (my-2nd-voice repo, branch feat/run-on-windows)

**All commands in this task run from `C:\project\training_ai\my-2nd-voice` on branch `feat/run-on-windows`.**

**Files:**
- Modify: `Cargo.toml` — move `hound = "3.5"` from `[dev-dependencies]` to `[dependencies]`; add `[[bin]]` and `[[test]]` entries.
- Create: `src/bin/eval_batch.rs`
- Create: `tests/integration/eval_batch_smoke.rs`

**Interfaces:**
- Consumes (all existing on this branch — construction mirrors `src/bin/translate.rs`):
  - `WhisperAsr::load(WhisperConfig { model_path: String, n_threads: None, use_gpu: bool, beam_size: Option<i32>, fixed_prompt: Option<String> })`, `normalize_episode(&[f32]) -> Vec<f32>` from `my_2nd_voice::pipeline::asr::whisper`.
  - `MarianMt::load(MarianConfig { model_dir: PathBuf, max_new_tokens: usize })` from `my_2nd_voice::pipeline::mt::marian`.
  - `Supertonic::load(SupertonicConfig { onnx_dir: String, voice_style_path: String, total_step: usize, speed: f32, cfg_strength: f32, use_gpu: bool })` from `my_2nd_voice::pipeline::tts::supertonic`.
  - Traits `AsrStage::transcribe(&mut self, samples, hint_lang) -> Result<Vec<AsrEvent>>`, `MtStage::translate(&mut self, text, src, dst) -> Result<String>`, `TtsStage::synthesize(&mut self, text, lang) -> Result<TtsFrame>`, `Lang(&'static str)` from `my_2nd_voice::pipeline::traits`.
  - `SincResampler::{new(src, dst), process(&[f32]), finish()}` from `my_2nd_voice::audio::resample`.
- Produces: the `eval_batch` binary honoring the **cross-repo contract** (File Structure Map).
- Iron-rule compliance: no `unwrap`/`expect` outside tests; `tracing` logs carry ids/timings/flags only — transcripts and translations go ONLY into `results.jsonl` (IR-4); no VAD/denoise/APM (deliberate, per spec).
- Parity notes: applies `normalize_episode` before ASR and the same `normalize_asr_text` post-processing as `translate.rs` (whitespace collapse + sentence-final punctuation for the Marian tokenizer). The live pipeline's hallucination blocklist is deliberately NOT applied — FLEURS clips are clean speech, and the eval should see raw model behavior; clause-splitting for TTS is skipped (batch mode has no latency constraint; FLEURS rows are single sentences).

- [ ] **Step 1: Update Cargo.toml**

Add after the existing `[[bin]] name = "incoming_spike"` block:

```toml
[[bin]]
name = "eval_batch"
path = "src/bin/eval_batch.rs"
```

Add after the existing `[[test]] name = "audio_quality"` block:

```toml
[[test]]
name = "eval_batch_smoke"
path = "tests/integration/eval_batch_smoke.rs"
```

Add `hound = "3.5"` to `[dependencies]` (next to `rubato`), and DELETE the `[dev-dependencies]` section's `hound = "3.5"` line (a `[dependencies]` entry is visible to dev targets too).

- [ ] **Step 2: Write src/bin/eval_batch.rs (implementation + unit tests in one file)**

```rust
//! Offline batch eval driver for voice-suite: manifest in → per-utterance
//! ASR→MT→TTS → results JSONL + output WAVs.
//!
//! Deliberately NO VAD/merge/denoise/APM: eval clips are clean single
//! utterances; skipping the live-audio front-end keeps runs deterministic
//! and measures AI quality (see voice-suite design doc).
//!
//! IR-4: transcripts/translations are written ONLY to the results file,
//! never to tracing logs.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use my_2nd_voice::audio::resample::SincResampler;
use my_2nd_voice::pipeline::asr::whisper::{normalize_episode, WhisperAsr, WhisperConfig};
use my_2nd_voice::pipeline::mt::marian::{MarianConfig, MarianMt};
use my_2nd_voice::pipeline::traits::{AsrEvent, AsrStage, Lang, MtStage, TtsStage};
use my_2nd_voice::pipeline::tts::supertonic::{Supertonic, SupertonicConfig};

const ASR_RATE: u32 = 16_000;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Ground-truth manifest (JSONL: id, audio_path, src_lang, dst_lang, …).
    #[arg(long)]
    manifest: PathBuf,
    /// Output dir for <id>.wav files and results.jsonl.
    #[arg(long)]
    out_dir: PathBuf,

    #[arg(long)]
    whisper_model: PathBuf,
    #[arg(long)]
    mt_dir: PathBuf,
    #[arg(long)]
    tts_onnx_dir: PathBuf,
    #[arg(long)]
    tts_voice_style: PathBuf,

    #[arg(long, default_value = "vi")]
    src: String,
    #[arg(long, default_value = "en")]
    dst: String,

    /// Whisper beam size (5 = beam search; matches the live pipeline default).
    #[arg(long, default_value_t = 5)]
    beam_size: i32,

    /// Fixed initial prompt anchoring Whisper's language/domain (live default).
    #[arg(
        long,
        default_value = "Đây là cuộc trò chuyện bằng tiếng Việt. Nội dung về công việc và kỹ thuật."
    )]
    whisper_prompt: String,
}

#[derive(Deserialize)]
struct ManifestRow {
    id: String,
    audio_path: String,
    src_lang: String,
    dst_lang: String,
}

#[derive(Serialize)]
struct ResultRow {
    id: String,
    asr_text: String,
    mt_text: String,
    audio_out: Option<String>,
    asr_ms: u64,
    mt_ms: u64,
    tts_ms: u64,
    error: Option<String>,
}

impl ResultRow {
    fn errored(id: &str, error: String) -> Self {
        Self {
            id: id.to_string(),
            asr_text: String::new(),
            mt_text: String::new(),
            audio_out: None,
            asr_ms: 0,
            mt_ms: 0,
            tts_ms: 0,
            error: Some(error),
        }
    }
}

/// Read a mono WAV as f32 samples at 16 kHz (resampling if needed).
fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!(
            "{}: expected mono, got {} channels",
            path.display(),
            spec.channels
        );
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| f32::from(v) / 32_768.0))
            .collect::<std::result::Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        (fmt, bits) => bail!(
            "{}: unsupported wav format {:?}/{} (want 16-bit int or 32-bit float)",
            path.display(),
            fmt,
            bits
        ),
    };
    if spec.sample_rate == ASR_RATE {
        return Ok(samples);
    }
    let mut resampler = SincResampler::new(spec.sample_rate, ASR_RATE)?;
    let mut out = resampler.process(&samples)?;
    out.extend(resampler.finish()?);
    Ok(out)
}

/// Write mono f32 samples as a 16-bit PCM WAV.
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32_767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Same light normalization the live pipeline applies before MT
/// (bin/translate.rs::normalize_asr_text): collapse whitespace and ensure
/// a sentence-final punctuation mark for the Marian tokenizer.
fn normalize_asr_text(s: &str) -> String {
    let collapsed = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() || collapsed.ends_with(['.', '?', '!', '…', '。']) {
        collapsed
    } else {
        format!("{collapsed}.")
    }
}

fn asr_events_to_text(events: Vec<AsrEvent>) -> String {
    let joined = events
        .into_iter()
        .filter_map(|e| match e {
            AsrEvent::Final { text, .. } | AsrEvent::Partial { text, .. } => Some(text),
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    normalize_asr_text(&joined)
}

fn read_manifest(path: &Path) -> Result<Vec<ManifestRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read manifest {}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: ManifestRow =
            serde_json::from_str(line).with_context(|| format!("manifest line {}", i + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn process_row(
    row: &ManifestRow,
    args: &Args,
    src: Lang,
    dst: Lang,
    asr: &mut WhisperAsr,
    mt: &mut MarianMt,
    tts: &mut Supertonic,
) -> Result<ResultRow> {
    if row.src_lang != args.src || row.dst_lang != args.dst {
        bail!(
            "direction mismatch: manifest row is {}→{}, this run is {}→{} \
             (Marian is single-pair)",
            row.src_lang,
            row.dst_lang,
            args.src,
            args.dst
        );
    }
    let samples = read_wav_16k_mono(Path::new(&row.audio_path))?;
    let normalized = normalize_episode(&samples);

    let t = Instant::now();
    let events = asr.transcribe(&normalized, Some(src))?;
    let asr_ms = t.elapsed().as_millis() as u64;
    let asr_text = asr_events_to_text(events);
    if asr_text.is_empty() {
        bail!("asr produced no text");
    }

    let t = Instant::now();
    let mt_text = mt.translate(&asr_text, src, dst)?;
    let mt_ms = t.elapsed().as_millis() as u64;

    let t = Instant::now();
    let frame = tts.synthesize(&mt_text, dst)?;
    let tts_ms = t.elapsed().as_millis() as u64;

    let wav_path = args.out_dir.join(format!("{}.wav", row.id));
    write_wav(&wav_path, &frame.samples, frame.sample_rate)?;

    Ok(ResultRow {
        id: row.id.clone(),
        asr_text,
        mt_text,
        audio_out: Some(wav_path.to_string_lossy().into_owned()),
        asr_ms,
        mt_ms,
        tts_ms,
        error: None,
    })
}

fn main() -> Result<()> {
    my_2nd_voice::logging::init();
    let args = Args::parse();

    let rows = read_manifest(&args.manifest)?;
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("create out dir {}", args.out_dir.display()))?;

    // Load all three stages ONCE — the whole point of batch mode.
    let t0 = Instant::now();
    let mut asr = WhisperAsr::load(WhisperConfig {
        model_path: args.whisper_model.to_string_lossy().into_owned(),
        n_threads: None,
        use_gpu: cfg!(target_os = "macos") || cfg!(feature = "cuda"),
        beam_size: if args.beam_size > 1 {
            Some(args.beam_size)
        } else {
            None
        },
        fixed_prompt: if args.whisper_prompt.is_empty() {
            None
        } else {
            Some(args.whisper_prompt.clone())
        },
    })?;
    let mut mt = MarianMt::load(MarianConfig {
        model_dir: args.mt_dir.clone(),
        max_new_tokens: 128,
    })?;
    let mut tts = Supertonic::load(SupertonicConfig {
        onnx_dir: args.tts_onnx_dir.to_string_lossy().into_owned(),
        voice_style_path: args.tts_voice_style.to_string_lossy().into_owned(),
        total_step: 8,
        speed: 1.05,
        cfg_strength: 0.3,
        use_gpu: false,
    })?;
    tracing::info!(
        load_ms = t0.elapsed().as_millis() as u64,
        utterances = rows.len(),
        "models_loaded"
    );

    let src = Lang(Box::leak(args.src.clone().into_boxed_str()));
    let dst = Lang(Box::leak(args.dst.clone().into_boxed_str()));

    let results_path = args.out_dir.join("results.jsonl");
    let mut out = std::fs::File::create(&results_path)
        .with_context(|| format!("create {}", results_path.display()))?;

    let total = rows.len();
    for (i, row) in rows.iter().enumerate() {
        let result = match process_row(row, &args, src, dst, &mut asr, &mut mt, &mut tts) {
            Ok(r) => r,
            // Per-utterance failure: record and continue (harness scores it 0).
            Err(e) => ResultRow::errored(&row.id, format!("{e:#}")),
        };
        // IR-4: ids and timings only — never transcript text.
        tracing::info!(
            utt = %row.id,
            n = i + 1,
            total,
            asr_ms = result.asr_ms,
            mt_ms = result.mt_ms,
            tts_ms = result.tts_ms,
            ok = result.error.is_none(),
            "utterance_done"
        );
        serde_json::to_writer(&mut out, &result)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("eval_batch_test_{tag}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn wav_roundtrip_16k_within_quantization() -> Result<()> {
        let path = tmp_dir("rt").join("rt.wav");
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 / 1600.0) - 0.5).collect();
        write_wav(&path, &samples, 16_000)?;
        let back = read_wav_16k_mono(&path)?;
        assert_eq!(back.len(), samples.len());
        for (a, b) in samples.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0 / 32_000.0, "quantization exceeded: {a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn wav_read_resamples_44k_to_16k() -> Result<()> {
        let path = tmp_dir("rs").join("44k.wav");
        let samples = vec![0.25f32; 44_100]; // 1 s at 44.1 kHz
        write_wav(&path, &samples, 44_100)?;
        let back = read_wav_16k_mono(&path)?;
        let expected = 16_000i64;
        assert!(
            (back.len() as i64 - expected).abs() < 800,
            "expected ~{expected} samples, got {}",
            back.len()
        );
        Ok(())
    }

    #[test]
    fn manifest_parse_ignores_extra_fields_and_blank_lines() -> Result<()> {
        let path = tmp_dir("mf").join("m.jsonl");
        std::fs::write(
            &path,
            "{\"id\":\"u1\",\"audio_path\":\"a.wav\",\"src_lang\":\"vi\",\
             \"dst_lang\":\"en\",\"ref_transcript\":\"x\",\"ref_translation\":\"y\"}\n\n",
        )?;
        let rows = read_manifest(&path)?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "u1");
        assert_eq!(rows[0].src_lang, "vi");
        Ok(())
    }

    #[test]
    fn asr_text_normalized_with_final_punctuation() {
        let events = vec![
            AsrEvent::Final {
                text: "  xin   chào".into(),
                t0_ms: 0,
                t1_ms: 1,
                lang: None,
            },
            AsrEvent::Final {
                text: "các bạn ".into(),
                t0_ms: 1,
                t1_ms: 2,
                lang: None,
            },
        ];
        assert_eq!(asr_events_to_text(events), "xin chào các bạn.");
    }

    #[test]
    fn stereo_wav_rejected() {
        let path = tmp_dir("st").join("stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("create");
        for _ in 0..32 {
            w.write_sample(0i16).expect("write");
        }
        w.finalize().expect("finalize");
        let err = read_wav_16k_mono(&path).unwrap_err();
        assert!(err.to_string().contains("expected mono"));
    }
}
```

- [ ] **Step 3: Run the unit tests**

Run: `cargo test --bin eval_batch`
Expected: 5 passed (first run also compiles the new bin)

- [ ] **Step 4: Lint + format**

Run: `cargo clippy --bin eval_batch -- -D warnings && cargo fmt`
Expected: no warnings; fmt makes no functional change

- [ ] **Step 5: Write the env-gated smoke test**

`tests/integration/eval_batch_smoke.rs`:

```rust
//! Env-gated end-to-end smoke for the eval_batch binary.
//!
//! Skips (passes) unless ALL model env vars are set:
//!   M2V_WHISPER_MODEL, M2V_MT_DIR, M2V_TTS_ONNX_DIR, M2V_TTS_VOICE_STYLE
//! Run: cargo test --test eval_batch_smoke -- --nocapture
//!
//! The input is a synthetic tone, not speech — the contract under test is
//! plumbing (exit 0 + one JSONL row per manifest row), not ASR quality.

use std::process::Command;

#[test]
fn eval_batch_runs_one_utterance_end_to_end() {
    let vars = [
        "M2V_WHISPER_MODEL",
        "M2V_MT_DIR",
        "M2V_TTS_ONNX_DIR",
        "M2V_TTS_VOICE_STYLE",
    ];
    let Ok(values) = vars.iter().map(std::env::var).collect::<Result<Vec<_>, _>>() else {
        eprintln!("skipping smoke: set {vars:?} to run");
        return;
    };

    let dir = std::env::temp_dir().join("eval_batch_smoke");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let wav = dir.join("tone.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&wav, spec).expect("wav create");
    for i in 0..16_000 {
        let s = (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin() * 0.3;
        w.write_sample((s * 32_767.0) as i16).expect("write");
    }
    w.finalize().expect("finalize");

    let manifest = dir.join("manifest.jsonl");
    std::fs::write(
        &manifest,
        format!(
            "{{\"id\":\"smoke1\",\"audio_path\":{:?},\"src_lang\":\"vi\",\"dst_lang\":\"en\"}}\n",
            wav.to_string_lossy()
        ),
    )
    .expect("manifest");

    let out_dir = dir.join("out");
    let status = Command::new(env!("CARGO_BIN_EXE_eval_batch"))
        .arg("--manifest")
        .arg(&manifest)
        .arg("--out-dir")
        .arg(&out_dir)
        .args(["--whisper-model", &values[0]])
        .args(["--mt-dir", &values[1]])
        .args(["--tts-onnx-dir", &values[2]])
        .args(["--tts-voice-style", &values[3]])
        .status()
        .expect("spawn eval_batch");
    assert!(status.success(), "eval_batch exited non-zero");

    let results =
        std::fs::read_to_string(out_dir.join("results.jsonl")).expect("results.jsonl");
    assert!(results.contains("\"id\":\"smoke1\""));
}
```

- [ ] **Step 6: Run the smoke test (skips without models)**

Run: `cargo test --test eval_batch_smoke -- --nocapture`
Expected: 1 passed, with `skipping smoke: …` printed (unless the M2V_* env vars are set, in which case it drives the real pipeline)

- [ ] **Step 7: Build the release exe the impl config points at**

Run: `cargo build --release --bin eval_batch`
Expected: `target/release/eval_batch.exe` exists. (GPU variant later: `cargo build --release --features cuda --bin eval_batch`, cuDNN DLL placement per the app's run scripts.)

- [ ] **Step 8: Commit (in my-2nd-voice)**

```bash
git add Cargo.toml src/bin/eval_batch.rs tests/integration/eval_batch_smoke.rs
git commit -m "feat(eval): add eval_batch offline batch driver for voice-suite"
```

---
### Task 15: README — prerequisites, model downloads, quickstart, manual smoke

**Files:**
- Create: `README.md` (voice-suite repo)

**Interfaces:**
- Consumes: everything — this documents the finished system.
- Produces: the only setup document. No test cycle; the check is a read-through against the CLI flags and paths defined in Tasks 12–14.

- [ ] **Step 1: Write README.md**

```markdown
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

### 3. Supertonic TTS — `models/tts/supertonic/`

The download recipe exists on the app repo's `origin/feat/python-setup`
branch; read it and port the paths:

    git -C ../my-2nd-voice show origin/feat/python-setup:download_supertonic.sh

It fetches the `Supertone/supertonic` HF repo: the ONNX dir (`tts.json`,
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
    M2V_TTS_ONNX_DIR=models/tts/supertonic/onnx \
    M2V_TTS_VOICE_STYLE=models/tts/supertonic/voice_styles/M1.json \
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
```

- [ ] **Step 2: Read it back against the code**

Check every command/flag/path in the README against Tasks 12–14 (CLI flags, local.toml keys, env var names, model file lists). Fix any drift.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, model downloads, quickstart"
```

---

## Done criteria (whole plan)

1. `python -m pytest` green in voice-suite with no models/network (Tasks 1–13).
2. `cargo test --bin eval_batch` and `cargo clippy --bin eval_batch -- -D warnings` green in my-2nd-voice (Task 14).
3. With models on disk: the Manual smoke section of the README works end to end — `run --limit 2` produces `out/audio/m2v_default/*.wav` + raw records, `score --no-judge` produces scored records, `report` prints a one-row leaderboard.
