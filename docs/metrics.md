# voice-suite Metrics Reference

What every number in `out/report.md` (and the underlying `out/scored.jsonl`) actually
measures, how it's computed, and what makes it change between runs. Grounded directly
in the source: `voice_suite/scoring/wer.py`, `voice_suite/scoring/judge.py`,
`voice_suite/scoring/backtranscribe.py`, `voice_suite/scoring/__init__.py`,
`voice_suite/aggregate.py`, `voice_suite/protocol.py`.

## 1. Pipeline at a glance

```
vi audio ──ASR──▶ vi text ──MT──▶ en text ──TTS──▶ en audio
                    │                │                │
                    │                │                └─(scoring) back-transcription ──▶ en text'
                    │                │                                                    │
                    ▼                ▼                                                    ▼
                   WER          mt_adequacy                                          e2e_adequacy
                              mt_fluency (judge)                                       (judge)
```

- **WER** checks only the ASR stage (vi audio → vi text vs. the reference transcript).
- **mt_adequacy / mt_fluency** check only the MT stage (vi text → en text vs. the reference translation), independent of whether TTS/audio worked at all.
- **e2e_adequacy** checks whether the *whole* pipeline survived — it re-hears the synthesized output audio (via back-transcription) and asks whether the original meaning is still there after TTS.

## 2. Quick reference

| Metric | Range | Needs judge? | Needs back-transcription? | Computed in |
|---|---|---|---|---|
| `wer` | 0.0–1.0, lower is better | No | No | `scoring/wer.py` |
| `mt_adequacy` | 0.0–1.0, higher is better | Yes | No | `scoring/judge.py` (LLM) |
| `mt_fluency` | 0.0–1.0, higher is better | Yes | No | `scoring/judge.py` (LLM) |
| `e2e_adequacy` | 0.0–1.0, higher is better | Yes | Yes | `scoring/judge.py` (LLM) |
| `verdict` / `rationale` | free text | Yes | — | `scoring/judge.py` (LLM) |
| `back_transcript` / `back_error` | text / error string | No | Yes | `scoring/backtranscribe.py` (faster-whisper `large-v3`) |
| `asr_ms` / `mt_ms` / `tts_ms` | integer milliseconds | No | No | the impl adapter itself |
| `errors` | count | No | No | `aggregate.py` |
| `judge_errors` | count | Yes | No | `aggregate.py` |

`--no-judge` (on `score`) disables the judge **and** back-transcription together — one
flag, not two independent switches. That means `mt_adequacy`, `mt_fluency`,
`e2e_adequacy`, and `back_transcript` all go dark at once; only `wer` and the latency
columns are ever "free" (no LLM call, no extra model download).

## 3. WER — ASR transcription accuracy

```python
def norm_text(s: str) -> str:
    """Lowercase, punctuation → space, collapse whitespace."""
    s = unicodedata.normalize("NFC", s)
    s = _PUNCT_RE.sub(" ", s.lower())
    return " ".join(s.split())

def score_wer(ref: str, hyp: str) -> float:
    ref_n, hyp_n = norm_text(ref), norm_text(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return min(jiwer.wer(ref_n, hyp_n), 1.0)
```

- Standard word error rate (`jiwer.wer`): `(substitutions + deletions + insertions) / reference_word_count`.
- Both strings are normalized first: Unicode NFC, lowercased, punctuation replaced with spaces (`\w` is Unicode-aware, so Vietnamese diacritics like `đ`, `ệ`, `ỳ` are preserved as word characters, not stripped).
- **Capped at 1.0**, i.e. 100% — a ratio of edits to reference word count, not a count of words or seconds. Raw WER can exceed 1.0 when the hypothesis has a lot of extra (inserted) words — e.g. a hallucinating ASR. The cap keeps one runaway hallucination from dragging a whole leaderboard mean below every other row disproportionately.
- Edge case: empty reference → `0.0` if the hypothesis is also empty (both silent, trivially "correct"), else `1.0` (can't compute a ratio against zero reference words, so it's scored as fully wrong).
- Always computed, for every record, regardless of `--no-judge`. This is your only *guaranteed* signal — it never needs a judge, back-transcription, or network access, only the ASR text your impl already produced and the manifest's `ref_transcript`.
- A pipeline failure (impl crashed on that utterance) still gets a real WER: an empty `asr_text` against a non-empty reference computes to `1.0` (worst case), which is intentional — a config that crashes should rank worse, not get a free pass by being excluded.

## 4. Judge scores — `mt_adequacy`, `mt_fluency`, `e2e_adequacy`

These three floats (plus `verdict` and `rationale`) come from one LLM call per
utterance (Claude, via `--judge cli` or `--judge api`), using this exact rubric
(`voice_suite/scoring/judge.py`):

```
You are a strict, neutral judge of speech-translation quality. A system
transcribed Vietnamese speech (ASR), translated it to English (MT), and
synthesized the translation as English speech (TTS). The system's output
audio was transcribed back to text by a strong independent ASR.
Score three values, each between 0.0 and 1.0:
- mt_adequacy: does MT OUTPUT convey the meaning of the reference
translation, given the Vietnamese source? 1.0 = same meaning, 0.0 = unrelated.
- mt_fluency: is MT OUTPUT natural, grammatical English (regardless of accuracy)?
- e2e_adequacy: does OUTPUT AUDIO TRANSCRIPT still convey the reference
meaning? Penalize meaning lost between MT OUTPUT and the audio (dropped or
garbled words); do not penalize back-transcription spelling or punctuation
variance.
```

In plain terms:

| Field | Question it answers | Would penalize | Would NOT penalize |
|---|---|---|---|
| `mt_adequacy` | Did the *meaning* survive vi → en translation? | Mistranslation, dropped content, wrong facts | Awkward phrasing (that's `mt_fluency`'s job) |
| `mt_fluency` | Does the English *read naturally*? | Grammatical errors, unnatural phrasing | Being inaccurate but fluent |
| `e2e_adequacy` | Did the meaning survive all the way through TTS and back? | Meaning lost/garbled between the MT text and the synthesized audio | Spelling/punctuation quirks introduced by the back-transcriber itself |

The judge prompt is **bias-blind**: it never tells the model which impl produced the
output, only the raw texts — so it can't be swayed by an impl's name or reputation
(`build_judge_prompt`'s own docstring: "no impl identity in inputs").

### A record's judge score is always in one of four states

This is the part that trips people up, because all four states can render as
`0.000` in the leaderboard if you only look at the numbers:

| `verdict` | When it happens | The three floats |
|---|---|---|
| `"pipeline-error"` | The impl itself failed on this utterance (`VoiceResult.error is not None`). No judge call is made. | All `0.0` (pydantic default) |
| `"skipped"` | You ran `score --no-judge`. No judge call is made. | All `0.0` (pydantic default) |
| `"error"` (with `.error` set) | The judge call itself failed — CLI/API error, timeout, or a response the lenient JSON parser couldn't extract three floats from. | All `0.0` (default; the LLM's opinion, if any, is discarded) |
| *(a real short string, e.g. `"mostly faithful"`)* | The judge ran successfully. | Real floats from the model's JSON response |

**Only the third row (`verdict="error"`) is excluded from the leaderboard's judge-score
means.** `"skipped"` and `"pipeline-error"` are *included* as literal zeros — so a
`--no-judge` run's `mt_adequacy`/`mt_fluency`/`e2e_adequacy` of `0.000` means "never
measured," not "measured and scored zero." Check the `verdict` field (or the
worst-utterances appendix's `error:` line) to tell them apart — the leaderboard table
itself doesn't show `verdict`, so the zero looks identical either way.

### The `e2e_adequacy` force-zero rule

Even when the judge *did* run successfully, `e2e_adequacy` is still forced to `0.0`
after the fact if back-transcription failed or was skipped for that record
(`back_error is not None`):

```python
if back_error is not None and judge.error is None:
    # Output audio unusable: e2e is deterministically 0; MT scores stand.
    judge = judge.model_copy(update={"e2e_adequacy": 0.0})
```

`mt_adequacy`/`mt_fluency` are left untouched in this case — they only need the MT
text, not the output audio, so they still stand as real judge scores even when
`e2e_adequacy` is deterministically zeroed.

## 5. Back-transcription — did the meaning survive synthesis?

`voice_suite/scoring/backtranscribe.py` re-transcribes the pipeline's **output** audio
(the synthesized English speech) back to text, using `faster-whisper`'s `large-v3`
model — deliberately a *larger, independent* model than anything under test, so it
"out-hears" the pipeline instead of introducing its own mistakes into the comparison.

- `back_transcript`: what the independent ASR heard in the output audio. This is what
  the judge compares against the reference meaning to produce `e2e_adequacy`.
- `back_error`: why there's no transcript, one of:
  - `"back-transcription skipped"` — you ran `--no-judge` (`transcribe=None`).
  - `"no output audio"` (or the impl's own error string) — the impl produced no audio file for this utterance.
  - the raw exception text — the transcription call itself threw.
- **Cached by audio content hash (SHA-256), not by run identity.** This cache
  (`out/backtranscripts.jsonl`) survives judge/model changes entirely — re-scoring with
  a different `--judge`/`--model` never re-runs back-transcription for audio you've
  already transcribed, since it's a pure function of the audio bytes.
- This is the slowest scoring step (a ~3 GB model on CPU/int8) — the cache is what
  makes repeated `score` runs cheap.

## 6. Latency — `asr_ms` / `mt_ms` / `tts_ms`

Per-utterance wall-clock time for each pipeline stage, reported by the impl adapter
itself (not measured by voice-suite) and stored in `StageTimings`. The leaderboard
shows **P50/P95** per stage, computed as a nearest-rank percentile over every record
where the impl didn't fail (`run_error is None`):

```python
def _pct(sorted_vals: list[int], q: float) -> int:
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]
```

**Small-sample caveat:** with `n=2` (e.g. `--limit 2`), `int(2*0.50)=1` and
`int(2*0.95)=1` — both land on the same index, so P50 and P95 report the *same* value
(the larger of your two samples), not a true median. Don't read anything into
`P50 == P95` until your sample size is large enough for the two quantiles to actually
diverge (roughly `n ≥ 20` before P95 reliably differs from P50).

## 7. Error counts — `errors` vs `judge_errors`

Two different failure counters, easy to conflate:

| Column | Counts | Meaning |
|---|---|---|
| `errors` | `run_error is not None` | The impl/adapter itself failed to produce output for this utterance (crash, timeout, bad exit code). |
| `judge_errors` | `judge.error is not None` | The judge *call* failed (CLI/API error, malformed response) — separate from whether the impl succeeded. |

An impl can have `errors=0` and still show `judge_errors > 0` (impl worked fine, judge
had a bad day) — and vice versa (impl crashed, but that crash is scored as a
deterministic worst case, not a judge error).

## 8. How the leaderboard row is built

`voice_suite/aggregate.py`, grouped by `impl`:

```python
judged = [r.judge for r in recs if r.judge.error is None]   # excludes ONLY verdict="error"
ran    = [r for r in recs if r.run_error is None]

mean_wer      = mean(r.wer for r in recs)                    # every record, no filtering
mt_adequacy   = mean(j.mt_adequacy  for j in judged)
mt_fluency    = mean(j.mt_fluency   for j in judged)
e2e_adequacy  = mean(j.e2e_adequacy for j in judged)
asr_ms_p50/p95, mt_ms_p50/p95, tts_ms_p50/p95 = percentile(timings from `ran`)
errors        = count(r.run_error is not None for r in recs)
judge_errors  = len(recs) - len(judged)

rows.sort(key=lambda r: (-r["e2e_adequacy"], r["mean_wer"]))  # best e2e first, then lowest WER
```

- `n` in the table is just `len(recs)` — how many scored records went into that row.
- The sort (best `e2e_adequacy` first, WER as tiebreak) is what decides
  **`Best implementation`** at the bottom of the report. With only one impl, this sort
  is a no-op — the ranking machinery only matters once you're comparing 2+ impls.
  Since every `--no-judge` run has `e2e_adequacy = 0.0` for all impls, `--no-judge`
  leaderboards effectively rank by WER alone.
- If any records were judge-errored, the report adds a line: `"<impl>: N judge call(s)
  failed and are excluded from judge means."` — this is your signal to distinguish
  "judge scored it low" from "judge broke."

## 9. How "Worst utterances" is picked

```python
ranked = sorted(records, key=lambda r: (r.judge.e2e_adequacy, -r.wer))
```

Ascending `e2e_adequacy` (worst first), then descending `wer` as a tiebreak (so among
equally-bad e2e scores — e.g. every row in a `--no-judge` run, all tied at `0.0` — the
highest-WER utterance surfaces first). Top `WORST_K = 10` shown.

## 10. Caching — what forces a re-score

Every `score` run is idempotent: it skips any `(impl, utterance, output, judge
config)` combination it's already scored. The cache key is a hash over everything
that could change the result:

```python
def score_key(impl, utt_id, result, audio_sha, judge_id=""):
    payload = "\x1f".join([impl, utt_id, result.asr_text, result.mt_text,
                           audio_sha, JUDGE_VERSION, judge_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

A cache **miss** (forcing a real re-score) happens when any of these change:

- `result.asr_text` / `result.mt_text` — you re-ran `run` and the impl produced different text.
- `audio_sha` — the output audio bytes changed.
- `JUDGE_VERSION` (currently `"v1"`, in `judge.py`) — bumped whenever the rubric prompt itself changes.
- `judge_id` — `"no-judge"` for `--no-judge`, otherwise `"<backend>:<model>"` (e.g. `"cli:claude-opus-4-8"`). **This means a `--no-judge` run and a real judge run never share cache entries** — switching from one to the other always re-scores everything, it never reuses the other's rows (correctly, since they're not comparable).

Back-transcription has its own, separate cache keyed purely on `audio_sha256` — it
persists across judge/model/rubric changes since the audio bytes are the same
regardless of who's judging them.

## 11. Where the numbers live

| File | Contents |
|---|---|
| `data/manifest.jsonl` | Ground truth: `id`, `audio_path`, `src_lang`, `dst_lang`, `ref_transcript`, `ref_translation` |
| `out/raw_results.jsonl` | One `RawRecord` per `(impl, utterance)`: `asr_text`, `mt_text`, `audio_path`, `timings`, `error` — pipeline output, pre-scoring |
| `out/scored.jsonl` | One `ScoredRecord` per scored `(impl, utterance, judge_id)`: adds `wer`, `back_transcript`/`back_error`, `judge` (the `JudgeScore`), `cache_key` |
| `out/backtranscripts.jsonl` | `{sha256, transcript}` cache, independent of `score_key` |
| `out/report.md` | The rendered leaderboard + worst-utterances markdown (same text `report` prints to stdout) |

## 12. Flags that decide which metrics you get

| Flag (on `score`) | Effect |
|---|---|
| `--no-judge` | Free, offline: `wer` + latency only. Judge and back-transcription both skipped (`verdict="skipped"`, `back_error="back-transcription skipped"`). |
| `--judge cli` (default) | Judge via your local `claude` CLI login. Needs Claude Code installed and authenticated. |
| `--judge api` | Judge via `ANTHROPIC_API_KEY` + the Anthropic SDK directly. |
| `--model <id>` | Judge model id (default `claude-opus-4-8`); part of `judge_id`, so changing it re-scores everything under the new id. |
| `--workers N` | Parallel judge calls (network-bound); back-transcription always runs sequentially first regardless of `--workers`. |
| `--limit` / `--seed` | Must match the values used on `run` — they select which sampled utterances get scored, not how they're scored. |

## 13. Using a different judge model/provider

Both current backends (`--judge cli` / `--judge api`) call Claude — there is no
built-in support for other providers (GPT/Codex, Qwen, etc.) today. The seam to
add one already exists, though: `get_judge()` returns a `JudgeFn = Callable[[str],
dict]`, and `score_utterance()` only requires that function to accept the rubric
prompt and return a dict with `mt_adequacy`, `mt_fluency`, `e2e_adequacy`,
`verdict`, `rationale` (`voice_suite/scoring/judge.py`).

To add a provider: write one function with that signature (see `_cli_judge` /
`_anthropic_judge` for the shape), and add a branch to `get_judge()` that returns
it for a new `--judge` value.

Two things to know before doing this:

- **Cache is judge-aware already.** `judge_id` (`"<backend>:<model>"`) is part of
  `score_key` (section 10 above), so a new backend gets its own cache entries
  automatically — it can't collide with or silently reuse Claude's cached scores.
- **Scores aren't comparable across judges.** Different models calibrate
  `mt_adequacy`/`mt_fluency`/`e2e_adequacy` differently, so a leaderboard mixing
  rows judged by different backends isn't a fair comparison. Judge everything
  you intend to compare with the same `--judge`/`--model`.
