"""Command-line interface: ingest / run / score / report."""
from __future__ import annotations

import logging
import sys
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

# Reconfigure stdout/stderr to UTF-8 for Windows console compatibility
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

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


@app.command("asr-run")
def asr_run(
    engine: str = typer.Option(
        "sherpa", "--engine",
        help="ASR engine: sherpa (local), openai (OPENAI_API_KEY) or "
             "gemini (GEMINI_API_KEY)."),
    model: Optional[str] = typer.Option(
        None, "--model", help="Cloud model override (e.g. whisper-1)."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Only transcribe a deterministic sample of N."),
    seed: int = typer.Option(0, "--seed", help="Seed for --limit sampling."),
    threads: int = typer.Option(4, "--threads", min=1,
                                help="sherpa-onnx decode threads."),
):
    """ASR-only run over the manifest (no MT/TTS): WER benchmark input."""
    from voice_suite import asr_bench  # heavy sherpa-onnx import
    if engine not in asr_bench.ENGINES:
        raise typer.BadParameter(
            f"unknown engine {engine!r}; choose from {list(asr_bench.ENGINES)}")
    utts = sample_items(load_manifest(config.MANIFEST), limit, seed)
    if engine == "sherpa":
        hw_dir = asr_bench.hotwords_dir_from_config()
        transcribe = asr_bench.get_transcriber(asr_bench.sherpa_model_dir(),
                                               num_threads=threads,
                                               hotwords_dir=hw_dir)
    else:
        transcribe = asr_bench.get_cloud_transcriber(engine, model)
    out_path = asr_bench.results_path(engine)
    n = asr_bench.run_asr(utts, transcribe, out_path, progress=typer.echo)
    typer.echo(f"transcribed {n} new → {out_path}")


@app.command()
def review(
    engine: str = typer.Option("sherpa", "--engine",
                               help="Engine whose transcripts to review."),
    port: int = typer.Option(8765, "--port", min=1, max=65535),
):
    """Open the manual WER review page (accept non-errors per word op)."""
    import webbrowser

    from voice_suite import asr_bench, review as review_mod
    decisions_path = review_mod.decisions_path(engine)
    rows = review_mod.build_rows(
        load_manifest(config.MANIFEST),
        asr_bench.load_asr_results(asr_bench.results_path(engine)),
        review_mod.load_decisions(decisions_path))
    if not rows:
        raise typer.BadParameter(
            f"no ASR results for {engine!r} — run "
            f"`voice-suite asr-run --engine {engine}` first")
    server = review_mod.serve_review(rows, port, decisions_path=decisions_path)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"review page: {url} (Ctrl+C to stop; decisions save to "
               f"{decisions_path})")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


@app.command("wer-report")
def wer_report():
    """Render the WER report across every engine with results in out/."""
    from voice_suite import asr_bench, review as review_mod
    utts = load_manifest(config.MANIFEST)
    engines: dict[str, list[dict]] = {}
    for results_file in sorted(config.OUT_DIR.glob("asr_*.jsonl")):
        name = results_file.stem.removeprefix("asr_")
        engines[name] = review_mod.build_rows(
            utts, asr_bench.load_asr_results(results_file),
            review_mod.load_decisions(review_mod.decisions_path(name)))
    text = review_mod.render_wer_report(engines)
    out = config.OUT_DIR / "wer_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(text)


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
