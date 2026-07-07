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
