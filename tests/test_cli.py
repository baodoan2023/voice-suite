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
