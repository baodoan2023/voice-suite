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
