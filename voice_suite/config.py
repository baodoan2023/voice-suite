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
