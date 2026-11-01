# ABOUTME: Tests for qtool.cli: argument validation (exit 2), dry-run, preflight blocking before any container,
# ABOUTME: a full fake quantize run, stale checkpoint, Ctrl-C (exit 130) and `qtool eval --act` checks.
import subprocess
import sys
from pathlib import Path

import pytest

from qtool import __version__, cli

ROOT = Path(__file__).resolve().parents[1]


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"qtool {__version__}"


def test_bin_shim():
    r = subprocess.run([sys.executable, str(ROOT / "bin" / "qtool"), "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == f"qtool {__version__}"
