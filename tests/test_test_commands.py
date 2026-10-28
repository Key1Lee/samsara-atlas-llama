"""Command failures must survive the real pytest and Makefile entry points."""
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_pytest_empty_collection_fails(tmp_path):
    empty = tmp_path / "empty_tests"
    empty.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(empty), "-q"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 5, result.stdout + result.stderr
    assert "no tests ran" in result.stdout + result.stderr


def test_make_load_propagates_script_failure(tmp_path):
    shutil.copyfile(ROOT / "Makefile", tmp_path / "Makefile")
    binaries = tmp_path / ".venv" / "bin"
    binaries.mkdir(parents=True)
    (binaries / "python").symlink_to(sys.executable)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "load_test.py").write_text("raise SystemExit(7)\n")

    result = subprocess.run(
        ["make", "load"], cwd=tmp_path,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Error 7" in result.stderr


def test_make_test_propagates_pytest_empty_collection_exit(tmp_path):
    shutil.copyfile(ROOT / "Makefile", tmp_path / "Makefile")
    binaries = tmp_path / ".venv" / "bin"
    binaries.mkdir(parents=True)
    pytest = binaries / "pytest"
    pytest.write_text("#!/bin/sh\nexit 5\n")
    pytest.chmod(0o755)

    result = subprocess.run(
        ["make", "test"], cwd=tmp_path,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Error 5" in result.stderr
