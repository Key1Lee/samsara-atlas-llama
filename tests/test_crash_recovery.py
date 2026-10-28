"""BR-13: real process death, independent SQLite reads, and safe keyed recovery."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests" / "fixtures" / "crash_case.py"


@pytest.fixture
def crash_store(tmp_path):
    token = uuid4().hex
    filename = tmp_path / "crash.db"
    environment = {**os.environ, "DATABASE_URL": f"sqlite:///{filename}", "PROVIDER_TYPE": "stub",
        "ATLAS_ENV": "production", "ATLAS_EMPLOYEE_TOKEN_HASHES": json.dumps({
            "E10234": hashlib.sha256(token.encode()).hexdigest()}),
        "CRASH_TEST_TOKEN": token, "CRASH_TEST_KEY": uuid4().hex}

    def run(mode, expected_exit=0):
        result = subprocess.run([sys.executable, str(DRIVER), mode], cwd=ROOT,
            env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == expected_exit, result.stdout + result.stderr
        if expected_exit:
            return None
        lines = [line.removeprefix("CRASH_TEST_RESULT=") for line in result.stdout.splitlines()
                 if line.startswith("CRASH_TEST_RESULT=")]
        assert len(lines) == 1, result.stdout + result.stderr
        return json.loads(lines[0])

    def read():
        # This connection is independent of the terminated API process.
        with sqlite3.connect(filename) as db:
            db.row_factory = sqlite3.Row
            return {"cases": [dict(row) for row in db.execute("SELECT * FROM cases ORDER BY id")],
                    "steps": [dict(row) for row in db.execute("SELECT * FROM steps ORDER BY id")],
                    "notes": [dict(row) for row in db.execute("SELECT * FROM case_notes ORDER BY id")],
                    "events": [dict(row) for row in db.execute("SELECT * FROM case_audit_events ORDER BY id")]}

    def start(mode, extra_environment):
        # Keep the inherited environment out of fixture representations and
        # failure diagnostics; only synthetic subprocess configuration escapes.
        return subprocess.Popen([sys.executable, str(DRIVER), mode], cwd=ROOT,
            env={**environment, **extra_environment},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return run, read, filename, start


def test_normal_subprocess_case_completes_and_retry_reuses_it(crash_store):
    run, read, _, _ = crash_store
    first = run("normal")
    assert first["http_status"] == 200 and first["body"]["status"] == "completed"
    assert run("retry")["body"] == first["body"]
    saved = read()
    assert len(saved["cases"]) == len(saved["notes"]) == 1
    assert len(saved["steps"]) == 5 and all(step["status"] == "completed" for step in saved["steps"])
    assert sum(event["event_type"] == "note_saved" for event in saved["events"]) == 1


@pytest.mark.parametrize("boundary,note_count,completed_steps", [
    ("before_note", 0, 4), ("after_note", 1, 4), ("before_final_status", 1, 5),
])
def test_abrupt_crash_recovers_same_case_without_duplicate_note(crash_store, boundary, note_count, completed_steps):
    run, read, _, _ = crash_store
    run(boundary, expected_exit=73)
    interrupted = read()
    assert len(interrupted["cases"]) == 1
    original = interrupted["cases"][0]
    assert original["status"] == "running" and not json.loads(original["outputs_json"] or "{}")
    assert len(interrupted["notes"]) == note_count
    assert sum(step["status"] == "completed" for step in interrupted["steps"]) == completed_steps
    assert not any(event["event_type"] == "case_completed" for event in interrupted["events"])
    original_events = interrupted["events"]

    recovered = run("retry")
    assert recovered["http_status"] == 200
    assert recovered["body"]["case_id"] == original["id"]
    assert recovered["body"]["status"] == "completed"
    saved = read()
    assert len(saved["cases"]) == len(saved["notes"]) == 1
    assert saved["cases"][0]["id"] == original["id"]
    assert saved["cases"][0]["status"] == "completed"
    assert json.loads(saved["cases"][0]["outputs_json"]) == recovered["body"]["outputs"]
    assert all(step["status"] == "completed" for step in saved["steps"])
    assert len(saved["steps"]) == 5
    assert saved["notes"][0]["case_id"] == original["id"]
    assert saved["notes"][0]["employee_id"] == "E10234"
    assert saved["events"][:len(original_events)] == original_events
    assert sum(event["event_type"] == "note_saved" for event in saved["events"]) == 1
    assert sum(event["event_type"] == "case_completed" for event in saved["events"]) == 1
    if note_count:
        assert saved["notes"] == interrupted["notes"]

    assert run("retry")["body"] == recovered["body"]
    repeated = read()
    assert repeated["cases"] == saved["cases"]
    assert repeated["notes"] == saved["notes"]
    assert sum(event["event_type"] == "note_saved" for event in repeated["events"]) == 1


def test_unknown_external_write_requires_manual_reconciliation_without_replay(crash_store):
    run, read, filename, _ = crash_store
    run("unknown_external", expected_exit=73)
    interrupted = read()
    assert len(interrupted["cases"]) == 1 and not interrupted["notes"]
    original_id = interrupted["cases"][0]["id"]
    assert interrupted["cases"][0]["status"] == "running"
    for _ in range(2):
        result = run("retry")
        assert result["http_status"] == 200
        assert result["body"]["case_id"] == original_id
        assert result["body"]["status"] == "running"
        assert result["body"].get("recovery") == "manual_reconciliation_required"
        saved = read()
        assert len(saved["cases"]) == 1 and not saved["notes"]
        assert not any(event["event_type"] == "case_completed" for event in saved["events"])
        with sqlite3.connect(filename) as db:
            assert db.execute("SELECT count(*) FROM synthetic_external_effects").fetchone()[0] == 1


def test_live_process_owner_is_not_mistaken_for_interruption(crash_store, tmp_path):
    run, read, _, start = crash_store
    ready, release = tmp_path / "ready", tmp_path / "release"
    child = start("block_before_note", {"CRASH_TEST_READY": str(ready), "CRASH_TEST_RELEASE": str(release)})
    try:
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert child.poll() is None, child.communicate()
            assert time.monotonic() < deadline, "Owner did not reach its committed note-start boundary"
            time.sleep(0.02)
        active = read()
        original_id = active["cases"][0]["id"]
        assert active["cases"][0]["status"] == "running" and not active["notes"]
        observer = run("retry")
        assert observer["http_status"] == 200
        assert observer["body"]["case_id"] == original_id
        assert observer["body"]["status"] == "running"
        assert not observer["body"].get("recovery")
        assert len(read()["cases"]) == 1 and not read()["notes"]
        release.touch()
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, output + error
        completed = run("retry")
        assert completed["body"]["case_id"] == original_id
        assert completed["body"]["status"] == "completed"
        saved = read()
        assert len(saved["cases"]) == len(saved["notes"]) == 1
        assert sum(event["event_type"] == "note_saved" for event in saved["events"]) == 1
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


def test_two_recovery_processes_preserve_one_case_and_one_note(crash_store):
    run, read, _, _ = crash_store
    run("after_note", expected_exit=73)
    interrupted = read()
    original_id = interrupted["cases"][0]["id"]
    original_note = interrupted["notes"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run("retry"), range(2)))
    assert all(result["http_status"] == 200 for result in results)
    assert {result["body"]["case_id"] for result in results} == {original_id}
    assert {result["body"]["status"] for result in results} <= {"running", "completed"}
    assert run("retry")["body"]["status"] == "completed"
    saved = read()
    assert len(saved["cases"]) == 1 and saved["notes"] == original_note
    assert sum(event["event_type"] == "note_saved" for event in saved["events"]) == 1
    assert sum(event["event_type"] == "case_completed" for event in saved["events"]) == 1
