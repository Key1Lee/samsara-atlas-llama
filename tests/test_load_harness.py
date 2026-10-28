"""Verify harness diagnostics AND real child-process exit codes, not just prints."""
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'load_test.py'


@pytest.fixture
def run_probe(tmp_path):
    # Deliberately omit unique constraints: inject corruption independently of
    # application safeguards and prove the verifier can detect it.
    database = tmp_path / 'probe.db'
    with sqlite3.connect(database) as db:
        db.executescript('''
            CREATE TABLE cases (id INTEGER PRIMARY KEY, idempotency_key TEXT,
                                inputs_json TEXT, status TEXT, outputs_json TEXT);
            CREATE TABLE case_notes (id INTEGER PRIMARY KEY, case_id INTEGER,
                                     employee_id TEXT, note TEXT);
        ''')
    (tmp_path / 'requests.py').write_text((ROOT / 'tests/fixtures/load_probe_api.py').read_text())

    def run(fault='valid', *args):
        env = {**os.environ, 'PYTHONPATH': str(tmp_path), 'PROBE_FAULT': fault,
               'PROBE_DATABASE': str(database)}
        # No live API, credentials, model or persistent app database is used.
        result = subprocess.run([sys.executable, str(SCRIPT), '--database', str(database),
                                 '--requests', '3', '--workers', '2', '--expected-notes', '1',
                                 *args], env=env, capture_output=True, text=True, timeout=20)
        return result
    return run


def test_all_expected_records_pass(run_probe):
    result = run_probe()
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS' in result.stdout


@pytest.mark.parametrize('fault,diagnostic', [
    ('missing_case', 'persisted case count'),
    ('duplicate_case', 'persisted case count'),
    ('response_not_persisted', 'response case IDs'),
    ('outputs_mismatch', 'saved outputs'),
    ('warning_only', 'HTTP 500'),
    ('post_exception', 'RuntimeError'),
    ('post_assertion', 'AssertionError'),
    ('detail_exception', 'RuntimeError'),
    ('case_notes_duplicate', 'local note count'),
    ('note_missing', 'local note count'),
    ('wrong_employee', 'saved inputs'),
    ('incomplete_case', 'not completed'),
    ('missing_from_history', 'employee history'),
    ('note_content_mismatch', 'API local notes'),
    ('unexpected_rejection', 'HTTP 422'),
])
def test_integrity_failures_exit_nonzero(run_probe, fault, diagnostic):
    result = run_probe(fault)
    assert result.returncode == 1, result.stdout + result.stderr
    assert diagnostic in result.stdout
    assert 'FAIL' in result.stdout
    assert 'private exception content' not in result.stdout + result.stderr


def test_explicit_expected_rejection_passes(run_probe):
    result = run_probe('expected_rejection', '--expect-rejection', '422')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS' in result.stdout


def test_expected_quota_rejection_does_not_require_quota_exempt_get(run_probe):
    result = run_probe('expected_quota', '--expect-rejection', '429')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'expected HTTP 429 rejections; 0 saved cases' in result.stdout


def test_expected_rejection_still_fails_on_persistence(run_probe):
    result = run_probe('expected_rejection_persisted', '--expect-rejection', '422')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'persisted case count: expected 0, found 1' in result.stdout


def test_independent_identical_questions_remain_separate(run_probe):
    result = run_probe('valid', '--independent')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '3 intended submissions' in result.stdout


@pytest.mark.parametrize('args,expected_cases,expected_notes', [
    ([], 1, 1),  # Same-key retries, including legitimate running responses.
    (['--independent'], 3, 3),  # Matching questions are intentional separate cases.
    (['--question', 'How much PTO can I carry over to next year?', '--expected-notes', '0'], 1, 0),
    (['--employee-id', 'MISSING', '--expect-rejection', '422'], 0, 0),
    (['--expect-rejection', '429'], 0, 0),
])
def test_probe_checks_real_api_and_sqlite(tmp_path, args, expected_cases, expected_notes):
    import hashlib
    import json
    database = tmp_path / 'application.db'
    (tmp_path / 'requests.py').write_text((ROOT / 'tests/fixtures/load_probe_app.py').read_text())
    employee = 'MISSING' if '422' in args else 'E10234'
    token = 'synthetic-probe-token'
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(tmp_path), str(ROOT)]),
           'DATABASE_URL': f'sqlite:///{database}', 'ATLAS_ENV': 'production',
           'ATLAS_EMPLOYEE_TOKEN': token, 'ATLAS_EMPLOYEE_TOKEN_HASHES': json.dumps(
               {employee: hashlib.sha256(token.encode()).hexdigest()}),
           'PROBE_INITIAL_QUOTA': '200' if '429' in args else ''}
    result = subprocess.run([sys.executable, str(SCRIPT), '--database', str(database),
                             '--requests', '3', '--workers', '2', *args], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS' in result.stdout
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT count(*) FROM cases').fetchone()[0] == expected_cases
        assert db.execute('SELECT count(*) FROM case_notes').fetchone()[0] == expected_notes
        assert not db.execute("SELECT id FROM cases WHERE status != 'completed'").fetchall()
        assert all(json.loads(row[0])['employee_id'] == employee
                   for row in db.execute('SELECT inputs_json FROM cases'))
