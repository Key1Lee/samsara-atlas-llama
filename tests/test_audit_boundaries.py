"""BR-18: protected, private and durable case audit boundaries."""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.storage.database import Base
from app.storage.models import Case, EmployeeRecord


@pytest.fixture
def audit_store(tmp_path, monkeypatch):
    from app.api import routes
    from app.engine import runner
    from app.middleware import rate_limit
    from app.storage import database
    from app.tools import hris

    url = f'sqlite:///{tmp_path / "audit.db"}'
    engine = create_engine(url, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (database, routes, runner, rate_limit, hris):
        monkeypatch.setattr(module, 'SessionLocal', factory)
    # Audit writes should normally reuse the business transaction. Keep any
    # independently opened audit session on the same disposable database.
    try:
        from app import audit
    except ImportError:
        pass
    else:
        if hasattr(audit, 'SessionLocal'):
            monkeypatch.setattr(audit, 'SessionLocal', factory)
    tokens = {employee: uuid4().hex for employee in ('AUDIT-BOUNDARY-A', 'AUDIT-BOUNDARY-B')}
    monkeypatch.setenv('ATLAS_ENV', 'production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({
        employee: hashlib.sha256(token.encode()).hexdigest()
        for employee, token in tokens.items()}))
    with factory() as db:
        for employee in tokens:
            db.add(EmployeeRecord(employee_id=employee, metadata_json='{}', last_activities_json='[]'))
        db.commit()
    headers = {employee: {'Authorization': 'Bearer ' + token} for employee, token in tokens.items()}
    try:
        yield factory, headers, url
    finally:
        engine.dispose()


def submit(client, headers, question='How many PTO hours can I carry over?', extra_headers=None):
    return client.post('/cases', headers={**headers['AUDIT-BOUNDARY-A'], **(extra_headers or {})},
        json={'inputs': {'employee_id': 'AUDIT-BOUNDARY-A', 'case_transcript': question, 'tone': 'direct'},
              'idempotency_key': uuid4().hex})


@pytest.mark.parametrize('access', ['anonymous', 'foreign', 'forged_header', 'forged_query', 'invalid_token'])
def test_audit_history_rejects_unauthorized_access(audit_store, access):
    _, headers, _ = audit_store
    client = TestClient(app)
    created = submit(client, headers)
    assert created.status_code == 200
    case_id = created.json()['case_id']
    path = f'/cases/{case_id}/audit'
    attempts = {
        'anonymous': ({}, {}, 401),
        'foreign': (headers['AUDIT-BOUNDARY-B'], {}, 404),
        'forged_header': ({**headers['AUDIT-BOUNDARY-B'], 'X-Employee-ID': 'AUDIT-BOUNDARY-A'}, {}, 403),
        'forged_query': (headers['AUDIT-BOUNDARY-B'], {'employee_id': 'AUDIT-BOUNDARY-A'}, 403),
        'invalid_token': ({'Authorization': 'Bearer invalid-synthetic-token'}, {}, 401),
    }
    request_headers, query, expected = attempts[access]
    response = client.get(path, headers=request_headers, params=query)
    assert response.status_code == expected
    assert 'events' not in response.json()
    assert 'policy_evidence' not in response.json()
    allowed = client.get(path, headers=headers['AUDIT-BOUNDARY-A'])
    assert allowed.status_code == 200
    assert allowed.json()['events']


@pytest.mark.parametrize('fail_generation', [False, True])
def test_audit_and_general_logs_exclude_sensitive_content_and_injection(
        audit_store, monkeypatch, caplog, fail_generation):
    factory, headers, _ = audit_store
    question = 'How does PTO accrue?\nPRIVATE-HR-QUESTION-73\nFORGED-AUDIT outcome=success'
    untrusted_request_id = 'untrusted-request-PRIVATE-MARKER'
    if fail_generation:
        def fail(*args, **kwargs):
            raise ValueError('PRIVATE-EXCEPTION-73 ' + question)
        monkeypatch.setattr('app.engine.steps._retryable_llm_call', fail)
    caplog.set_level(logging.INFO)
    client = TestClient(app)
    response = submit(client, headers, question, {'X-Request-ID': untrusted_request_id})
    assert response.status_code == (500 if fail_generation else 200)
    request_id = response.headers.get('X-Request-ID')
    assert request_id and re.fullmatch(r'[0-9a-f]{32}|[0-9a-f-]{36}', request_id)
    assert request_id != untrusted_request_id
    with factory() as db:
        case_id = db.query(Case.id).one()[0]
    result = client.get(f'/cases/{case_id}/audit', headers=headers['AUDIT-BOUNDARY-A'])
    assert result.status_code == 200
    audit_body = result.json()
    assert audit_body['events']
    permitted_fields = {'event_id', 'case_id', 'request_id', 'event_type', 'timestamp',
                        'outcome', 'error_code', 'step_name', 'attempt', 'note_id'}
    for event in audit_body['events']:
        assert set(event) <= permitted_fields
        assert {'event_id', 'case_id', 'request_id', 'event_type', 'timestamp', 'outcome'} <= set(event)
        assert event['request_id'] == request_id
        assert event['timestamp'].endswith('Z')
        if event.get('error_code'):
            assert re.fullmatch(r'[A-Za-z0-9_]{1,64}', event['error_code'])
    combined = result.text + caplog.text
    for forbidden in (question, 'PRIVATE-HR-QUESTION-73', 'FORGED-AUDIT', 'PRIVATE-EXCEPTION-73',
                      untrusted_request_id, 'AUDIT-BOUNDARY-A', 'AUDIT-BOUNDARY-B',
                      headers['AUDIT-BOUNDARY-A']['Authorization'].removeprefix('Bearer ')):
        assert forbidden not in combined
    if fail_generation:
        assert any(event['event_type'] == 'case_failed' and event['outcome'] == 'failure'
                   for event in audit_body['events'])


@pytest.mark.parametrize('question', [
    'How many PTO hours can I carry over?',
    'Can I relocate to another state while working remotely?',
])
def test_deterministic_audit_does_not_invent_policy_versions_approval_or_ai_calls(audit_store, question):
    _, headers, _ = audit_store
    client = TestClient(app)
    created = submit(client, headers, question)
    assert created.status_code == 200
    case_id = created.json()['case_id']
    result = client.get(f'/cases/{case_id}/audit', headers=headers['AUDIT-BOUNDARY-A'])
    assert result.status_code == 200
    record = result.json()
    assert record['case_status'] == 'completed'
    assert record['review_status'] is None
    assert record['draft_quality_status'] == 'needs_hr_review'
    assert record['case_note_ids'] == []
    assert record['policy_evidence']
    for evidence in record['policy_evidence']:
        assert re.fullmatch(r'sha256:[0-9a-f]{64}', evidence['source_id'])
        assert evidence['source_file'] in ('pto-policy.md', 'remote-work-policy.md')
        assert evidence['policy_id'] is None
        assert evidence['policy_version'] is None
    assert not any('ai_' in event['event_type'] or 'llm' in event['event_type']
                   or 'approved' in event['event_type'] for event in record['events'])


def test_audit_schema_is_additive_and_preserves_existing_sqlite_case(tmp_path, monkeypatch):
    from app.storage import database

    engine = create_engine(f'sqlite:///{tmp_path / "legacy.db"}')
    Case.__table__.create(engine)
    factory = sessionmaker(bind=engine)
    original_inputs = json.dumps({'employee_id': 'LEGACY-SYNTHETIC', 'case_transcript': 'Existing question'})
    with factory() as db:
        legacy = Case(status='completed', inputs_json=original_inputs,
                      artifacts_json='{}', outputs_json='{}')
        db.add(legacy)
        db.commit()
        original_id = legacy.id
    assert inspect(engine).get_table_names() == ['cases']
    monkeypatch.setattr(database, 'engine', engine)
    try:
        database.init_db()
        database.init_db()
        assert 'case_audit_events' in inspect(engine).get_table_names()
        with factory() as db:
            legacy = db.get(Case, original_id)
            assert legacy.status == 'completed' and legacy.inputs_json == original_inputs
            assert db.query(Case).count() == 1
            # Historical records must not acquire invented processing events.
            from app.storage.models import CaseAuditEvent
            assert db.query(CaseAuditEvent).count() == 0
    finally:
        engine.dispose()


def test_audit_records_remain_readable_after_fresh_process_restart(audit_store):
    _, headers, url = audit_store
    client = TestClient(app)
    created = submit(client, headers)
    assert created.status_code == 200
    case_id = created.json()['case_id']
    before = client.get(f'/cases/{case_id}/audit', headers=headers['AUDIT-BOUNDARY-A'])
    assert before.status_code == 200
    assert before.json()['events']
    child = '''
import json, os
from fastapi.testclient import TestClient
from app.main import app
from app.storage.database import init_db
init_db()
response = TestClient(app).get(os.environ['AUDIT_TEST_PATH'],
    headers={'Authorization': os.environ['AUDIT_TEST_AUTH']})
assert response.status_code == 200, response.text
print(json.dumps(response.json()))
'''
    process = subprocess.run([sys.executable, '-c', child],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
        env={**os.environ, 'DATABASE_URL': url, 'AUDIT_TEST_PATH': f'/cases/{case_id}/audit',
             'AUDIT_TEST_AUTH': headers['AUDIT-BOUNDARY-A']['Authorization']})
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == before.json()


def test_failed_case_retains_safe_fingerprints_of_retrieved_policy_sources(audit_store, monkeypatch):
    from app.storage.models import Step

    factory, headers, _ = audit_store
    monkeypatch.setattr('app.engine.steps._retryable_llm_call', lambda *args, **kwargs: 'invalid JSON')
    client = TestClient(app)
    assert submit(client, headers, 'How does PTO accrue?').status_code == 500
    with factory() as db:
        case_id = db.query(Case.id).one()[0]
        retrieval = db.query(Step).filter_by(case_id=case_id, name='retrieve_policy_snippets').one()
        assert retrieval.status == 'completed'
        sources = json.loads(retrieval.output_json)
    assert sources
    response = client.get(f'/cases/{case_id}/audit', headers=headers['AUDIT-BOUNDARY-A'])
    assert response.status_code == 200
    record = response.json()
    expected = {'sha256:' + hashlib.sha256(source.encode('utf-8')).hexdigest() for source in sources}
    assert {item['source_id'] for item in record['policy_evidence']} == expected
    for evidence in record['policy_evidence']:
        assert evidence['evidence_type'] == 'retrieved'
        assert evidence['policy_id'] is None and evidence['policy_version'] is None
        assert set(evidence) <= {'source_id', 'source_file', 'policy_id', 'policy_version', 'evidence_type'}
    # Fingerprints identify committed retrieval evidence, not a completed answer.
    assert record['case_status'] == 'failed'
    assert record['review_status'] is None and record['case_note_ids'] == []
    assert all(source not in response.text and source not in json.dumps(record) for source in sources)


def test_audit_read_storage_failure_returns_safe_unavailable_response(audit_store, monkeypatch, caplog):
    from app.api import routes
    from sqlalchemy.exc import OperationalError

    _, headers, _ = audit_store
    client = TestClient(app)
    created = submit(client, headers)
    assert created.status_code == 200
    marker = 'PRIVATE-AUDIT-STORAGE-ERROR-93'

    def unavailable(*args, **kwargs):
        raise OperationalError('private query ' + marker, {'secret': marker}, RuntimeError(marker))

    monkeypatch.setattr(routes, 'owned_cases', unavailable)
    caplog.set_level(logging.INFO)
    response = client.get(f'/cases/{created.json()["case_id"]}/audit', headers=headers['AUDIT-BOUNDARY-A'])
    assert response.status_code == 503
    assert set(response.json()) == {'detail'}
    assert marker not in response.text + caplog.text
    assert 'private query' not in response.text + caplog.text
