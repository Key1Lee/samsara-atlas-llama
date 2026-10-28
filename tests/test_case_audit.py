"""BR-18 request correlation and truthful transactional event history."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.main import app
from app.storage.database import Base
from app.storage.models import Case, CaseNote, EmployeeRecord, Step


@pytest.fixture
def audit_store(tmp_path, monkeypatch):
    from app.api import routes
    from app.engine import runner
    from app.middleware import rate_limit
    from app.storage import database
    from app.tools import hris
    engine = create_engine(f'sqlite:///{tmp_path / "audit.db"}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (database, routes, runner, rate_limit, hris):
        monkeypatch.setattr(module, 'SessionLocal', factory)
    tokens = {employee: uuid4().hex for employee in ('AUDIT-A', 'AUDIT-B')}
    monkeypatch.setenv('ATLAS_ENV', 'production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({
        employee: hashlib.sha256(token.encode()).hexdigest() for employee, token in tokens.items()}))
    with factory() as db:
        db.add_all([EmployeeRecord(employee_id=employee, metadata_json='{}', last_activities_json='[]')
                    for employee in tokens])
        db.commit()
    yield factory, {employee: {'Authorization': 'Bearer ' + token} for employee, token in tokens.items()}
    engine.dispose()


def submit(client, headers, employee='AUDIT-A', key=None, question='How does PTO accrue?'):
    return client.post('/cases', headers=headers[employee], json={
        'inputs': {'employee_id': employee, 'case_transcript': question, 'tone': 'direct'},
        'idempotency_key': key or uuid4().hex})


def audit(client, headers, case_id, employee='AUDIT-A'):
    response = client.get(f'/cases/{case_id}/audit', headers=headers[employee])
    assert response.status_code == 200, response.text
    return response.json()


def test_successful_submission_has_ordered_committed_lifecycle(audit_store):
    factory, headers = audit_store
    client = TestClient(app)
    result = submit(client, headers)
    assert result.status_code == 200
    request_id = result.headers.get('X-Request-ID')
    assert request_id and len(request_id) == 32
    case_id = result.json()['case_id']
    trail = audit(client, headers, case_id)
    assert trail['case_status'] == 'completed'
    events = trail['events']
    assert [row['event_type'] for row in events] == [
        'case_created', 'step_started', 'step_completed', 'step_started', 'step_completed',
        'step_started', 'step_completed', 'step_started', 'step_completed',
        'step_started', 'note_saved', 'step_completed', 'case_completed']
    assert all(row['request_id'] == request_id and row['case_id'] == case_id for row in events)
    assert all(row['timestamp'].endswith('Z') for row in events)
    assert [row['event_id'] for row in events] == sorted(row['event_id'] for row in events)
    assert events[-1]['outcome'] == 'success'
    with factory() as db:
        assert db.get(Case, case_id).status == 'completed'
        note = db.query(CaseNote).filter_by(case_id=case_id).one()
        assert trail['case_note_ids'] == [note.id]
        assert next(row for row in events if row['event_type'] == 'note_saved')['note_id'] == note.id


def test_independent_requests_and_same_key_retry_have_distinct_request_ids(audit_store):
    factory, headers = audit_store
    client = TestClient(app)
    key = uuid4().hex
    first = submit(client, headers, key=key)
    retry = submit(client, headers, key=key)
    other = submit(client, headers)
    assert first.json() == retry.json()  # Preserve the idempotent response contract.
    assert other.json()['case_id'] != first.json()['case_id']
    assert len({r.headers['X-Request-ID'] for r in (first, retry, other)}) == 3
    trail = audit(client, headers, first.json()['case_id'])
    reused = trail['events'][-1]
    assert reused['event_type'] == 'request_reused'
    assert reused['request_id'] == retry.headers['X-Request-ID']
    assert len([row for row in trail['events'] if row['event_type'] == 'note_saved']) == 1
    with factory() as db:
        assert db.query(Case).count() == 2


def test_concurrent_requests_do_not_mix_request_context(audit_store, monkeypatch):
    from app.engine import runner
    factory, headers = audit_store
    barrier = Barrier(2)
    original = runner.STEPS[0][1]

    def overlap(ctx):
        barrier.wait(timeout=10)
        return original(ctx)

    monkeypatch.setattr(runner, 'STEPS', [(runner.STEPS[0][0], overlap), *runner.STEPS[1:]])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda employee: submit(TestClient(app), headers, employee), ['AUDIT-A', 'AUDIT-B']))
    assert all(response.status_code == 200 for response in results)
    assert len({r.headers['X-Request-ID'] for r in results}) == 2
    for employee, result in zip(('AUDIT-A', 'AUDIT-B'), results):
        rows = audit(TestClient(app), headers, result.json()['case_id'], employee)['events']
        assert {row['request_id'] for row in rows} == {result.headers['X-Request-ID']}
    from app.audit import current_request_id
    assert current_request_id.get() is None


def test_failure_then_retry_keeps_prior_failure_and_note_receipt(audit_store, monkeypatch):
    from app.tools.registry import get_tool_registry
    factory, headers = audit_store
    registry = get_tool_registry()
    original = registry.get('post_case_note_to_hris')
    first_call = True

    def saved_then_lost(**kwargs):
        nonlocal first_call
        result = original(**kwargs)
        if first_call:
            first_call = False
            raise RuntimeError('PRIVATE lost acknowledgement')
        return result

    monkeypatch.setitem(registry._tools, 'post_case_note_to_hris', saved_then_lost)
    client = TestClient(app)
    key = uuid4().hex
    first = submit(client, headers, key=key)
    assert first.status_code == 500
    case_id = client.get('/cases', headers=headers['AUDIT-A']).json()['case_ids'][0]
    failed = audit(client, headers, case_id)
    assert failed['case_status'] == 'failed'
    assert failed['events'][-1]['event_type'] == 'case_failed'
    assert failed['events'][-1]['error_code'] == 'workflow_error'
    note_event = next(row for row in failed['events'] if row['event_type'] == 'note_saved')
    assert note_event['outcome'] == 'success'
    retried = submit(client, headers, key=key)
    assert retried.status_code == 200 and retried.json()['case_id'] == case_id
    completed = audit(client, headers, case_id)
    assert sum(row['event_type'] == 'note_saved' for row in completed['events']) == 1
    assert any(row['event_type'] == 'step_failed' and row['attempt'] == 1 for row in completed['events'])
    assert any(row['event_type'] == 'step_completed' and row['attempt'] == 2 for row in completed['events'])
    assert note_event in completed['events']
    with factory() as db:
        assert db.query(CaseNote).count() == 1


def test_initial_commit_failure_has_no_case_or_success_event(audit_store, monkeypatch):
    from app.storage.models import CaseAuditEvent
    factory, headers = audit_store
    original = Session.commit

    def fail_after_flush(db):
        if any(isinstance(row, CaseAuditEvent) and row.event_type == 'case_created' for row in db.new):
            db.flush()
            raise OperationalError('PRIVATE initial write failure', {}, RuntimeError())
        return original(db)

    monkeypatch.setattr(Session, 'commit', fail_after_flush)
    response = submit(TestClient(app), headers)
    assert response.status_code == 500
    assert response.headers['X-Request-ID']
    with factory() as db:
        assert db.query(Case).count() == db.query(CaseAuditEvent).count() == 0


@pytest.mark.parametrize('failed_event,completed_steps', [('step_completed', 0), ('note_saved', 4)])
def test_mandatory_audit_failure_rolls_back_step_success(audit_store, failed_event, completed_steps):
    from app.storage.models import CaseAuditEvent
    factory, headers = audit_store
    failed_once = False

    def reject_event(db, *_):
        nonlocal failed_once
        if not failed_once and any(isinstance(row, CaseAuditEvent) and row.event_type == failed_event
                                   for row in db.new):
            failed_once = True
            raise OperationalError('PRIVATE audit unavailable', {}, RuntimeError())

    event.listen(Session, 'before_flush', reject_event)
    try:
        client = TestClient(app)
        response = submit(client, headers)
        assert response.status_code == 500
        case_id = client.get('/cases', headers=headers['AUDIT-A']).json()['case_ids'][0]
        trail = audit(client, headers, case_id)
        assert trail['case_status'] == 'failed'
        assert sum(row['event_type'] == 'step_completed' for row in trail['events']) == completed_steps
        assert not any(row['event_type'] in ('note_saved', 'case_completed') for row in trail['events'])
        assert trail['events'][-1]['error_code'] == 'storage_error'
        with factory() as db:
            assert db.query(Step).filter_by(case_id=case_id, status='failed').count() == 1
            assert db.query(CaseNote).count() == 0
    finally:
        event.remove(Session, 'before_flush', reject_event)


def test_optional_diagnostic_error_does_not_replace_workflow_error(audit_store, monkeypatch):
    from app.engine import runner
    factory, headers = audit_store

    def failed_step(ctx):
        raise ValueError('PRIVATE invalid operation')

    def failed_log(*args, **kwargs):
        raise OSError('PRIVATE log unavailable')

    monkeypatch.setattr(runner, 'STEPS', [('synthetic_failure', failed_step)])
    monkeypatch.setattr(runner.logger, 'error', failed_log)
    with pytest.raises(ValueError, match='PRIVATE invalid operation'):
        runner.run_workflow({'employee_id': 'AUDIT-A', 'case_transcript': 'How does PTO accrue?', 'tone': 'direct'})
    with factory() as db:
        assert db.query(Case).one().status == 'failed'
        from app.storage.models import CaseAuditEvent
        assert all(row.request_id is None for row in db.query(CaseAuditEvent))


def test_failed_final_case_commit_cannot_record_completion(audit_store):
    from app.storage.models import CaseAuditEvent
    factory, headers = audit_store
    failed_once = False

    def fail_completion(db, *_):
        nonlocal failed_once
        if not failed_once and any(isinstance(row, CaseAuditEvent) and row.event_type == 'case_completed'
                                   for row in db.new):
            failed_once = True
            raise OperationalError('PRIVATE final commit failed', {}, RuntimeError())

    event.listen(Session, 'before_flush', fail_completion)
    try:
        client = TestClient(app)
        response = submit(client, headers)
        assert response.status_code == 500
        case_id = client.get('/cases', headers=headers['AUDIT-A']).json()['case_ids'][0]
        record = audit(client, headers, case_id)
        assert record['case_status'] == 'failed'
        assert not any(row['event_type'] == 'case_completed' for row in record['events'])
        assert record['events'][-1]['event_type'] == 'case_failed'
        assert record['events'][-1]['error_code'] == 'storage_error'
        # Previously committed step/note successes remain truthful.
        assert sum(row['event_type'] == 'step_completed' for row in record['events']) == 5
        assert not any(row['event_type'] == 'step_failed' for row in record['events'])
        assert len(record['case_note_ids']) == 1
        with factory() as db:
            assert db.query(Step).filter_by(case_id=case_id, status='completed').count() == 5
    finally:
        event.remove(Session, 'before_flush', fail_completion)
