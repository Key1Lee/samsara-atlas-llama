"""Recovery authorization, unknown outcomes and non-keyed case preservation."""
from contextlib import contextmanager
import hashlib
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.storage.database import Base
from app.storage.models import Case, CaseAuditEvent, CaseNote, EmployeeRecord, Step

INPUTS = {'employee_id': 'RECOVERY-A', 'case_transcript': 'How does PTO accrue?', 'tone': 'direct'}


@pytest.fixture
def store(tmp_path, monkeypatch):
    from app.api import routes
    from app.engine import runner
    from app.middleware import rate_limit
    from app.storage import database
    from app.tools import hris
    engine = create_engine(f'sqlite:///{tmp_path / "boundaries.db"}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (routes, runner, rate_limit, database, hris):
        monkeypatch.setattr(module, 'SessionLocal', factory)
    tokens = {employee: uuid4().hex for employee in ('RECOVERY-A', 'RECOVERY-B')}
    monkeypatch.setenv('ATLAS_ENV', 'production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({
        employee: hashlib.sha256(token.encode()).hexdigest() for employee, token in tokens.items()}))
    with factory() as db:
        db.add_all([EmployeeRecord(employee_id=employee, metadata_json='{}', last_activities_json='[]')
                    for employee in tokens])
        db.commit()
    yield factory, {employee: {'Authorization': 'Bearer ' + token} for employee, token in tokens.items()}
    engine.dispose()


def create(client, headers, key=None):
    return client.post('/cases', headers=headers['RECOVERY-A'], json={'inputs': INPUTS, 'idempotency_key': key})


def test_case_without_key_recovers_by_authorized_id_without_repeating_confirmed_write(store, monkeypatch):
    from app.engine import runner
    from app.tools.registry import get_tool_registry
    factory, headers = store
    registry = get_tool_registry()
    original = registry.get('post_case_note_to_hris')
    calls = []

    class Interrupted(BaseException):
        pass

    def saved_then_interrupted(**kwargs):
        calls.append(kwargs['case_id'])
        original(**kwargs)
        raise Interrupted()

    monkeypatch.setitem(registry._tools, 'post_case_note_to_hris', saved_then_interrupted)
    with pytest.raises(Interrupted):
        runner.run_workflow(INPUTS)  # Deliberately no idempotency key.
    with factory() as db:
        case = db.query(Case).one()
        case_id = case.id
        assert case.status == 'running' and case.idempotency_key is None
        original_note = db.query(CaseNote).one().id
    client = TestClient(app)
    monkeypatch.setattr(runner, 'retrieve_policy_snippets', lambda *_: [])
    recovered = client.post(f'/cases/{case_id}/recover', headers=headers['RECOVERY-A'])
    assert recovered.status_code == 200 and recovered.json()['status'] == 'completed'
    assert recovered.json()['case_id'] == case_id and calls == [case_id]
    assert client.post(f'/cases/{case_id}/recover', headers=headers['RECOVERY-A']).json() == recovered.json()
    with factory() as db:
        assert db.query(Case).count() == 1
        assert db.query(CaseNote).one().id == original_note
        assert db.query(CaseAuditEvent).filter_by(event_type='note_saved').count() == 1
        assert db.query(Step).filter_by(status='completed').count() == 5


@pytest.mark.parametrize('actor,selection,expected', [
    (None, None, 401), ('RECOVERY-B', None, 404),
    ('RECOVERY-B', 'RECOVERY-A', 403),
])
def test_recovery_endpoint_denies_unauthorized_callers_without_mutation(store, actor, selection, expected):
    factory, headers = store
    client = TestClient(app)
    created = create(client, headers).json()
    with factory() as db:
        before = (db.query(Case).count(), db.query(CaseNote).count(), db.query(CaseAuditEvent).count())
    attempt_headers = dict(headers[actor]) if actor else {}
    if selection:
        attempt_headers['X-Employee-ID'] = selection
    response = client.post(f'/cases/{created["case_id"]}/recover', headers=attempt_headers)
    assert response.status_code == expected and 'outputs' not in response.json()
    with factory() as db:
        assert before == (db.query(Case).count(), db.query(CaseNote).count(), db.query(CaseAuditEvent).count())


def test_failed_unknown_write_is_not_retried_just_because_exception_was_caught(store, monkeypatch):
    from app.tools.registry import get_tool_registry
    factory, headers = store
    calls = []

    def unknown_write(**kwargs):
        calls.append(kwargs['case_id'])
        raise RuntimeError('PRIVATE remote outcome unknown')

    monkeypatch.setitem(get_tool_registry()._tools, 'post_case_note_to_hris', unknown_write)
    client = TestClient(app)
    key = uuid4().hex
    assert create(client, headers, key).status_code == 500
    with factory() as db:
        case_id = db.query(Case).one().id
    for _ in range(2):
        response = create(client, headers, key)
        assert response.status_code == 200 and response.json()['status'] == 'failed'
        assert response.json()['recovery'] == 'manual_reconciliation_required'
        assert response.json()['recovery_reason'] == 'note_outcome_unknown'
    assert calls == [case_id]
    with factory() as db:
        assert db.query(CaseNote).count() == 0
        assert db.query(Step).filter_by(name='post_case_note_to_hris').one().attempts == 1
        assert db.query(CaseAuditEvent).filter_by(event_type='case_completed').count() == 0


@pytest.mark.parametrize('field,value', [('employee_id', 'RECOVERY-B'), ('note', 'Incorrect saved note')])
def test_mismatched_local_note_requires_manual_reconciliation(store, field, value):
    factory, headers = store
    client = TestClient(app)
    key = uuid4().hex
    case_id = create(client, headers, key).json()['case_id']
    with factory() as db:
        db.get(Case, case_id).status = 'failed'
        db.query(Step).filter_by(case_id=case_id, name='post_case_note_to_hris').one().status = 'failed'
        setattr(db.query(CaseNote).one(), field, value)
        db.commit()
    response = client.post(f'/cases/{case_id}/recover', headers=headers['RECOVERY-A'])
    assert response.status_code == 200 and response.json()['status'] == 'failed'
    assert response.json()['recovery_reason'] == 'note_receipt_mismatch'
    with factory() as db:
        assert db.query(CaseNote).count() == 1
        assert db.get(Case, case_id).status == 'failed'


def test_unavailable_execution_guard_does_not_run_a_workflow(store, monkeypatch):
    from app.engine import runner
    from app.engine.recovery import GuardUnavailable
    factory, headers = store

    @contextmanager
    def unavailable(*args):
        raise GuardUnavailable()
        yield

    monkeypatch.setattr(runner, 'execution_guard', unavailable)
    response = create(TestClient(app), headers, uuid4().hex)
    assert response.status_code == 200 and response.json()['status'] == 'running'
    assert response.json()['recovery_reason'] == 'execution_guard_unavailable'
    with factory() as db:
        assert db.query(Case).count() == 1
        assert db.query(Step).count() == db.query(CaseNote).count() == 0
        assert db.query(CaseAuditEvent).filter_by(event_type='case_completed').count() == 0


def test_recovery_revalidates_directory_membership(store):
    factory, headers = store
    client = TestClient(app)
    case_id = create(client, headers).json()['case_id']
    with factory() as db:
        db.query(EmployeeRecord).filter_by(employee_id='RECOVERY-A').delete()
        db.commit()
    response = client.post(f'/cases/{case_id}/recover', headers=headers['RECOVERY-A'])
    assert response.status_code == 422
    with factory() as db:
        assert db.query(CaseNote).count() == 1


def test_recovery_lookup_failure_is_safe_and_does_not_leak_sql(store, monkeypatch, caplog):
    from app.api import routes
    _, headers = store
    client = TestClient(app)
    case_id = create(client, headers).json()['case_id']

    def unavailable(*args):
        raise OperationalError('PRIVATE recovery query', {'employee': 'PRIVATE marker'}, RuntimeError('PRIVATE'))

    monkeypatch.setattr(routes, 'owned_cases', unavailable)
    result = client.post(f'/cases/{case_id}/recover', headers=headers['RECOVERY-A'])
    assert result.status_code == 503
    assert 'PRIVATE' not in result.text + caplog.text
