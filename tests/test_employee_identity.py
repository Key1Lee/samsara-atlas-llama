"""BR-02: validate the synthetic employee directory before creating business records."""
import hashlib
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.main import app
from app.storage.models import Case, CaseNote, EmployeeRecord, Step


@pytest.fixture
def employee(db_session):
    record = EmployeeRecord(employee_id='IDENTITY-' + uuid4().hex,
                            metadata_json='{}', last_activities_json='[]')
    db_session.add(record)
    db_session.commit()
    return record


def payload(employee_id, question='How many PTO hours can I carry over?'):
    return {'inputs': {'employee_id': employee_id, 'case_transcript': question, 'tone': 'direct'},
            'idempotency_key': uuid4().hex}


def business_records(db):
    return tuple(db.query(model).count() for model in (Case, Step, CaseNote))


@pytest.mark.parametrize('question', [
    'How many PTO hours can I carry over?',
    'Can I relocate to another state while working remotely?',
    'Can I bring an iguana to the office?',
    'How does PTO accrue?',
])
def test_unknown_id_cannot_create_case_on_any_workflow(db_session, question):
    before = business_records(db_session)
    response = TestClient(app).post('/cases', json=payload('UNKNOWN-' + uuid4().hex, question))
    assert business_records(db_session) == before, 'Unknown ID created case/step/note records'
    assert response.status_code == 422
    assert response.json()['detail'] == 'Employee not found. Refresh the employee list before submitting.'


@pytest.mark.parametrize('question', [
    'How many PTO hours can I carry over?', 'Can I bring an iguana to the office?', 'How does PTO accrue?',
])
def test_valid_directory_employee_preserves_submission(db_session, employee, question):
    client = TestClient(app)
    assert employee.employee_id in client.get('/employees').json()['employees']
    request = payload(employee.employee_id, question)
    result = client.post('/cases', json=request)
    assert result.status_code == 200
    assert result.json()['status'] == 'completed'
    assert client.post('/cases', json=request).json() == result.json()
    case = db_session.get(Case, result.json()['case_id'])
    assert json.loads(case.inputs_json)['employee_id'] == employee.employee_id


@pytest.mark.parametrize('bad_id', [None, '', '   ', 123])
def test_malformed_employee_id_creates_no_business_records(db_session, bad_id):
    before = business_records(db_session)
    response = TestClient(app).post('/cases', json=payload(bad_id))
    assert response.status_code == 422
    assert business_records(db_session) == before


def test_missing_employee_id_creates_no_business_records(db_session):
    request = payload('unused')
    del request['inputs']['employee_id']
    before = business_records(db_session)
    assert TestClient(app).post('/cases', json=request).status_code == 422
    assert business_records(db_session) == before


def test_stale_selection_is_revalidated_before_creation(db_session, employee):
    client = TestClient(app)
    selected = employee.employee_id
    assert selected in client.get('/employees').json()['employees']
    db_session.delete(employee)
    db_session.commit()
    before = business_records(db_session)
    response = client.post('/cases', headers={'X-Employee-ID': selected}, json=payload(selected))
    assert business_records(db_session) == before
    assert response.status_code == 422


def test_directory_outage_is_explicit_safe_and_creates_no_case(db_session, employee, monkeypatch, caplog):
    before = business_records(db_session)
    original = Session.query
    marker = 'SYNTHETIC-PRIVATE-DIRECTORY-' + uuid4().hex

    def unavailable(session, *entities, **kwargs):
        if entities == (EmployeeRecord,):
            raise OperationalError('employee directory lookup', {}, RuntimeError(marker))
        return original(session, *entities, **kwargs)

    monkeypatch.setattr(Session, 'query', unavailable)
    client = TestClient(app, raise_server_exceptions=False)
    listing = client.get('/employees')
    submission = client.post('/cases', json=payload(employee.employee_id))
    assert business_records(db_session) == before
    assert listing.status_code == submission.status_code == 503
    assert marker not in listing.text + submission.text + caplog.text
    assert 'Employee lookup unavailable' in submission.json()['detail']


def test_authenticated_but_absent_employee_is_not_a_valid_directory_identity(db_session, monkeypatch):
    employee_id = 'UNSEEDED-' + uuid4().hex
    token = uuid4().hex
    monkeypatch.setenv('ATLAS_ENV', 'production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({
        employee_id: hashlib.sha256(token.encode()).hexdigest()}))
    headers = {'Authorization': 'Bearer ' + token}
    client = TestClient(app)
    assert client.get('/employees', headers=headers).json() == {'employees': []}
    before = business_records(db_session)
    response = client.post('/cases', headers=headers, json=payload(employee_id))
    assert business_records(db_session) == before
    assert response.status_code == 422


def test_runner_entry_cannot_bypass_identity_validation(db_session):
    from app.engine.runner import run_workflow
    request = payload('DIRECT-' + uuid4().hex)
    before = business_records(db_session)
    with pytest.raises(ValueError, match='Employee not found'):
        run_workflow(request['inputs'], request['idempotency_key'])
    assert business_records(db_session) == before


def test_replay_revalidates_deleted_employee_and_preserves_original_case(db_session, employee):
    selected = employee.employee_id
    request = payload(selected)
    original = Case(status='completed', inputs_json=json.dumps(request['inputs']), outputs_json='{}')
    db_session.add(original)
    db_session.commit()
    original_id = original.id
    db_session.delete(employee)
    db_session.commit()
    before = business_records(db_session)
    response = TestClient(app).post(f'/cases/{original_id}/replay', headers={'X-Employee-ID': selected})
    assert business_records(db_session) == before
    assert response.status_code == 422
    assert db_session.get(Case, original_id).status == 'completed'


def test_intentional_seeded_demo_data_remains_supported(db_session, monkeypatch):
    from data.seed import load_fixture, seed_employee_records
    monkeypatch.setenv('ATLAS_ENV', 'development')
    fixture = load_fixture()
    seed_employee_records(db_session, fixture)
    client = TestClient(app)
    ids = client.get('/employees').json()['employees']
    assert all(record['employee_id'] in ids for record in fixture['employees'])
    assert client.post('/cases', json=payload(fixture['employees'][0]['employee_id'])).status_code == 200
