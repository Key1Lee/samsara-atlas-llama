"""Employee ownership is enforced by SQL and authenticated identity."""
import hashlib
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.models import Case, EmployeeRecord


@pytest.fixture
def employee_cases(db_session):
    employees = ['EMP-A-' + uuid4().hex, 'EMP-B-' + uuid4().hex]
    records = {}
    for employee in employees:
        db_session.add(EmployeeRecord(employee_id=employee, metadata_json='{}', last_activities_json='[]'))
        records[employee] = []
        for i in range(3):
            case = Case(status='completed', inputs_json=json.dumps({
                'employee_id': employee, 'case_transcript': f'{employee}: question {i}', 'tone': 'direct'}),
                artifacts_json='{}', outputs_json='{}')
            db_session.add(case)
            db_session.flush()
            records[employee].append(case.id)
    db_session.commit()
    return employees, records


def credentials(monkeypatch, employees):
    tokens = {employee: uuid4().hex for employee in employees}
    monkeypatch.setenv('ATLAS_ENV', 'production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({
        employee: hashlib.sha256(token.encode()).hexdigest() for employee, token in tokens.items()}))
    return {employee: {'Authorization': 'Bearer ' + token} for employee, token in tokens.items()}


def test_authenticated_lists_details_and_cross_employee_denials(db_session, monkeypatch, employee_cases):
    employees, records = employee_cases
    headers = credentials(monkeypatch, employees)
    client = TestClient(app)
    for employee, other in (employees, employees[::-1]):
        own = client.get('/cases', headers=headers[employee])
        assert own.status_code == 200
        data = own.json()
        assert set(data['case_ids']) == set(records[employee])
        assert all(row['question'].startswith(employee) for row in data['cases'])
        assert all(row['submitted_at'] and row['status'] == 'completed' for row in data['cases'])
        assert client.get('/employees', headers=headers[employee]).json()['employees'] == [employee]
        assert client.get(f'/cases/{records[employee][0]}', headers=headers[employee]).status_code == 200
        for route in (f'/cases/{records[other][0]}', f'/cases/{records[other][0]}/replay'):
            method = client.post if route.endswith('replay') else client.get
            assert method(route, headers=headers[employee]).status_code == 404
        assert client.get('/cases', headers={**headers[employee], 'X-Employee-ID': other}).status_code == 403
        assert client.get('/cases', params={'employee_id': other}, headers=headers[employee]).status_code == 403
        assert client.post('/cases', headers=headers[employee], json={'inputs': {
            'employee_id': other, 'case_transcript': 'PTO carryover?', 'tone': 'direct'}}).status_code == 403
    assert client.get('/cases').status_code == 401
    assert client.get('/cases', headers={'Authorization': 'Bearer wrong'}).status_code == 401


def test_production_without_credentials_fails_closed(db_session, monkeypatch):
    monkeypatch.setenv('ATLAS_ENV', 'production')
    assert TestClient(app).get('/cases', headers={'X-Employee-ID': 'E10234'}).status_code == 503


def test_demo_requires_scope_and_returns_all_cases(db_session):
    employee = 'ALL-' + uuid4().hex
    for i in range(55):
        db_session.add(Case(status='completed', inputs_json=json.dumps({
            'employee_id': employee, 'case_transcript': str(i)})))
    db_session.commit()
    client = TestClient(app)
    assert client.get('/cases').status_code == 400
    result = client.get('/cases', headers={'X-Employee-ID': employee}).json()
    assert len(result['cases']) == len(set(result['case_ids'])) == 55
    assert result['case_ids'] == sorted(result['case_ids'], reverse=True)


def test_authenticated_new_submission_appears_once(db_session, monkeypatch, employee_cases):
    employees, records = employee_cases
    headers = credentials(monkeypatch, employees)
    employee = employees[0]
    client = TestClient(app)
    payload = {'inputs': {'employee_id': employee, 'case_transcript': 'How many PTO hours can I carry over?',
                          'tone': 'direct'}, 'idempotency_key': uuid4().hex}
    first = client.post('/cases', headers=headers[employee], json=payload)
    assert first.status_code == 200
    assert client.post('/cases', headers=headers[employee], json=payload).json() == first.json()
    cases = client.get('/cases', headers=headers[employee]).json()['case_ids']
    assert cases.count(first.json()['case_id']) == 1
    assert len(cases) == len(records[employee]) + 1
