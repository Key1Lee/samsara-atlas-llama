from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4
import json

from fastapi.testclient import TestClient

from app.main import app
from app.middleware import rate_limit
from app.storage.models import EmployeeDailyRequests, Case


def prime(db, employee, count, day=None):
    db.add(EmployeeDailyRequests(employee_id=employee,
        day=day or rate_limit.utc_now().date().isoformat(), request_count=count))
    db.commit()


def test_shared_limit_across_methods_and_independent_employees(db_session):
    employee = uuid4().hex
    prime(db_session, employee, 198)
    client = TestClient(app)
    headers = {'X-Employee-ID': employee}
    assert client.get('/info', headers=headers).headers['X-RateLimit-Remaining'] == '1'
    # Even unsupported methods consume the same quota (they return 405).
    assert client.patch('/info', headers=headers).status_code == 405
    blocked = client.delete('/info', headers=headers)
    assert blocked.status_code == 429
    assert int(blocked.headers['Retry-After']) > 0
    assert blocked.headers['X-RateLimit-Remaining'] == '0'
    assert client.get('/info', headers={'X-Employee-ID': uuid4().hex}).status_code == 200
    assert client.get('/info').status_code == 200


def test_body_only_submission_and_replay_cannot_bypass_limit(db_session):
    employee = uuid4().hex
    prime(db_session, employee, 200)
    case = Case(status='completed', inputs_json=json.dumps({'employee_id': employee}))
    db_session.add(case)
    db_session.commit()
    client = TestClient(app)
    assert client.post('/cases', json={'inputs': {'employee_id': employee,
        'case_transcript': 'Can I bring an iguana?', 'tone': 'direct'}}).status_code == 429
    assert client.post(f'/cases/{case.id}/replay').status_code == 429
    assert client.post('/cases', headers={'X-Employee-ID': 'different'}, json={
        'inputs': {'employee_id': employee}}).status_code == 400


def test_daily_reset_keeps_previous_day_count(db_session, monkeypatch):
    employee = uuid4().hex
    prime(db_session, employee, 200, '2026-10-26')
    monkeypatch.setattr(rate_limit, 'utc_now', lambda: datetime(2026, 10, 27, tzinfo=timezone.utc))
    response = TestClient(app).get('/info', headers={'X-Employee-ID': employee})
    assert response.status_code == 200
    assert response.headers['X-RateLimit-Remaining'] == '199'
    assert db_session.get(EmployeeDailyRequests, (employee, '2026-10-26')).request_count == 200


def test_atomic_counter_survives_new_sessions_and_simultaneous_requests(db_session):
    employee = uuid4().hex
    day = rate_limit.utc_now().date().isoformat()
    prime(db_session, employee, 199)
    with ThreadPoolExecutor(max_workers=2) as pool:
        counts = list(pool.map(lambda _: rate_limit.consume_request(employee, day), range(2)))
    assert counts.count(200) == 1
    assert counts.count(None) == 1
    assert rate_limit.consume_request(employee, day) is None
