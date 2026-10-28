"""BR-04: real committed case history shared by API reads and workflow context."""
import hashlib
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.main import app
from app.storage.database import Base
from app.storage.models import Case, CaseNote, EmployeeRecord, Step


@pytest.fixture
def history_store(tmp_path, monkeypatch):
    # Each test gets a separate file and real independent SQLAlchemy sessions.
    from app.api import routes
    from app.engine import runner
    from app.middleware import rate_limit
    from app.storage import database
    from app.tools import hris
    engine = create_engine(f'sqlite:///{tmp_path / "history.db"}',
                           connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (database, routes, runner, rate_limit, hris):
        monkeypatch.setattr(module, 'SessionLocal', factory)
    tokens = {employee: uuid4().hex for employee in ('HISTORY-A', 'HISTORY-B')}
    monkeypatch.setenv('ATLAS_ENV', 'production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({
        employee: hashlib.sha256(token.encode()).hexdigest() for employee, token in tokens.items()}))
    with factory() as db:
        for employee in tokens:
            db.add(EmployeeRecord(employee_id=employee, metadata_json='{}',
                last_activities_json=json.dumps([{'type': 'case', 'summary': 'Seeded activity', 'date': '2026-01-01'}])))
        db.commit()
    try:
        yield factory, {employee: {'Authorization': 'Bearer ' + token} for employee, token in tokens.items()}
    finally:
        engine.dispose()


def submit(client, headers, question, employee='HISTORY-A'):
    return client.post('/cases', headers=headers[employee], json={'inputs': {
        'employee_id': employee, 'case_transcript': question, 'tone': 'direct'},
        'idempotency_key': uuid4().hex})


@pytest.mark.parametrize('question', [
    'How many PTO hours can I carry over?',
    'Can I relocate to another state while working remotely?',
    'Can I bring an iguana to the office?',
    'How does PTO accrue?',
])
def test_committed_case_is_visible_once_in_new_requests_and_workflow_history(history_store, question):
    from app.tools.hris import retrieve_employee_context
    factory, headers = history_store
    created = submit(TestClient(app), headers, question)
    assert created.status_code == 200
    case_id = created.json()['case_id']
    with factory() as db:
        assert db.get(Case, case_id).status == 'completed'
    # A fresh client, tool call and DB session must see the committed record.
    client = TestClient(app)
    listing = client.get('/cases', headers=headers['HISTORY-A']).json()
    assert listing['case_ids'] == [case_id]
    context = retrieve_employee_context('HISTORY-A')
    history = context.get('case_history', [])
    assert [row['case_id'] for row in history] == listing['case_ids']
    assert history[0]['question'] == question
    assert history[0]['status'] == 'completed'
    assert history[0]['submitted_at'] == listing['cases'][0]['submitted_at']
    assert history[0]['source'] == 'cases'
    assert 'body' not in history[0] and 'action_items' not in history[0]
    assert any(activity['summary'] == 'Seeded activity' for activity in context['last_activities'])
    assert client.get('/cases', headers=headers['HISTORY-B']).json()['case_ids'] == []
    assert client.get(f'/cases/{case_id}', headers=headers['HISTORY-B']).status_code == 404
    assert client.get('/cases', headers={**headers['HISTORY-B'], 'X-Employee-ID': 'HISTORY-A'}).status_code == 403


def test_distinct_cases_and_notes_do_not_duplicate_history_and_reach_later_workflow(history_store, monkeypatch):
    from app.engine import steps
    from app.tools.hris import retrieve_employee_context
    factory, headers = history_store
    client = TestClient(app)
    first = submit(client, headers, 'How many PTO hours can I carry over?').json()['case_id']
    second = submit(client, headers, 'How does PTO accrue?').json()['case_id']
    other_case = submit(client, headers, 'Can I relocate to another state while working remotely?',
                        employee='HISTORY-B').json()['case_id']
    own = client.get('/cases', headers=headers['HISTORY-A']).json()
    assert set(own['case_ids']) == {first, second}
    context = retrieve_employee_context('HISTORY-A')
    assert [row['case_id'] for row in context.get('case_history', [])] == own['case_ids']
    with factory() as db:
        assert db.query(CaseNote).filter_by(case_id=first).count() == 0
        assert db.query(CaseNote).filter_by(case_id=second).count() == 1
    # Existing model steps serialize the context; no prompt changes are needed.
    prompts = []
    complete = steps._retryable_llm_call

    def capture(prompt, **kwargs):
        prompts.append(prompt)
        return complete(prompt, **kwargs)

    monkeypatch.setattr(steps, '_retryable_llm_call', capture)
    third = submit(TestClient(app), headers, 'Explain PTO accrual for this question.').json()['case_id']
    stored = TestClient(app).get(f'/cases/{third}', headers=headers['HISTORY-A']).json()
    saved_context = json.loads(next(step['output'] for step in stored['steps']
                                    if step['name'] == 'retrieve_employee_context'))
    assert [row['case_id'] for row in saved_context.get('case_history', [])] == own['case_ids']
    assert third not in [row['case_id'] for row in saved_context['case_history']]
    assert saved_context['fact_verification'] == 'unverified'
    assert len(prompts) == 2
    assert all('How many PTO hours can I carry over?' in prompt and 'How does PTO accrue?' in prompt
               for prompt in prompts)
    assert all('HISTORY-B' not in prompt for prompt in prompts)
    other = retrieve_employee_context('HISTORY-B')
    assert [row['case_id'] for row in other.get('case_history', [])] == [other_case]


def test_rolled_back_case_write_does_not_appear_as_saved(history_store, monkeypatch):
    from app.tools.hris import retrieve_employee_context
    factory, headers = history_store
    original = Session.commit
    flushed_ids = []

    def reject_case_insert(db):
        pending = [record for record in db.new if isinstance(record, Case)]
        if pending:
            db.flush()
            flushed_ids.extend(record.id for record in pending)
            raise OperationalError('synthetic write failure', {}, RuntimeError('no commit'))
        return original(db)

    monkeypatch.setattr(Session, 'commit', reject_case_insert)
    client = TestClient(app)
    response = submit(client, headers, 'How many PTO hours can I carry over?')
    assert response.status_code == 500
    assert 'outputs' not in response.json()
    assert len(flushed_ids) == 1 and flushed_ids[0] is not None
    with factory() as db:
        assert db.query(Case).count() == db.query(CaseNote).count() == db.query(Step).count() == 0
    assert client.get('/cases', headers=headers['HISTORY-A']).json()['case_ids'] == []
    assert retrieve_employee_context('HISTORY-A').get('case_history', []) == []


def test_committed_failed_case_is_history_but_not_a_successful_draft(history_store, monkeypatch):
    from app.engine import steps
    from app.tools.hris import retrieve_employee_context
    factory, headers = history_store
    monkeypatch.setattr(steps, '_retryable_llm_call', lambda *a, **kw: 'invalid JSON')
    client = TestClient(app)
    response = submit(client, headers, 'How does PTO accrue?')
    assert response.status_code == 500
    listing = client.get('/cases', headers=headers['HISTORY-A']).json()
    assert len(listing['cases']) == 1 and listing['cases'][0]['status'] == 'failed'
    history = retrieve_employee_context('HISTORY-A').get('case_history', [])
    assert len(history) == 1 and history[0]['status'] == 'failed'
    with factory() as db:
        assert db.query(CaseNote).count() == 0


def test_tied_dates_have_stable_order_in_api_and_workflow_history(history_store):
    from datetime import datetime
    from app.tools.hris import retrieve_employee_context
    factory, headers = history_store
    stamp = datetime(2026, 1, 1)
    with factory() as writer:
        for question in ('One', 'Two'):
            writer.add(Case(status='running', created_at=stamp, inputs_json=json.dumps({
                'employee_id': 'HISTORY-A', 'case_transcript': question})))
        writer.commit()
    listing = TestClient(app).get('/cases', headers=headers['HISTORY-A']).json()
    ids = listing['case_ids']
    assert len(ids) == 2 and ids == sorted(ids, reverse=True)
    assert [row['case_id'] for row in retrieve_employee_context('HISTORY-A').get('case_history', [])] == ids
