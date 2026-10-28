"""Global benchmark denominators, review evidence, topics, privacy and navigation."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from app.benchmarks import global_benchmark, fingerprint, question_topics, REVIEWS
from app.main import app
from app.storage.database import Base
from app.storage.models import Case, CaseAuditEvent, EmployeeRecord
from tests.test_ui import api_bridge, continue_to_question, UI  # noqa: F401

TOKEN = 'separate-analytics-test-token'
HEADERS = {'Authorization': f'Bearer {TOKEN}'}


@pytest.fixture
def analytics_auth(monkeypatch):
    monkeypatch.setenv('ATLAS_ANALYTICS_TOKEN_HASH', hashlib.sha256(TOKEN.encode()).hexdigest())


@pytest.fixture
def benchmark_db(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path / "benchmarks.db"}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr('app.api.routes.SessionLocal', factory)
    monkeypatch.setattr('app.engine.runner.SessionLocal', factory)
    monkeypatch.setattr('app.tools.hris.SessionLocal', factory)
    monkeypatch.setattr('app.middleware.rate_limit.SessionLocal', factory)
    with factory() as db:
        yield db
    engine.dispose()


def add_case(db, employee, question, *, status='completed', outputs=None):
    row = Case(inputs_json=json.dumps({'employee_id': employee, 'case_transcript': question, 'tone': 'empathetic'}),
               outputs_json=json.dumps(outputs or {'body': 'Unreviewed draft', 'quality_status': 'approved'}),
               artifacts_json='{}', status=status, idempotency_key=uuid4().hex)
    db.add(row)
    db.commit()
    return row


def record_reviews(monkeypatch, tmp_path, cases_and_categories):
    # Only fixture-authored review evidence can grant quality scores, never model labels.
    source = json.loads(REVIEWS.read_text())
    source['results'] = [
        {'inputs_sha256': fingerprint(json.loads(row.inputs_json)),
         'key_sha256': fingerprint(row.idempotency_key),
         'outputs_sha256': fingerprint(json.loads(row.outputs_json)), 'category': category}
        for row, category in cases_and_categories]
    path = tmp_path / 'reviews.json'
    path.write_text(json.dumps(source))
    monkeypatch.setattr('app.benchmarks.REVIEWS', path)


def test_global_scores_include_both_employees_errors_and_unreviewed(benchmark_db, monkeypatch, tmp_path):
    db = benchmark_db
    good = add_case(db, 'A', 'How much PTO carries over?', outputs={'body': '40 hours, subject to local law.'})
    bad = add_case(db, 'B', 'When do benefits start?', outputs={'body': 'Wrong date.'})
    absent = add_case(db, 'A', 'Can I bring an iguana?', outputs={'body': 'No information available; ask HR.'})
    add_case(db, 'B', 'My expense request?', status='failed')
    add_case(db, 'B', 'Remote work and PTO?', outputs={'body': 'Confident answer', 'quality_status': 'approved'})
    record_reviews(monkeypatch, tmp_path, [(good, 'Correct answer'), (bad, 'Incorrect answer'),
                                         (absent, 'No information available')])
    result = global_benchmark(db)
    assert result['summary']['total'] == 5
    assert list(result['summary']['counts'].values()) == [1, 1, 1, 1, 1]
    assert set(result['summary']['percentages'].values()) == {20.0}
    assert result['summary']['appropriate_outcome_percent'] == 40.0
    assert result['summary']['review_coverage_percent'] == 60.0
    assert result['topic_counts'] == {'PTO': 2, 'Remote work': 1, 'Benefits': 1,
                                      'Expenses': 1, 'Leave': 0, 'Other/unknown': 1}
    # Recovering/changing an answer invalidates its old review.
    good.outputs_json = json.dumps({'body': 'A different draft'})
    db.commit()
    updated = global_benchmark(db)['summary']
    assert updated['counts']['Correct answer'] == 0
    assert updated['counts']['Awaiting review'] == 2


def test_empty_database_missing_reviews_and_invalid_records(benchmark_db, monkeypatch, tmp_path):
    monkeypatch.setattr('app.benchmarks.REVIEWS', tmp_path / 'absent.json')
    result = global_benchmark(benchmark_db)
    assert result['summary']['total'] == 0
    assert set(result['summary']['percentages'].values()) == {0.0}
    assert not result['review_evidence_available']
    add_case(benchmark_db, 'A', 'What is PTO?', status='running')
    benchmark_db.add(Case(inputs_json='not JSON', status='completed'))
    benchmark_db.commit()
    summary = global_benchmark(benchmark_db)['summary']
    assert summary['total'] == 1 and summary['counts']['Awaiting review'] == 1
    assert summary['in_progress'] == 1 and summary['excluded_invalid_records'] == 1


def test_policy_change_prevents_stale_review_scores(benchmark_db, monkeypatch, tmp_path):
    good = add_case(benchmark_db, 'A', 'PTO?', outputs={'body': 'Reviewed answer'})
    record_reviews(monkeypatch, tmp_path, [(good, 'Correct answer')])
    monkeypatch.setenv('DOCS_DIR', str(tmp_path / 'changed-policies'))
    result = global_benchmark(benchmark_db)
    assert not result['review_evidence_available']
    assert result['summary']['counts']['Awaiting review'] == 1


@pytest.mark.parametrize('question,expected', [
    ('PTO PTO and remote work', ['PTO', 'Remote work']),
    ('A reimbursement receipt for relocation expenses', ['Remote work', 'Expenses']),
    ('Health coverage and parental leave', ['Benefits', 'Leave']),
    ('A 401(k) retirement question', ['Benefits']),
    ('My iguana?', ['Other/unknown']),
    ('Please help', ['Other/unknown']),
])
def test_deterministic_multilabel_topic_counts(question, expected):
    assert question_topics(question) == expected


def test_new_submissions_retries_and_gets_change_only_relevant_counts(benchmark_db, analytics_auth):
    for employee in ('A', 'B'):
        benchmark_db.add(EmployeeRecord(employee_id=employee, metadata_json='{}', last_activities_json='[]'))
    benchmark_db.commit()
    with TestClient(app) as client:
        payload = {'inputs': {'employee_id': 'A', 'case_transcript': 'How many PTO hours can I carry over?',
                              'tone': 'empathetic'}, 'idempotency_key': 'one-submission'}
        first = client.post('/cases', json=payload)
        assert first.status_code == 200
        before = client.get('/benchmarks/latest', headers=HEADERS).json()
        assert before['summary']['total'] == 1
        assert before['summary']['counts']['Awaiting review'] == 1
        for _ in range(2):
            retry = client.post('/cases', json=payload)
            assert retry.json()['case_id'] == first.json()['case_id']
            client.get('/info')
            client.get('/cases', headers={'X-Employee-ID': 'A'})
        retry_result = client.get('/benchmarks/latest', headers=HEADERS).json()
        assert retry_result['summary']['total'] == 1
        assert retry_result['summary']['retry_attempts_recorded'] == 2
        payload['inputs']['employee_id'] = 'B'
        payload['inputs']['case_transcript'] = 'Can I relocate to another state while working remotely?'
        payload['idempotency_key'] = 'independent-submission'
        assert client.post('/cases', json=payload).status_code == 200
        after = client.get('/benchmarks/latest', headers=HEADERS).json()
        assert after['summary']['total'] == 2
        assert after['topic_counts']['PTO'] == 1 and after['topic_counts']['Remote work'] == 1
        assert after['summary']['retry_attempts_recorded'] == 2


def test_retry_events_are_distinct_requests_not_steps(benchmark_db):
    row = add_case(benchmark_db, 'A', 'PTO?')
    for kind, request_id in [('request_reused', 'r1'), ('case_resumed', 'r1'),
                             ('step_started', 'r1'), ('request_reused', None)]:
        benchmark_db.add(CaseAuditEvent(case_id=row.id, event_type=kind, request_id=request_id, outcome='observed'))
    benchmark_db.commit()
    assert global_benchmark(benchmark_db)['summary']['retry_attempts_recorded'] == 1


@pytest.mark.parametrize('environment', ['development', 'production'])
def test_global_analytics_authorization_is_separate_from_employee_access(benchmark_db, analytics_auth, monkeypatch, environment):
    monkeypatch.setenv('ATLAS_ENV', environment)
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({'A': hashlib.sha256(b'employee-token').hexdigest()}))
    with TestClient(app) as client:
        assert client.get('/benchmarks/latest', headers={'X-Employee-ID': 'A'}).status_code == 401
        for token in ('employee-token', 'samsara', 'wrong'):
            assert client.get('/benchmarks/latest', headers={'Authorization': f'Bearer {token}'}).status_code == 403
        assert client.get('/benchmarks/latest', headers=HEADERS).status_code == 200
        # Analytics permission never grants access to employee records.
        assert client.get('/cases', headers=HEADERS).status_code == 401


def test_unconfigured_analytics_fails_closed(benchmark_db, monkeypatch):
    monkeypatch.delenv('ATLAS_ANALYTICS_TOKEN_HASH', raising=False)
    with TestClient(app) as client:
        assert client.get('/benchmarks/latest', headers=HEADERS).status_code == 503


def test_global_response_ignores_employee_selection_and_does_not_expose_payloads(benchmark_db, analytics_auth):
    add_case(benchmark_db, 'PRIVATE-EMPLOYEE-A', 'A private PTO question', outputs={'body': 'Private draft'})
    add_case(benchmark_db, 'PRIVATE-EMPLOYEE-B', 'Another private expense question')
    with TestClient(app) as client:
        a = client.get('/benchmarks/latest', headers={**HEADERS, 'X-Employee-ID': 'A'}).json()
        b = client.get('/benchmarks/latest?employee_id=B', headers=HEADERS).json()
        assert a['summary'] == b['summary'] and a['topic_counts'] == b['topic_counts']
        assert a['summary']['total'] == 2
        serialized = json.dumps(a)
        for sensitive in ('PRIVATE-EMPLOYEE', 'A private PTO question', 'Private draft', 'case_id', 'inputs_sha256'):
            assert sensitive not in serialized


def test_global_benchmark_ui_refresh_chart_and_employee_switch(api_bridge, analytics_auth, monkeypatch, db_session):
    def no_submission(*args, **kwargs):
        raise AssertionError('Viewing analytics must not submit cases')
    monkeypatch.setattr(requests, 'post', no_submission)
    ui = AppTest.from_file(UI).run()
    continue_to_question(ui, 'E11342')
    ui.switch_page('app_pages/benchmark.py').run()
    assert not ui.metric
    assert any('authorized analytics token' in i.value for i in ui.info)
    ui.text_input(key='analytics_access_token').set_value(TOKEN).run()
    assert not ui.exception
    assert not ui.error, [item.value for item in ui.error]
    assert ui.metric, repr(ui)
    assert ui.get('vega_lite_chart')
    expected = api_bridge.get('/benchmarks/latest', headers=HEADERS).json()
    assert ui.metric[0].value == str(expected['summary']['total'])
    assert len(next(table.value for table in ui.dataframe if 'Topic' in table.value.columns)) == 6
    add_case(db_session, 'UI-GLOBAL-' + uuid4().hex, 'An expense receipt question')
    next(b for b in ui.button if b.label == 'Refresh benchmark').click().run()
    assert not ui.exception
    assert ui.metric[0].value == str(expected['summary']['total'] + 1)
    topics = next(table.value for table in ui.dataframe if 'Topic' in table.value.columns)
    assert int(topics.set_index('Topic').loc['Expenses', 'Submissions']) == expected['topic_counts']['Expenses'] + 1
    ui = AppTest.from_file(UI).run()
    continue_to_question(ui, 'E10234')
    ui.switch_page('app_pages/benchmark.py').run()
    # A second employee selection still needs the separate analytics credential.
    ui.text_input(key='analytics_access_token').set_value(TOKEN).run()
    assert not ui.exception
    assert ui.metric[0].value == str(expected['summary']['total'] + 1)
    assert not any('Case ID' in table.value.columns for table in ui.dataframe)


def test_global_page_requires_no_employee_selection(api_bridge, analytics_auth):
    ui = AppTest.from_file(UI).run()
    assert 'selected_employee' not in ui.session_state
    ui.switch_page('app_pages/benchmark.py').run()
    ui.text_input(key='analytics_access_token').set_value(TOKEN).run()
    assert not ui.exception and ui.metric
    assert 'selected_employee' not in ui.session_state
    assert not ui.text_area


def test_employee_quota_cannot_block_separately_authorized_global_analytics(api_bridge, analytics_auth, db_session):
    from app.storage.models import EmployeeDailyRequests
    from app.middleware.rate_limit import utc_now
    ui = AppTest.from_file(UI).run()
    continue_to_question(ui, 'E11342')
    quota = db_session.query(EmployeeDailyRequests).filter_by(employee_id='E11342', day=utc_now().date().isoformat()).one()
    quota.request_count = 200
    db_session.commit()
    ui.switch_page('app_pages/benchmark.py').run()
    ui.text_input(key='analytics_access_token').set_value(TOKEN).run()
    assert not ui.exception and ui.metric
    assert not any('Daily limit' in error.value for error in ui.error)
    assert api_bridge.get('/cases', headers={'X-Employee-ID': 'E11342'}).status_code == 429

