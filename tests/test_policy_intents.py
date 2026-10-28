import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from app.engine import steps

GOLDEN = json.loads((Path(__file__).parent / 'fixtures/policy_intents.json').read_text())
DOCS = Path(__file__).resolve().parents[1] / 'data/docs'


@pytest.fixture
def policy_employee(db_session):
    from app.storage.models import EmployeeRecord
    employee_id = 'POLICY-' + uuid4().hex
    db_session.add(EmployeeRecord(employee_id=employee_id, metadata_json='{}', last_activities_json='[]'))
    db_session.commit()
    return employee_id


@pytest.mark.parametrize('example', GOLDEN)
def test_golden_classification(example):
    assert steps.classify_policy_intent(example['question']) == example['intent']

@pytest.mark.parametrize('question,expected', [
    ('  WHAT’S the PTO rollover limit!!! ', 'PTO_POLICY'),
    ('How much paid time off can roll over?', 'PTO_POLICY'),
    ('Can I move to a different state?', 'REMOTE_WORK'),
    ('Can I work remotely?', None),
    ('What remote work equipment do I receive?', None),
    ('Explain PTO carryover and relocating to another state', None),
])
def test_conservative_paraphrases(question, expected):
    assert steps.classify_policy_intent(question) == expected

@pytest.mark.parametrize('question,filename,required', [
    (GOLDEN[0]['question'], 'pto-policy.md', ['40 hours','forfeited unless local law requires payout']),
    (GOLDEN[2]['question'], 'remote-work-policy.md', ['notify HR before relocating','tax withholding','benefits eligibility','compensation zone','does not automatically authorize']),
])
@pytest.mark.parametrize('tone', ['empathetic', 'direct', 'formal'])
def test_api_policy_response_without_model_hris_or_bm25(db_session, policy_employee, monkeypatch, question, filename, required, tone):
    from app.main import app
    def forbidden(*a, **kw):
        pytest.fail('Policy intents must not call a model, HRIS, or BM25')
    monkeypatch.setattr(steps, '_retryable_llm_call', forbidden)
    monkeypatch.setattr(steps, 'get_tool_registry', forbidden)
    client = TestClient(app)
    payload = {'inputs': {'employee_id':policy_employee, 'case_transcript':question,'tone':tone}, 'idempotency_key':uuid4().hex}
    response = client.post('/cases',json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result['status']=='completed'
    output=result['outputs']
    for fact in required: assert fact in output['body']
    assert output['tone']==tone and output['quality_status']=='needs_hr_review'
    source=(DOCS/filename).read_text()
    assert output['cited_snippets']
    assert all(quote in source for quote in output['cited_snippets'])
    assert output['execution_receipts']==[]
    assert client.post('/cases',json=payload).json()==result
    stored=client.get(f"/cases/{result['case_id']}", headers={"X-Employee-ID": payload["inputs"]["employee_id"]}).json()
    assert stored['case_notes']==[] and len(stored['steps'])==1
    assert stored['steps'][0]['name']=='generate_structured_response'
    for evidence in output['citation_evidence']:
        assert source[evidence['start']:evidence['end']]==evidence['quote']


@pytest.mark.parametrize('question', [
    'Can I bring an iguana to the office?',
    'Can I bring a pet snake to a team meeting?',
    'Can I keep bees at home?',
    'What is the company policy for keeping bees at home?',
])
def test_unsupported_policy_question_gets_deterministic_hr_referral(db_session, policy_employee, monkeypatch, question):
    from app.main import app
    def forbidden(*a, **kw):
        pytest.fail('Unsupported policy questions must not call a model or HRIS')
    monkeypatch.setattr(steps, '_retryable_llm_call', forbidden)
    monkeypatch.setattr(steps, 'get_tool_registry', forbidden)
    client = TestClient(app)
    payload = {'inputs': {'employee_id':policy_employee,'case_transcript':question,'tone':'direct'},
               'idempotency_key':uuid4().hex}
    response = client.post('/cases', json=payload)
    assert response.status_code == 200
    result = response.json()
    output = result['outputs']
    assert result['status'] == 'completed'
    assert output['body'] == steps.unsupported_policy_response('direct')['body']
    assert output['cited_snippets'] == []
    assert output['quality_status'] == 'needs_hr_review'
    stored = client.get(f"/cases/{result['case_id']}", headers={"X-Employee-ID": payload["inputs"]["employee_id"]}).json()
    assert stored['case_notes'] == [] and len(stored['steps']) == 1
    assert stored['steps'][0]['name'] == 'generate_structured_response'
    # Same question/version, different case and key still has the same answer.
    payload['idempotency_key']=uuid4().hex
    assert client.post('/cases',json=payload).json()['outputs']==output

@pytest.mark.parametrize('intent,filename', [('PTO_POLICY','pto-policy.md'),('REMOTE_WORK','remote-work-policy.md')])
def test_missing_file_or_statement_fails_safely(tmp_path, monkeypatch, intent, filename):
    monkeypatch.setenv('DOCS_DIR',str(tmp_path))
    missing=steps.deterministic_policy_response(intent,'direct')
    assert 'could not be verified' in missing['body'] and missing['cited_snippets']==[]
    (tmp_path/filename).write_text('# Policy\n- Unrelated statement\n')
    assert steps.deterministic_policy_response(intent,'direct')==missing


def test_updated_policy_changes_answer_without_code_change(tmp_path, monkeypatch):
    monkeypatch.setenv('DOCS_DIR',str(tmp_path))
    source=(DOCS/'pto-policy.md').read_text().replace('40 hours','64 hours')
    (tmp_path/'pto-policy.md').write_text(source)
    output=steps.deterministic_policy_response('PTO_POLICY','direct')
    assert '64 hours' in output['body'] and '40 hours' not in output['body']
    assert all(q in source for q in output['cited_snippets'])


def test_partial_policy_does_not_make_up_exception(tmp_path, monkeypatch):
    monkeypatch.setenv('DOCS_DIR',str(tmp_path))
    (tmp_path/'pto-policy.md').write_text('# PTO\n- Unused PTO carries over up to 40 hours into the next calendar year\n')
    output=steps.deterministic_policy_response('PTO_POLICY','direct')
    assert 'could not be verified' in output['body'] and output['cited_snippets']==[]


def test_both_intents_keep_existing_workflow(db_session):
    from app.main import app
    from data.seed import load_fixture, seed_employee_records
    seed_employee_records(db_session, load_fixture())
    client = TestClient(app)
    response = client.post('/cases', json={'inputs':{
        'employee_id':'E10234', 'case_transcript':'Explain PTO carryover and relocating to another state',
        'tone':'direct'}, 'idempotency_key':uuid4().hex})
    assert response.status_code == 200
    stored=client.get(f"/cases/{response.json()['case_id']}", headers={"X-Employee-ID": "E10234"}).json()
    assert len(stored['steps'])==5 and len(stored['case_notes'])==1


def test_missing_policy_through_api(db_session, policy_employee, monkeypatch, tmp_path):
    from app.main import app
    monkeypatch.setenv('DOCS_DIR', str(tmp_path))
    client=TestClient(app)
    response=client.post('/cases', json={'inputs':{'employee_id':policy_employee,
        'case_transcript':GOLDEN[0]['question'], 'tone':'formal'}, 'idempotency_key':uuid4().hex})
    assert response.status_code==200
    output=response.json()['outputs']
    assert 'could not be verified' in output['body']
    assert output['cited_snippets']==[] and output['quality_status']=='needs_hr_review'
