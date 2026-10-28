"""Bounded end-to-end scenarios with controlled model output, not model quality evals."""
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize('question,policy', [
    ('I report 30 unused PTO hours. How does PTO accrue?', 'Paid Time Off'),
    ('I do not know my PTO balance. Can HR confirm it?', 'Paid Time Off'),
    ('What is parental leave eligibility and duration?', 'Parental Leave'),
    ('Can I bring a pet iguana to the office?', None),
])
def test_reviewable_hr_scenario_persists_once(db_session, monkeypatch, question, policy):
    from app.main import app
    from app.engine import steps
    from data.seed import load_fixture, seed_employee_records
    seed_employee_records(db_session, load_fixture())
    prompts = []

    def complete(prompt, **kwargs):
        prompts.append(prompt)
        result = {'summary': 'Employee asks HR to review the question.',
                  'action_items': ['Ask HR to verify employee facts and policy applicability'],
                  'next_steps': ['HR review required before sharing the draft']}
        if kwargs:
            result = {'subject': 'HR policy review', 'body': 'Please ask HR to verify your balance and applicable policy before making a decision.',
                      'tone': kwargs['tone'], 'cited_snippets': []}
        return json.dumps(result)

    monkeypatch.setattr(steps, '_retryable_llm_call', complete)
    client = TestClient(app, headers={"X-Employee-ID": "E10234"})
    payload = {'inputs': {'employee_id': 'E10234', 'case_transcript': question, 'tone': 'empathetic'},
               'idempotency_key': uuid4().hex}
    response = client.post('/cases', json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'completed'
    assert result['outputs']['quality_status'] == 'needs_hr_review'
    assert result['outputs']['execution_receipts'] == []
    stored = client.get(f"/cases/{result['case_id']}").json()
    if not policy:
        assert result['outputs']['body'] == steps.unsupported_policy_response('empathetic')['body']
        assert len(stored['steps']) == 1
        assert stored['steps'][0]['name'] == 'generate_structured_response'
        assert stored['case_notes'] == []
        assert client.post('/cases', json=payload).json() == result
        return
    policies = json.loads(stored['steps'][1]['output'])
    assert bool(policies)
    assert any(policy in p for p in policies)
    assert all(p in prompt for p in policies for prompt in prompts)
    assert len(stored['steps']) == 5
    assert len(stored['case_notes']) == 1
    assert 'Unapproved' in stored['case_notes'][0]['note']
    assert client.post('/cases', json=payload).json() == result
    assert len(client.get(f"/cases/{result['case_id']}").json()['case_notes']) == 1
