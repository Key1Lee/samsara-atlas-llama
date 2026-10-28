import pytest
from app.engine.context import ExecutionContext
from app.engine.steps import generate_structured_response_step

@pytest.mark.parametrize('tone',['empathetic','direct','formal'])
def test_missing_pto_balance_returns_reviewable_answer_without_guessing(monkeypatch,tone):
    def unexpected(*a,**kw):
        pytest.fail('Missing authoritative balance must not require model guessing')
    monkeypatch.setattr('app.engine.steps._retryable_llm_call',unexpected)
    ctx=ExecutionContext(case_id=1,inputs={'employee_id':'E10234','case_transcript':'How many hours of PTO do I have','tone':tone},artifacts={},step_outputs={'retrieve_employee_context':{'found':True,'fact_verification':'unverified','metadata':{},'last_activities':[{'summary':'Previously mentioned 30 hours'}]}})
    result=generate_structured_response_step(ctx)
    assert 'cannot confirm' in result['body']
    assert 'HR partner' in result['body']
    assert '30' not in result['body']
    assert result['quality_status']=='needs_hr_review'
    assert result['tone']==tone
    assert result['execution_receipts']==[]
