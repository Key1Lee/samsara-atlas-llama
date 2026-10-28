import builtins
import json
from pathlib import Path
from uuid import uuid4
import pytest
from app.engine import steps
from app.engine.context import ExecutionContext
from app.providers.base import CompletionResponse
from app.providers.ollama import OllamaProvider
from app.diagnostics import failure_diagnostic
from app.audit import current_request_id

SOURCE = (Path(__file__).resolve().parents[1] / 'data/docs/workplace-accommodation-policy.md').read_text()

def context():
    return ExecutionContext(0, {'employee_id':'SYNTHETIC','case_transcript':'Explain accommodation review.','tone':'direct'}, {},
        {'retrieve_policy_snippets':[SOURCE]})

def response(**fields):
    return dict(subject='Draft',body='Ask HR to review the policy.',tone='direct',cited_snippets=[],**fields)

def install(monkeypatch, payload):
    class Fake:
        def complete(self, prompt, **kw):
            return CompletionResponse(payload)
    monkeypatch.setattr(steps,'get_provider',lambda:Fake())

@pytest.mark.parametrize('text,rule,subtype', [
    ('{','json_syntax','json_parse'), ('[]','json_object_required','format_validation'),
    ('{}','schema_fields','schema_validation'),
    (json.dumps({**response(),'cited_snippets':['PRIVATE fabricated source']}),'citation_absent','semantic_validation'),
    (json.dumps({**response(),'tone':'formal'}),'tone_mismatch','semantic_validation'),
    (json.dumps({**response(),'body':'I have sent the email.'}),'unsupported_execution_claim','semantic_validation'),
])
def test_invalid_completions_have_safe_rules(monkeypatch,text,rule,subtype):
    install(monkeypatch,text)
    token=current_request_id.set('a'*32)
    try:
        with pytest.raises(ValueError) as caught: steps.generate_structured_response_step(context())
        d=caught.value.atlas_diagnostic
        assert d['validation_rule']==rule and d['error_subtype']==subtype
        assert d['schema_version']=='structured_response.v1' and d['request_id']=='a'*32
        assert 'PRIVATE' not in json.dumps(d)
    finally:current_request_id.reset(token)

@pytest.mark.parametrize('fence',['```','```json'])
def test_fenced_json_is_parsed(monkeypatch,fence):
    install(monkeypatch,fence+'\n'+json.dumps(response())+'\n```')
    assert steps.generate_structured_response_step(context())['subject']=='Draft'

def test_empty_retrieval_does_not_call_model(monkeypatch):
    monkeypatch.setattr(steps,'get_provider',lambda:pytest.fail('model called'))
    ctx=context();ctx.step_outputs['retrieve_policy_snippets']=[]
    assert steps.generate_structured_response_step(ctx)['answerability']['reason']=='unsupported_policy_topic'

@pytest.mark.parametrize('payload,status',[
    ({'response':json.dumps(response()),'done':True,'done_reason':'length'},'truncated'),
    ({'response':'','done':True,'done_reason':'refusal','refusal':'PRIVATE'},'refused'),
])
def test_provider_completion_state_rejected_before_acceptance(monkeypatch,payload,status):
    class HTTP:
        status_code=200
        def json(self):return payload
    monkeypatch.setattr('requests.post',lambda *a,**k:HTTP())
    monkeypatch.setattr(steps,'get_provider',lambda:OllamaProvider())
    with pytest.raises(ValueError) as caught:steps.generate_structured_response_step(context())
    assert caught.value.atlas_diagnostic['model_completion_status']==status
    assert 'PRIVATE' not in json.dumps(caught.value.atlas_diagnostic)

def test_adapter_requests_schema(monkeypatch):
    def post(*a,**kw):
        assert kw['json']['format']['type']=='object'
        class HTTP:
            status_code=200
            def json(self):return {'response':json.dumps(response()),'done':True,'done_reason':'stop'}
        return HTTP()
    monkeypatch.setattr('requests.post',post)
    monkeypatch.setattr(steps,'get_provider',lambda:OllamaProvider())
    assert steps.generate_structured_response_step(context())['subject']=='Draft'

def test_missing_ollama_dependency_is_diagnosed(monkeypatch):
    from app.providers.factory import get_provider
    original=builtins.__import__
    def importing(name,*a,**kw):
        if name=='app.providers.ollama':raise ModuleNotFoundError('PRIVATE',name='requests')
        return original(name,*a,**kw)
    monkeypatch.setenv('PROVIDER_TYPE','ollama');monkeypatch.setattr(builtins,'__import__',importing)
    monkeypatch.setattr(steps,'get_provider',get_provider)
    with pytest.raises(ImportError) as caught:steps.summarize_and_extract_actions_step(context())
    d=caught.value.atlas_diagnostic
    assert d['validation_rule']=='dependency_import' and d['dependency_module']=='requests'
    assert d['model_completion_status']=='not_called' and 'PRIVATE' not in json.dumps(d)

def test_summary_missing_fields_is_schema_error(monkeypatch):
    install(monkeypatch,'{}')
    with pytest.raises(ValueError) as caught:steps.summarize_and_extract_actions_step(context())
    assert caught.value.atlas_diagnostic['schema_version']=='extract_actions.v1'
    assert caught.value.atlas_diagnostic['validation_rule']=='schema_fields'

def test_failure_diagnostic_persisted_and_no_note(db_session,monkeypatch,caplog):
    from app.engine.runner import run_workflow
    from app.storage.models import EmployeeRecord,Case,CaseNote
    emp='DIAG-'+uuid4().hex
    db_session.add(EmployeeRecord(employee_id=emp,metadata_json='{}',last_activities_json='[]'));db_session.commit()
    install(monkeypatch,'{PRIVATE')
    with pytest.raises(ValueError):run_workflow({'employee_id':emp,'case_transcript':'Explain accommodation review.','tone':'direct'})
    case=db_session.query(Case).order_by(Case.id.desc()).first()
    d=json.loads(case.artifacts_json)['last_failure_diagnostic']
    assert case.status=='failed' and d['validation_rule']=='json_syntax'
    assert db_session.query(CaseNote).filter_by(case_id=case.id).count()==0
    assert 'PRIVATE' not in caplog.text and emp not in caplog.text

def test_quality_evaluation_is_not_human_review(db_session):
    from app.storage.models import Case
    from app.benchmarks import case_quality_evaluation
    case=Case(status='completed',inputs_json='{}',outputs_json='{}')
    assert case_quality_evaluation(case)['human_review_status']=='not_recorded'
    assert case_quality_evaluation(case)['status']=='not_evaluated'

@pytest.mark.parametrize('text',['{', '```json\n{"summary":'])
def test_summary_malformed_json(monkeypatch,text):
    install(monkeypatch,text)
    with pytest.raises(ValueError) as caught:steps.summarize_and_extract_actions_step(context())
    assert caught.value.atlas_diagnostic['validation_rule']=='json_syntax'

def test_missing_symbol_is_allowlisted():
    exc=ImportError("cannot import name 'CompletionResponse' from PRIVATE",name='app.providers.base')
    d=failure_diagnostic(exc)
    assert d['dependency_symbol']=='CompletionResponse'
    assert 'PRIVATE' not in json.dumps(d)

def test_unknown_import_message_is_not_logged():
    d=failure_diagnostic(ImportError('PRIVATE employee data',name='PRIVATE'))
    assert d['dependency_module']=='unknown' and d['dependency_symbol']=='unknown'
    assert 'PRIVATE' not in json.dumps(d)
