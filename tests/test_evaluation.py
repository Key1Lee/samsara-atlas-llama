"""Offline evaluator contracts, authorization of evidence, privacy and degradation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from dataclasses import replace
import pytest
import requests
from app.evaluation.artifacts import ArtifactStore
from app.evaluation.checks import validate
from app.evaluation.cli import dataset, run_cases
from app.evaluation.contracts import SemanticScores, SemanticVerdict, Finding, Evaluation
from app.evaluation.evidence import CaseEvidence, snapshots
from app.evaluation.harness import evaluate, OfflineQueue
from app.evaluation.local import Config, LocalEvaluator, EvaluationFailure
from app.evaluation.reporting import agreement

FIXTURE=Path(__file__).parent/'fixtures/evaluation_v1.json'
DIGEST='a'*64

@pytest.fixture
def case():return dataset(FIXTURE,'dev')[0][0]

def verdict(findings=()):
    return SemanticVerdict(scores=SemanticScores(**{k:1 for k in SemanticScores.model_fields}),findings=list(findings),confidence=.7,requires_human_review=True)

class Judge:
    def __init__(self,result=None,error=None):self.result=result or verdict();self.error=error;self.calls=0
    def evaluate(self,payload):
        self.calls+=1
        if self.error:raise self.error
        return self.result,{'latency_seconds':.01,'eval_count':20}

@pytest.mark.parametrize('split',['dev','test'])
def test_fixture_deterministic_contracts(split):
    cases,labels,data=dataset(FIXTURE,split)
    rows={r['case']['case_id']:r for r in data['cases']}
    for c in cases:
        assert {ch.code for ch in validate(c) if ch.status=='fail'}==set(rows[c.case_id]['expected_failed_checks'])

def test_splits_are_disjoint_and_labels_not_human():
    a,_,d=dataset(FIXTURE,'dev');b,_,_=dataset(FIXTURE,'test')
    assert not {c.case_id for c in a}&{c.case_id for c in b}
    assert all(not row['human_reviewed'] for row in d['cases'])

def test_disabled_and_failed_cases_never_call_judge(case):
    judge=Judge();cfg=Config()
    assert evaluate(case,cfg,judge,True).evaluation_status=='disabled'
    case.state='failed';case.answer=None
    assert evaluate(case,replace(cfg,enabled=True),judge,True).evaluation_status=='skipped_no_answer'
    assert judge.calls==0

@pytest.mark.parametrize('status',['timeout','unavailable','truncated','refused','invalid_output','context_limit'])
def test_evaluator_failure_does_not_change_case(case,status):
    original=copy.deepcopy(case)
    result=evaluate(case,Config(enabled=True),Judge(error=EvaluationFailure(status)),True)
    assert result.evaluation_status==status and case==original
    assert result.confirmed_root_cause is None and result.requires_human_review

def test_invented_evidence_rejected(case):
    judge=Judge(verdict([Finding(category='unsupported_claim',assessment='concern',evidence_ids=['invented'])]))
    assert evaluate(case,Config(enabled=True),judge,True).evaluation_status=='invalid_output'

def test_semantic_findings_require_exact_sources_and_outputs(case):
    refs,_=case.catalog();source=next(r.id for r in refs if r.kind=='source');output=next(r.id for r in refs if r.kind=='output')
    incomplete=Judge(verdict([Finding(category='grounding',assessment='concern',evidence_ids=[source])]))
    assert evaluate(case,Config(enabled=True),incomplete,True).evaluation_status=='invalid_output'
    judge=Judge(verdict([Finding(category='grounding',assessment='concern',evidence_ids=[source,output])]))
    r=evaluate(case,Config(enabled=True),judge,True)
    assert r.evaluation_status=='completed' and r.confirmed_root_cause is None
    assert r.requires_human_review and r.human_review_status=='not_recorded'
    for ref in r.evidence_references:
        if ref.start is not None:
            assert ref.end>ref.start and len(ref.sha256)==64

def test_unauthorized_source_skips_semantics(case):
    case.sources[0]['authorized']=False;judge=Judge()
    result=evaluate(case,Config(enabled=True),judge,True)
    assert result.evaluation_status=='evidence_invalid' and judge.calls==0
    assert not any(r.kind=='source' for r in result.evidence_references)

def test_schema_error_has_paths_without_content(case):
    case.answer={'subject':'PRIVATE'}
    checks=validate(case);s=next(c for c in checks if c.code=='response_schema')
    assert '/body' in {e['path'] for e in s.details['errors']}
    assert 'PRIVATE' not in s.model_dump_json()

def test_artifacts_private_and_no_raw(case,tmp_path):
    root=tmp_path/'private';store=ArtifactStore(root)
    result=evaluate(case,Config(),None)
    store.write('case-1.json',result.model_dump())
    assert root.stat().st_mode&0o777==0o700
    assert (root/'case-1.json').stat().st_mode&0o777==0o600
    data=(root/'case-1.json').read_text()
    assert case.question not in data and case.answer['body'] not in data and case.sources[0]['text'] not in data
    (root/'link.json').symlink_to(root/'case-1.json')
    with pytest.raises(ValueError):store.write('link.json',{})
    with pytest.raises(ValueError):store.write('../elsewhere.json',{})
    os.chmod(root,0o755)
    with pytest.raises(ValueError):ArtifactStore(root)

@pytest.mark.parametrize('url',['https://example.com','http://192.168.0.1:11434','http://127.0.0.1.evil.test:11434','http://user:pass@localhost:11434','http://localhost:11434/cloud'])
def test_nonlocal_endpoints_rejected(url):
    with pytest.raises(ValueError):Config(endpoint=url)

def fake_session(monkeypatch,evaluator,payload,wrong_digest=False):
    def request(method,url,**kwargs):
        assert kwargs['allow_redirects'] is False
        assert kwargs['timeout'][1]<=300
        if url.endswith('/api/version'):data={'version':'test-runtime'}
        elif url.endswith('/api/tags'):data={'models':[{'name':'gpt-oss:20b','digest':('b'*64 if wrong_digest else DIGEST),'size':123}]}
        elif url.endswith('/api/show'):data={'details':{'family':'gptoss','quantization_level':'MXFP4'},'capabilities':['completion']}
        elif url.endswith('/api/ps'):data={'models':[]}
        else:
            assert kwargs['json']['format']['type']=='object'
            assert kwargs['json']['model']=='gpt-oss:20b'
            assert 'tools' not in kwargs['json']
            data=payload
        class Response:
            status_code=200
            def json(self):return data
        return Response()
    monkeypatch.setattr(evaluator.session,'request',request)

def test_local_adapter_valid_json_and_metrics(monkeypatch):
    e=LocalEvaluator(Config(enabled=True,digest=DIGEST));fake_session(monkeypatch,e,{'done':True,'done_reason':'stop','message':{'content':verdict().model_dump_json()},'eval_count':12})
    result,metrics=e.evaluate({'question':'SYNTHETIC'})
    assert metrics['digest']==DIGEST and metrics['eval_count']==12
    assert e.session.trust_env is False

@pytest.mark.parametrize('payload,status',[
    ({'done':True,'done_reason':'length','message':{'content':'{}'}},'truncated'),
    ({'done':True,'done_reason':'refusal','refusal':'PRIVATE'},'refused'),
    ({'done':True,'done_reason':'stop','message':{'content':'PRIVATE'}},'invalid_output')])
def test_local_adapter_handles_finish_and_parse(monkeypatch,payload,status):
    e=LocalEvaluator(Config(enabled=True,digest=DIGEST));fake_session(monkeypatch,e,payload)
    with pytest.raises(EvaluationFailure) as caught:e.evaluate({})
    assert caught.value.status==status and 'PRIVATE' not in json.dumps(caught.value.metrics)

def test_digest_pin_and_timeout(monkeypatch):
    e=LocalEvaluator(Config(enabled=True,digest=DIGEST));fake_session(monkeypatch,e,{},wrong_digest=True)
    with pytest.raises(EvaluationFailure) as caught:e.preflight()
    assert caught.value.status=='unavailable'
    def timeout(*a,**kw):raise requests.Timeout('PRIVATE')
    monkeypatch.setattr(e.session,'request',timeout)
    with pytest.raises(EvaluationFailure) as caught:e.preflight()
    assert caught.value.status=='timeout' and 'PRIVATE' not in str(caught.value)

def test_bounded_queue_backpressure(case):
    event=threading.Event()
    class Slow(Judge):
        def evaluate(self,payload):event.wait(3);return super().evaluate(payload)
    queue=OfflineQueue(Config(enabled=True,queue_capacity=1),Slow())
    try:
        f=queue.submit(case)
        assert queue.submit(case) is None and queue.rejected==1
        event.set();assert f.result(timeout=3).evaluation_status=='completed'
    finally:event.set();queue.close()

def test_agreement_counts_false_positives_false_negatives(case):
    refs,_=case.catalog();output=next(r.id for r in refs if r.kind=='output')
    r=evaluate(case,Config(enabled=True),Judge(verdict([Finding(category='unsupported_claim',assessment='concern',evidence_ids=[output])])),True)
    a=agreement([r],{'1':['completeness']})
    assert a['per_category']['unsupported_claim']['fp']==1
    assert a['per_category']['completeness']['fn']==1 and a['disagreement_rate']==1

def test_readonly_snapshot_and_scope(db_session,tmp_path):
    from app.storage.models import Case
    row=Case(status='failed',inputs_json=json.dumps({'employee_id':'SYNTH-A','case_transcript':'PRIVATE','tone':'direct'}),artifacts_json='{}')
    db_session.add(row);db_session.commit()
    database=Path(db_session.bind.url.database)
    rows=list(snapshots(database,tmp_path,{row.id},'SYNTH-A'))
    assert len(rows)==1 and rows[0].question=='PRIVATE'
    assert list(snapshots(database,tmp_path,{row.id},'OTHER'))==[]
    assert db_session.get(Case,row.id).status=='failed'

def test_opt_in_retrieval_observation_does_not_change_rank(monkeypatch):
    from app.evaluation.observation import begin,end
    from app.tools.retrieval import retrieve_policy_snippets
    before=retrieve_policy_snippets('How does PTO accrue?')
    monkeypatch.setenv('ATLAS_EVAL_DIAGNOSTICS','1')
    obs,token=begin('retrieve_policy_snippets')
    try:after=retrieve_policy_snippets('How does PTO accrue?')
    finally:end(obs,token)
    assert before==after and obs['retrieval']
    assert obs['retrieval'][0]['rank']==1 and isinstance(obs['retrieval'][0]['score'],float)
    assert all('text' not in r for r in obs['retrieval'])

def test_failure_stack_and_schema_paths_do_not_leak(monkeypatch):
    from app.engine import steps
    from app.engine.context import ExecutionContext
    from app.providers.base import CompletionResponse
    class Provider:
        def complete(self,*a,**kw):return CompletionResponse('{PRIVATE')
    monkeypatch.setattr(steps,'get_provider',lambda:Provider())
    c=ExecutionContext(0,{'employee_id':'PRIVATE','case_transcript':'Question','tone':'direct'},{})
    with pytest.raises(ValueError) as caught:steps.summarize_and_extract_actions_step(c)
    d=caught.value.atlas_diagnostic
    assert d['stack_trace'] and d['failing_code_location']['file'].startswith('app/')
    assert 'PRIVATE' not in json.dumps(d)

def test_opt_in_failure_evidence_is_persisted_without_content(db_session,monkeypatch):
    from app.engine.runner import run_workflow
    from app.engine import steps
    from app.providers.base import CompletionResponse
    from app.storage.models import Case,EmployeeRecord
    monkeypatch.setenv('ATLAS_EVAL_DIAGNOSTICS','1')
    db_session.add(EmployeeRecord(employee_id='OBS-SYNTH',metadata_json='{}',last_activities_json='[]'));db_session.commit()
    class Invalid:
        def complete(self,*a,**kw):return CompletionResponse('{PRIVATE')
    monkeypatch.setattr(steps,'get_provider',lambda:Invalid())
    with pytest.raises(ValueError):run_workflow({'employee_id':'OBS-SYNTH','case_transcript':'Explain accommodation review.','tone':'direct'})
    row=db_session.query(Case).order_by(Case.id.desc()).first()
    metadata=json.loads(row.artifacts_json)
    assert metadata['step_diagnostics'][-1]['parsing']['succeeded'] is False
    assert metadata['step_diagnostics'][1]['retrieval']
    assert 'PRIVATE' not in row.artifacts_json and 'OBS-SYNTH' not in row.artifacts_json

def test_missing_duplicate_and_transition_checks(case):
    case.record_count=0;case.duplicate_key_count=2
    case.events=[{'id':1,'event_type':'case_failed','outcome':'failure'}]
    checks={c.code:c.status for c in validate(case)}
    assert checks['record_count']=='fail' and checks['duplicate_key']=='fail'
    assert checks['workflow_state_transition']=='fail'

def test_explicit_missing_case_record_is_reported(db_session,tmp_path):
    database=Path(db_session.bind.url.database)
    result=list(snapshots(database,tmp_path,{999999999}))
    assert len(result)==1 and result[0].record_count==0
    assert next(c for c in validate(result[0]) if c.code=='record_count').status=='fail'
