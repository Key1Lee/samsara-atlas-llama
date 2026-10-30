"""Real persistent Chroma and LangChain splitters; deterministic fake embeddings."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
import pytest

from app.rag.config import Settings
from app.rag.documents import load_sources, split_sources, verify_span
from app.rag.embeddings import validate_vectors
from app.rag.errors import (EmbeddingInvalid, EmbeddingUnavailable, IndexMissing,
    IndexStale, ModelMismatch, PolicyConflict, SourceUnauthorized, SourceUnavailable, IndexCorrupt)
from app.rag.index import ingest, read_manifest, open_store
from app.rag.retrieval import RetrievalService, get_service, verified_citation

ROOT = Path(__file__).resolve().parents[1]
EXPENSE = (ROOT / 'data/docs/expense-policy.md').read_text()
PTO = (ROOT / 'data/docs/pto-policy.md').read_text()
REMOTE = (ROOT / 'data/docs/remote-work-policy.md').read_text()
QUOTE = 'Submit expenses within 30 days of the expense date'


class FakeEmbeddings(Embeddings):
    """An explicitly synthetic embedding fixture, never an accuracy benchmark."""
    def __init__(self):
        self.version = 'f' * 64
        self.calls = []
        self.fail = False
        self.dimensions = 4
    def identity(self):
        return self.version
    def embed_documents(self, texts):
        self.calls.extend(texts)
        if self.fail:
            raise EmbeddingUnavailable()
        output = []
        for text in texts:
            text = text.lower()
            axis = (0 if any(w in text for w in ('expense', 'reimbursement', 'receipt', 'claim', 'spend', 'disputes', 'travel', 'submission'))
                    else 1 if any(w in text for w in ('pto', 'accrual', 'vacation'))
                    else 2 if any(w in text for w in ('remote', 'zone', 'relocating')) else 3)
            v = [0.0] * self.dimensions
            v[axis] = 1.0
            output.append(v)
        return output
    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def rag(tmp_path, monkeypatch):
    docs = tmp_path / 'policies'
    docs.mkdir()
    (docs / 'expense-policy.md').write_text(EXPENSE)
    (docs / 'pto-policy.md').write_text(PTO)
    (docs / 'remote-work-policy.md').write_text(REMOTE)
    settings = Settings(docs_dir=docs, index_dir=tmp_path / 'vectors', model='fake-test-only', min_similarity=.8)
    monkeypatch.setenv('DOCS_DIR', str(docs))
    monkeypatch.setenv('ATLAS_RAG_INDEX_DIR', str(settings.index_dir))
    monkeypatch.setenv('ATLAS_EMBEDDING_MODEL', settings.model)
    monkeypatch.setenv('ATLAS_RAG_MIN_SIMILARITY', '.8')
    provider = FakeEmbeddings()
    monkeypatch.setattr('app.rag.retrieval.LocalEmbeddings', lambda settings: provider)
    get_service.cache_clear()
    yield settings, provider
    get_service.cache_clear()


def test_ingest_exact_metadata_recursive_and_crlf(rag):
    settings, provider = rag
    sub = settings.docs_dir / 'nested'
    sub.mkdir()
    (sub / 'benefit.md').write_bytes(b'# Benefit\r\n\r\n## Eligibility\r\n- Keep every condition, unless local law says otherwise.\r\n')
    result = ingest(settings, provider=provider)
    assert result['documents'] == 4
    sources = load_sources(settings)
    chunks = split_sources(sources, settings, provider.identity())
    for chunk in chunks:
        assert {'document_id','chunk_id','source_file','source_path','document_title','section_heading',
                'content_hash','chunk_index','embedding_model','embedding_version','index_version'} <= set(chunk.metadata)
        verify_span(settings, chunk.metadata, chunk.page_content, {'employee_id':'E-test'})
    assert any('\r\n' in c.page_content for c in chunks if c.metadata['source_path'] == 'nested/benefit.md')
    assert all('approval_status' not in c.metadata for c in chunks)


def test_incremental_reindex_has_no_duplicate_embeddings_or_records(rag):
    settings, provider = rag
    first = ingest(settings, provider=provider)
    count = len(provider.calls)
    again = ingest(settings, provider=provider)
    assert len(provider.calls) == count and again['embedded_chunks'] == 0
    assert again['index_version'] == first['index_version']
    path = settings.docs_dir / 'expense-policy.md'
    path.write_text(EXPENSE + '## Audit\n- Keep records for review.\n')
    update = ingest(settings, provider=provider)
    assert update['changed_documents'] == 1 and update['embedded_chunks'] == 1
    assert update['index_version'] != first['index_version']
    (settings.docs_dir / 'remote-work-policy.md').unlink()
    deleted = ingest(settings, provider=provider)
    assert deleted['deleted_documents'] == 1 and deleted['embedded_chunks'] == 0
    manifest = read_manifest(settings)
    assert len(open_store(str(settings.index_dir),manifest['collection']).get()['ids']) == deleted['chunks']
    assert 'remote-work-policy.md' not in manifest['sources']


def test_failed_rebuild_preserves_active_index(rag):
    settings, provider = rag
    ingest(settings, provider=provider)
    before = (settings.index_dir / 'active.json').read_bytes()
    provider.fail = True
    with pytest.raises(EmbeddingUnavailable):
        ingest(settings, provider=provider, rebuild=True)
    assert before == (settings.index_dir / 'active.json').read_bytes()


@pytest.mark.parametrize('mode', ['semantic','hybrid'])
def test_paraphrase_bypasses_lexical_gate_and_citations_are_exact(rag, mode):
    settings, provider = rag
    ingest(settings, provider=provider)
    query = 'By when must I file a claim?'
    from app.tools.retrieval import policy_topic_matches
    assert not policy_topic_matches(query, EXPENSE)
    result = RetrievalService(settings,provider).retrieve(query, {'employee_id':'E-test'},mode=mode)
    assert result.evidence and all(h['metadata']['source_file']=='expense-policy.md' for h in result.evidence)
    cite = verified_citation(QUOTE, result.passages, result.metadata, settings, {'employee_id':'E-test'})
    assert EXPENSE[cite['start']:cite['end']] == QUOTE and cite['source_verification']=='matched'
    assert cite['approval_status']=='unknown'


def test_complete_governing_section_survives_small_chunks(rag):
    settings,provider=rag
    settings=replace(settings,chunk_size=100,chunk_overlap=10)
    ingest(settings,provider=provider)
    result=RetrievalService(settings,provider).retrieve('PTO accrual',{'employee_id':'E-test'},top_k=1,mode='semantic')
    text='\n'.join(result.passages)
    assert '40 hours' in text and 'unless local law requires payout' in text
    assert 'based on level and tenure' in text


def test_unknown_evidence_is_not_infrastructure_failure(rag):
    settings,provider=rag
    # An unknown fixture vector is orthogonal to all policy topic vectors except
    # unmatched source headings; use only the expense corpus for this assertion.
    (settings.docs_dir/'pto-policy.md').unlink()
    (settings.docs_dir/'remote-work-policy.md').unlink()
    ingest(settings,provider=provider)
    result=RetrievalService(settings,provider).retrieve('Iguana habitat temperature?',{},mode='semantic')
    assert result.evidence==[] and result.metadata['outcome']=='no_relevant_evidence'


def test_missing_stale_corrupt_model_changed_and_unavailable_are_distinct(rag):
    settings,provider=rag
    service=RetrievalService(settings,provider)
    with pytest.raises(IndexMissing): service.retrieve('expense',{})
    ingest(settings,provider=provider)
    provider.version='e'*64
    with pytest.raises(ModelMismatch):service.retrieve('expense',{})
    provider.version='f'*64
    provider.dimensions=5
    with pytest.raises(EmbeddingInvalid):service.retrieve('expense',{})
    provider.dimensions=4
    provider.fail=True
    with pytest.raises(EmbeddingUnavailable):service.retrieve('expense',{})
    provider.fail=False
    (settings.docs_dir/'expense-policy.md').write_text(EXPENSE+'\nChanged\n')
    with pytest.raises(IndexStale):service.retrieve('expense',{})
    (settings.index_dir/'active.json').write_text('{}')
    with pytest.raises(IndexCorrupt):service.retrieve('expense',{})


@pytest.mark.parametrize('vectors', [[], [[]], [[float('nan')]], [[float('inf')]], [[0,0]], [[True]], [[1],[1,2]]])
def test_invalid_vectors_rejected(vectors):
    with pytest.raises(EmbeddingInvalid):validate_vectors(vectors,1)


def test_model_change_requires_explicit_rebuild(rag):
    settings,provider=rag
    ingest(settings,provider=provider)
    provider.version='e'*64
    with pytest.raises(ModelMismatch):ingest(settings,provider=provider)
    rebuilt=ingest(settings,provider=provider,rebuild=True)
    assert rebuilt['embedding_version']=='e'*64


def test_symlink_outside_policy_root_is_rejected(rag,tmp_path):
    settings,_=rag
    secret=tmp_path/'employee-record.md'
    secret.write_text('PRIVATE NOT POLICY')
    (settings.docs_dir/'escape.md').symlink_to(secret)
    with pytest.raises(SourceUnavailable):load_sources(settings)


def test_missing_unreadable_and_non_utf8_sources_explicit(rag,monkeypatch):
    settings,_=rag
    with pytest.raises(SourceUnavailable):load_sources(replace(settings,docs_dir=settings.docs_dir/'missing'))
    (settings.docs_dir/'bad.md').write_bytes(b'\xff')
    with pytest.raises(SourceUnavailable):load_sources(settings)
    (settings.docs_dir/'bad.md').unlink()
    original=Path.open
    def denied(path,*args,**kwargs):
        if path.name=='expense-policy.md':raise PermissionError()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',denied)
    with pytest.raises(SourceUnavailable):load_sources(settings)


def test_conflicting_versions_and_outdated_policy_rejected(rag,tmp_path):
    settings,provider=rag
    (settings.docs_dir/'old-expense.md').write_text(EXPENSE+'\n- Deadline is 60 days.\n')
    with pytest.raises(PolicyConflict):ingest(settings,provider=provider)
    (settings.docs_dir/'old-expense.md').unlink()
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'expense-policy.md':{'approval_status':'approved','expires_at':'2000-01-01'}}))
    settings=replace(settings,metadata_file=str(metadata))
    ingest(settings,provider=provider)
    with pytest.raises(PolicyConflict):RetrievalService(settings,provider).retrieve('expense',{})


def test_authorization_applies_before_returning_sources(rag,tmp_path):
    settings,provider=rag
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'expense-policy.md':{'access_scope':['E-allowed']}}))
    settings=replace(settings,metadata_file=str(metadata))
    ingest(settings,provider=provider)
    with pytest.raises(SourceUnauthorized):RetrievalService(settings,provider).retrieve('expense',{'employee_id':'E-denied'})
    result=RetrievalService(settings,provider).retrieve('expense',{'employee_id':'E-allowed'})
    assert result.evidence


def test_changed_source_cannot_validate_saved_citation(rag):
    settings,provider=rag
    ingest(settings,provider=provider)
    result=RetrievalService(settings,provider).retrieve('expense',{})
    (settings.docs_dir/'expense-policy.md').write_text(EXPENSE.replace('30 days','60 days'))
    with pytest.raises(IndexStale):verified_citation(QUOTE,result.passages,result.metadata,settings,{})


def test_no_network_config_outside_loopback():
    with pytest.raises(ValueError):Settings(base_url='https://external.example')


def test_diagnostics_disabled_and_requires_operator_plus_employee_scope(rag,monkeypatch):
    from app.main import app
    settings,provider=rag
    ingest(settings,provider=provider)
    client=TestClient(app)
    body={'question':'By when must I file a claim?','mode':'semantic'}
    assert client.post('/diagnostics/retrieval',json=body).status_code==404
    monkeypatch.setenv('ATLAS_RAG_DIAGNOSTICS','1')
    monkeypatch.setenv('ATLAS_RAG_DIAGNOSTICS_TOKEN_HASH',hashlib.sha256(b'developer-test').hexdigest())
    assert client.post('/diagnostics/retrieval',json=body).status_code==403
    headers={'X-Atlas-Diagnostics-Token':'developer-test','X-Employee-ID':'RAG-test'}
    response=client.post('/diagnostics/retrieval',json=body,headers=headers)
    assert response.status_code==200
    data=response.json()
    assert data['candidates'] and data['request_id']
    assert 'passage' not in response.text and 'source_path' not in response.text
    monkeypatch.setenv('ATLAS_ENV','production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES',json.dumps({'RAG-test':hashlib.sha256(b'employee-test').hexdigest()}))
    assert client.post('/diagnostics/retrieval',json=body,headers=headers).status_code==401
    headers['Authorization']='Bearer employee-test'
    assert client.post('/diagnostics/retrieval',json=body,headers=headers).status_code==200

@pytest.fixture
def workflow_client(rag, db_session, monkeypatch):
    from app.main import app
    from app.storage.models import EmployeeRecord
    settings,provider=rag
    ingest(settings,provider=provider)
    monkeypatch.setenv('ATLAS_RETRIEVAL_MODE','hybrid')
    db_session.add(EmployeeRecord(employee_id='RAG-'+settings.index_dir.parent.name,
        metadata_json='{}',last_activities_json='[]'))
    db_session.commit()
    employee='RAG-'+settings.index_dir.parent.name
    with TestClient(app,headers={'X-Employee-ID':employee}) as client:
        yield client,employee


def _completion(prompt,**kwargs):
    if 'subject' not in prompt.lower():
        return json.dumps({'summary':'Expense filing deadline question','action_items':[], 'next_steps':[]})
    return json.dumps({'subject':'Expense filing deadline',
        'body':'Submit expenses within 30 days of the expense date. Ask HR to confirm applicability.',
        'tone':'empathetic','cited_snippets':[QUOTE]})


def test_end_to_end_paraphrase_retry_citation_and_single_note(workflow_client,monkeypatch):
    client,employee=workflow_client
    monkeypatch.setattr('app.engine.steps._retryable_llm_call',_completion)
    payload={'inputs':{'employee_id':employee,'case_transcript':'By when must I file a claim?','tone':'empathetic'},
             'idempotency_key':'rag-e2e-'+employee}
    response=client.post('/cases',json=payload)
    assert response.status_code==200,response.text
    first=response.json()
    assert client.post('/cases',json=payload).json()==first
    outputs=first['outputs']
    assert outputs['citation_evidence'][0]['quote']==QUOTE
    assert outputs['quality_status']=='needs_hr_review' and first['review_status'] is None
    stored=client.get('/cases/'+str(first['case_id'])).json()
    assert len(stored['case_notes'])==1 and len(stored['steps'])==5


@pytest.mark.parametrize('bad', ['not JSON','{}','[]','fabricated_citation','unsupported_action','missing_citation','missing_import'])
def test_generation_failures_are_separate_and_never_create_note(workflow_client,monkeypatch,bad):
    client,employee=workflow_client
    def complete(prompt,**kwargs):
        if 'subject' not in prompt.lower():return _completion(prompt,**kwargs)
        if bad=='missing_import':raise ImportError("cannot import name 'CompletionResponse'",name='app.providers.base')
        if bad in {'not JSON','{}','[]'}:return bad
        answer=json.loads(_completion(prompt,**kwargs))
        if bad=='fabricated_citation':answer['cited_snippets']=['Made up policy']
        if bad=='unsupported_action':answer['body']='I have approved your reimbursement.'
        if bad=='missing_citation':answer['cited_snippets']=[]
        return json.dumps(answer)
    monkeypatch.setattr('app.engine.steps._retryable_llm_call',complete)
    response=client.post('/cases',json={'inputs':{'employee_id':employee,'case_transcript':'By when must I file a claim?','tone':'empathetic'},'idempotency_key':employee+bad})
    assert response.status_code==500
    case_id=client.get('/cases').json()['case_ids'][0]
    stored=client.get(f'/cases/{case_id}').json()
    assert stored['case_notes']==[]
    assert next(s for s in stored['steps'] if s['name']=='generate_structured_response')['status']=='failed'
    from app.storage.database import SessionLocal
    from app.storage.models import Case
    with SessionLocal() as db:
        diagnostic=json.loads(db.get(Case,case_id).artifacts_json)['last_failure_diagnostic']
    expected='json_parse' if bad=='not JSON' else 'schema_validation' if bad=='{}' else 'format_validation' if bad=='[]' else 'dependency' if bad=='missing_import' else 'semantic_validation'
    assert diagnostic['error_subtype']==expected
    assert diagnostic['request_id'] and diagnostic['schema_version']=='structured_response.v1'


def test_empty_retrieval_skips_model_and_note(workflow_client,rag,monkeypatch):
    client,employee=workflow_client
    settings,provider=rag
    # Every expense passage embeds on axis zero; unknown query on axis three.
    (settings.docs_dir/'pto-policy.md').unlink()
    (settings.docs_dir/'remote-work-policy.md').unlink()
    ingest(settings,provider=provider)
    def must_not_call(*args,**kwargs):raise AssertionError('Unexpected generation')
    monkeypatch.setattr('app.engine.steps._retryable_llm_call',must_not_call)
    response=client.post('/cases',json={'inputs':{'employee_id':employee,'case_transcript':'Aquarium water salinity?','tone':'empathetic'}})
    assert response.status_code==200
    stored=client.get('/cases/'+str(response.json()['case_id'])).json()
    assert stored['case_notes']==[]
    assert not any(s['name']=='summarize_and_extract_actions' for s in stored['steps'])
    assert response.json()['outputs']['answerability']['reason']=='unsupported_policy_topic'


def test_index_failure_persisted_on_retrieval_step_and_same_key_can_resume(workflow_client,rag,monkeypatch):
    client,employee=workflow_client
    settings,provider=rag
    (settings.index_dir/'active.json').unlink()
    payload={'inputs':{'employee_id':employee,'case_transcript':'By when must I file a claim?','tone':'empathetic'},'idempotency_key':employee+'recover'}
    response=client.post('/cases',json=payload)
    assert response.status_code==503 and response.json()['detail']['error_code']=='rag_index_missing'
    case_id=client.get('/cases').json()['case_ids'][0]
    stored=client.get(f'/cases/{case_id}').json()
    assert next(s for s in stored['steps'] if s['name']=='retrieve_policy_snippets')['status']=='failed'
    assert stored['case_notes']==[]
    ingest(settings,provider=provider)
    monkeypatch.setattr('app.engine.steps._retryable_llm_call',_completion)
    response=client.post('/cases',json=payload)
    assert response.status_code==200 and response.json()['case_id']==case_id
    assert len(client.get(f'/cases/{case_id}').json()['case_notes'])==1


def test_deterministic_pto_route_preserves_access_controls(workflow_client,rag,tmp_path,monkeypatch):
    client,employee=workflow_client
    metadata=tmp_path/'access.json'
    metadata.write_text(json.dumps({'pto-policy.md':{'access_scope':['different-employee']}}))
    monkeypatch.setenv('ATLAS_POLICY_METADATA',str(metadata))
    response=client.post('/cases',json={'inputs':{'employee_id':employee,'case_transcript':'How much PTO can I carry over?','tone':'empathetic'}})
    assert response.status_code==403 and response.json()['detail']['error_code']=='rag_source_unauthorized'


def test_context_limit_blocks_generation_without_dropping_evidence(workflow_client,rag,monkeypatch):
    client,employee=workflow_client
    monkeypatch.setenv('OLLAMA_CONTEXT_TOKENS','1100')
    monkeypatch.setattr('app.engine.steps._retryable_llm_call',lambda *a,**kw:pytest.fail('Model should not run'))
    response=client.post('/cases',json={'inputs':{'employee_id':employee,'case_transcript':'By when must I file a claim?','tone':'empathetic'}})
    assert response.status_code==500
    stored=client.get('/cases/'+str(client.get('/cases').json()['case_ids'][0])).json()
    assert stored['case_notes']==[]
    assert next(s for s in stored['steps'] if s['name']=='summarize_and_extract_actions')['error']=='ContextBudgetExceeded'


def test_chroma_staging_write_failure_keeps_old_collection(rag,monkeypatch):
    from langchain_chroma import Chroma
    settings,provider=rag
    ingest(settings,provider=provider)
    before=(settings.index_dir/'active.json').read_bytes()
    monkeypatch.setattr(Chroma,'add_documents',lambda *a,**kw:(_ for _ in ()).throw(OSError('disk error')))
    with pytest.raises(OSError):ingest(settings,provider=provider,rebuild=True)
    assert (settings.index_dir/'active.json').read_bytes()==before
    assert RetrievalService(settings,provider).retrieve('expense',{}).evidence


def test_embedding_adapter_sets_no_truncation(monkeypatch):
    from app.rag.embeddings import StrictOllamaEmbeddings
    adapter=StrictOllamaEmbeddings(model='test',base_url='http://127.0.0.1:11434')
    calls=[]
    class Client:
        def embed(self,*args,**kwargs):
            calls.append(kwargs)
            return {'embeddings':[[1.0,0.0]]}
    adapter._client=Client()
    assert adapter.embed_documents(['text'])==[[1.0,0.0]]
    assert calls[0]['truncate'] is False
