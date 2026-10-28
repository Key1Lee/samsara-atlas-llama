"""Read-only case snapshots. Sensitive text stays in memory, not in reports."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sqlite3
from app.tools.retrieval import _load_docs
from .contracts import EvidenceReference

def digest(value):
    if not isinstance(value,str): value=json.dumps(value,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(value.encode()).hexdigest()

@dataclass
class CaseEvidence:
    case_id: int
    state: str
    question: str
    answer: dict | None = None
    sources: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    raw_completion: str | None = None
    criteria: list[str] = field(default_factory=list)
    correlation_id: str | None = None
    record_count: int = 1
    duplicate_key_count: int = 1
    inputs_valid: bool = True
    expected_source_ids: list[str] = field(default_factory=list)

    def catalog(self):
        """Exact full-paragraph spans; no synthetic claims about pre-existing chunks."""
        records=[]; texts={}
        def add(kind,target,text):
            offset=0
            for block in text.splitlines(keepends=True):
                end=offset+len(block)
                if block.strip():
                    key=f'{target}:{offset}:{end}'
                    records.append(EvidenceReference(id=key,kind=kind,target=target,start=offset,end=end,sha256=digest(block)))
                    texts[key]=block
                offset=end
        add('question','question',self.question)
        for s in self.sources:
            if s.get('authorized') and s.get('available'):add('source',s['id'],s['text'])
        if self.answer:
            for key in ('subject','body'):
                if isinstance(self.answer.get(key),str):add('output','answer:'+key,self.answer[key])
        for i,event in enumerate(self.events):
            key=f'event:{event.get("id",i)}'
            records.append(EvidenceReference(id=key,kind='event',target=key,sha256=digest(event)))
            texts[key]=json.dumps(event,sort_keys=True)
        for i,text in enumerate(self.criteria):add('criteria',f'criterion:{i}',text)
        return records,texts

def snapshots(database: Path, docs: Path, case_ids=None, employee_id=None):
    """OS-authorized operator CLI only. Never initializes, recovers or writes a case."""
    known={digest(text) for _,text in _load_docs(str(docs))}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        seen=set()
        for row in db.execute('SELECT * FROM cases ORDER BY id'):
            seen.add(row['id'])
            if case_ids is not None and row['id'] not in case_ids:continue
            try:inputs=json.loads(row['inputs_json'] or '{}')
            except ValueError:inputs={}
            if not isinstance(inputs,dict):inputs={}
            if employee_id and inputs.get('employee_id')!=employee_id:continue
            try:answer=json.loads(row['outputs_json']) if row['outputs_json'] else None
            except ValueError:answer=None
            step_rows=[dict(s) for s in db.execute('SELECT * FROM steps WHERE case_id=? ORDER BY id',(row['id'],))]
            events=[dict(e) for e in db.execute('SELECT id,request_id,event_type,step_name,outcome,error_code,attempt,timestamp FROM case_audit_events WHERE case_id=? ORDER BY id',(row['id'],))]
            selected=[]
            for step in step_rows:
                if step['name']=='retrieve_policy_snippets' and step['output_json']:
                    try:selected=json.loads(step['output_json'])
                    except ValueError:selected=[]
            if not isinstance(selected,list):selected=[]
            # Deterministic routes can skip stored retrieval; exact cited snapshots
            # can be resolved only against the operator-authorized current corpus.
            if not selected and isinstance(answer,dict):
                cited={e.get('source_id') for e in answer.get('citation_evidence',[]) if isinstance(e,dict)}
                selected=[text for _,text in _load_docs(str(docs)) if 'sha256:'+digest(text) in cited]
            sources=[]
            for rank,text in enumerate(selected,1):
                if not isinstance(text,str):continue
                h=digest(text)
                sources.append({'id':'sha256:'+h,'chunk_id':'sha256:'+h+':whole',
                    'text':text,'rank':rank,'score':None,'authorized':h in known,'available':h in known,
                    'score_provenance':'not_recorded'})
            try:art=json.loads(row['artifacts_json'] or '{}')
            except ValueError:art={}
            if not isinstance(art,dict):art={}
            for stage in art.get('step_diagnostics',[]):
                for observed in stage.get('retrieval',[]):
                    for source in sources:
                        if source['id']==observed.get('source_id'):
                            source['score']=observed.get('score');source['score_provenance']='recorded'
            duplicate=1
            if row['idempotency_key'] is not None:
                duplicate=db.execute('SELECT COUNT(*) FROM cases WHERE idempotency_key=?',(row['idempotency_key'],)).fetchone()[0]
            yield CaseEvidence(row['id'],row['status'],inputs.get('case_transcript',''),
                answer=answer if isinstance(answer,dict) else None,sources=sources,events=events,steps=step_rows,
                diagnostics=art,correlation_id=next((e['request_id'] for e in reversed(events) if e['request_id']),None),
                duplicate_key_count=duplicate,inputs_valid=bool(inputs.get('employee_id') and inputs.get('case_transcript')))

        if case_ids is not None and employee_id is None:
            for missing in sorted(case_ids-seen):
                yield CaseEvidence(missing,'missing','',record_count=0,inputs_valid=False)
