"""Deterministic execution and format facts; never infer semantic correctness."""
import json
from pydantic import ValidationError
from app.schemas import StructuredResponse
from .contracts import Check
from .evidence import digest

def validate(case):
    checks=[]
    def check(code,status,refs=(),**details):
        checks.append(Check(code=code,status=status,evidence_ids=list(refs),details=details))
    event_refs=[f'event:{e.get("id",i)}' for i,e in enumerate(case.events)]
    check('record_count','pass' if case.record_count==1 else 'fail',count=case.record_count)
    check('duplicate_key','pass' if case.duplicate_key_count<=1 else 'fail',count=case.duplicate_key_count)
    check('case_inputs','pass' if case.inputs_valid else 'fail')
    check('workflow_completed','pass' if case.state=='completed' else 'fail',event_refs,state=case.state if case.state in {'failed','completed','running','pending'} else 'unknown')
    failures=[s for s in case.steps if s.get('status')=='failed']
    known={'ValueError','JSONDecodeError','ImportError','ValidationError','ContextBudgetExceeded','ProviderTimeout','RetryError'}
    for step in failures:
        error=step.get('error')
        code={'JSONDecodeError':'historical_json_parse','ImportError':'historical_import','ValueError':'historical_value_error',
              'ValidationError':'historical_schema','ProviderTimeout':'provider_timeout'}.get(error,'workflow_failure')
        check(code,'fail',event_refs,error_type=error if error in known else 'unknown',
              stage=step.get('name') if step.get('name') in {'retrieve_employee_context','retrieve_policy_snippets','summarize_and_extract_actions','generate_structured_response','post_case_note_to_hris'} else 'unknown',
              attempts=step.get('attempts'))
    retrieval=[s for s in case.steps if s.get('name')=='retrieve_policy_snippets']
    check('retrieval_execution','pass' if retrieval and retrieval[-1].get('status')=='completed' else ('fail' if retrieval else 'not_applicable'),event_refs)
    check('retrieval_nonempty','pass' if case.sources else 'unknown',selected_count=len(case.sources),note='nonempty_is_not_usefulness')
    if case.expected_source_ids:
        check('retrieval_expected_sources','pass' if set(case.expected_source_ids)<={s['id'] for s in case.sources} else 'fail',expected_ids=case.expected_source_ids)
    check('source_access','pass' if all(s.get('authorized') and s.get('available') for s in case.sources) else 'fail',
          available_count=sum(bool(s.get('authorized') and s.get('available')) for s in case.sources))
    answer=case.answer
    if case.raw_completion is not None:
        try:answer=json.loads(case.raw_completion);check('json_parse','pass')
        except ValueError:check('json_parse','fail');answer=None
    else:check('json_parse','unknown' if case.answer is None else 'pass',provenance='stored_parsed_output' if case.answer else 'raw_completion_not_retained')
    check('raw_generation_evidence','pass' if case.raw_completion is not None else 'unknown')
    generation=next((s for s in case.steps if s.get('name')=='generate_structured_response'),{})
    retained=bool(generation.get('output_json') or case.answer)
    check('generation_output_retained','pass' if retained or case.raw_completion is not None else 'unknown')
    if answer is None:
        check('response_schema','unknown');check('citation_validity','unknown')
    else:
        try:StructuredResponse.model_validate(answer);check('response_schema','pass')
        except ValidationError as exc:
            fields={'subject','body','tone','cited_snippets'}
            errors=[{'path':'/'+str(e['loc'][0]) if e['loc'] and e['loc'][0] in fields else '/',
                     'type':e['type'] if e['type'] in {'missing','string_type','string_too_short','list_type','literal_error','model_type'} else 'invalid'}
                    for e in exc.errors(include_input=False,include_context=False)]
            check('response_schema','fail',errors=errors)
        if isinstance(answer,dict):
            citations=answer.get('cited_snippets',[]);bad=[]
            for i,quote in enumerate(citations if isinstance(citations,list) else []):
                matches=[s for s in case.sources if isinstance(quote,str) and quote.strip() and quote in s['text'] and s.get('available') and s.get('authorized')]
                if len({s['id'] for s in matches})!=1:bad.append(i)
            evidence=answer.get('citation_evidence',[])
            if isinstance(evidence,list):
                for i,e in enumerate(evidence):
                    if not isinstance(e,dict):bad.append(i);continue
                    match=next((s for s in case.sources if s['id']==e.get('source_id') and s.get('available') and s.get('authorized')),None)
                    start,end=e.get('start'),e.get('end')
                    if not match or not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(match['text']) or match['text'][start:end]!=e.get('quote'):bad.append(i)
            check('citation_validity','fail' if bad else 'pass',bad_indices=sorted(set(bad)))
    # Audit failures are confirmed observations; hidden completion/timeout is unknown.
    latest=case.diagnostics.get('last_failure_diagnostic',{})
    from app.diagnostics import RULES
    rule=latest.get('validation_rule')
    if rule in RULES:
        check('recorded_validation_rule','fail' if rule!='unclassified' else 'unknown',event_refs,rule=rule)
    status=latest.get('model_completion_status','unknown')
    check('generation_finish','fail' if status in {'truncated','refused'} else ('pass' if status=='completed' else 'unknown'),finish_status=status if status in {'truncated','refused','completed','not_called','unknown','requested'} else 'unknown')
    timeout=any(e.get('error_code')=='provider_timeout' for e in case.events)
    check('provider_timeout','fail' if timeout else 'unknown',event_refs)
    transitions=[e['event_type'] for e in case.events]
    terminal='case_completed' if case.state=='completed' else 'case_failed'
    check('terminal_event','pass' if terminal in transitions else 'unknown',event_refs)
    terminals=[t for t in transitions if t in {'case_completed','case_failed'}]
    check('workflow_state_transition', 'fail' if terminals and terminals[-1]!=terminal else ('pass' if terminals else 'unknown'),event_refs)
    return checks
