"""Independent deterministic and advisory layers with bounded offline scheduling."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import time
from .checks import validate
from .contracts import Evaluation
from .evidence import digest
from .local import EvaluationFailure

ACTIONS={'completed':'inspect_evidence_and_human_review','skipped_no_answer':'inspect_execution_evidence',
         'deterministic_only':'semantic_evaluation_not_run','disabled':'enable_evaluator_for_offline_run'}

def evaluate(case,config,adapter=None,semantic=False):
    start=time.monotonic();checks=validate(case);references,texts=case.catalog()
    failures=list(dict.fromkeys(c.code for c in checks if c.status=='fail'))
    failed_step=next((s.get('name') for s in reversed(case.steps) if s.get('status')=='failed'),None)
    allowed_stages={'retrieve_employee_context','retrieve_policy_snippets','summarize_and_extract_actions','generate_structured_response','post_case_note_to_hris'}
    result=Evaluation(case_id=case.case_id,correlation_id=case.correlation_id,
        workflow_stage=failed_step if failed_step in allowed_stages else None,
        evaluator_model=config.model,evaluator_version=config.digest or None,
        evaluation_status='deterministic_only',deterministic_checks=checks,failure_categories=failures,
        evidence_references=references,recommended_next_action='inspect_execution_evidence' if failures else 'semantic_evaluation_not_run',
        metrics={'snapshot_sha256':digest(asdict(case)),'raw_completion_retained':False})
    if not semantic:status='deterministic_only'
    elif case.state!='completed' or case.answer is None:status='skipped_no_answer'
    elif not config.enabled:status='disabled'
    elif any(c.status=='fail' and c.code in {'source_access','response_schema','json_parse','record_count','duplicate_key'} for c in checks):status='evidence_invalid'
    else:
        try:
            payload={'case_id':case.case_id,'question':case.question,'answer':case.answer,
                     'evidence_catalog':[{'id':r.id,'kind':r.kind,'text':texts[r.id]} for r in references],
                     'deterministic_checks':[c.model_dump() for c in checks],
                     'criteria':case.criteria,'execution_state':case.state,
                     'caution':'No human approval is recorded. Unknown execution evidence must remain unknown.'}
            verdict,metrics=adapter.evaluate(payload)
            by_id={r.id:r for r in references}
            for finding in verdict.findings:
                if any(ref not in by_id for ref in finding.evidence_ids):raise EvaluationFailure('invalid_output',{**metrics,'failure_rule':'evidence_id_unknown'})
                kinds={by_id[ref].kind for ref in finding.evidence_ids}
                if finding.category in {'grounding','contradiction','citation_alignment','policy_interpretation'}:
                    if 'output' not in kinds or (case.sources and 'source' not in kinds):raise EvaluationFailure('invalid_output',{**metrics,'failure_rule':'source_and_output_required'})
                if finding.category=='completeness' and finding.assessment=='concern' and 'criteria' not in kinds:raise EvaluationFailure('invalid_output',metrics)
                if finding.category in {'unsupported_claim','execution_claim'} and 'output' not in kinds:raise EvaluationFailure('invalid_output',metrics)
            result.semantic_scores=verdict.scores;result.confidence=verdict.confidence;result.findings=verdict.findings
            result.metrics.update(metrics)
            result.failure_categories += ['semantic_'+f.category for f in verdict.findings if f.assessment=='concern']
            result.suspected_cause='semantic_answer_issue' if verdict.findings else None
            # Always advisory: even a judge's high scores cannot grant approval.
            status='completed'
        except EvaluationFailure as exc:status=exc.status;result.metrics.update(exc.metrics)
        except Exception as exc:
            from app.diagnostics import failure_diagnostic
            result.metrics['failure_diagnostic']=failure_diagnostic(exc,'atlas.evaluation.v1')
            # Graceful degradation only at the evaluator boundary; no employee workflow affected.
            status='runtime_error'
    result.evaluation_status=status
    result.recommended_next_action=ACTIONS.get(status,'inspect_evaluator_failure_and_retry_offline')
    result.metrics['total_seconds']=time.monotonic()-start
    return result

class OfflineQueue:
    """One inference at a time, finite pending list; explicit backpressure."""
    def __init__(self,config,adapter):
        self.config=config;self.adapter=adapter;self.pool=ThreadPoolExecutor(max_workers=1)
        self.pending=[];self.rejected=0

    def submit(self,case,semantic=True):
        self.pending=[f for f in self.pending if not f.done()]
        if len(self.pending)>=self.config.queue_capacity:
            self.rejected+=1;return None
        future=self.pool.submit(evaluate,case,self.config,self.adapter,semantic)
        self.pending.append(future);return future

    def close(self):self.pool.shutdown(wait=True,cancel_futures=False)
