from collections import Counter

def report(results):
    n=len(results);categories=Counter(c for r in results for c in r.failure_categories)
    workflow=Counter(next((c.details.get('state') for c in r.deterministic_checks if c.code=='workflow_completed'),'unknown') for r in results)
    latencies=sorted(r.metrics['latency_seconds'] for r in results if 'latency_seconds' in r.metrics)
    return {'schema_version':'atlas.evaluation-report.v1','total_cases':n,'workflow_states':dict(workflow),
        'workflow_failures':workflow.get('failed',0),'completed_cases':workflow.get('completed',0),
        'failure_categories':dict(categories),'evaluator_statuses':dict(Counter(r.evaluation_status for r in results)),
        'counts_by_stage':dict(Counter(r.workflow_stage for r in results if r.workflow_stage)),
        'confirmed_root_causes':dict(Counter(r.confirmed_root_cause for r in results if r.confirmed_root_cause)),
        'unresolved_cases':sum(r.confirmed_root_cause is None and any(c.status=='fail' for c in r.deterministic_checks) for r in results),
        'human_review_cases':sum(r.requires_human_review for r in results),
        'repeated_signatures':dict(Counter('|'.join([r.workflow_stage or 'none']+[c.code for c in r.deterministic_checks if c.status=='fail']) for r in results)),
        'json_failures':categories.get('historical_json_parse',0)+categories.get('json_parse',0),
        'schema_failures':categories.get('historical_schema',0)+categories.get('response_schema',0),
        'retrieval_execution_failures':categories.get('retrieval_execution',0),
        'grounding_concerns':categories.get('semantic_grounding',0),'citation_failures':categories.get('citation_validity',0),
        'semantic_judgments_are_advisory':True,'human_label_agreement':None,
        'latency_seconds':{'samples':len(latencies),'total':sum(latencies),'p50':latencies[len(latencies)//2] if latencies else None},
        'compute':{'prompt_tokens':sum(r.metrics.get('prompt_eval_count') or 0 for r in results),
                   'output_tokens':sum(r.metrics.get('eval_count') or 0 for r in results),'energy_joules':None,'monetary_cost':None},
        'limitations':['Missing historical failure completions cannot be reconstructed.',
            'Nonempty retrieval and exact citations do not prove semantic support or policy approval.',
            'Evaluator confidence and semantic scores are not calibrated probabilities.',
            'No human labels or review approvals are inferred.']}

def agreement(results,labels):
    """Per-category error detection. Skip unlabelled or unjudged samples explicitly."""
    from .contracts import SemanticScores
    counts={k:{'tp':0,'fp':0,'fn':0,'tn':0} for k in SemanticScores.model_fields};labelled=0;disagreements=0
    for result in results:
        expected=labels.get(str(result.case_id))
        if expected is None or result.evaluation_status!='completed':continue
        labelled+=1
        predicted={f.category for f in result.findings if f.assessment=='concern'}
        actual=set(expected)
        disagreements+=predicted!=actual
        for category,row in counts.items():
            row['tp' if category in actual and category in predicted else 'fp' if category in predicted else 'fn' if category in actual else 'tn']+=1
    for row in counts.values():
        row['precision']=row['tp']/(row['tp']+row['fp']) if row['tp']+row['fp'] else None
        row['recall']=row['tp']/(row['tp']+row['fn']) if row['tp']+row['fn'] else None
    return {'labelled_samples_total':len(labels),'non_graded_statuses':dict(Counter(r.evaluation_status for r in results if str(r.case_id) in labels and r.evaluation_status!='completed')),'labelled_and_evaluated':labelled,'disagreements':disagreements,
            'disagreement_rate':disagreements/labelled if labelled else None,'per_category':counts}
