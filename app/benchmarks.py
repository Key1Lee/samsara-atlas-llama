"""Live aggregate metrics. Answer quality requires matching documented review evidence."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re

from sqlalchemy import func

from app.storage.models import Case, CaseAuditEvent

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / 'data/benchmarks/e11342-live-2026-10-27.json'
STAGES = ROOT / 'data/benchmarks/stages.json'
CATEGORIES = ('Correct answer', 'Incorrect or incomplete answer', 'No information available',
              'Workflow error', 'Awaiting review')
# Independent multi-label matches: a mixed PTO/remote question counts in both bars.
TOPICS = {
    'PTO': r'\b(?:pto|paid time off|vacation|carryover|carry over|rollover)\b',
    'Remote work': r'\b(?:remote|remotely|relocat\w*|work from home|working from home|work location|moving states?|moving to another state|laptop|peripherals)\b',
    'Benefits': r'\b(?:benefits?|insurance|health coverage|medical coverage|dental|vision|retirement|401\s*\(?k\)?)\b',
    'Expenses': r'\b(?:expenses?|reimburse\w*|receipts?)\b',
    'Leave': r'\b(?:leave|parental|maternity|paternity|fmla|cfra|bereavement)\b',
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def question_topics(question):
    matches = [name for name, pattern in TOPICS.items() if re.search(pattern, question, re.IGNORECASE)]
    return matches or ['Other/unknown']


def documented_reviews():
    """Missing/stale evidence removes quality scores, never the live submissions."""
    try:
        review = json.loads(REVIEWS.read_text(encoding='utf-8'))
        docs = Path(os.getenv('DOCS_DIR', str(ROOT / 'data/docs')))
        current = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in docs.glob('*.md')}
        if current != review['source_sha256']:
            return {}, False
        matched = {}
        for row in review['results']:
            if row['category'] in ('Correct answer', 'Incorrect answer', 'No information available'):
                key = (row['inputs_sha256'], row['key_sha256'], row['outputs_sha256'])
                matched[key] = ('Incorrect or incomplete answer' if row['category'] == 'Incorrect answer'
                                else row['category'])
        return matched, True
    except (OSError, ValueError, KeyError, TypeError):
        return {}, False


def case_quality_evaluation(case):
    """Documented quality evidence is separate from human approval."""
    reviews, available = documented_reviews()
    try:
        key = (fingerprint(json.loads(case.inputs_json or '{}')), fingerprint(case.idempotency_key),
               fingerprint(json.loads(case.outputs_json or '{}')))
        category = reviews.get(key) if case.status == 'completed' else None
    except (ValueError, TypeError):
        category = None
    return {'status': 'documented' if category else 'not_evaluated',
            'category': category, 'evidence_available': available,
            'human_review_status': 'not_recorded'}


def staged_benchmarks():
    """Aggregate-only stage reports; no inferred scores for unrun experiments."""
    try:
        reports = json.loads(STAGES.read_text(encoding='utf-8'))
        if not isinstance(reports, dict):
            raise ValueError()
    except (OSError, ValueError):
        reports = {}
    stages = []
    for name, questions, rpm in [('baseline', 100, 10), ('expanded', 350, 30), ('stress', None, None)]:
        report = reports.get(name, {})
        metrics = report.get('metrics') if isinstance(report, dict) else None
        metadata = report.get('metadata', {}) if isinstance(report, dict) else {}
        # Deliberately omit per-question results, response bodies and record IDs.
        allowed_metrics = ('planned_questions', 'attempted', 'target_rpm', 'elapsed_seconds', 'completed',
                           'workflow_success_percent', 'answer_checks_passed', 'answer_checks_failed',
                           'no_information_checks_passed', 'answer_contract_percent', 'retrieval_checked',
                           'retrieval_passed', 'retrieval_percent', 'http_errors', 'http_429', 'generation_failures',
                           'application_failures', 'persistence_checks_passed', 'p50_seconds', 'p95_seconds',
                           'max_seconds', 'completed_per_minute', 'observed_offered_rpm', 'max_schedule_lag_seconds')
        safe_metrics = {key: metrics.get(key) for key in allowed_metrics} if isinstance(metrics, dict) else None
        safe_metadata = {key: metadata.get(key) for key in ('hardware', 'deployment', 'provider', 'model',
                         'model_digest', 'fixture_sha256', 'code_sha256', 'max_inflight')} if isinstance(metadata, dict) else {}
        stages.append({'stage': name, 'planned_questions': questions, 'target_rpm': rpm,
                       'status': report.get('status', 'Not run') if isinstance(report, dict) else 'Not run',
                       'metrics': safe_metrics, 'configuration': safe_metadata})
    return {'stages': stages, 'capacity': 'Unknown — these runs do not establish sustainable deployment capacity.',
            'method': 'Keep BM25 and the current Llama model unchanged for the initial baseline. The initial fixture contains 100 unique synthetic questions covering 25 fact/topic groups with four wording variants each; expectations are source-checked, not HR-approved, and variants are correlated. Retrieval checks measure expected source/required-clause coverage using BM25 top 5; in the 100-question baseline, four personal-fact probes are excluded from retrieval scoring. Answer scores are conservative pattern and citation contracts against documented expectations, not a human-reviewed accuracy score. Deterministic workflow routes may bypass BM25 or Llama; they are included in end-to-end results but the standalone BM25 probe is reported separately. Failure areas identify the persisted failed step, not a proven underlying root cause. Load is scheduled at the target rate with bounded concurrency; schedule lag and achieved completions are reported instead of assuming the target was sustained.',
            'next_steps': 'Investigate baseline failures before expanding to 350 source-checked questions at 30 requests/minute. Stress testing follows the expanded run with explicit rates, duration and stopping limits; no stress score is inferred.'}


def global_benchmark(db):
    reviews, available = documented_reviews()
    counts = dict.fromkeys(CATEGORIES, 0)
    topics = dict.fromkeys([*TOPICS, 'Other/unknown'], 0)
    ids = []
    excluded = 0
    in_progress = 0
    for case in db.query(Case).order_by(Case.id).yield_per(100):
        try:
            inputs = json.loads(case.inputs_json or '{}')
            question = inputs['case_transcript']
            employee = inputs['employee_id']
            if not isinstance(question, str) or not question.strip() or not isinstance(employee, str) or not employee.strip():
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            excluded += 1
            continue
        ids.append(case.id)
        for topic in question_topics(question):
            topics[topic] += 1
        if case.status == 'failed':
            category = 'Workflow error'
        elif case.status != 'completed':
            category = 'Awaiting review'
            in_progress += 1
        else:
            try:
                outputs = json.loads(case.outputs_json or '{}')
                key = (fingerprint(inputs), fingerprint(case.idempotency_key), fingerprint(outputs))
                category = reviews.get(key, 'Awaiting review')
            except (ValueError, TypeError):
                category = 'Awaiting review'
        counts[category] += 1
    # Audit events identify retries/resumes without counting every step as a request.
    # Events without request IDs are legacy/direct calls and cannot prove HTTP attempts.
    retries = db.query(func.count(func.distinct(CaseAuditEvent.request_id))).filter(
        CaseAuditEvent.case_id.in_(ids),
        CaseAuditEvent.event_type.in_(['request_reused', 'case_resumed']),
        CaseAuditEvent.request_id.isnot(None),
    ).scalar() or 0
    total = len(ids)
    percent = lambda count: round(100 * count / total, 1) if total else 0.0
    appropriate = counts['Correct answer'] + counts['No information available']
    reviewed = counts['Correct answer'] + counts['Incorrect or incomplete answer'] + counts['No information available']
    return {
        'scope': 'application', 'updated_at': datetime.now(timezone.utc).isoformat(),
        'summary': {'total': total, 'counts': counts,
                    'percentages': {name: percent(count) for name, count in counts.items()},
                    'appropriate_outcomes': appropriate, 'appropriate_outcome_percent': percent(appropriate),
                    'quality_evaluated_answers': reviewed, 'documented_human_reviews': None,
                    'human_review_status': 'not_recorded', 'reviewed_answers': reviewed, 'review_coverage_percent': percent(reviewed),
                    'retry_attempts_recorded': retries, 'in_progress': in_progress,
                    'excluded_invalid_records': excluded},
        'topic_counts': topics, 'review_evidence_available': available,
        'staged_benchmark': staged_benchmarks(),
        'scoring_method': 'One persisted HR-question submission counts once across all employees. Idempotent retries reuse that submission; an explicit replay/new key is a new submission. Category percentage = category count / total submissions × 100. Page views, GET requests and rejected requests without a persisted case are excluded. Failed workflows count as workflow errors. A completed answer receives a quality score only when its full inputs, submission key and full outputs match documented review evidence and policy files are unchanged; all others await review. Model confidence, answerability labels and HTTP 200 do not establish correctness.',
        'topic_method': 'Case-insensitive word/phrase rules: PTO (PTO, paid time off, vacation, carryover/rollover); remote work (remote, relocation, work from home, work location, moving states, laptop/peripherals); benefits (benefits, insurance, health/medical coverage, dental, vision, retirement/401k); expenses (expense, reimbursement, receipt); leave (leave, parental, maternity/paternity, FMLA/CFRA, bereavement). Each question counts at most once per matching topic; multiple topics may match, so bars can sum above total submissions. Other/unknown means no match. No LLM or embeddings are used.',
        'limitations': ['Scores describe recorded submissions and documented reviews, not HR approval or production accuracy.',
                        'Awaiting review includes unfinished workflows; these remain in the denominator.',
                        'Retry counts include only audited HTTP reuse/resume events with request IDs. Legacy attempts without these records cannot be reconstructed.',
                        'Missing, changed or unavailable review evidence leaves completed answers awaiting review; live totals and chart still work.',
                        'Keyword topics are approximate; they are not a semantic relevance or retrieval-quality score.'],
    }
