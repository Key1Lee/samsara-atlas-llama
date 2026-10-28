"""Independent source expectations, stage reporting and failure attribution."""
import copy
import json
from pathlib import Path

import pytest

from app.benchmarks import staged_benchmarks
from scripts.benchmark_stages import validate_fixture, retrieval_check, answer_check, summarize, percentile

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/hr_baseline_100.json'
DOCS = ROOT / 'data/docs'


def test_100_unique_questions_have_source_checked_expectations():
    fixture = json.loads(FIXTURE.read_text())
    questions = validate_fixture(fixture, DOCS, 100)
    assert len(questions) == 100
    assert len({q['fact_group'] for q in questions}) == 25
    assert sum(q['expected'] == 'supported' for q in questions) == 80
    assert sum(q['retrieval_evaluable'] for q in questions) == 96
    broken = copy.deepcopy(fixture)
    broken['questions'][0]['required_passages'][0]['text'] = 'A rule not present in a source'
    with pytest.raises(ValueError, match='clause missing'):
        validate_fixture(broken, DOCS, 100)
    with pytest.raises(ValueError, match='planned number'):
        validate_fixture(fixture, DOCS, 350)


def test_expected_answer_contract_requires_rule_exception_and_citation():
    item = json.loads(FIXTURE.read_text())['questions'][0]
    output = {'body': 'Up to 40 hours carry over, with the local law exception.', 'tone': 'empathetic',
              'execution_receipts': [], 'citation_evidence': [
                  {'source_file': passage['doc_id'] + '.md', 'quote': passage['text']}
                  for passage in item['required_passages']]}
    assert answer_check(item, output)
    assert not answer_check(item, {**output, 'body': 'Up to 40 hours carry over.'})
    assert not answer_check(item, {**output, 'citation_evidence': []})
    assert not answer_check(item, {**output, 'body': "I've sent an email. Up to 40 hours, local law exception."})


def test_unsupported_answer_contract_requires_explicit_referral_and_no_fabricated_policy():
    item = next(q for q in json.loads(FIXTURE.read_text())['questions'] if q['id'] == 'office_pets_1')
    output = {'body': "We don't have verified information. Please contact HR.", 'tone': 'empathetic',
              'answerability': {'status': 'hr_review_required'}, 'cited_snippets': []}
    assert answer_check(item, output)
    assert not answer_check(item, {**output, 'body': 'Yes, iguanas are permitted.'})
    assert not answer_check(item, {**output, 'cited_snippets': ['Invented office pet rule']})


def test_bm25_probe_and_personal_fact_exclusion():
    questions = json.loads(FIXTURE.read_text())['questions']
    assert retrieval_check(questions[0], DOCS)['passed']
    personal = next(q for q in questions if q['id'] == 'personal_pto_1')
    assert retrieval_check(personal, DOCS) == {'evaluated': False, 'passed': None}


def test_stage_metrics_keep_errors_in_denominator_and_separate_failure_areas():
    results = [
        {'retrieval': {'evaluated': True, 'passed': True}, 'workflow_completed': True,
         'answer_passed': True, 'expected': 'supported', 'http_status': 200, 'latency_seconds': 2,
         'schedule_lag_seconds': 0, 'persistence_consistent': True, 'failure_area': ''},
        {'retrieval': {'evaluated': True, 'passed': False}, 'workflow_completed': False,
         'answer_passed': False, 'expected': 'supported', 'http_status': 500, 'latency_seconds': 10,
         'schedule_lag_seconds': 1, 'persistence_consistent': True, 'failure_area': 'generation/validation'},
        {'retrieval': {'evaluated': False, 'passed': None}, 'workflow_completed': False,
         'answer_passed': False, 'expected': 'no_information', 'http_status': 429, 'latency_seconds': .1,
         'schedule_lag_seconds': 0, 'persistence_consistent': False, 'failure_area': 'application/rate_limit'},
    ]
    metrics = summarize(results, 60, 10, 100)
    assert metrics['attempted'] == 3 and metrics['planned_questions'] == 100
    assert metrics['answer_contract_percent'] == 33.3 and metrics['retrieval_percent'] == 50.0
    assert metrics['generation_failures'] == 1 and metrics['application_failures'] == 1
    assert metrics['persistence_checks_passed'] == 2 and metrics['http_429'] == 1
    assert metrics['p95_seconds'] == 10 and metrics['completed_per_minute'] == 1
    assert percentile([], .95) is None
    assert summarize([], 0, 10, 100)['answer_contract_percent'] is None


def test_stage_endpoint_never_invents_unrun_scores_or_exposes_results(monkeypatch, tmp_path):
    path = tmp_path / 'stages.json'
    monkeypatch.setattr('app.benchmarks.STAGES', path)
    stages = staged_benchmarks()['stages']
    assert [s['status'] for s in stages] == ['Not run'] * 3
    assert [s['planned_questions'] for s in stages] == [100, 350, None]
    assert [s['target_rpm'] for s in stages] == [10, 30, None]
    assert all(s['metrics'] is None for s in stages)
    path.write_text(json.dumps({'baseline': {'status': 'Running', 'results': [{'question': 'Private question'}],
                                             'metrics': {'attempted': 3, 'question': 'Private question'},
                                             'metadata': {'employee_id': 'Secret employee'}}}))
    result = staged_benchmarks()
    assert result['stages'][0]['metrics']['attempted'] == 3
    assert 'Private question' not in json.dumps(result) and 'Secret employee' not in json.dumps(result)
    path.write_text('not JSON')
    assert all(stage['metrics'] is None for stage in staged_benchmarks()['stages'])
