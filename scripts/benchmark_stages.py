#!/usr/bin/env python3
"""Bounded, paced local baseline; writes aggregate stage scores without changing BM25/Llama."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from app.tools.retrieval import retrieve_policy_snippets

STAGES_FILE = ROOT / 'data/benchmarks/stages.json'


def validate_fixture(fixture, docs, expected_count):
    questions = fixture['questions']
    if len(questions) != expected_count or len({q['question'] for q in questions}) != expected_count or len({q['id'] for q in questions}) != expected_count:
        raise ValueError('Fixture requires the planned number of unique IDs and questions')
    if fixture['source_sha256'] != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in docs.glob('*.md')}:
        raise ValueError('Policy files differ from source-checked expectations')
    for item in questions:
        for passage in item['required_passages']:
            if passage['text'] not in (docs / (passage['doc_id'] + '.md')).read_text():
                raise ValueError('Expected clause missing from source')
        if item['expected'] not in ('supported', 'no_information'):
            raise ValueError('Unknown expected answer category')
        for pattern in item['answer_patterns']:
            re.compile(pattern)
    return questions


def retrieval_check(item, docs):
    if not item.get('retrieval_evaluable', True):
        return {'evaluated': False, 'passed': None}
    snippets = retrieve_policy_snippets(item['question'], top_k=5)
    documents = {p.stem: p.read_text() for p in docs.glob('*.md')}
    retrieved = {name for name, source in documents.items() if any(s and s in source for s in snippets)}
    expected = set(item['expected_doc_ids'])
    missing = [p for p in item['required_passages'] if not any(p['text'] in s and s in documents[p['doc_id']] for s in snippets)]
    return {'evaluated': True, 'passed': retrieved == expected and not missing,
            'missing_clauses': len(missing), 'unexpected_sources': len(retrieved - expected)}


def answer_check(item, outputs):
    body = outputs.get('body', '')
    if not isinstance(body, str) or not body.strip() or outputs.get('tone') != 'empathetic':
        return False
    if outputs.get('execution_receipts') or re.search(r"\b(?:I|we)(?:'ve| have)? (?:sent|notified|booked|scheduled|updated|approved)\b", body, re.I):
        return False
    if item['expected'] == 'no_information':
        return (outputs.get('answerability', {}).get('status') == 'hr_review_required'
                and not outputs.get('cited_snippets')
                and bool(re.search(r"don't have verified information|does not have a verified current PTO balance", body, re.I))
                and bool(re.search(r'\bHR\b', body)))
    normalized = ' '.join(body.split())
    if not all(re.search(pattern, normalized, re.I) for pattern in item['answer_patterns']):
        return False
    evidence = outputs.get('citation_evidence', [])
    return all(any(e.get('source_file') == p['doc_id'] + '.md' and p['text'] in e.get('quote', '') for e in evidence)
               for p in item['required_passages'])


def persistence(database, key, inputs, response):
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=10) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('select * from cases where idempotency_key=?', (key,)).fetchall()
        if len(rows) != 1:
            return {'consistent': False, 'saved_cases': len(rows), 'failure_area': 'application/storage'}
        row = rows[0]
        notes = db.execute('select employee_id from case_notes where case_id=?', (row['id'],)).fetchall()
        steps = db.execute('select name,status,error from steps where case_id=?', (row['id'],)).fetchall()
        failed = next((s for s in reversed(steps) if s['status'] == 'failed'), None)
        completed = row['status'] == 'completed'
        expected_notes = int(any(s['name'] == 'post_case_note_to_hris' and s['status'] == 'completed' for s in steps))
        consistent = (json.loads(row['inputs_json']) == inputs and len(notes) == expected_notes
                      and all(n['employee_id'] == inputs['employee_id'] for n in notes))
        outputs = json.loads(row['outputs_json'] or '{}')
        if response is not None and completed:
            consistent = (consistent and response.get('case_id') == row['id']
                          and response.get('status') == row['status'] and response.get('outputs') == outputs)
        area = None
        if not consistent:
            area = 'application/storage'
        elif failed:
            area = 'generation/validation' if failed['name'] in ('summarize_and_extract_actions', 'generate_structured_response') else 'application/tool/storage'
        elif not completed:
            area = 'application/incomplete'
        return {'consistent': consistent, 'saved_cases': 1, 'completed': completed,
                'failed_step': failed['name'] if failed else None, 'error_code': failed['error'] if failed else None,
                'failure_area': area, 'outputs': outputs}


def percentile(values, fraction):
    return round(sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)], 3) if values else None


def summarize(results, elapsed, target_rpm, expected_total):
    n = len(results)
    checked = [r for r in results if r['retrieval']['evaluated']]
    completed = [r for r in results if r.get('workflow_completed')]
    latencies = [r['latency_seconds'] for r in results]
    arrivals = [r['submitted_offset_seconds'] for r in results if 'submitted_offset_seconds' in r]
    passed = sum(r.get('answer_passed', False) for r in results)
    return {'planned_questions': expected_total, 'attempted': n, 'target_rpm': target_rpm,
            'elapsed_seconds': round(elapsed, 2), 'completed': len(completed),
            'workflow_success_percent': round(100 * len(completed) / n, 1) if n else None,
            'answer_checks_passed': passed, 'answer_checks_failed': len(completed) - passed,
            'no_information_checks_passed': sum(r.get('answer_passed', False) and r['expected'] == 'no_information' for r in results),
            'answer_contract_percent': round(100 * passed / n, 1) if n else None,
            'retrieval_checked': len(checked), 'retrieval_passed': sum(r['retrieval']['passed'] for r in checked),
            'retrieval_percent': round(100 * sum(r['retrieval']['passed'] for r in checked) / len(checked), 1) if checked else None,
            'http_errors': sum(r.get('http_status') != 200 for r in results),
            'http_429': sum(r.get('http_status') == 429 for r in results),
            'generation_failures': sum(r.get('failure_area') == 'generation/validation' for r in results),
            'application_failures': sum(r.get('failure_area', '').startswith('application') for r in results),
            'persistence_checks_passed': sum(r.get('persistence_consistent', False) for r in results),
            'p50_seconds': percentile(latencies, .5), 'p95_seconds': percentile(latencies, .95),
            'max_seconds': max(latencies) if latencies else None,
            'completed_per_minute': round(60 * len(completed) / elapsed, 2) if elapsed else 0,
            'observed_offered_rpm': round(60 * len(arrivals) / (max(arrivals) - min(arrivals) + 60 / target_rpm), 2) if arrivals else None,
            'max_schedule_lag_seconds': round(max((r['schedule_lag_seconds'] for r in results), default=0), 3)}


def hardware():
    data = {'os': platform.system(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count()}
    if sys.platform == 'darwin':
        for name, field in [('hw.memsize', 'memory_bytes'), ('machdep.cpu.brand_string', 'cpu')]:
            result = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True)
            data[field] = result.stdout.strip() if result.returncode == 0 else 'unavailable'
    return data


def main():
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['baseline', 'expanded', 'stress'], default='baseline')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/hr_baseline_100.json')
    parser.add_argument('--api-base', default='http://127.0.0.1:8000')
    parser.add_argument('--database', type=Path, default=ROOT / 'data/atlas.db')
    parser.add_argument('--employees', default='E10234,E10897,E11342')
    parser.add_argument('--rpm', type=int)
    parser.add_argument('--requests', type=int)
    parser.add_argument('--max-inflight', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    planned, rpm = {'baseline': (100, 10), 'expanded': (350, 30), 'stress': (args.requests, args.rpm)}[args.stage]
    if not planned or not rpm or rpm < 1 or planned < 1 or args.max_inflight < 1:
        parser.error('stress requires explicit positive --requests and --rpm; concurrency must be positive')
    if args.stage != 'stress' and ((args.requests is not None and args.requests != planned) or (args.rpm is not None and args.rpm != rpm)):
        parser.error('use the fixed planned question count and RPM for this stage')
    load_dotenv(ROOT / '.env')
    docs = ROOT / 'data/docs'
    os.environ['DOCS_DIR'] = str(docs)
    fixture = json.loads(args.fixture.read_text())
    questions = validate_fixture(fixture, docs, planned)
    provider = requests.get(args.api_base + '/info', timeout=10).json()
    if provider != {'provider': 'ollama', 'mode': 'live'}:
        raise ValueError('Baseline requires the unchanged live Ollama provider')
    # Fail before any submission if the exact SQLite database is unreadable.
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('select idempotency_key from cases limit 0')
    employees = [x.strip() for x in args.employees.split(',') if x.strip()]
    if not employees:
        parser.error('provide synthetic employee IDs authorized on this API')
    token = os.getenv('ATLAS_EMPLOYEE_TOKEN')
    run_id = uuid4().hex
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'hardware': hardware(), 'deployment': 'Existing local API, SQLite and Ollama on this host; competing local activity is not controlled.',
                'max_inflight': args.max_inflight,
                'provider': provider['provider'], 'model': os.getenv('OLLAMA_MODEL', 'llama3.2'),
                'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                'source_sha256': fixture['source_sha256'],
                'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('app/tools/retrieval.py', 'app/providers/ollama.py')}}
    try:
        tags = requests.get(os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434') + '/api/tags', timeout=10).json()
        metadata['model_digest'] = next(m['digest'] for m in tags['models'] if m['name'].split(':')[0] == metadata['model'].split(':')[0])
    except (requests.RequestException, KeyError, StopIteration, ValueError):
        metadata['model_digest'] = 'unverified'
    results = []
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    def publish(status):
        report = {'status': status, 'run_id': run_id, 'started_at': started_at,
                  'updated_at': datetime.now(timezone.utc).isoformat(), 'metadata': metadata,
                  'metrics': summarize(results, time.monotonic() - started, rpm, planned)}
        stages = json.loads(STAGES_FILE.read_text()) if STAGES_FILE.exists() else {}
        stages[args.stage] = report
        temporary = STAGES_FILE.with_suffix('.tmp')
        temporary.write_text(json.dumps(stages, indent=2) + '\n')
        temporary.replace(STAGES_FILE)
        (args.output_dir / 'results.json').write_text(json.dumps({**report, 'results': results}, indent=2) + '\n')
    def submit(index, item):
        due = started + index * 60 / rpm
        sent = time.monotonic()
        inputs = {'employee_id': employees[index % len(employees)], 'case_transcript': item['question'], 'tone': 'empathetic'}
        key = f'baseline-{run_id}-{index}'
        result = {'id': item['id'], 'expected': item['expected'], 'schedule_lag_seconds': max(0, sent - due),
                  'submitted_offset_seconds': sent - started,
                  'retrieval': retrieval_check(item, docs)}
        http_start = time.monotonic()
        try:
            response = requests.post(args.api_base + '/cases', json={'inputs': inputs, 'idempotency_key': key},
                                     headers={'X-Employee-ID': inputs['employee_id'], **({'Authorization': 'Bearer ' + token} if token else {})}, timeout=180)
            result['http_status'] = response.status_code
            payload = response.json() if response.status_code == 200 else None
            saved = persistence(args.database, key, inputs, payload)
            result.update(workflow_completed=response.status_code == 200 and payload.get('status') == 'completed' and saved.get('completed', False),
                          persistence_consistent=saved['consistent'], failure_area=saved.get('failure_area') or '',
                          failed_step=saved.get('failed_step'), error_code=saved.get('error_code'))
            result['answer_passed'] = result['workflow_completed'] and answer_check(item, saved.get('outputs', {}))
            if response.status_code == 429:
                result['failure_area'] = 'application/rate_limit'
        except requests.RequestException:
            result.update(http_status=None, workflow_completed=False, answer_passed=False,
                          persistence_consistent=False, failure_area='application/transport')
        except (sqlite3.Error, ValueError, TypeError):
            result.update(workflow_completed=False, answer_passed=False,
                          persistence_consistent=False, failure_area='application/storage_or_protocol')
        result['latency_seconds'] = round(time.monotonic() - http_start, 3)
        return result
    publish('Running')
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        pending = set()
        def collect(done):
            for future in done:
                results.append(future.result())
            publish('Running')
            print(f'{len(results)}/{planned} attempted; {sum(r.get("workflow_completed", False) for r in results)} completed', flush=True)
        for index, item in enumerate(questions):
            while len(pending) >= args.max_inflight:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
            delay = started + index * 60 / rpm - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            done = {f for f in pending if f.done()}
            pending -= done
            if done:
                collect(done)
            pending.add(pool.submit(submit, index, item))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect(done)
    publish('Completed')
    print(json.dumps(summarize(results, time.monotonic() - started, rpm, planned), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
