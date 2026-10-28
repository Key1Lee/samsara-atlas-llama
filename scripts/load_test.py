#!/usr/bin/env python3
"""Small local reliability probe: HTTP results must agree with saved records.

Default: eight retries of one fresh submission, one case and one local draft
note. Use --independent for intentional separate submissions, even when their
questions match. This is a correctness probe, not a capacity benchmark.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sqlite3
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = 'How does PTO accrue?'


class VerificationFailure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VerificationFailure(message)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', default='http://127.0.0.1:8000')
    parser.add_argument('--employee-id', default='E10234')
    parser.add_argument('--database', type=Path, help='SQLite file used by this API; opened read-only')
    parser.add_argument('--requests', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--independent', action='store_true')
    parser.add_argument('--question', default=DEFAULT_QUESTION)
    parser.add_argument('--expected-notes', type=int,
                        help='Local notes per case; required with a custom question (deterministic routes use 0)')
    parser.add_argument('--expect-rejection', type=int, choices=[409, 422, 429],
                        help='Explicitly expect every POST to reject and persist no new records')
    args = parser.parse_args()
    if args.requests < 1 or args.workers < 1 or (args.expected_notes is not None and args.expected_notes < 0):
        parser.error('requests/workers must be positive and expected-notes nonnegative')
    if args.expected_notes is None:
        if args.question != DEFAULT_QUESTION:
            parser.error('custom questions require an explicit --expected-notes contract')
        args.expected_notes = 1
    if args.database is None:
        url = os.environ.get('DATABASE_URL', f'sqlite:///{ROOT / "data/atlas.db"}')
        if not url.startswith('sqlite:///'):
            parser.error('this local probe requires a SQLite --database file')
        args.database = Path(url[len('sqlite:///'):])
    return args


def post_case(args, key, inputs, headers):
    response = requests.post(f'{args.api_base}/cases',
                             json={'inputs': inputs, 'idempotency_key': key},
                             headers=headers, timeout=60)
    # A non-JSON successful response is an unexpected protocol failure.
    return key, response.status_code, response.json() if response.status_code == 200 else None


def verify(args):
    inputs = {'employee_id': args.employee_id, 'case_transcript': args.question, 'tone': 'empathetic'}
    headers = {'X-Employee-ID': args.employee_id}
    token = os.environ.get('ATLAS_EMPLOYEE_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    prefix = f'atlas-probe-{uuid.uuid4().hex}'
    keys = [f'{prefix}-{i if args.independent else 0}' for i in range(args.requests)]
    unique_keys = set(keys)
    expected_cases = 0 if args.expect_rejection else len(unique_keys)
    print(f'{len(unique_keys)} intended submissions; {args.requests} HTTP attempts; '
          f'expected {expected_cases} saved cases and {expected_cases * args.expected_notes} local notes.')
    # Fail before submitting if the local verifier cannot read the chosen store.
    # Read-only mode never creates a file or modifies application records.
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        placeholders = ','.join('?' for _ in unique_keys)
        query = f'SELECT * FROM cases WHERE idempotency_key IN ({placeholders})'
        require(not db.execute(query, tuple(unique_keys)).fetchall(), 'probe keys already exist')
        db.execute('SELECT id FROM case_notes LIMIT 0')
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(post_case, args, key, inputs, headers) for key in keys]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        for _, status, data in results:
            expected_http = args.expect_rejection or 200
            require(status == expected_http, f'POST /cases HTTP {status}; expected {expected_http}')
            if not args.expect_rejection:
                require(isinstance(data, dict) and type(data.get('case_id')) is int,
                        'POST response missing integer case_id')
                require(data.get('status') in ('running', 'completed'), 'POST response has invalid workflow status')
        rows = db.execute(query, tuple(unique_keys)).fetchall()
        require(len(rows) == expected_cases,
                f'persisted case count: expected {expected_cases}, found {len(rows)}')
        if args.expect_rejection:
            # A deliberately exhausted quota also rejects GETs. The independent
            # read-only store check proves that these rejected keys saved nothing.
            print(f'PASS: {len(results)} expected HTTP {args.expect_rejection} rejections; 0 saved cases.')
            return
        ids = {row['id'] for row in rows}
        if not args.expect_rejection:
            require({data['case_id'] for _, _, data in results} == ids,
                    'response case IDs do not match persisted case IDs')
            require({row['idempotency_key'] for row in rows} == unique_keys,
                    'persisted cases do not match intended submission keys')
        history = requests.get(f'{args.api_base}/cases', headers=headers, timeout=10)
        require(history.status_code == 200, f'employee history HTTP {history.status_code}; expected 200')
        history_ids = history.json()['case_ids']
        require(len(history_ids) == len(set(history_ids)), 'employee history contains duplicate IDs')
        require(ids.issubset(set(history_ids)), 'persisted cases missing from employee history')
        for row in rows:
            case_id = row['id']
            require(json.loads(row['inputs_json']) == inputs, f'case {case_id}: saved inputs/employee mismatch')
            require(row['status'] == 'completed', f'case {case_id}: saved workflow is not completed')
            outputs = json.loads(row['outputs_json'])
            require(isinstance(outputs, dict) and bool(outputs), f'case {case_id}: saved outputs missing')
            matching = [data for key, _, data in results if key == row['idempotency_key']]
            require(all(data['case_id'] == case_id for data in matching),
                    f'case {case_id}: retry returned another submission\'s case')
            completed = [data for data in matching if data['status'] == 'completed']
            require(bool(completed), f'case {case_id}: no completed response after all requests finished')
            require(all(data.get('outputs') == outputs for data in completed), f'case {case_id}: saved outputs mismatch')
            notes = db.execute('SELECT * FROM case_notes WHERE case_id=?', (case_id,)).fetchall()
            require(len(notes) == args.expected_notes,
                    f'case {case_id}: local note count: expected {args.expected_notes}, found {len(notes)}')
            require(all(note['employee_id'] == args.employee_id for note in notes),
                    f'case {case_id}: local note employee mismatch')
            detail = requests.get(f'{args.api_base}/cases/{case_id}', headers=headers, timeout=10)
            require(detail.status_code == 200, f'case {case_id}: detail HTTP {detail.status_code}; expected 200')
            saved = detail.json()
            require(saved['case']['id'] == case_id and saved['case']['status'] == row['status'],
                    f'case {case_id}: API status/identity differs from persistence')
            require(json.loads(saved['case']['inputs']) == inputs, f'case {case_id}: API inputs/employee mismatch')
            require(json.loads(saved['case']['outputs']) == outputs, f'case {case_id}: API saved outputs mismatch')
            require({(note['id'], note['employee_id'], note['note']) for note in saved['case_notes']}
                    == {(note['id'], note['employee_id'], note['note']) for note in notes}
                    and len(saved['case_notes']) == len(notes)
                    and all(note['employee_id'] == args.employee_id for note in saved['case_notes']),
                    f'case {case_id}: API local notes differ from persistence')
    print(f'PASS: {len(results)} expected HTTP results; {len(rows)} verified saved cases.')


def main():
    args = arguments()
    try:
        verify(args)
    except VerificationFailure as exc:
        print(f'FAIL: {exc}')
        return 1
    except Exception as exc:
        # Exception messages can contain credentials, case text or HTTP payloads.
        print(f'FAIL: unexpected {type(exc).__name__}; could not verify request/persistence. '
              'Check API access and the matching SQLite file.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
