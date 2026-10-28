"""Controlled requests replacement for subprocess tests; never opens a socket.

Writes real disposable SQLite rows so response claims and persistence can be
faulted independently. This is a harness fixture, not an application mock test.
"""
import json as jsonlib
import os
import sqlite3
from threading import RLock
from urllib.parse import urlsplit

_lock = RLock()
_fault = os.environ['PROBE_FAULT']
_db = os.environ['PROBE_DATABASE']


class ConnectionError(Exception):
    pass


class exceptions:
    ConnectionError = ConnectionError


class Response:
    def __init__(self, data, status=200):
        self.status_code = status
        self.ok = 200 <= status < 400
        self.data = data

    def json(self):
        return self.data


def post(url, json, **kwargs):
    if _fault == 'post_exception':
        raise RuntimeError('private exception content must not be printed')
    if _fault == 'post_assertion':
        raise AssertionError('private exception content must not be printed')
    if _fault == 'expected_quota':
        return Response({'detail': 'Synthetic quota rejection'}, 429)
    if _fault in ('expected_rejection', 'unexpected_rejection'):
        return Response({'detail': 'Synthetic validation rejection'}, 422)
    if _fault == 'warning_only':
        return Response({'detail': 'Synthetic operation failed'}, 500)
    inputs = json['inputs']
    key = json['idempotency_key']
    outputs = {'subject': 'Synthetic draft', 'body': 'Review with HR.', 'tone': inputs['tone']}
    if _fault == 'missing_case':
        return Response({'case_id': 1, 'status': 'completed', 'outputs': outputs})
    with _lock, sqlite3.connect(_db) as db:
        row = db.execute('SELECT id FROM cases WHERE idempotency_key=?', (key,)).fetchone()
        if row:
            case_id = row[0]
        else:
            stored_inputs = dict(inputs)
            if _fault == 'wrong_employee':
                stored_inputs['employee_id'] = 'FOREIGN'
            status = 'running' if _fault == 'incomplete_case' else 'completed'
            stored_outputs = {'subject': 'Different saved output'} if _fault == 'outputs_mismatch' else outputs
            case_id = db.execute('INSERT INTO cases (idempotency_key, inputs_json, status, outputs_json) VALUES (?, ?, ?, ?)',
                                 (key, jsonlib.dumps(stored_inputs), status, jsonlib.dumps(stored_outputs))).lastrowid
            if _fault == 'duplicate_case':
                db.execute('INSERT INTO cases (idempotency_key, inputs_json, status, outputs_json) VALUES (?, ?, ?, ?)',
                           (key, jsonlib.dumps(stored_inputs), status, jsonlib.dumps(stored_outputs)))
            note_count = 0 if _fault == 'note_missing' else 2 if _fault == 'case_notes_duplicate' else 1
            for _ in range(note_count):
                db.execute('INSERT INTO case_notes (case_id, employee_id, note) VALUES (?, ?, ?)',
                           (case_id, stored_inputs['employee_id'], 'Synthetic local draft note'))
        response_id = 999 if _fault == 'response_not_persisted' else case_id
        if _fault == 'expected_rejection_persisted':
            return Response({'detail': 'Synthetic rejection after incorrect persistence'}, 422)
        return Response({'case_id': response_id,
                         'status': 'running' if _fault == 'incomplete_case' else 'completed', 'outputs': outputs})


def get(url, **kwargs):
    path = urlsplit(url).path.rstrip('/')
    if _fault == 'expected_quota':
        return Response({'detail': 'Synthetic quota rejection'}, 429)
    if path == '/cases':
        with sqlite3.connect(_db) as db:
            ids = [row[0] for row in db.execute('SELECT id FROM cases')]
        if _fault == 'missing_from_history':
            ids = []
        return Response({'case_ids': ids})
    if _fault == 'detail_exception':
        raise RuntimeError('private exception content must not be printed')
    case_id = int(path.rsplit('/', 1)[1])
    with sqlite3.connect(_db) as db:
        row = db.execute('SELECT id, status, inputs_json, outputs_json FROM cases WHERE id=?', (case_id,)).fetchone()
        if not row:
            return Response({'detail': 'Case not found'}, 404)
        notes = [{'id': n[0], 'employee_id': n[1], 'note': n[2]} for n in db.execute(
            'SELECT id, employee_id, note FROM case_notes WHERE case_id=?', (case_id,))]
    if _fault == 'note_content_mismatch':
        notes[0]['note'] = 'Different API note'
    return Response({'case': {'id': row[0], 'status': row[1], 'inputs': row[2], 'outputs': row[3]},
                     'case_notes': notes})
