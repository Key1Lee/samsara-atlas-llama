"""Conservative recovery for the existing single-host, file-backed SQLite store."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

from app.storage.models import CaseNote, Step
from app.tools.hris import post_case_note_to_hris
from app.tools.registry import get_tool_registry
from app.engine.answer_safety import no_operations

GUARD = 'local_flock_v1'
LOCAL_NOTE = 'sqlite_case_note_v1'


class GuardUnavailable(Exception):
    pass


@contextmanager
def execution_guard(db, case_id):
    # Kernel ownership outlives neither process death nor descriptor close.
    # Never unlink these files while the database is in use: changing the inode
    # could let two processes lock different files for the same case.
    try:
        import fcntl
        bind = db.get_bind()
        filename = bind.url.database
        if bind.dialect.name != 'sqlite' or not filename or filename == ':memory:':
            raise GuardUnavailable()
        directory = Path(str(Path(filename).resolve()) + '.case-locks')
        directory.mkdir(mode=0o700, exist_ok=True)
        descriptor = os.open(directory / f'{case_id}.lock', os.O_CREAT | os.O_RDWR, 0o600)
    except (ImportError, OSError):
        raise GuardUnavailable() from None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        except OSError:
            raise GuardUnavailable() from None
        yield True
    finally:
        os.close(descriptor)


def note_backend():
    return LOCAL_NOTE if get_tool_registry().get('post_case_note_to_hris') is post_case_note_to_hris else 'unknown'


def confirmed_note(db, case_id, employee_id):
    """Return a verified local receipt; never call an uncertain write to find out."""
    note = db.query(CaseNote).filter_by(case_id=case_id).first()
    if note is None:
        return None, None
    draft = db.query(Step).filter_by(case_id=case_id, name='generate_structured_response', status='completed').first()
    if not draft:
        return None, 'note_receipt_mismatch'
    response = json.loads(draft.output_json)
    expected = f"Unapproved response draft (needs HR review): {response.get('subject', '')} - {response.get('body', '')[:200]}"
    if note.employee_id != employee_id or note.note != expected:
        return None, 'note_receipt_mismatch'
    return {'id': note.id, 'employee_id': note.employee_id, 'note': note.note[:100]}, None


def case_note_status(db, case, inputs):
    """Read actual local evidence; a model or tool-output ID is never a receipt."""
    result = no_operations()
    step = db.query(Step).filter_by(case_id=case.id, name='post_case_note_to_hris').first()
    try:
        receipt, mismatch = confirmed_note(db, case.id, inputs['employee_id'])
    except (ValueError, TypeError, KeyError, AttributeError):
        receipt, mismatch = None, 'note_receipt_mismatch'
    if receipt:
        result['case_note'] = {'status': 'succeeded', 'attempted': True, 'receipt': {
            'operation': 'save_case_note', 'status': 'succeeded', 'source': 'sqlite_case_notes',
            'case_id': case.id, 'note_id': receipt['id'],
        }}
    elif step is not None or mismatch:
        try:
            artifacts = json.loads(case.artifacts_json or '{}')
        except (ValueError, TypeError):
            artifacts = {}
        operation = artifacts.get('note_operation', {}) if isinstance(artifacts, dict) else {}
        if not isinstance(operation, dict):
            operation = {}
        failed_local = (not mismatch and step is not None and step.status == 'failed'
                        and operation.get('backend') == LOCAL_NOTE
                        and operation.get('operation_id') == case.id and note_backend() == LOCAL_NOTE)
        result['case_note'] = {'status': 'failed' if failed_local else 'unknown',
                               'attempted': True, 'receipt': None}
    return result


def recovery_blocker(db, case, inputs):
    step = db.query(Step).filter_by(case_id=case.id, name='post_case_note_to_hris').first()
    if step is None or step.status == 'completed':
        return None
    receipt, mismatch = confirmed_note(db, case.id, inputs['employee_id'])
    if mismatch or receipt:
        return mismatch
    operation = json.loads(case.artifacts_json or '{}').get('note_operation', {})
    if operation.get('backend') != LOCAL_NOTE or operation.get('operation_id') != case.id:
        return 'note_outcome_unknown'
    if note_backend() != LOCAL_NOTE:
        return 'note_backend_changed'
    return None
