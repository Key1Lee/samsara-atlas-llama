"""Small request context and transactional case-event helpers; no payload logging."""
from contextvars import ContextVar
import hashlib
import re

from sqlalchemy.exc import SQLAlchemyError

from app.storage.models import CaseAuditEvent

current_request_id = ContextVar('atlas_request_id', default=None)


def safe_step_name(name):
    return name if isinstance(name, str) and re.fullmatch(r'[a-z_]{1,64}', name) else 'unknown'


def error_code(exc):
    # Fixed categories, never untrusted exception messages or class names.
    from app.providers.errors import ProviderError
    from tenacity import RetryError
    if isinstance(exc, RetryError):
        exc = exc.last_attempt.exception()
    if isinstance(exc, ProviderError):
        return type(exc).code
    from app.rag.errors import RAGError
    if isinstance(exc, RAGError):
        return exc.code
    if isinstance(exc, SQLAlchemyError):
        return 'storage_error'
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return 'provider_or_tool_unavailable'
    if isinstance(exc, ValueError):
        return 'invalid_output'
    return 'workflow_error'


def record_event(db, case_id, event_type, outcome, *, step_name=None, attempt=None,
                 error_code=None, note_id=None):
    # Caller commits: a rolled-back business transition rolls back its event.
    event = CaseAuditEvent(case_id=case_id, request_id=current_request_id.get(),
                           event_type=event_type, outcome=outcome,
                           step_name=safe_step_name(step_name) if step_name else None,
                           attempt=attempt, error_code=error_code, note_id=note_id)
    db.add(event)
    return event


def policy_references(outputs, retrieved_sources=()):
    """Expose known content fingerprints, never promote them to approved versions."""
    references = []
    for evidence in outputs.get('citation_evidence', []):
        source_id = evidence.get('source_id', '')
        if not isinstance(source_id, str) or not re.fullmatch(r'sha256:[a-f0-9]{64}', source_id):
            continue
        filename = evidence.get('source_file')
        if not isinstance(filename, str) or not re.fullmatch(r'[a-z0-9-]+\.md', filename):
            filename = None
        reference = {'source_id': source_id, 'source_file': filename,
                     'policy_id': None, 'policy_version': None, 'evidence_type': 'cited'}
        if reference not in references:
            references.append(reference)
    for source in retrieved_sources:
        source_id = 'sha256:' + hashlib.sha256(source.encode('utf-8')).hexdigest()
        if not any(reference['source_id'] == source_id for reference in references):
            references.append({'source_id': source_id, 'source_file': None,
                               'policy_id': None, 'policy_version': None, 'evidence_type': 'retrieved'})
    return references
