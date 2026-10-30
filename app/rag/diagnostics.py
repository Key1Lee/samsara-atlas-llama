"""Developer diagnostics require both operator authority and employee scope."""
import hashlib
import hmac
import os
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.access import employee_access, require_scope
from .config import Settings
from .errors import RAGError
from .retrieval import get_service

router = APIRouter()


def developer_access(request: Request):
    if os.getenv('ATLAS_RAG_DIAGNOSTICS', '0') != '1':
        raise HTTPException(404, 'Not found')
    expected = os.getenv('ATLAS_RAG_DIAGNOSTICS_TOKEN_HASH', '')
    if not re.fullmatch(r'[a-fA-F0-9]{64}', expected):
        raise HTTPException(503, 'Retrieval diagnostics authentication is not configured')
    token = request.headers.get('X-Atlas-Diagnostics-Token', '')
    if not token or not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), expected.lower()):
        raise HTTPException(403, 'Retrieval diagnostics access denied')


class DiagnosticQuery(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: str = Field(pattern='^(lexical|semantic|hybrid)$')
    top_k: int = Field(default=5, ge=1, le=20)


@router.post('/diagnostics/retrieval', dependencies=[Depends(developer_access)])
def diagnose(query: DiagnosticQuery, employee=Depends(employee_access)):
    employee = require_scope(employee)
    if not query.question.strip():
        raise HTTPException(422, 'A question is required')
    try:
        result = get_service(Settings()).retrieve(query.question, {'employee_id': employee}, query.top_k, query.mode)
    except RAGError as error:
        raise HTTPException(error.status_code, {'error_code': error.code}) from None
    # No passages, raw prompts, employee attributes or absolute source paths.
    metadata = {key: value for key, value in result.metadata.items() if key != 'context_spans'}
    return {'query': query.question, **metadata}
