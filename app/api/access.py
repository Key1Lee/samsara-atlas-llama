"""Employee scope for the local demo and credential-based production access."""
import hashlib
import hmac
import json
import os
import re

from fastapi import HTTPException, Request


def authentication_required():
    # Anonymous synthetic access must be an explicit development-only choice.
    return bool(os.getenv('ATLAS_EMPLOYEE_TOKEN_HASHES')) or os.getenv('ATLAS_ENV', 'production').strip().lower() != 'development'


def employee_access(request: Request) -> str | None:
    if hasattr(request.state, 'employee_scope'):
        return request.state.employee_scope
    header = request.headers.get('X-Employee-ID', '').strip() or None
    query = request.query_params.get('employee_id', '').strip() or None
    if header and query and header != query:
        raise HTTPException(403, 'Employee scope does not match.')
    selected = header or query
    configured = os.getenv('ATLAS_EMPLOYEE_TOKEN_HASHES', '')
    if authentication_required():
        try:
            credentials = json.loads(configured)
            if not isinstance(credentials, dict) or not credentials or any(
                not isinstance(k, str) or not k or k != k.strip()
                or not isinstance(v, str) or re.fullmatch(r'[0-9a-fA-F]{64}', v) is None
                for k, v in credentials.items()
            ):
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(503, 'Employee authentication is not configured.') from None
        scheme, _, token = request.headers.get('Authorization', '').partition(' ')
        if scheme.lower() != 'bearer' or not token:
            raise HTTPException(401, 'Employee authentication required.', headers={'WWW-Authenticate': 'Bearer'})
        digest = hashlib.sha256(token.encode()).hexdigest()
        identities = [employee for employee, expected in credentials.items()
                      if hmac.compare_digest(digest, expected.lower())]
        if len(identities) != 1:
            raise HTTPException(401, 'Invalid employee credential.', headers={'WWW-Authenticate': 'Bearer'})
        employee = identities[0]
        if selected and selected != employee:
            raise HTTPException(403, 'You cannot access another employee\'s cases.')
        selected = employee
    request.state.employee_scope = selected
    return selected


def require_scope(employee):
    if not employee:
        raise HTTPException(400, 'Select an employee ID before retrieving cases.')
    return employee


def analytics_access(request: Request):
    """A separate operator credential; employee/demo access never grants analytics."""
    expected = os.getenv('ATLAS_ANALYTICS_TOKEN_HASH', '')
    if re.fullmatch(r'[a-fA-F0-9]{64}', expected) is None:
        raise HTTPException(503, 'Analytics authentication is not configured.')
    scheme, _, token = request.headers.get('Authorization', '').partition(' ')
    if scheme.lower() != 'bearer' or not token:
        raise HTTPException(401, 'Analytics authentication required.', headers={'WWW-Authenticate': 'Bearer'})
    if not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), expected.lower()):
        raise HTTPException(403, 'Analytics access denied.')
