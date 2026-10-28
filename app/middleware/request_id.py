"""Server-generated HTTP identity, independent of case and idempotency IDs."""
from uuid import uuid4

from app.audit import current_request_id


async def correlate_request(request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    token = current_request_id.set(request_id)
    try:
        response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        return response
    finally:
        current_request_id.reset(token)
