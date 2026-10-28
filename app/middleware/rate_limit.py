"""Persistent daily quotas for requests attributable to a selected employee."""
import json
import re
from datetime import datetime, timedelta, timezone

from fastapi import Request, HTTPException
from app.api.access import employee_access, analytics_access
from fastapi.responses import JSONResponse
from sqlalchemy.dialects.sqlite import insert
from starlette.concurrency import run_in_threadpool

from app.storage.database import SessionLocal
from app.storage.models import Case, EmployeeDailyRequests

DAILY_LIMIT = 200


def utc_now():
    return datetime.now(timezone.utc)


def consume_request(employee_id, day):
    # One atomic statement prevents simultaneous requests exceeding the quota.
    with SessionLocal() as db:
        statement = insert(EmployeeDailyRequests).values(
            employee_id=employee_id, day=day, request_count=1,
        ).on_conflict_do_update(
            index_elements=['employee_id', 'day'],
            set_={'request_count': EmployeeDailyRequests.request_count + 1},
            where=EmployeeDailyRequests.request_count < DAILY_LIMIT,
        ).returning(EmployeeDailyRequests.request_count)
        count = db.execute(statement).scalar_one_or_none()
        db.commit()
        return count


def replay_employee(case_id):
    with SessionLocal() as db:
        case = db.get(Case, case_id)
        return json.loads(case.inputs_json or '{}').get('employee_id') if case else None


async def employee_daily_limit(request: Request, call_next):
    # This exact aggregate endpoint has its own operator credential. Do not
    # interpret it as an employee bearer token or charge an employee's quota.
    if request.url.path == '/benchmarks/latest':
        try:
            analytics_access(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail}, headers=exc.headers)
        return await call_next(request)
    try:
        employee_id = employee_access(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail}, headers=exc.headers)
    if request.method == 'POST' and request.url.path == '/cases':
        try:
            payload = await request.json()
            body_id = payload.get('inputs', {}).get('employee_id')
        except (ValueError, AttributeError):
            body_id = None
        if isinstance(body_id, str) and body_id.strip():
            body_id = body_id.strip()
            if employee_id and employee_id != body_id:
                return JSONResponse(status_code=403 if request.headers.get('Authorization') else 400, content={'detail': 'Employee header must match the case employee ID.'})
            employee_id = body_id
    if not employee_id and request.method == 'POST':
        match = re.fullmatch(r'/cases/(\d+)/replay', request.url.path)
        if match:
            employee_id = await run_in_threadpool(replay_employee, int(match.group(1)))
    if not employee_id:
        return await call_next(request)

    now = utc_now()
    reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    count = await run_in_threadpool(consume_request, employee_id, now.date().isoformat())
    headers = {'X-RateLimit-Limit': str(DAILY_LIMIT),
               'X-RateLimit-Remaining': str(max(0, DAILY_LIMIT - (count or DAILY_LIMIT))),
               'X-RateLimit-Reset': str(int(reset.timestamp()))}
    if count is None:
        headers['Retry-After'] = str(max(1, int((reset - now).total_seconds())))
        return JSONResponse(status_code=429, headers=headers, content={
            'detail': 'Daily limit of 200 requests for this employee reached. Try again after midnight UTC.'})
    response = await call_next(request)
    response.headers.update(headers)
    return response
