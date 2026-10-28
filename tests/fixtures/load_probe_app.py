"""requests adapter to the real local API; only the model is stubbed."""
from urllib.parse import urlsplit
from datetime import datetime, timezone
import os

from fastapi.testclient import TestClient

from app.engine import steps
from app.main import app
from app.providers.stub import StubProvider
from app.storage.database import init_db, SessionLocal
from app.storage.models import EmployeeRecord, EmployeeDailyRequests

init_db()
with SessionLocal() as db:
    db.add(EmployeeRecord(employee_id='E10234', metadata_json='{}', last_activities_json='[]'))
    if os.environ.get('PROBE_INITIAL_QUOTA'):
        db.add(EmployeeDailyRequests(employee_id='E10234', day=datetime.now(timezone.utc).date().isoformat(),
                                     request_count=int(os.environ['PROBE_INITIAL_QUOTA'])))
    db.commit()
steps.get_provider = lambda: StubProvider(timeout_chance=0)
_client = TestClient(app)


def post(url, **kwargs):
    return _client.post(urlsplit(url).path, **kwargs)


def get(url, **kwargs):
    return _client.get(urlsplit(url).path, **kwargs)
