"""Disposable-process workflow driver with real abrupt exits at commit boundaries."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, os.environ.get("CRASH_TEST_REPO_ROOT", str(Path(__file__).resolve().parents[2])))

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from app.engine import steps
from app.main import app
from app.providers.stub import StubProvider
from app.storage.database import SessionLocal, engine, init_db
from app.storage.models import CaseAuditEvent, EmployeeRecord, Step
from app.tools.registry import get_tool_registry


INPUTS = {"employee_id": "E10234", "case_transcript": "How does PTO accrue?", "tone": "direct"}


def main():
    mode = sys.argv[1]
    init_db()
    with SessionLocal() as db:
        if db.query(EmployeeRecord).filter_by(employee_id=INPUTS["employee_id"]).first() is None:
            db.add(EmployeeRecord(employee_id=INPUTS["employee_id"], metadata_json="{}", last_activities_json="[]"))
            db.commit()
    steps.get_provider = lambda: StubProvider(timeout_chance=0)

    if mode == "unknown_external":
        def unknown_write(**kwargs):
            # A separate committed operation simulates an unqueryable remote
            # service. The application must not assume this operation failed.
            with sqlite3.connect(engine.url.database) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS synthetic_external_effects (case_id INTEGER)")
                connection.execute("INSERT INTO synthetic_external_effects VALUES (?)", (kwargs["case_id"],))
            os._exit(73)
        get_tool_registry().register("post_case_note_to_hris", unknown_write)

    if mode in ("before_note", "after_note", "before_final_status", "block_before_note"):
        @event.listens_for(Session, "before_commit")
        def tag_boundary(session):
            records = tuple(session.new) + tuple(session.dirty)
            session.info["crash_boundary"] = (
                any(isinstance(row, Step) and row.name == "post_case_note_to_hris"
                    and row.status == "running" for row in records) if mode in ("before_note", "block_before_note") else
                # The HRIS explicitly flushes CaseNote before commit; its
                # mandatory note_saved event is still pending in that same
                # transaction and proves which commit to interrupt after.
                any(isinstance(row, CaseAuditEvent) and row.event_type == "note_saved"
                    for row in session.new) if mode == "after_note" else
                any(isinstance(row, Step) and row.name == "post_case_note_to_hris"
                    and row.status == "completed" for row in records)
            )

        @event.listens_for(Session, "after_commit")
        def crash_after_confirmed_commit(session):
            if session.info.pop("crash_boundary", False):
                if mode == "block_before_note":
                    Path(os.environ["CRASH_TEST_READY"]).touch()
                    deadline = time.monotonic() + 20
                    while not Path(os.environ["CRASH_TEST_RELEASE"]).exists():
                        if time.monotonic() > deadline:
                            os._exit(74)
                        time.sleep(0.02)
                    return
                # Bypass Python exception handling, finalizers and lock cleanup.
                os._exit(73)

    with TestClient(app) as client:
        response = client.post("/cases", headers={"Authorization": "Bearer " + os.environ["CRASH_TEST_TOKEN"]},
            json={"inputs": INPUTS, "idempotency_key": os.environ["CRASH_TEST_KEY"]})
        print("CRASH_TEST_RESULT=" + json.dumps({"http_status": response.status_code, "body": response.json()}), flush=True)


if __name__ == "__main__":
    main()
