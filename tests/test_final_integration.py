"""Phase12 authenticated UI/API/SQLite integration; controlled model, no live HRIS."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from app.storage.database import Base
from app.storage.models import Case, CaseNote, EmployeeRecord, Step
from app.providers.base import CompletionResponse

ROOT = Path(__file__).resolve().parents[1]


def test_authenticated_ui_to_committed_history_and_audit(tmp_path, monkeypatch):
    from app.api import routes
    from app.engine import runner, steps
    from app.middleware import rate_limit
    from app.storage import database
    from app.tools import hris
    from app.main import app

    engine = create_engine(f"sqlite:///{tmp_path / 'integration.db'}",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (routes, runner, rate_limit, database, hris):
        monkeypatch.setattr(module, "SessionLocal", factory)
    tokens = {name: uuid4().hex for name in ("INTEGRATION-A", "INTEGRATION-B")}
    monkeypatch.setenv("ATLAS_ENV", "production")
    monkeypatch.setenv("ATLAS_EMPLOYEE_TOKEN_HASHES", json.dumps({
        name: hashlib.sha256(token.encode()).hexdigest() for name, token in tokens.items()}))
    with factory() as db:
        db.add_all([EmployeeRecord(employee_id=name, metadata_json="{}", last_activities_json="[]")
                    for name in tokens])
        db.commit()

    sql_ms, http, model = [], [], []
    @event.listens_for(engine, "before_cursor_execute")
    def sql_start(conn, cursor, statement, parameters, context, executemany):
        context.phase12_start = perf_counter()
    @event.listens_for(engine, "after_cursor_execute")
    def sql_end(conn, cursor, statement, parameters, context, executemany):
        sql_ms.append((perf_counter() - context.phase12_start) * 1000)

    notice = "Submit requests through Workday at least 2 weeks in advance for planned time off"
    approval = "Manager approval is required before time off is confirmed"
    body = notice + ". " + approval + "."
    class ControlledProvider:
        def complete(self, prompt, **kwargs):
            start = perf_counter()
            with factory() as db:
                case = db.query(Case).one()
                assert case.status == "running"  # Intake committed before provider invocation.
            if "cited_snippets" in prompt:
                data = {"subject": "Planned PTO request", "body": body,
                        "tone": kwargs["tone"], "cited_snippets": [notice, approval]}
            else:
                data = {"summary": "Employee asks about planned PTO requests.",
                        "action_items": ["Ask the manager to review the request."], "next_steps": []}
            model.append({"prompt_bytes": len(prompt.encode()), "latency_ms": (perf_counter()-start)*1000})
            return CompletionResponse(json.dumps(data))
    monkeypatch.setattr(steps, "get_provider", lambda: ControlledProvider())

    client = TestClient(app)
    def bridge(method, url, **kw):
        kw.pop("timeout", None)
        start = perf_counter()
        response = client.request(method, url.replace("http://localhost:8000", ""), **kw)
        http.append({"method": method, "path": url.replace("http://localhost:8000", ""),
                     "status": response.status_code, "latency_ms": (perf_counter()-start)*1000,
                     "request_id": response.headers.get("X-Request-ID")})
        result = requests.Response()
        result.status_code = response.status_code
        result._content = response.content
        return result
    monkeypatch.setattr(requests, "get", lambda url, **kw: bridge("GET", url, **kw))
    monkeypatch.setattr(requests, "post", lambda url, **kw: bridge("POST", url, **kw))
    try:
        ui = AppTest.from_file(str(ROOT/"ui/streamlit_app.py")).run()
        next(w for w in ui.text_input if w.label == "Employee access token").set_value(tokens["INTEGRATION-A"]).run()
        ui.selectbox[0].select("INTEGRATION-A").run()
        next(w for w in ui.text_input if w.label == "Password").set_value("samsara").run()
        next(b for b in ui.button if b.label == "Continue").click().run()
        question = "How do I request planned PTO?"
        ui.text_area[0].set_value(question)
        assert not ui.selectbox and not ui.number_input
        next(b for b in ui.button if b.label == "Run Workflow").click().run()
        assert not ui.exception
        result = ui.session_state["case_result"]
        case_id = result["case_id"]
        assert result["status"] == "completed" and result["review_status"] is None
        assert result["outputs"]["body"] == body
        assert any("HR review is required" in w.value for w in ui.warning)
        assert len(model) == 2
        headers = {"Authorization": "Bearer " + tokens["INTEGRATION-A"]}
        detail = client.get(f"/cases/{case_id}", headers=headers).json()
        assert detail["review_status"] is None
        with factory() as db:
            case = db.get(Case, case_id)
            assert json.loads(case.inputs_json)["case_transcript"] == question
            assert json.loads(case.outputs_json) == result["outputs"]
            records = db.query(Step).filter_by(case_id=case_id).order_by(Step.id).all()
            assert len(records) == 5 and all(s.status == "completed" for s in records)
            selected = json.loads(records[1].output_json)
            assert any("unless local law requires payout" in source for source in selected)
            assert len(result["outputs"]["citation_evidence"]) == 2
            assert all(e["source_file"] == "pto-policy.md" and e["source_verification"] == "matched"
                       for e in result["outputs"]["citation_evidence"])
            note = db.query(CaseNote).one()
            assert note.employee_id == "INTEGRATION-A" and note.case_id == case_id
            assert result["outputs"]["operation_status"]["case_note"]["receipt"]["note_id"] == note.id
            assert result["outputs"]["execution_receipts"] == []
            assert result["outputs"]["operation_status"]["external_actions"]["status"] == "not_attempted"
            durations = {s.name: (s.finished_at-s.started_at).total_seconds()*1000 for s in records}
        history = hris.retrieve_employee_context("INTEGRATION-A")["case_history"]
        assert [r["case_id"] for r in history] == [case_id]
        trail = client.get(f"/cases/{case_id}/audit", headers=headers).json()
        assert trail["case_status"] == "completed" and trail["review_status"] is None
        assert trail["policy_evidence"] and trail["events"][-1]["event_type"] == "case_completed"
        post = next(r for r in http if r["method"] == "POST" and r["path"] == "/cases")
        assert all(e["request_id"] == post["request_id"] for e in trail["events"])
        assert client.get(f"/cases/{case_id}").status_code == 401
        assert client.get(f"/cases/{case_id}", headers={
            "Authorization": "Bearer " + tokens["INTEGRATION-B"]}).status_code == 404
        next(b for b in ui.button if b.label == "Run Workflow").click().run()
        assert ui.session_state["case_result"] == result and len(model) == 2
        with factory() as db:
            assert db.query(Case).count() == db.query(CaseNote).count() == 1
        target = os.getenv("ATLAS_PHASE12_EVIDENCE")
        if target:
            Path(target).write_text(json.dumps({
                "case_id": case_id, "processing": "completed", "review_status": None,
                "case_rows": 1, "step_rows": 5, "local_note_rows": 1,
                "external_actions": 0, "model": "controlled-fixture-phase12",
                "model_calls": model, "http": http, "step_duration_ms": durations,
                "sql_operations": len(sql_ms), "sql_total_ms": sum(sql_ms),
                "sql_max_ms": max(sql_ms), "audit_events": trail["events"],
                "policy_evidence": trail["policy_evidence"], "retry_model_calls": 0,
                "integrity_violations": 0, "history_case_ids": [case_id],
                "limitations": "Synthetic self-access credential; no HR-wide role, approved-policy registry or live inference."
            }, indent=2))
    finally:
        engine.dispose()

