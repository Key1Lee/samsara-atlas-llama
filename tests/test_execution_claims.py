"""BR-08: application-owned action state and actual local note receipts."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.engine import steps
from app.engine.context import ExecutionContext
from app.storage.database import Base
from app.storage.models import Case, CaseNote, EmployeeRecord, Step


DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
INPUTS = {"employee_id": "EXECUTION-A", "case_transcript": "Please explain accommodation review.",
          "tone": "direct"}


def _draft(monkeypatch, body, *, subject="Accommodation review", extra=None, question=None):
    monkeypatch.setattr(steps, "_retryable_llm_call", lambda *args, **kwargs: json.dumps({
        "subject": subject, "body": body, "tone": "direct", "cited_snippets": [], **(extra or {}),
    }))
    context = ExecutionContext(case_id=0, inputs={
        **INPUTS, "case_transcript": question or INPUTS["case_transcript"],
    }, artifacts={}, step_outputs={
        "retrieve_employee_context": {"found": True, "fact_verification": "unverified", "metadata": {}},
        "retrieve_policy_snippets": [(DOCS / "workplace-accommodation-policy.md").read_text()],
    })
    return steps.generate_structured_response_step(context)


def _external_not_executed(operation_status):
    assert operation_status["external_actions"]["status"] == "not_attempted"
    assert operation_status["external_actions"]["supported_operations"] == []


def test_proposed_action_has_no_execution_or_model_fabricated_receipt(monkeypatch):
    body = "Ask your HR partner to schedule a meeting and review the interactive process."
    output = _draft(monkeypatch, body, extra={
        "operation_status": {
            "case_note": {"status": "succeeded", "attempted": True, "receipt": {"note_id": 999999}},
            "external_actions": {"status": "succeeded", "supported_operations": ["send_email"]},
        },
        "execution_receipts": [{"operation": "send_email", "status": "succeeded"}],
    })
    assert output["body"] == body
    assert output["execution_receipts"] == []
    status = output["operation_status"]
    assert status["case_note"] == {"status": "not_attempted", "attempted": False, "receipt": None}
    _external_not_executed(status)


@pytest.mark.parametrize("claim", [
    "The email is sent.",
    "HR has been notified.",
    "Your PTO balance increased to 80 hours.",
    "Booked your meeting.",
])
def test_passive_and_elided_completion_claims_require_hr_review(monkeypatch, claim):
    output = _draft(monkeypatch, claim)
    assert claim not in output["body"]
    assert output["answerability"]["status"] == "hr_review_required"
    assert "hr" in output["body"].lower()
    assert output["execution_receipts"] == []
    assert output["operation_status"]["case_note"]["receipt"] is None
    _external_not_executed(output["operation_status"])


def test_completion_in_subject_cannot_bypass_the_action_check(monkeypatch):
    subject = "HR has been notified"
    output = _draft(monkeypatch, "Ask HR to review the interactive process.", subject=subject)
    assert output["subject"] != subject
    assert output["answerability"]["status"] == "hr_review_required"
    _external_not_executed(output["operation_status"])


@pytest.fixture
def store(tmp_path, monkeypatch):
    from app.api import routes
    from app.engine import runner
    from app.middleware import rate_limit
    from app.storage import database
    from app.tools import hris

    engine = create_engine(f'sqlite:///{tmp_path / "execution.db"}', connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (routes, runner, rate_limit, database, hris):
        monkeypatch.setattr(module, "SessionLocal", factory)
    tokens = {employee: uuid4().hex for employee in ("EXECUTION-A", "EXECUTION-B")}
    monkeypatch.setenv("ATLAS_ENV", "production")
    monkeypatch.setenv("ATLAS_EMPLOYEE_TOKEN_HASHES", json.dumps({
        employee: hashlib.sha256(token.encode()).hexdigest() for employee, token in tokens.items()}))
    with factory() as db:
        db.add_all([EmployeeRecord(employee_id=employee, metadata_json="{}", last_activities_json="[]")
                    for employee in tokens])
        db.commit()
    yield factory, {employee: {"Authorization": "Bearer " + token} for employee, token in tokens.items()}
    engine.dispose()


def _create(client, headers, key):
    return client.post("/cases", headers=headers["EXECUTION-A"],
                       json={"inputs": INPUTS, "idempotency_key": key})


def _receipt(status, case_id, note_id):
    assert status["case_note"]["status"] == "succeeded"
    assert status["case_note"]["attempted"] is True
    assert status["case_note"]["receipt"] == {
        "operation": "save_case_note", "status": "succeeded", "source": "sqlite_case_notes",
        "case_id": case_id, "note_id": note_id,
    }
    _external_not_executed(status)


def test_successful_local_note_receipt_is_persisted_reused_and_owner_scoped(store):
    from app.main import app

    factory, headers = store
    client = TestClient(app)
    key = uuid4().hex
    response = _create(client, headers, key)
    assert response.status_code == 200
    result = response.json()
    with factory() as db:
        note = db.query(CaseNote).one()
        note_id = note.id
        assert note.employee_id == "EXECUTION-A" and note.case_id == result["case_id"]
        assert json.loads(db.get(Case, result["case_id"]).outputs_json) == result["outputs"]
    _receipt(result["outputs"]["operation_status"], result["case_id"], note_id)
    assert result["outputs"]["execution_receipts"] == []
    assert _create(client, headers, key).json() == result
    detail = client.get(f'/cases/{result["case_id"]}', headers=headers["EXECUTION-A"])
    assert detail.status_code == 200
    _receipt(detail.json()["operation_status"], result["case_id"], note_id)
    assert client.get(f'/cases/{result["case_id"]}', headers=headers["EXECUTION-B"]).status_code == 404
    with factory() as db:
        assert db.query(Case).count() == db.query(CaseNote).count() == 1


def test_failed_local_write_has_no_success_receipt(store):
    from app.main import app

    factory, headers = store

    def fail_note_flush(session, flush_context, instances):
        if any(isinstance(record, CaseNote) for record in session.new):
            raise RuntimeError("Synthetic local note failure before commit")

    event.listen(factory, "before_flush", fail_note_flush)
    try:
        client = TestClient(app)
        assert _create(client, headers, uuid4().hex).status_code == 500
        with factory() as db:
            case = db.query(Case).one()
            assert case.status == "failed" and db.query(CaseNote).count() == 0
            case_id = case.id
        detail = client.get(f"/cases/{case_id}", headers=headers["EXECUTION-A"]).json()
        status = detail["operation_status"]
        assert status["case_note"] == {"status": "failed", "attempted": True, "receipt": None}
        _external_not_executed(status)
    finally:
        event.remove(factory, "before_flush", fail_note_flush)


def test_lost_acknowledgement_uses_actual_note_even_when_step_failed(store, monkeypatch):
    from app.main import app
    from app.tools.registry import get_tool_registry

    factory, headers = store
    registry = get_tool_registry()
    original = registry.get("post_case_note_to_hris")
    calls = []

    def saved_then_failed(**kwargs):
        calls.append(kwargs["case_id"])
        original(**kwargs)
        raise RuntimeError("Synthetic acknowledgement lost after local commit")

    monkeypatch.setitem(registry._tools, "post_case_note_to_hris", saved_then_failed)
    client = TestClient(app)
    key = uuid4().hex
    assert _create(client, headers, key).status_code == 500
    with factory() as db:
        case = db.query(Case).one()
        note = db.query(CaseNote).one()
        case_id, note_id = case.id, note.id
        assert case.status == "failed"
    detail = client.get(f"/cases/{case_id}", headers=headers["EXECUTION-A"]).json()
    _receipt(detail["operation_status"], case_id, note_id)
    recovered = _create(client, headers, key)
    assert recovered.status_code == 200
    _receipt(recovered.json()["outputs"]["operation_status"], case_id, note_id)
    assert calls == [case_id]
    with factory() as db:
        assert db.query(CaseNote).count() == 1


def _record(factory, *, note_status="failed", backend="sqlite_case_note_v1", note=None):
    response = {"subject": "HR review", "body": "Ask HR to review the interactive process.",
                "tone": "direct", "cited_snippets": []}
    with factory() as db:
        case = Case(status="failed", inputs_json=json.dumps(INPUTS))
        db.add(case)
        db.flush()
        case.artifacts_json = json.dumps({"note_operation": {"backend": backend, "operation_id": case.id}})
        db.add(Step(case_id=case.id, name="generate_structured_response", status="completed",
                    output_json=json.dumps(response)))
        db.add(Step(case_id=case.id, name="post_case_note_to_hris", status=note_status,
                    output_json=json.dumps({"id": 999999, "employee_id": INPUTS["employee_id"]})))
        if note:
            expected = f'Unapproved response draft (needs HR review): {response["subject"]} - {response["body"][:200]}'
            db.add(CaseNote(case_id=case.id, employee_id=note.get("employee_id", INPUTS["employee_id"]),
                            note=note.get("note", expected)))
        db.commit()
        return case.id


@pytest.mark.parametrize("note_status,backend", [
    ("failed", "unknown"), ("running", "sqlite_case_note_v1"), ("completed", "sqlite_case_note_v1"),
])
def test_uncertain_or_missing_receipt_is_unknown_not_success(store, note_status, backend):
    from app.engine.recovery import case_note_status

    factory, _ = store
    case_id = _record(factory, note_status=note_status, backend=backend)
    with factory() as db:
        status = case_note_status(db, db.get(Case, case_id), INPUTS)
    assert status["case_note"] == {"status": "unknown", "attempted": True, "receipt": None}
    _external_not_executed(status)


@pytest.mark.parametrize("note", [
    {"employee_id": "EXECUTION-B"}, {"note": "This is unrelated to the saved draft."},
])
def test_mismatched_note_cannot_establish_a_success_receipt(store, note):
    from app.engine.recovery import case_note_status

    factory, _ = store
    case_id = _record(factory, note=note)
    with factory() as db:
        status = case_note_status(db, db.get(Case, case_id), INPUTS)
    assert status["case_note"] == {"status": "unknown", "attempted": True, "receipt": None}
    _external_not_executed(status)


@pytest.mark.parametrize("artifacts,draft", [
    ("not-json", None), ("[]", None), ('{"note_operation":[]}', None), (None, "[]"),
])
def test_malformed_receipt_metadata_remains_unknown(store, artifacts, draft):
    from app.engine.recovery import case_note_status

    factory, _ = store
    case_id = _record(factory, note={} if draft is None else {"employee_id": "EXECUTION-A"})
    with factory() as db:
        case = db.get(Case, case_id)
        if artifacts is not None:
            case.artifacts_json = artifacts
        if draft is not None:
            db.query(Step).filter_by(case_id=case_id, name="generate_structured_response").one().output_json = draft
        db.commit()
        status = case_note_status(db, case, INPUTS)
    assert status["case_note"] == {"status": "unknown", "attempted": True, "receipt": None}


def test_deterministic_policy_route_has_no_attempted_note(monkeypatch):
    output = _draft(monkeypatch, "An untrusted provider would invent execution.",
                    question="How many PTO hours can I carry over?")
    assert output["operation_status"]["case_note"] == {
        "status": "not_attempted", "attempted": False, "receipt": None,
    }
    assert output["execution_receipts"] == []
    _external_not_executed(output["operation_status"])
