"""Acceptance tests for employee context and committed case-history consistency."""
import json
from uuid import uuid4


def _employee(db):
    from app.storage.models import EmployeeRecord
    employee_id = f"HR-{uuid4().hex}"
    fixture_activity = {"type": "case", "date": "2026-06-02", "summary": "Fixture history"}
    db.add(EmployeeRecord(employee_id=employee_id, metadata_json=json.dumps({"department": "Engineering"}), last_activities_json=json.dumps([fixture_activity])))
    db.commit()
    return employee_id, fixture_activity


def _case(db, employee_id):
    from app.storage.models import Case
    case = Case(status="completed", inputs_json=json.dumps({"employee_id": employee_id}))
    db.add(case)
    db.commit()
    return case.id


def test_committed_note_appears_once_in_next_employee_lookup(db_session):
    from app.tools.hris import retrieve_employee_context, post_case_note_to_hris
    employee_id, fixture_activity = _employee(db_session)
    case_id = _case(db_session, employee_id)
    before = retrieve_employee_context(employee_id)
    receipt = post_case_note_to_hris(case_id, employee_id, "PTO draft recorded")
    retried_receipt = post_case_note_to_hris(case_id, employee_id, "PTO draft recorded")
    after = retrieve_employee_context(employee_id)
    assert before["last_activities"] == [fixture_activity]
    assert receipt["id"] == retried_receipt["id"]
    persisted = [a for a in after["last_activities"] if a.get("case_id") == case_id]
    assert len(persisted) == 1
    assert persisted[0]["summary"] == "PTO draft recorded"
    assert persisted[0]["source"] == "case_notes"
    assert persisted[0]["case_note_id"] == receipt["id"]
    assert persisted[0]["date"].endswith("+00:00")
    assert fixture_activity in after["last_activities"]
    assert retrieve_employee_context(employee_id) == after


def test_history_lookup_does_not_leak_another_employees_note(db_session):
    from app.tools.hris import retrieve_employee_context, post_case_note_to_hris
    employee_id, _ = _employee(db_session)
    other_id, _ = _employee(db_session)
    own_case = _case(db_session, employee_id)
    other_case = _case(db_session, other_id)
    post_case_note_to_hris(own_case, employee_id, "Own employee note")
    post_case_note_to_hris(other_case, other_id, "Other employee confidential note")
    context = retrieve_employee_context(employee_id)
    assert any(a.get("case_id") == own_case for a in context["last_activities"])
    assert not any(a.get("case_id") == other_case for a in context["last_activities"])
    assert "Other employee confidential note" not in json.dumps(context)
