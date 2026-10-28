import json
import os
from datetime import timezone
from sqlalchemy import func, case as sql_case
from sqlalchemy.exc import IntegrityError

from app.storage.database import SessionLocal
from app.storage.models import EmployeeRecord, CaseNote, Case
from app.audit import record_event
from app.employee_facts import unavailable_pto_balance


def retrieve_employee_context(employee_id: str, exclude_case_id: int | None = None) -> dict:
    """Read committed cases and supplementary HRIS notes, not verified HR facts.

    Workflow callers exclude their active case; direct reads include all statuses.
    """
    db = SessionLocal()
    try:
        emp = db.query(EmployeeRecord).filter_by(employee_id=employee_id).first()
        if not emp:
            return {
                "employee_id": employee_id,
                "metadata": {},
                "last_activities": [],
                "case_history": [],
                "found": False,
                "pto_balance": unavailable_pto_balance(employee_id),
            }
        # Match the API's authoritative case ownership and deterministic order.
        # Notes are supplementary: deterministic/failed cases may have no note.
        owner = func.json_extract(sql_case((func.json_valid(Case.inputs_json) == 1,
                                           Case.inputs_json), else_='{}'), '$.employee_id')
        cases = db.query(Case).filter(owner == employee_id)
        if exclude_case_id is not None:
            cases = cases.filter(Case.id != exclude_case_id)
        history = [
            {"case_id": record.id,
             "question": json.loads(record.inputs_json).get("case_transcript", ""),
             "status": record.status,
             "submitted_at": record.created_at.isoformat() + "Z" if record.created_at else None,
             "source": "cases"}
            for record in cases.order_by(Case.created_at.desc(), Case.id.desc()).all()
        ]
        notes = (
            db.query(CaseNote)
            .filter_by(employee_id=employee_id)
            .order_by(CaseNote.created_at.desc(), CaseNote.id.desc())
            .all()
        )
        # Read committed notes directly rather than copying them into fixture JSON.
        # The unique case_id constraint gives retries the same history entry.
        activities = [
            {
                "type": "case",
                "case_id": note.case_id,
                "case_note_id": note.id,
                "source": "case_notes",
                "date": note.created_at.replace(tzinfo=timezone.utc).isoformat(),
                "summary": note.note,
            }
            for note in notes
        ]
        activities.extend(json.loads(emp.last_activities_json or "[]"))
        return {
            "employee_id": emp.employee_id,
            "metadata": json.loads(emp.metadata_json or "{}"),
            "last_activities": activities,
            "case_history": history,
            "found": True,
            "pto_balance": unavailable_pto_balance(employee_id),
        }
    finally:
        db.close()


def post_case_note_to_hris(case_id: int, employee_id: str, note: str) -> dict:
    """Write at most one HRIS note per case, including after a lost acknowledgement."""
    db = SessionLocal()
    try:
        existing = db.query(CaseNote).filter_by(case_id=case_id).first()
        if existing:
            return {"id": existing.id, "employee_id": existing.employee_id, "note": existing.note[:100]}
        case_note = CaseNote(
            case_id=case_id,
            employee_id=employee_id,
            note=note,
        )
        db.add(case_note)
        try:
            db.flush()
            record_event(db, case_id, 'note_saved', 'success', note_id=case_note.id)
            db.commit()
        except IntegrityError:
            db.rollback()
            case_note = db.query(CaseNote).filter_by(case_id=case_id).one()
        db.refresh(case_note)
        return {"id": case_note.id, "employee_id": case_note.employee_id, "note": case_note.note[:100]}
    finally:
        db.close()
