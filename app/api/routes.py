from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.schemas import CaseInputs, IdempotencyKey

from app.engine.runner import run_workflow, recover_workflow, IdempotencyConflict, EmployeeNotFound, EmployeeLookupUnavailable
from app.storage.database import SessionLocal
from app.storage.models import Case, Step, CaseNote, EmployeeRecord, CaseAuditEvent
from app.benchmarks import case_quality_evaluation
from app.audit import policy_references
from app.engine.recovery import case_note_status
from datetime import timezone

import os
import json
from sqlalchemy import func, case as sql_case
from sqlalchemy.exc import SQLAlchemyError
from app.api.access import employee_access, require_scope, authentication_required, analytics_access
from app.benchmarks import global_benchmark, BenchmarkWindowUnavailable
from app.rag.errors import RAGError

router = APIRouter()


@router.get("/employees")
def list_employees(employee: str | None = Depends(employee_access)):
    db = SessionLocal()
    try:
        query = db.query(EmployeeRecord)
        if employee and authentication_required():
            query = query.filter(EmployeeRecord.employee_id == employee)
        employees = query.order_by(EmployeeRecord.employee_id).all()
        return {"employees": [e.employee_id for e in employees]}
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Employee lookup unavailable. Try again later.") from None
    finally:
        db.close()


@router.get("/info")
def get_info():
    """Returns current provider type - use to verify Ollama is active."""
    provider = os.getenv("PROVIDER_TYPE", "stub").lower()
    return {"provider": provider, "mode": "demo" if provider in {"stub", "local"} else
            "live" if provider == "ollama" else "configuration_error"}


@router.get("/benchmarks/latest")
def get_global_benchmark(include_history: bool = False, _=Depends(analytics_access)):
    try:
        with SessionLocal() as db:
            return global_benchmark(db, include_history=include_history)
    except (SQLAlchemyError, BenchmarkWindowUnavailable):
        raise HTTPException(503, "Benchmark data unavailable.") from None


def owned_cases(db, employee):
    # Existing cases retain their original storage format; no migration needed.
    owner = func.json_extract(sql_case((func.json_valid(Case.inputs_json) == 1,
                                       Case.inputs_json), else_='{}'), '$.employee_id')
    return db.query(Case).filter(owner == employee)


@router.get("/cases")
def list_cases(employee: str | None = Depends(employee_access)):
    employee = require_scope(employee)
    db = SessionLocal()
    try:
        cases = owned_cases(db, employee).order_by(Case.created_at.desc(), Case.id.desc()).all()
        return {"case_ids": [c.id for c in cases], "cases": [
            {"case_id": c.id, "question": json.loads(c.inputs_json).get("case_transcript", ""),
             "status": c.status, "submitted_at": c.created_at.isoformat() + "Z" if c.created_at else None}
            for c in cases]}
    finally:
        db.close()


class CreateCaseRequest(BaseModel):
    inputs: CaseInputs
    idempotency_key: IdempotencyKey | None = None


@router.post("/cases")
def create_case(req: CreateCaseRequest, employee: str | None = Depends(employee_access)):
    if employee and employee != req.inputs.employee_id:
        raise HTTPException(403, "You cannot submit a case for another employee.")
    try:
        result = run_workflow(
            inputs=req.inputs.model_dump(),
            idempotency_key=req.idempotency_key,
        )
        return result
    except EmployeeNotFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except EmployeeLookupUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except RAGError as exc:
        raise HTTPException(status_code=exc.status_code, detail={'error_code': exc.code,
            'guidance': 'Check policy retrieval configuration; retry with the same idempotency key.'}) from None
    except Exception:
        raise HTTPException(status_code=500, detail="Workflow failed; inspect the case steps and retry with the same idempotency key.") from None


@router.get("/cases/{case_id}")
def get_case(case_id: int, employee: str | None = Depends(employee_access)):
    employee = require_scope(employee)
    db = SessionLocal()
    try:
        case = owned_cases(db, employee).filter(Case.id == case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        steps = db.query(Step).filter_by(case_id=case_id).order_by(Step.id).all()
        case_notes = db.query(CaseNote).filter_by(case_id=case_id, employee_id=employee).all()
        return {
            "case": {
                "id": case.id,
                "status": case.status,
                "inputs": case.inputs_json,
                "artifacts": case.artifacts_json,
                "outputs": case.outputs_json,
            },
            "steps": [
                {
                    "id": s.id,
                    "name": s.name,
                    "status": s.status,
                    "started_at": str(s.started_at) if s.started_at else None,
                    "finished_at": str(s.finished_at) if s.finished_at else None,
                    "attempts": s.attempts,
                    "output": s.output_json,
                    "error": s.error,
                }
                for s in steps
            ],
            "case_notes": [
                {"id": n.id, "employee_id": n.employee_id, "note": n.note, "created_at": str(n.created_at)}
                for n in case_notes
            ],
            "operation_status": case_note_status(db, case, {"employee_id": employee}),
            "review_status": None,  # Consistent with audit: no verified review decision.
        }
    finally:
        db.close()


@router.post("/cases/{case_id}/replay")
def replay_case(case_id: int, employee: str | None = Depends(employee_access)):
    employee = require_scope(employee)
    db = SessionLocal()
    try:
        case = owned_cases(db, employee).filter(Case.id == case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        import json
        inputs = json.loads(case.inputs_json or "{}")
        # Replay is an explicit new run; keyed retries resume the original.
        result = create_case(CreateCaseRequest(inputs=inputs), employee=employee)
        return result
    finally:
        db.close()


@router.get("/cases/{case_id}/audit")
def get_case_audit(case_id: int, employee: str | None = Depends(employee_access)):
    employee = require_scope(employee)
    try:
        with SessionLocal() as db:
            case = owned_cases(db, employee).filter_by(id=case_id).first()
            if case is None:
                raise HTTPException(404, "Case not found")
            outputs = json.loads(case.outputs_json or '{}')
            retrieved = db.query(Step).filter_by(case_id=case_id, name='retrieve_policy_snippets').first()
            sources = json.loads(retrieved.output_json) if retrieved and retrieved.output_json else []
            events = db.query(CaseAuditEvent).filter_by(case_id=case_id).order_by(CaseAuditEvent.id).all()
            return {
                'case_id': case_id, 'case_status': case.status,
                'review_status': None,  # No human-review transaction exists.
                'failure_diagnostic': json.loads(case.artifacts_json or '{}').get('last_failure_diagnostic'),
                'human_review_status': 'not_recorded',
                'quality_evaluation': case_quality_evaluation(case),
                'draft_quality_status': 'needs_hr_review' if outputs.get('quality_status') == 'needs_hr_review' else None,
                'policy_evidence': policy_references(outputs, sources),
                'case_note_ids': [note.id for note in db.query(CaseNote).filter_by(
                    case_id=case_id, employee_id=employee).order_by(CaseNote.id)],
                'events': [{
                    'event_id': event.id, 'case_id': event.case_id, 'request_id': event.request_id,
                    'event_type': event.event_type,
                    'timestamp': event.timestamp.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z'),
                    'outcome': event.outcome, 'error_code': event.error_code,
                    'step_name': event.step_name, 'attempt': event.attempt, 'note_id': event.note_id,
                } for event in events],
            }
    except SQLAlchemyError:
        raise HTTPException(503, "Audit history unavailable. Try again later.") from None


@router.post("/cases/{case_id}/recover")
def recover_case(case_id: int, employee: str | None = Depends(employee_access)):
    employee = require_scope(employee)
    try:
        with SessionLocal() as db:
            if owned_cases(db, employee).filter_by(id=case_id).first() is None:
                raise HTTPException(404, "Case not found")
        return recover_workflow(case_id)
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(503, "Case recovery unavailable. Try again later.") from None
    except EmployeeNotFound as exc:
        raise HTTPException(422, str(exc)) from None
    except EmployeeLookupUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception:
        raise HTTPException(500, "Recovery failed. Inspect the case audit before retrying.") from None
