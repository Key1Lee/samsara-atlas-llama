import json
import logging
import os
from datetime import datetime, timezone

from pydantic import TypeAdapter
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.engine.context import ExecutionContext
from app.engine.steps import STEPS, classify_policy_intent
from app.tools.retrieval import retrieve_policy_snippets
from app.schemas import CaseInputs, IdempotencyKey
from app.storage.database import SessionLocal
from app.storage.models import Case, Step, EmployeeRecord
from app.audit import record_event, error_code, safe_step_name
from app.engine.recovery import GUARD, execution_guard, GuardUnavailable, note_backend, confirmed_note, recovery_blocker, case_note_status

logger = logging.getLogger(__name__)


class IdempotencyConflict(ValueError):
    pass


class EmployeeNotFound(ValueError):
    pass


class EmployeeLookupUnavailable(RuntimeError):
    pass


def _result(case):
    # No authoritative human-review decision store exists. Model quality is not approval.
    return {"case_id": case.id, "status": case.status, "review_status": None,
            "outputs": json.loads(case.outputs_json or "{}")}


def _validate_employee(db, employee_id):
    try:
        employee = db.query(EmployeeRecord).filter_by(employee_id=employee_id).first()
    except SQLAlchemyError:
        raise EmployeeLookupUnavailable("Employee lookup unavailable. Try again later.") from None
    if employee is None:
        raise EmployeeNotFound("Employee not found. Refresh the employee list before submitting.")


def run_workflow(inputs: dict, idempotency_key: str | None = None) -> dict:
    inputs = CaseInputs.model_validate(inputs).model_dump()
    idempotency_key = TypeAdapter(IdempotencyKey | None).validate_python(idempotency_key)
    db = SessionLocal()
    try:
        # Validate before creating a case or choosing a workflow shortcut.
        # EmployeeRecord is the local synthetic directory, not a live HRIS.
        _validate_employee(db, inputs["employee_id"])
        case = None
        if idempotency_key is not None:
            case = db.query(Case).filter_by(idempotency_key=idempotency_key).first()
        if case is None:
            case = Case(
                status="running", idempotency_key=idempotency_key,
                inputs_json=json.dumps(inputs),
                artifacts_json=json.dumps({"prompt_version": "v1", "provider_type": os.getenv("PROVIDER_TYPE", "stub"),
                                           "execution_guard": GUARD}),
            )
            db.add(case)
            try:
                record_event(db, case.id, 'case_created', 'success').case = case
                db.commit()
            except IntegrityError:
                db.rollback()
                if idempotency_key is None:
                    raise
                case = db.query(Case).filter_by(idempotency_key=idempotency_key).one()
                return _resume_or_return(db, case, inputs)
            return _resume_or_return(db, case, inputs, newly_created=True)
        return _resume_or_return(db, case, inputs)
    finally:
        db.close()


def recover_workflow(case_id):
    with SessionLocal() as db:
        case = db.get(Case, case_id)
        if case is None:
            raise ValueError("Case not found")
        inputs = CaseInputs.model_validate(json.loads(case.inputs_json)).model_dump()
        _validate_employee(db, inputs['employee_id'])
        return _resume_or_return(db, case, inputs)


def _manual_reconciliation(db, case, reason):
    record_event(db, case.id, 'recovery_blocked', 'failure', error_code=reason)
    db.commit()
    return {**_result(case), 'recovery': 'manual_reconciliation_required', 'recovery_reason': reason}


def _resume_or_return(db, case, inputs, newly_created=False):
    if json.loads(case.inputs_json) != inputs:
        raise IdempotencyConflict("Idempotency key already used with different inputs")
    if case.status not in ('failed', 'running'):
        record_event(db, case.id, 'request_reused', 'observed')
        db.commit()
        return _result(case)
    if case.status == 'running' and json.loads(case.artifacts_json or '{}').get('execution_guard') != GUARD:
        return _manual_reconciliation(db, case, 'ownership_unverified')
    try:
        with execution_guard(db, case.id) as acquired:
            db.refresh(case)
            if not acquired or case.status not in ('failed', 'running'):
                record_event(db, case.id, 'request_reused', 'observed')
                db.commit()
                return _result(case)
            reason = recovery_blocker(db, case, inputs)
            if reason:
                return _manual_reconciliation(db, case, reason)
            if not newly_created:
                if case.status == 'failed':
                    claimed = db.query(Case).filter_by(id=case.id, status='failed').update({'status': 'running'})
                    if not claimed:
                        db.refresh(case)
                        return _result(case)
                artifacts = json.loads(case.artifacts_json or '{}')
                artifacts['execution_guard'] = GUARD
                case.artifacts_json = json.dumps(artifacts)
                record_event(db, case.id, 'case_resumed', 'success')
                db.commit()
            return _execute(db, case, inputs)
    except GuardUnavailable:
        return _manual_reconciliation(db, case, 'execution_guard_unavailable')


def _execute(db, case, inputs):
    ctx = ExecutionContext(case_id=case.id, inputs=inputs, artifacts=json.loads(case.artifacts_json or "{}"))
    step_record = None
    try:
        workflow_steps = STEPS
        previous_steps = {name for (name,) in db.query(Step.name).filter_by(case_id=case.id)}
        # Keep the already-started route on recovery. Re-running retrieval to
        # choose a route could skip a pending note if policy files have changed.
        deterministic_only = (previous_steps == {'generate_structured_response'} if previous_steps else
                              bool(classify_policy_intent(inputs["case_transcript"])
                                   or not retrieve_policy_snippets(inputs["case_transcript"])))
        if deterministic_only:
            # Deterministic policy answers and unsupported questions need no
            # model, employee context, or HRIS write.
            workflow_steps = [(name, fn) for name, fn in STEPS if name == "generate_structured_response"]
        for step_name, step_fn in workflow_steps:
            step_record = db.query(Step).filter_by(case_id=case.id, name=step_name).first()
            if step_record and step_record.status == "completed":
                ctx.step_outputs[step_name] = json.loads(step_record.output_json)
                step_record = None
                continue
            if step_record is None:
                step_record = Step(case_id=case.id, name=step_name, attempts=0)
                db.add(step_record)
            step_record.status = "running"
            step_record.started_at = datetime.now(timezone.utc)
            step_record.finished_at = None
            step_record.error = None
            step_record.attempts += 1
            if step_name == 'post_case_note_to_hris':
                ctx.artifacts['note_operation'] = {'backend': note_backend(), 'operation_id': case.id}
                case.artifacts_json = json.dumps(ctx.artifacts)
            record_event(db, case.id, 'step_started', 'started',
                         step_name=step_name, attempt=step_record.attempts)
            db.commit()

            receipt = None
            if step_name == 'post_case_note_to_hris':
                receipt, mismatch = confirmed_note(db, case.id, inputs['employee_id'])
                if mismatch:
                    raise ValueError('Stored note receipt does not match this case')
            from app.evaluation.observation import begin, end
            observation, observation_token = begin(step_name)
            try:
                output = receipt if receipt is not None else step_fn(ctx)
            finally:
                end(observation, observation_token)
                if observation is not None:
                    ctx.artifacts.setdefault('step_diagnostics', []).append(observation)
                    ctx.artifacts['step_diagnostics'] = ctx.artifacts['step_diagnostics'][-50:]
                    case.artifacts_json = json.dumps(ctx.artifacts)
            ctx.step_outputs[step_name] = output
            step_record.output_json = json.dumps(output)
            step_record.status = "completed"
            step_record.finished_at = datetime.now(timezone.utc)
            record_event(db, case.id, 'step_completed', 'success',
                         step_name=step_name, attempt=step_record.attempts)
            db.commit()
            step_record = None

        outputs = {**ctx.step_outputs["generate_structured_response"],
                   "operation_status": case_note_status(db, case, inputs)}
        case.status = "completed"
        case.outputs_json = json.dumps(outputs)
        record_event(db, case.id, 'case_completed', 'success')
        db.commit()
        return _result(case)
    except Exception as exc:
        db.rollback()
        from app.diagnostics import failure_diagnostic
        diagnostic = getattr(exc, "atlas_diagnostic", None) or failure_diagnostic(exc)
        saved_artifacts = {**json.loads(case.artifacts_json or "{}"), **ctx.artifacts}
        saved_artifacts["last_failure_diagnostic"] = diagnostic
        case.artifacts_json = json.dumps(saved_artifacts)
        case.status = "failed"
        if step_record is not None:
            step_record.status = "failed"
            step_record.finished_at = datetime.now(timezone.utc)
            # Validation exceptions can include the original case text or model output.
            step_record.error = type(exc).__name__
            record_event(db, case.id, 'step_failed', 'failure', step_name=step_record.name,
                         attempt=step_record.attempts, error_code=error_code(exc))
        record_event(db, case.id, 'case_failed', 'failure', error_code=error_code(exc))
        db.commit()
        try:
            logger.error("Case %s step %s failed (%s)", case.id,
                         safe_step_name(step_record.name if step_record else None), error_code(exc))
            logger.error("atlas_workflow_failure %s", json.dumps({"case_id": case.id,
                         "step": safe_step_name(step_record.name if step_record else None), **diagnostic}))
        except Exception:
            pass  # Optional diagnostics cannot replace the persisted failure.
        raise
