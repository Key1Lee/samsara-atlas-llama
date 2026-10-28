"""Acceptance contracts for policy-aware proposals and unapproved draft output.

No external model, service, or live database is used here.
"""
import json

import pytest

from app.engine.context import ExecutionContext
from app.engine import steps


def context():
    return ExecutionContext(
        case_id=0,
        inputs={"employee_id": "E10234", "case_transcript": "Please explain accommodation review.", "tone": "empathetic"},
        artifacts={},
        step_outputs={
            "retrieve_employee_context": {"employee_id": "E10234", "found": True, "metadata": {"department": "People"}},
            "retrieve_policy_snippets": ["# Workplace Accommodation Policy\nUse the interactive process; HR determines eligibility."],
        },
    )


def model_response(monkeypatch, body):
    monkeypatch.setattr(steps, "_retryable_llm_call", lambda *args, **kwargs: json.dumps({
        "subject": "Accommodation review", "body": body, "tone": "empathetic", "cited_snippets": [],
    }))


def test_action_extraction_receives_employee_and_policy_evidence(monkeypatch):
    prompts = []
    def complete(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"summary": "Accommodation question", "action_items": ["Ask HR to review eligibility"], "next_steps": ["Discuss the interactive process with HR"]})
    monkeypatch.setattr(steps, "_retryable_llm_call", complete)
    ctx = context()
    steps.summarize_and_extract_actions_step(ctx)
    assert "People" in prompts[0]
    assert ctx.step_outputs["retrieve_policy_snippets"][0] in prompts[0]
    assert "propos" in prompts[0].lower()
    assert "execut" in prompts[0].lower()


@pytest.mark.parametrize("body", [
    "I have scheduled your meeting with HR.",
    "I sent the policy document to your email.",
    "I updated your PTO balance to 30 hours.",
    "I've notified your manager team.",
])
def test_unexecuted_action_claim_is_rejected(monkeypatch, body):
    model_response(monkeypatch, body)
    with pytest.raises(ValueError, match="(?i)(action|execut|receipt)"):
        steps.generate_structured_response_step(context())


def test_action_proposal_remains_a_reviewable_draft(monkeypatch):
    model_response(monkeypatch, "Ask your HR partner to schedule a meeting and review the interactive process.")
    output = steps.generate_structured_response_step(context())
    assert output["quality_status"] == "needs_hr_review"
    assert output["execution_receipts"] == []
    assert "schedule a meeting" in output["body"]


def test_unsupported_completion_in_subject_is_rejected(monkeypatch):
    monkeypatch.setattr(steps, "_retryable_llm_call", lambda *a, **kw: json.dumps({
        "subject": "I have scheduled your meeting", "body": "Ask HR to review.",
        "tone": "empathetic", "cited_snippets": [],
    }))
    with pytest.raises(ValueError, match="execution receipt"):
        steps.generate_structured_response_step(context())


def test_model_cannot_self_approve_or_invent_receipts(monkeypatch):
    monkeypatch.setattr(steps, "_retryable_llm_call", lambda *a, **kw: json.dumps({
        "subject": "Review question", "body": "Ask HR to review.",
        "tone": "empathetic", "cited_snippets": [],
        "quality_status": "approved", "execution_receipts": [{"action": "email", "status": "sent"}],
    }))
    output = steps.generate_structured_response_step(context())
    assert output["quality_status"] == "needs_hr_review"
    assert output["execution_receipts"] == []


def test_citation_evidence_resolves_exact_quote_and_source(monkeypatch):
    import hashlib
    ctx = context()
    source = ctx.step_outputs["retrieve_policy_snippets"][0]
    quote = "Use the interactive process"
    monkeypatch.setattr(steps, "_retryable_llm_call", lambda *a, **kw: json.dumps({
        "subject": "Review question", "body": "Ask HR to review the interactive process.",
        "tone": "empathetic", "cited_snippets": [quote],
    }))
    output = steps.generate_structured_response_step(ctx)
    evidence = output["citation_evidence"][0]
    assert evidence["source_id"] == "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    assert source[evidence["start"]:evidence["end"]] == evidence["quote"] == quote


def test_employee_metadata_is_not_labeled_as_verified(monkeypatch):
    class Registry:
        def execute(self, *args, **kwargs):
            return {"employee_id": "E10234", "metadata": {"pto_balance": 12}, "found": True}
    monkeypatch.setattr(steps, "get_tool_registry", lambda: Registry())
    output = steps.retrieve_employee_context_step(context())
    assert output["metadata"]["pto_balance"] == 12
    assert output["fact_verification"] == "unverified"


def test_unknown_employee_stops_before_model_or_case_note(monkeypatch, db_session):
    from uuid import uuid4
    from app.engine.runner import run_workflow
    from app.storage.models import Case, CaseNote, Step
    calls = []
    monkeypatch.setattr(steps, "_retryable_llm_call", lambda *args, **kwargs: calls.append(args))
    inputs = {**context().inputs, "employee_id": "MISSING-" + uuid4().hex}
    key = "missing-" + uuid4().hex
    before = (db_session.query(Step).count(), db_session.query(CaseNote).count())
    with pytest.raises(ValueError, match="Employee not found"):
        run_workflow(inputs, key)
    assert calls == []
    assert db_session.query(Case).filter_by(idempotency_key=key).count() == 0
    assert (db_session.query(Step).count(), db_session.query(CaseNote).count()) == before
