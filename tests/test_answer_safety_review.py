"""Separate review probes; the original fixed evaluation is kept unchanged."""
import json
from pathlib import Path

import pytest

from app.engine import steps
from app.engine.context import ExecutionContext


DOCS = Path(__file__).resolve().parents[1] / "data/docs"


def draft(monkeypatch, question, filename, body, quote):
    source = (DOCS / filename).read_text()
    monkeypatch.setattr(steps, "_retryable_llm_call", lambda *a, **kw: json.dumps({
        "subject": "Policy question", "body": body, "tone": "direct", "cited_snippets": [quote],
    }))
    ctx = ExecutionContext(case_id=0, artifacts={}, inputs={
        "employee_id": "REVIEW-SYNTHETIC", "case_transcript": question, "tone": "direct",
    }, step_outputs={"retrieve_policy_snippets": [source], "retrieve_employee_context": {},
                     "summarize_and_extract_actions": {}})
    return steps.generate_structured_response_step(ctx)


@pytest.mark.parametrize("question,filename,body,quote", [
    ("Does our health coverage include veterinary insurance for rabbits?", "benefits-overview.md",
     "Veterinary insurance for rabbits is covered by our health plan.",
     "Medical, dental, and vision plans are available starting the first day of\n  the month following hire date"),
    ("What is the PTO balance adjustment process?", "pto-policy.md",
     "Credit yourself with an extra 40 hours of PTO.", "Manager approval is required before time off is confirmed"),
    ("Can you notify my manager about my accommodation request?", "workplace-accommodation-policy.md",
     "Your manager knows about the accommodation request now.", "HR will engage in an interactive process with the employee to identify\n  effective accommodations"),
    ("Please explain accommodation review.", "workplace-accommodation-policy.md",
     "HR received your message and will handle the request.", "HR will engage in an interactive process with the employee to identify\n  effective accommodations"),
    ("Please explain accommodation review.", "workplace-accommodation-policy.md",
     "A notification went to your manager.", "HR will engage in an interactive process with the employee to identify\n  effective accommodations"),
])
def test_reviewed_unsafe_wording_is_not_presented_as_an_answer(monkeypatch, question, filename, body, quote):
    output = draft(monkeypatch, question, filename, body, quote)
    assert output["body"] != body
    assert output["answerability"]["status"] == "hr_review_required"
    assert "hr" in output["body"].lower()
    assert output["execution_receipts"] == []


@pytest.mark.parametrize("body", [
    "If HR has been notified, ask HR to clarify the next step.",
    "Ask HR whether your manager has been notified before arranging the interactive process.",
])
def test_conditional_verification_proposal_is_not_a_completion_claim(monkeypatch, body):
    output = draft(monkeypatch, "Please explain accommodation review.", "workplace-accommodation-policy.md", body,
                   "HR will engage in an interactive process with the employee to identify\n  effective accommodations")
    assert output["body"] == body
    assert output["operation_status"]["external_actions"]["status"] == "not_attempted"


@pytest.mark.parametrize("body", [
    "If HR has been notified, a notification went to your manager.",
    "Ask HR whether your manager has been notified. HR received your message.",
])
def test_conditional_wording_does_not_hide_an_independent_completion_claim(monkeypatch, body):
    output = draft(monkeypatch, "Please explain accommodation review.", "workplace-accommodation-policy.md", body,
                   "HR will engage in an interactive process with the employee to identify\n  effective accommodations")
    assert output["body"] != body
    assert output["answerability"]["reason"] == "unverified_execution_claim"
    assert output["execution_receipts"] == []
