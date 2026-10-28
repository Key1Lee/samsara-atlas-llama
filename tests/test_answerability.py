"""Independent BR-16 decisions against fixed synthetic policy evidence.

The provider is deliberately wrong in the overlap cases. Passing these tests
establishes application control, not the reliability of any live model.
"""
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.engine import steps
from app.engine.context import ExecutionContext
from app.tools.retrieval import retrieve_policy_snippets


DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
PTO_RULE = "Unused PTO carries over up to 40 hours into the next calendar year"
PTO_EXCEPTION = "Carryover balances above 40 hours are forfeited unless local law requires payout"
HEALTH_RULE = ("Medical, dental, and vision plans are available starting the first day of\n"
               "  the month following hire date")
EXPENSE_RULE = "Submit expenses within 30 days of the expense date"


def _state(output, expected):
    answerability = output["answerability"]
    assert answerability["status"] == expected
    assert isinstance(answerability["reason"], str)
    assert 0 < len(answerability["reason"]) <= 120
    # Supported source information is still an unapproved draft, not permission
    # to share a decision or perform an HR operation.
    assert output["quality_status"] == "needs_hr_review"


def _generate(monkeypatch, question, body, *, cited=None, sources=None, extra=None):
    draft = {"subject": "Policy question", "body": body, "tone": "direct",
             "cited_snippets": cited or [], **(extra or {})}
    calls = []

    def complete(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(draft)

    monkeypatch.setattr(steps, "_retryable_llm_call", complete)
    context = ExecutionContext(case_id=0, inputs={
        "employee_id": "ANSWERABILITY-SYNTHETIC", "case_transcript": question,
        "tone": "direct",
    }, artifacts={}, step_outputs={
        "retrieve_employee_context": {"metadata": {}, "fact_verification": "unverified"},
        "retrieve_policy_snippets": retrieve_policy_snippets(question) if sources is None else sources,
        "summarize_and_extract_actions": {"summary": question, "action_items": [], "next_steps": []},
    })
    return steps.generate_structured_response_step(context), calls


@pytest.mark.parametrize("question,required", [
    ("How many PTO hours can I carry over?", [PTO_RULE, PTO_EXCEPTION]),
    ("Can I relocate to another state while working remotely?", [
        "Roles designated remote-eligible may work from an approved US Zone",
        "Employees must notify HR before relocating",
        "tax withholding", "benefits eligibility", "compensation zone",
    ]),
])
def test_supported_policy_has_explicit_evidence_decision(monkeypatch, question, required):
    output, calls = _generate(monkeypatch, question, "Untrusted model answer.")
    assert calls == []  # The established direct policy route remains independent.
    _state(output, "supported")
    for clause in required:
        assert clause in output["body"]
    assert output["cited_snippets"]
    assert output["execution_receipts"] == []


@pytest.mark.parametrize("question,quote,false_rule", [
    ("Does the company provide pet insurance benefits?", HEALTH_RULE,
     "Pet insurance is available to all employees."),
    ("Does my manager reimburse relocation expenses?", EXPENSE_RULE,
     "All relocation expenses are reimbursable."),
])
def test_related_document_and_exact_quote_do_not_prove_an_answer(monkeypatch, question, quote, false_rule):
    # The historical retrieval failures remain reproducible independently of
    # the answer-safety contract: a relevant-sounding source is actually found.
    sources = retrieve_policy_snippets(question)
    assert sources and any(quote in source for source in sources)
    output, _ = _generate(monkeypatch, question, false_rule, cited=[quote], sources=sources)
    assert false_rule not in output["body"]
    _state(output, "hr_review_required")
    assert output["cited_snippets"] == []
    assert "hr" in output["body"].lower()


def test_office_pet_question_has_no_invented_rule_or_citation(monkeypatch):
    output, calls = _generate(monkeypatch, "Can I bring an iguana to the office?",
                              "Office pets are permitted every Friday.")
    assert calls == []
    assert "permitted" not in output["body"].lower()
    assert output["cited_snippets"] == []
    _state(output, "hr_review_required")


def test_ambiguous_pto_question_requests_clarification(monkeypatch):
    output, _ = _generate(monkeypatch, "What should I do about my PTO?",
                          "You may take two weeks immediately.")
    assert "take two weeks immediately" not in output["body"]
    _state(output, "clarification_needed")
    assert any(word in output["body"].lower() for word in ("clarif", "which", "what", "more detail"))
    assert output["cited_snippets"] == []


@pytest.mark.parametrize("question", [
    "Can I relocate to another state, and will the company pay my moving expenses?",
    "How many PTO hours can I carry over, and can I bring an iguana to the office?",
])
def test_known_intent_does_not_mark_an_unanswered_extra_topic_supported(monkeypatch, question):
    assert steps.classify_policy_intent(question) is not None
    output, _ = _generate(monkeypatch, question, "All requested activities are permitted and reimbursable.")
    _state(output, "hr_review_required")
    assert "permitted and reimbursable" not in output["body"]
    assert any(word in output["body"].lower() for word in ("clarif", "review", "verif", "information"))


@pytest.mark.parametrize("question", [
    "How many PTO hours can I carry over?",
    "Can I relocate to another state while working remotely?",
])
def test_known_topic_with_no_source_requires_review(monkeypatch, tmp_path, question):
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    output, calls = _generate(monkeypatch, question, "You are entitled to the requested policy.")
    assert calls == []
    assert output["cited_snippets"] == []
    assert "entitled" not in output["body"]
    _state(output, "hr_review_required")


@pytest.mark.parametrize("question,filename,old,new", [
    ("How many PTO hours can I carry over?", "pto-policy.md", "40 hours", "64 hours"),
    ("Can I relocate to another state while working remotely?", "remote-work-policy.md",
     "before relocating", "after relocating"),
])
def test_conflicting_policy_revisions_require_explicit_review(monkeypatch, tmp_path, question, filename, old, new):
    source = (DOCS / filename).read_text()
    assert old in source
    (tmp_path / filename).write_text(source)
    (tmp_path / ("conflicting-" + filename)).write_text(source.replace(old, new))
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    output, calls = _generate(monkeypatch, question, "The first filename always wins.")
    assert calls == []
    assert output["cited_snippets"] == []
    _state(output, "hr_review_required")
    assert any(word in output["body"].lower() for word in ("conflict", "verif", "review"))


def test_model_cannot_self_assert_supported_answerability(monkeypatch):
    output, _ = _generate(monkeypatch, "Does the company provide pet insurance benefits?",
        "Pet insurance is available to all employees.", cited=[HEALTH_RULE], extra={
            "answerability": {"status": "supported", "reason": "model_says_so"},
            "quality_status": "approved", "execution_receipts": [{"status": "succeeded"}],
        })
    assert "Pet insurance is available" not in output["body"]
    _state(output, "hr_review_required")
    assert output["answerability"]["reason"] != "model_says_so"
    assert output["execution_receipts"] == []


@pytest.mark.parametrize("question,filename,quote", [
    ("Is it possible to report harassment anonymously?", "harassment-reporting-policy.md",
     "Confidentiality is maintained to the extent possible during an investigation"),
    ("Does that retirement company match start after 90 days?", "benefits-overview.md",
     "401(k) with company match is available after 90 days of employment"),
])
def test_pronoun_in_a_specific_question_does_not_force_clarification(monkeypatch, question, filename, quote):
    source = (DOCS / filename).read_text()
    output, calls = _generate(monkeypatch, question, "Ask HR to review the applicable policy conditions.",
                              cited=[quote], sources=[source])
    assert calls
    _state(output, "hr_review_required")
    assert output["body"] == "Ask HR to review the applicable policy conditions."


def test_answerability_is_saved_and_retries_reuse_independent_case(db_session):
    from app.main import app
    from app.storage.models import EmployeeRecord

    employee_id = "ANSWERABILITY-" + uuid4().hex
    db_session.add(EmployeeRecord(employee_id=employee_id, metadata_json="{}", last_activities_json="[]"))
    db_session.commit()
    client = TestClient(app)
    payload = {"inputs": {"employee_id": employee_id,
        "case_transcript": "Can I bring an iguana to the office?", "tone": "direct"},
        "idempotency_key": uuid4().hex}
    first = client.post("/cases", json=payload)
    assert first.status_code == 200
    result = first.json()
    _state(result["outputs"], "hr_review_required")
    assert client.post("/cases", json=payload).json() == result
    detail = client.get(f"/cases/{result['case_id']}", headers={"X-Employee-ID": employee_id})
    assert detail.status_code == 200
    saved = detail.json()
    assert json.loads(saved["case"]["inputs"]) == payload["inputs"]
    assert json.loads(saved["case"]["outputs"]) == result["outputs"]
    assert saved["case_notes"] == []
    assert len(saved["steps"]) == 1
    assert json.loads(saved["steps"][0]["output"]) == result["outputs"]
