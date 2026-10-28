"""Policy risk acceptance checks; no live service or application DB required."""
import json
from pathlib import Path

import pytest

from app.engine.context import ExecutionContext
from app.engine.steps import generate_structured_response_step
from app.tools.retrieval import retrieve_policy_snippets

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


def test_pto_query_does_not_include_unrelated_policy_noise():
    snippets = retrieve_policy_snippets(
        "Asked how PTO carryover works going into next year. Has about 30 unused hours."
    )
    assert snippets
    assert all("# Paid Time Off (PTO) Policy" in snippet for snippet in snippets)


def test_unknown_policy_abstains_instead_of_matching_common_words():
    assert retrieve_policy_snippets("What is the policy for bringing my pet iguana to the office?") == []


def test_leave_retrieval_keeps_complete_guardrail():
    source = (DOCS / "leave-of-absence-policy.md").read_text()
    snippets = retrieve_policy_snippets("medical leave eligibility and hours worked")
    assert source in snippets
    assert "Always route eligibility determinations through an HR partner" in "\n".join(snippets)


def _draft(monkeypatch, snippet, cited):
    response = {"subject": "PTO question", "body": "Ask HR to confirm the applicable PTO policy.",
                "tone": "empathetic", "cited_snippets": cited}
    prompts = []
    def complete(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(response)
    monkeypatch.setattr("app.engine.steps._retryable_llm_call", complete)
    ctx = ExecutionContext(case_id=1, inputs={"employee_id": "E10234", "tone": "empathetic"}, artifacts={},
        step_outputs={"retrieve_employee_context": {}, "retrieve_policy_snippets": [snippet],
                      "summarize_and_extract_actions": {"summary": "PTO question", "action_items": [], "next_steps": []}})
    return generate_structured_response_step(ctx), prompts


def test_drafting_keeps_complete_policy_guardrail(monkeypatch):
    source = (DOCS / "leave-of-absence-policy.md").read_text()
    _, prompts = _draft(monkeypatch, source, [])
    assert source in prompts[0]


def test_fabricated_citation_fails_validation(monkeypatch):
    source = (DOCS / "pto-policy.md").read_text()
    with pytest.raises(ValueError, match="[Cc]itation|[Ss]ource"):
        _draft(monkeypatch, source, ["POLICY-DOES-NOT-EXIST"])


def test_real_policy_quote_remains_accepted(monkeypatch):
    source = (DOCS / "pto-policy.md").read_text()
    quote = "Carryover balances above 40 hours are forfeited unless local law requires payout"
    response, _ = _draft(monkeypatch, source, [quote])
    assert response["cited_snippets"] == [quote]


def test_missing_sick_day_policy_does_not_substitute_other_leave_policy():
    assert retrieve_policy_snippets("What sick leave can I use for a day off with the flu?") == []


def test_parental_question_does_not_substitute_generic_leave_policy():
    snippets = retrieve_policy_snippets("What parental leave eligibility and duration apply after the birth of my child?")
    assert snippets
    assert all("# Parental Leave Policy" in snippet for snippet in snippets)
