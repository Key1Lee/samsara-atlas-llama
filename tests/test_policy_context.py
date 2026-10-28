"""Complete policy conditions at the actual provider boundary, without a model."""
import json
from pathlib import Path

import pytest

from app.engine import steps
from app.engine.context import ExecutionContext
from app.providers.base import CompletionResponse
from app.tools.retrieval import retrieve_policy_snippets


DOCS = Path(__file__).resolve().parents[1] / "data/docs"


class RecordingProvider:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        result = ({"subject": "Leave review", "body": "Ask HR to review all applicable conditions.",
                   "tone": kwargs["tone"], "cited_snippets": []} if "tone" in kwargs else
                  {"summary": "Leave question", "action_items": ["Ask HR to verify eligibility"],
                   "next_steps": ["HR review"]})
        return CompletionResponse(json.dumps(result))


def context(source):
    return ExecutionContext(case_id=0, artifacts={}, inputs={
        "employee_id": "CONTEXT-TEST", "case_transcript": "Explain medical leave conditions.", "tone": "direct",
    }, step_outputs={"retrieve_policy_snippets": [source], "retrieve_employee_context": {},
                     "summarize_and_extract_actions": {}})


@pytest.mark.parametrize("step", [steps.summarize_and_extract_actions_step, steps.generate_structured_response_step])
def test_complete_long_clause_and_end_qualification_reach_provider(tmp_path, monkeypatch, step):
    # Synthetic approved-for-this-test source; this is not production approval metadata.
    source = (DOCS / "leave-of-absence-policy.md").read_text()
    # Keep the source-authored rule, eligibility, local-law exception and final
    # HR qualification separated by more than the historic 500-character cutoff.
    source = source.replace("## Job Protection", "## Additional conditions\n- " + "Review documentation. " * 40 + "\n\n## Job Protection")
    (tmp_path / "leave-of-absence-policy.md").write_text(source)
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    snippets = retrieve_policy_snippets("medical leave eligibility")
    assert snippets == [source]
    provider = RecordingProvider()
    monkeypatch.setattr(steps, "get_provider", lambda: provider)
    step(context(snippets[0]))
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    assert source in prompt
    for condition in ["12 months of employment", "1,250 hours worked", "whichever\n  is more generous",
                      "Always route eligibility determinations through an HR partner", "confirm eligibility informally"]:
        assert condition in prompt


@pytest.mark.parametrize("step", [steps.summarize_and_extract_actions_step, steps.generate_structured_response_step])
def test_oversized_policy_is_explicitly_blocked_without_truncation(monkeypatch, step):
    source = (DOCS / "leave-of-absence-policy.md").read_text()
    source += "\n## Unrelated equipment manual\n" + "Unrelated manual material. " * 1000
    provider = RecordingProvider()
    monkeypatch.setattr(steps, "get_provider", lambda: provider)
    with pytest.raises(ValueError, match="(?i)(context|budget)"):
        step(context(source))
    assert provider.prompts == []


@pytest.mark.parametrize("field", ["retrieve_employee_context", "summarize_and_extract_actions"])
def test_nonpolicy_context_also_counts_toward_complete_prompt_budget(monkeypatch, field):
    ctx = context((DOCS / "leave-of-absence-policy.md").read_text())
    ctx.step_outputs[field] = {"unverified_history": "Synthetic history. " * 1000}
    provider = RecordingProvider()
    monkeypatch.setattr(steps, "get_provider", lambda: provider)
    with pytest.raises(ValueError, match="(?i)(context|budget)"):
        steps.generate_structured_response_step(ctx)
    assert provider.prompts == []


@pytest.mark.parametrize("step", [steps.summarize_and_extract_actions_step, steps.generate_structured_response_step])
def test_large_optional_history_is_explicitly_omitted_before_policy_evidence(monkeypatch, step):
    source = (DOCS / "leave-of-absence-policy.md").read_text()
    ctx = context(source)
    employee = {"metadata": {"department": "People"}, "fact_verification": "unverified",
                "case_history": [{"question": "OLD-QUESTION"}] * 1000,
                "last_activities": [{"summary": "OLD-DRAFT"}] * 1000}
    ctx.step_outputs["retrieve_employee_context"] = employee
    original = json.dumps(employee)
    provider = RecordingProvider()
    monkeypatch.setattr(steps, "get_provider", lambda: provider)
    step(ctx)
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    assert source in prompt and "People" in prompt and "unverified" in prompt
    assert "context_omissions" in prompt and "context_budget" in prompt
    assert "case_history" in prompt and "last_activities" in prompt
    assert "OLD-QUESTION" not in prompt and "OLD-DRAFT" not in prompt
    assert len(prompt.encode()) <= 6000
    assert json.dumps(employee) == original  # Persisted history is not changed.


def test_budget_counts_utf8_bytes_and_allows_exact_boundary(monkeypatch):
    provider = RecordingProvider()
    monkeypatch.setattr(steps, "get_provider", lambda: provider)
    steps._retryable_llm_call("a" * 6000)
    assert provider.prompts == ["a" * 6000]
    with pytest.raises(ValueError, match="(?i)(context|budget)"):
        steps._retryable_llm_call("a" * 6001)
    with pytest.raises(ValueError, match="(?i)(context|budget)"):
        steps._retryable_llm_call("界" * 2001)
    assert len(provider.prompts) == 1


def test_ollama_has_explicit_context_and_output_budgets(monkeypatch):
    from app.providers.ollama import OllamaProvider
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "{}"}

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("app.providers.ollama.requests.post", post)
    provider = OllamaProvider()
    provider.complete("Complete policy clause")
    assert calls[0]["prompt"] == "Complete policy clause"
    assert calls[0]["options"] == {"num_ctx": 8192, "num_predict": 1024}
    with pytest.raises(ValueError, match="(?i)(context|budget)"):
        provider.complete("x" * 6001)
    assert len(calls) == 1


def test_oversized_policy_failure_is_inspectable_and_retry_keeps_one_case(db_session, tmp_path, monkeypatch):
    from uuid import uuid4
    from fastapi.testclient import TestClient
    from app.main import app
    from app.storage.models import Case, CaseNote, EmployeeRecord, Step

    employee_id = "POLICY-CONTEXT-" + uuid4().hex
    db_session.add(EmployeeRecord(employee_id=employee_id, metadata_json="{}", last_activities_json="[]"))
    db_session.commit()
    source = (DOCS / "leave-of-absence-policy.md").read_text() + "\n## Unrelated manual\n" + "Documentation. " * 1000
    (tmp_path / "leave-of-absence-policy.md").write_text(source)
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    provider = RecordingProvider()
    monkeypatch.setattr(steps, "get_provider", lambda: provider)
    payload = {"inputs": {"employee_id": employee_id, "case_transcript": "Explain medical leave conditions.",
                          "tone": "direct"}, "idempotency_key": uuid4().hex}
    client = TestClient(app)
    for expected_attempts in (1, 2):
        assert client.post("/cases", json=payload).status_code == 500
        db_session.expire_all()
        case = db_session.query(Case).filter_by(idempotency_key=payload["idempotency_key"]).one()
        assert case.status == "failed"
        failed = db_session.query(Step).filter_by(case_id=case.id, name="summarize_and_extract_actions").one()
        assert failed.error == "ContextBudgetExceeded" and failed.attempts == expected_attempts
        retrieved = db_session.query(Step).filter_by(case_id=case.id, name="retrieve_policy_snippets").one()
        assert json.loads(retrieved.output_json) == [source]
        assert db_session.query(CaseNote).filter_by(case_id=case.id).count() == 0
    assert provider.prompts == []
