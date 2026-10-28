"""Controlled provider boundary probes; no live-model inference."""
import json
from uuid import uuid4
import pytest
import requests
from app.providers.factory import get_provider
from app.providers.ollama import OllamaProvider
from app.providers.stub import StubProvider


@pytest.mark.parametrize("name", ["typo", ""])
def test_invalid_configuration_never_silently_simulates(monkeypatch, name):
    monkeypatch.setenv("PROVIDER_TYPE", name)
    with pytest.raises(Exception, match="Provider configuration"):
        get_provider()


def test_explicit_stub_remains_identifiable(monkeypatch):
    monkeypatch.setenv("PROVIDER_TYPE", "stub")
    provider = get_provider()
    provider.timeout_chance = 0
    assert isinstance(provider, StubProvider)
    assert provider.complete("prompt").raw["model"] == "stub"


def test_missing_optional_local_adapter_never_falls_back(monkeypatch):
    import builtins
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == "app.providers.local_stub":
            raise ImportError("synthetic missing optional adapter")
        return original(name, *args, **kwargs)
    monkeypatch.setenv("PROVIDER_TYPE", "local")
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(Exception, match="Provider configuration"):
        get_provider()


@pytest.mark.parametrize("status,payload,expected", [
    (429, {}, "provider_rate_limited"), (403, {}, "provider_rejected"),
    (503, {}, "provider_unavailable"), (200, {}, "provider_invalid_output"),
    (200, {"response": ""}, "provider_invalid_output"),
    (200, {"response": "   "}, "provider_invalid_output"),
    (200, {"response": 12}, "provider_invalid_output"),
    (200, [], "provider_invalid_output"),
])
def test_provider_outcomes_are_classified_without_payloads(monkeypatch, status, payload, expected):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    with pytest.raises(Exception) as caught:
        OllamaProvider().complete("PRIVATE PROMPT")
    assert getattr(caught.value, "code", None) == expected
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize("error,expected", [
    (requests.Timeout("PRIVATE URL"), "provider_timeout"),
    (requests.ConnectionError("PRIVATE URL"), "provider_unavailable"),
])
def test_transport_failure_is_safe(monkeypatch, error, expected):
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(Exception) as caught:
        OllamaProvider().complete("PRIVATE PROMPT")
    assert caught.value.code == expected
    assert "PRIVATE" not in str(caught.value)


def test_valid_adapter_response_is_used(monkeypatch):
    from pathlib import Path
    from app.engine import steps
    from app.engine.context import ExecutionContext
    response = requests.Response()
    response.status_code = 200
    body = "Ask HR to review the interactive process."
    response._content = json.dumps({"response": json.dumps({
        "subject": "Draft", "body": body, "tone": "direct", "cited_snippets": []})}).encode()
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    monkeypatch.setattr(steps, "get_provider", lambda: OllamaProvider())
    ctx = ExecutionContext(case_id=0, artifacts={}, inputs={"employee_id": "PROVIDER-A",
        "case_transcript": "Explain accommodation review.", "tone": "direct"}, step_outputs={
        "retrieve_policy_snippets": [(Path(__file__).resolve().parents[1] /
            "data/docs/workplace-accommodation-policy.md").read_text()]})
    output = steps.generate_structured_response_step(ctx)
    assert output["body"] == body
    assert output["quality_status"] == "needs_hr_review"


def test_configuration_failure_persists_failed_case_without_note(db_session, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.engine import steps
    from app.storage.models import EmployeeRecord, Case, Step, CaseNote
    employee = "PROVIDER-" + uuid4().hex
    db_session.add(EmployeeRecord(employee_id=employee, metadata_json="{}", last_activities_json="[]"))
    db_session.commit()
    monkeypatch.setattr(steps, "get_provider", get_provider)
    monkeypatch.setenv("PROVIDER_TYPE", "invalid")
    response = TestClient(app).post("/cases", json={"inputs": {
        "employee_id": employee, "case_transcript": "Explain accommodation review.", "tone": "direct"},
        "idempotency_key": uuid4().hex})
    assert response.status_code == 500
    case = db_session.query(Case).filter(Case.inputs_json.contains(employee)).one()
    assert case.status == "failed"
    assert json.loads(case.outputs_json or "{}") == {}
    assert db_session.query(CaseNote).filter_by(case_id=case.id).count() == 0
    assert db_session.query(Step).filter_by(case_id=case.id, status="failed").one()


@pytest.mark.parametrize("status,attempts", [(429, 1), (403, 1), (503, 3)])
def test_retries_are_bounded_and_rejections_are_not_retried(monkeypatch, status, attempts):
    from app.engine import steps
    from tenacity import wait_none
    from app.audit import error_code
    response = requests.Response()
    response.status_code = status
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **kw: calls.append(1) or response)
    monkeypatch.setattr(steps, "get_provider", lambda: OllamaProvider())
    with pytest.raises(Exception) as caught:
        steps._retryable_llm_call.retry_with(wait=wait_none())("prompt")
    assert len(calls) == attempts
    assert error_code(caught.value) == {429: "provider_rate_limited",
        403: "provider_rejected", 503: "provider_unavailable"}[status]


def test_malformed_transport_json_is_safe(monkeypatch):
    response = requests.Response()
    response.status_code = 200
    response._content = b"PRIVATE NOT JSON"
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    with pytest.raises(Exception) as caught:
        OllamaProvider().complete("prompt")
    assert caught.value.code == "provider_invalid_output"
    assert "PRIVATE" not in str(caught.value)
