"""Core contract regressions; providers and failures are deterministic."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

INPUTS = {"employee_id": "E10234", "case_transcript": "How does PTO accrue?", "tone": "empathetic"}


@pytest.fixture
def client(db_session):
    from app.storage.models import EmployeeRecord
    from app.main import app
    if not db_session.query(EmployeeRecord).filter_by(employee_id="E10234").first():
        db_session.add(EmployeeRecord(employee_id="E10234", metadata_json='{"department":"People"}', last_activities_json="[]"))
        db_session.commit()
    with TestClient(app, headers={"X-Employee-ID": "E10234"}) as client:
        yield client


def create(client, key=None, inputs=None):
    return client.post("/cases", json={"inputs": inputs or INPUTS, "idempotency_key": key})


@pytest.mark.parametrize("field,value", [("employee_id", ""), ("employee_id", "  "), ("case_transcript", "\n "), ("tone", "friendly")])
def test_invalid_inputs_rejected(client, field, value):
    response = create(client, inputs={**INPUTS, field: value})
    assert response.status_code == 422


def test_empty_key_rejected(client):
    assert create(client, key=" ").status_code == 422


def test_unkeyed_submissions_are_independent(client):
    first, second = create(client).json(), create(client).json()
    assert first["case_id"] != second["case_id"]


def test_same_key_returns_same_result_and_rejects_different_inputs(client):
    first = create(client, "core-key").json()
    assert create(client, "core-key").json() == first
    assert create(client, "core-key", {**INPUTS, "tone": "direct"}).status_code == 409
    stored = client.get(f"/cases/{first['case_id']}").json()
    assert len(stored["steps"]) == 5
    assert len(stored["case_notes"]) == 1


def test_persisted_results(client):
    result = create(client, "persisted").json()
    stored = client.get(f"/cases/{result['case_id']}").json()
    assert stored["case"]["status"] == "completed"
    assert json.loads(stored["case"]["inputs"]) == INPUTS
    assert json.loads(stored["case"]["outputs"]) == result["outputs"]
    assert len(stored["steps"]) == 5
    assert all(s["status"] == "completed" and s["finished_at"] for s in stored["steps"])
    assert json.loads(stored["steps"][0]["output"])["found"] is True
    snippets = json.loads(stored["steps"][1]["output"])
    assert snippets and any("PTO" in s for s in snippets)
    assert len(stored["case_notes"]) == 1
    assert stored["case_notes"][0]["employee_id"] == INPUTS["employee_id"]
    assert result["outputs"]["subject"] in stored["case_notes"][0]["note"]


@pytest.mark.parametrize("output", ['{"summary":"ok","action_items":"not a list","next_steps":[]}', '{}', '[]', 'not JSON'])
def test_invalid_summary_fails_without_note(client, monkeypatch, output):
    monkeypatch.setattr("app.engine.steps._retryable_llm_call", lambda *a, **kw: output)
    response = create(client, "bad-summary-" + output)
    assert response.status_code == 500
    case_id = client.get("/cases").json()["case_ids"][0]
    stored = client.get(f"/cases/{case_id}").json()
    assert stored["case"]["status"] == "failed"
    assert stored["steps"][-1]["status"] == "failed"
    assert stored["steps"][-1]["finished_at"]
    assert stored["case_notes"] == []
    assert "not a list" not in response.text


@pytest.mark.parametrize("output", [
    {"subject": "", "body": "", "tone": "empathetic", "cited_snippets": []},
    {"subject": "Valid", "body": "Valid", "tone": "direct", "cited_snippets": []},
    {"subject": "Valid", "body": "Valid", "tone": "empathetic", "cited_snippets": "invalid"},
    {},
])
def test_invalid_response_fails_without_note(client, monkeypatch, output):
    from app.providers.base import CompletionResponse
    from app.providers.stub import StubProvider
    class InvalidResponse(StubProvider):
        def complete(self, prompt, **kwargs):
            if kwargs:
                return CompletionResponse(json.dumps(output))
            return super().complete(prompt, **kwargs)
    monkeypatch.setattr("app.engine.steps.get_provider", lambda: InvalidResponse(timeout_chance=0))
    assert create(client, "bad-response-" + json.dumps(output)).status_code == 500
    case_id = client.get("/cases").json()["case_ids"][0]
    stored = client.get(f"/cases/{case_id}").json()
    assert stored["case"]["status"] == "failed"
    assert stored["case_notes"] == []


def test_failure_after_note_commit_can_retry_without_duplicates(client, monkeypatch):
    from app.tools.registry import get_tool_registry
    registry = get_tool_registry()
    original = registry.get("post_case_note_to_hris")
    calls = 0
    def commit_then_fail(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == 1:
            raise RuntimeError("Simulated lost acknowledgement")
        return result
    monkeypatch.setitem(registry._tools, "post_case_note_to_hris", commit_then_fail)
    assert create(client, "note-retry").status_code == 500
    result = create(client, "note-retry").json()
    assert result["status"] == "completed"
    stored = client.get(f"/cases/{result['case_id']}").json()
    assert len(stored["case_notes"]) == 1
    assert len(stored["steps"]) == 5
    assert stored["steps"][-1]["attempts"] == 2
    assert all(s["status"] == "completed" for s in stored["steps"])


def test_transient_llm_failure_retries_only_llm(client, monkeypatch):
    from app.providers.stub import StubProvider
    calls = 0
    class TransientProvider(StubProvider):
        def complete(self, prompt, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary")
            return super().complete(prompt, **kwargs)
    monkeypatch.setattr("app.engine.steps.get_provider", lambda: TransientProvider(timeout_chance=0))
    response = create(client, "transient")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert calls == 3
    stored = client.get(f"/cases/{response.json()['case_id']}").json()
    assert len(stored["steps"]) == 5 and len(stored["case_notes"]) == 1


def test_concurrent_same_key_runs_once(client, monkeypatch):
    from app.engine import runner
    started, release = Event(), Event()
    original = runner.STEPS[0][1]
    def blocked(ctx):
        started.set()
        assert release.wait(10)
        return original(ctx)
    monkeypatch.setattr(runner, "STEPS", [(runner.STEPS[0][0], blocked), *runner.STEPS[1:]])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(create, client, "concurrent")
        try:
            assert started.wait(10)
            second = create(client, "concurrent")
            assert second.status_code == 200
            assert second.json()["status"] == "running"
        finally:
            release.set()
        completed = first.result().json()
    assert second.json()["case_id"] == completed["case_id"]
    stored = client.get(f"/cases/{completed['case_id']}").json()
    assert len(stored["steps"]) == 5 and len(stored["case_notes"]) == 1


def test_exhausted_llm_retries_fail_without_note(client, monkeypatch):
    calls = 0
    class Unavailable:
        def complete(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("provider unavailable")
    monkeypatch.setattr("app.engine.steps.get_provider", lambda: Unavailable())
    assert create(client, "exhausted").status_code == 500
    assert calls == 3
    case_id = client.get("/cases").json()["case_ids"][0]
    stored = client.get(f"/cases/{case_id}").json()
    assert stored["case"]["status"] == "failed"
    assert stored["steps"][-1]["finished_at"]
    assert stored["case_notes"] == []


def test_replay_creates_separate_case(client):
    original = create(client, "replay-original").json()
    replay = client.post(f"/cases/{original['case_id']}/replay").json()
    assert replay["case_id"] != original["case_id"]
    assert replay["status"] == "completed"
    for result in [original, replay]:
        assert len(client.get(f"/cases/{result['case_id']}").json()["case_notes"]) == 1


def test_database_rejects_duplicate_key_and_note(client, db_session):
    from app.storage.models import Case, CaseNote
    from sqlalchemy.exc import IntegrityError
    result = create(client, "db-unique").json()
    db_session.add(Case(idempotency_key="db-unique"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
    db_session.add(CaseNote(case_id=result["case_id"], employee_id="E10234", note="duplicate"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


@pytest.mark.parametrize("error_name", ["Timeout", "ConnectionError"])
def test_ollama_transport_retries(monkeypatch, error_name):
    import requests
    from app.providers.ollama import OllamaProvider
    from app.engine.steps import _retryable_llm_call
    calls = 0
    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise getattr(requests.exceptions, error_name)("temporary")
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"response":"valid text"}'
        return response
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("app.engine.steps.get_provider", lambda: OllamaProvider())
    assert _retryable_llm_call("prompt") == "valid text"
    assert calls == 2


def test_provider_normalized_text_is_used(monkeypatch):
    from app.providers.base import CompletionResponse
    from app.engine.steps import _retryable_llm_call
    class Provider:
        def complete(self, *args, **kwargs):
            return CompletionResponse("normalized", raw={"choices": [{"message": {"content": "raw"}}]})
    monkeypatch.setattr("app.engine.steps.get_provider", lambda: Provider())
    assert _retryable_llm_call("prompt") == "normalized"


def test_simultaneous_key_creation_is_unique(client):
    from threading import Barrier
    from sqlalchemy import event
    from app.storage.database import engine
    barrier = Barrier(2)
    def synchronize_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO cases ") and "creation-race" in parameters:
            barrier.wait(timeout=10)
    event.listen(engine, "before_cursor_execute", synchronize_insert)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: create(client, "creation-race"), range(2)))
    finally:
        event.remove(engine, "before_cursor_execute", synchronize_insert)
    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json()["case_id"] == responses[1].json()["case_id"]
    stored = client.get(f"/cases/{responses[0].json()['case_id']}").json()
    assert stored["case"]["status"] == "completed"
    assert len(stored["steps"]) == 5 and len(stored["case_notes"]) == 1
