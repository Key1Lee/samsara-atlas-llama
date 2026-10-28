"""R12/R13/R14 acceptance checks; blocked contracts are explicit expected failures."""
import json
from pathlib import Path
from uuid import uuid4

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

UI = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")


@pytest.fixture
def seeded_employee(db_session):
    from data.seed import load_fixture, seed_employee_records
    seed_employee_records(db_session, load_fixture())


@pytest.mark.parametrize("mode", ["http_500", "timeout", "empty", "malformed", "invalid_ids"])
def test_employee_lookup_failure_disables_submission(monkeypatch, mode):
    posts = []

    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        if url.endswith("/employees"):
            if mode == "timeout":
                raise requests.Timeout("Synthetic employee lookup timeout")
            response.status_code = 500 if mode == "http_500" else 200
            data = {"employees": []}
            if mode == "malformed":
                data = {"employees": "E10234"}
            elif mode == "invalid_ids":
                data = {"employees": [""]}
        elif url.endswith("/info"):
            data = {"provider": "stub"}
        else:
            data = {"case_ids": []}
        response._content = json.dumps(data).encode()
        return response

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: posts.append(kwargs))
    app = AppTest.from_file(UI).run()
    assert not app.exception
    employee = next(widget for widget in app.selectbox if widget.label == "Employee ID")
    submit = next(widget for widget in app.button if widget.label == "Continue")
    assert employee.options == []
    assert employee.disabled
    assert submit.disabled
    assert app.error or app.warning
    with pytest.raises(AppTestError, match="disabled"):
        submit.click()
    assert posts == []


def test_employee_dropdown_uses_successful_api_values(monkeypatch):
    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        data = {"employees": ["EMP-DATABASE-ONLY"]} if url.endswith("/employees") else {"case_ids": [], "provider": "stub"}
        response._content = json.dumps(data).encode()
        return response

    monkeypatch.setattr(requests, "get", get)
    app = AppTest.from_file(UI).run()
    assert not app.exception
    employee = next(widget for widget in app.selectbox if widget.label == "Employee ID")
    submit = next(widget for widget in app.button if widget.label == "Continue")
    assert employee.options == ["EMP-DATABASE-ONLY"]
    assert not employee.disabled and submit.disabled
    employee.select("EMP-DATABASE-ONLY").run()
    assert next(b for b in app.button if b.label == "Continue").disabled
    next(w for w in app.text_input if w.label == "Password").set_value("samsara").run()
    assert not next(b for b in app.button if b.label == "Continue").disabled


def _running_case(db):
    from app.storage.models import Case
    inputs = {"employee_id": "E10234", "case_transcript": "Synthetic crash recovery question", "tone": "direct"}
    key = f"interrupted-{uuid4().hex}"
    case = Case(status="running", idempotency_key=key, inputs_json=json.dumps(inputs), artifacts_json="{}")
    db.add(case)
    db.commit()
    return case, inputs, key


def test_running_retry_does_not_start_competing_execution(db_session, seeded_employee, monkeypatch):
    from app.engine import runner
    case, inputs, key = _running_case(db_session)

    def competing_execution(*args):
        pytest.fail("A running case must not be blindly restarted")

    monkeypatch.setattr(runner, "_execute", competing_execution)
    result = runner.run_workflow(inputs, key)
    assert result == {"case_id": case.id, "status": "running", "outputs": {}, "review_status": None,
                      "recovery": "manual_reconciliation_required", "recovery_reason": "ownership_unverified"}



def test_workflow_failure_keeps_sensitive_text_out_of_logs_and_error(db_session, seeded_employee, monkeypatch, caplog):
    from app.main import app
    from app.engine import runner
    secret = "SYNTHETIC-PRIVATE-MARKER-" + uuid4().hex

    def fail(ctx):
        raise ValueError(secret)

    monkeypatch.setattr(runner, "STEPS", [("synthetic_failure", fail)])
    response = TestClient(app).post("/cases", json={"inputs": {"employee_id": "E10234", "case_transcript": secret, "tone": "direct"}, "idempotency_key": uuid4().hex})
    assert response.status_code == 500
    assert secret not in response.text
    assert secret not in caplog.text


@pytest.mark.parametrize("quality_status", [None, "needs_hr_review"])
def test_completed_draft_shows_review_warning(monkeypatch, quality_status):
    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        data = {"employees": ["EMP-REVIEW"]} if url.endswith("/employees") else {"case_ids": [], "provider": "stub"}
        response._content = json.dumps(data).encode()
        return response

    def post(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        outputs = {"subject": "Synthetic draft", "body": "Please review this draft."}
        if quality_status is not None:
            outputs["quality_status"] = quality_status
        response._content = json.dumps({"case_id": 900, "status": "completed", "outputs": outputs}).encode()
        return response

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(UI).run()
    app.selectbox[0].select("EMP-REVIEW").run()
    next(w for w in app.text_input if w.label == "Password").set_value("samsara").run()
    next(b for b in app.button if b.label == "Continue").click().run()
    next(b for b in app.button if b.label == "Run Workflow").click().run()
    assert not app.exception
    assert any("Case completed" in message.value for message in app.success)
    assert any("HR review is required" in message.value for message in app.warning)
