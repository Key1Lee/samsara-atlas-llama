"""Processing success never creates an HR approval decision."""
import json
from uuid import uuid4
from pathlib import Path
import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("fail", [False, True])
def test_review_status_is_independent_of_processing(db_session, monkeypatch, fail):
    from app.main import app
    from app.engine import steps
    from app.storage.models import EmployeeRecord, Case
    employee = "REVIEW-" + uuid4().hex
    db_session.add(EmployeeRecord(employee_id=employee, metadata_json="{}", last_activities_json="[]"))
    db_session.commit()
    if fail:
        def broken(*a, **kw):
            raise ValueError("synthetic provider failure")
        monkeypatch.setattr(steps, "_retryable_llm_call", broken)
    client = TestClient(app)
    response = client.post("/cases", json={"inputs": {
        "employee_id": employee, "case_transcript": "Explain accommodation review.", "tone": "direct",
        "review_status": "approved"}, "idempotency_key": uuid4().hex, "review_status": "approved"})
    assert response.status_code == (500 if fail else 200)
    case = db_session.query(Case).filter(Case.inputs_json.contains(employee)).one()
    assert case.status == ("failed" if fail else "completed")
    if not fail:
        assert response.json()["review_status"] is None
        assert response.json()["outputs"]["quality_status"] == "needs_hr_review"
    detail = client.get(f"/cases/{case.id}", headers={"X-Employee-ID": employee})
    assert detail.json()["review_status"] is None
    assert client.post(f"/cases/{case.id}/approve", headers={"X-Employee-ID": employee}).status_code == 404
    assert "review_status" not in json.loads(case.inputs_json)


def test_ui_never_promotes_forged_quality_to_approval(monkeypatch):
    def get(url, **kw):
        response = requests.Response()
        response.status_code = 200
        data = {"employees": ["REVIEW-UI"]} if url.endswith("/employees") else {
            "case_ids": [], "provider": "stub", "mode": "demo"}
        response._content = json.dumps(data).encode()
        return response
    def post(*a, **kw):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"case_id": 1, "status": "completed",
            "review_status": None, "outputs": {"subject": "Draft", "body": "Review me",
            "quality_status": "approved"}}).encode()
        return response
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"ui/streamlit_app.py")).run()
    app.selectbox[0].select("REVIEW-UI").run()
    next(w for w in app.text_input if w.label == "Password").set_value("samsara").run()
    next(b for b in app.button if b.label == "Continue").click().run()
    next(b for b in app.button if b.label == "Run Workflow").click().run()
    assert not app.exception
    assert any("HR review is required" in w.value for w in app.warning)
    assert any("Demo provider" in w.value for w in app.warning)

