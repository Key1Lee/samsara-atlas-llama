"""Exercise the existing Streamlit form through FastAPI and the real runner/HRIS."""
from pathlib import Path
from uuid import uuid4

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

UI = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")


@pytest.fixture
def api_bridge(db_session, monkeypatch):
    from data.seed import seed_employee_records, load_fixture
    from app.main import app
    seed_employee_records(db_session, load_fixture())
    client = TestClient(app)
    def request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        response = client.request(method, url.replace("http://localhost:8000", ""), **kwargs)
        result = requests.Response()
        result.status_code = response.status_code
        result._content = response.content
        return result
    monkeypatch.setattr(requests, "get", lambda url, **kw: request("GET", url, **kw))
    monkeypatch.setattr(requests, "post", lambda url, **kw: request("POST", url, **kw))
    return client


def test_streamlit_submission_persists_case(api_bridge):
    app = AppTest.from_file(UI).run()
    assert not app.exception
    continue_to_question(app)
    app.text_area[0].set_value("How does PTO accrue?")
    run_button(app).click().run()
    assert not app.exception
    assert any("Case completed" in message.value for message in app.success)
    case_id = app.session_state["last_case_id"]
    stored = api_bridge.get(f"/cases/{case_id}", headers={"X-Employee-ID": "E10234"}).json()
    assert stored["case"]["status"] == "completed"
    assert len(stored["steps"]) == 5 and len(stored["case_notes"]) == 1
    run_button(app).click().run()
    assert app.session_state["last_case_id"] == case_id
    assert len(api_bridge.get(f"/cases/{case_id}", headers={"X-Employee-ID": "E10234"}).json()["case_notes"]) == 1


def test_streamlit_handles_api_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("API unavailable")
    monkeypatch.setattr(requests, "get", unavailable)
    app = AppTest.from_file(UI).run()
    assert not app.exception
    assert any("API not reachable" in warning.value for warning in app.warning)


def test_streamlit_does_not_report_running_case_as_completed(api_bridge, monkeypatch):
    def running(*args, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"case_id":999,"status":"running","outputs":{}}'
        return response
    monkeypatch.setattr(requests, "post", running)
    app = AppTest.from_file(UI).run()
    continue_to_question(app)
    run_button(app).click().run()
    assert not app.exception
    assert not any("Case completed" in message.value for message in app.success)
    assert any("running" in message.value.lower() for message in app.info)


@pytest.mark.parametrize("question,expected", [
    ("How many PTO hours can I carry over?", "40 hours"),
    ("Can I relocate to another state while working remotely?", "notify HR before relocating"),
])
def test_streamlit_deterministic_policy_submission(api_bridge, monkeypatch, question, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("Policy answers must not invoke model or HRIS tools")
    monkeypatch.setattr("app.engine.steps._retryable_llm_call", forbidden)
    monkeypatch.setattr("app.engine.steps.get_tool_registry", forbidden)
    app = AppTest.from_file(UI).run()
    continue_to_question(app)
    app.text_area[0].set_value(question)
    run_button(app).click().run()
    assert not app.exception
    result = app.session_state["case_result"]
    assert result["status"] == "completed" and expected in result["outputs"]["body"]
    stored = api_bridge.get(f"/cases/{result['case_id']}", headers={"X-Employee-ID": "E10234"}).json()
    assert len(stored["steps"]) == 1 and stored["case_notes"] == []


def continue_to_question(app, employee="E10234"):
    app.selectbox[0].select(employee).run()
    next(w for w in app.text_input if w.label == "Password").set_value("samsara").run()
    next(button for button in app.button if button.label == "Continue").click().run()
    assert not app.exception


def run_button(app):
    return next(button for button in app.button if button.label == "Ask Atlas")


def test_employee_selection_precedes_question_and_employee_cases(api_bridge):
    app = AppTest.from_file(UI).run()
    assert not app.text_area and not app.number_input
    assert not any(s.value == "Case Explorer" for s in app.subheader)
    assert next(b for b in app.button if b.label == "Continue").disabled
    continue_to_question(app, "E10897")
    assert app.session_state["selected_employee"] == "E10897"
    assert len(app.text_area) == 1
    assert any(s.value == "Your question history" for s in app.subheader)
    assert not app.number_input and not app.text_input and not app.selectbox
    assert app.title[0].value == "Increase the safety, efficiency, and sustainability of the operations that power the global economy."
    assert app.text_area[0].label == "Case question"
    assert not any(b.label in {"Load Case", "Replay Case"} for b in app.button)
    app.text_area[0].set_value("How many PTO hours can I carry over?")
    run_button(app).click().run()
    case_id = app.session_state["last_case_id"]
    import json
    stored = api_bridge.get(f"/cases/{case_id}", headers={"X-Employee-ID": "E10897"}).json()
    assert json.loads(stored["case"]["inputs"])["employee_id"] == "E10897"
    assert json.loads(stored["case"]["inputs"])["tone"] == "empathetic"
    assert not any("Submission key:" in c.value for c in app.caption)
    assert list(app.dataframe[0].value["Case ID"]).count(case_id) == 1
    next(b for b in app.button if b.label == "Change employee").click().run()
    assert not app.text_area and not app.number_input
    assert "case_result" not in app.session_state
    assert "loaded_case" not in app.session_state
    continue_to_question(app, "E11342")
    assert app.session_state["selected_employee"] == "E11342"
    assert "case_result" not in app.session_state
    assert not app.text_input


def test_authenticated_employee_switch_refreshes_case_list(api_bridge, db_session, monkeypatch):
    import hashlib
    import json
    from uuid import uuid4
    from app.storage.models import Case, EmployeeRecord
    employees = ['UI-A-' + uuid4().hex, 'UI-B-' + uuid4().hex]
    tokens = {employee: uuid4().hex for employee in employees}
    for employee in employees:
        db_session.add(EmployeeRecord(employee_id=employee, metadata_json='{}', last_activities_json='[]'))
        for i in range(2):
            db_session.add(Case(status='completed', inputs_json=json.dumps({
                'employee_id': employee, 'case_transcript': f'{employee} question {i}', 'tone': 'direct'})))
    db_session.commit()
    monkeypatch.setenv('ATLAS_ENV', 'production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES', json.dumps({
        employee: hashlib.sha256(token.encode()).hexdigest() for employee, token in tokens.items()}))
    app = AppTest.from_file(UI).run()
    token_input = lambda: next(w for w in app.text_input if w.label == 'Employee access token')
    token_input().set_value(tokens[employees[0]]).run()
    assert app.selectbox[0].options == [employees[0]]
    continue_to_question(app, employees[0])
    assert len(app.dataframe[0].value) == 2
    assert all(employees[0] in q for q in app.dataframe[0].value['Question'])
    app.text_area[0].set_value('How many PTO hours can I carry over?')
    run_button(app).click().run()
    assert len(app.dataframe[0].value) == 3
    new_case = app.session_state['last_case_id']
    run_button(app).click().run()
    assert list(app.dataframe[0].value['Case ID']).count(new_case) == 1
    token_input().set_value(tokens[employees[1]]).run()
    assert 'case_result' not in app.session_state
    assert app.selectbox[0].options == [employees[1]]
    continue_to_question(app, employees[1])
    assert len(app.dataframe[0].value) == 2
    assert all(employees[1] in q for q in app.dataframe[0].value['Question'])
    assert not app.exception


def test_unspecified_mode_shows_credentials_and_blocks_employee_selection(api_bridge, monkeypatch):
    monkeypatch.delenv('ATLAS_ENV', raising=False)
    app = AppTest.from_file(UI).run()
    assert not app.exception
    assert any(widget.label == 'Employee access token' for widget in app.text_input)
    assert next(button for button in app.button if button.label == 'Continue').disabled
    assert not app.text_area


def cases(client):
    return client.get('/cases', headers={'X-Employee-ID': 'E10234'}).json()['case_ids']


@pytest.mark.parametrize('directory_outage', [False, True])
def test_blank_ui_key_reuses_saved_case_after_lost_response(api_bridge, monkeypatch, directory_outage):
    original = requests.post
    sent_keys = []
    lose_reply = True

    def save_then_timeout(*args, **kwargs):
        nonlocal lose_reply
        sent_keys.append(kwargs['json']['idempotency_key'])
        response = original(*args, **kwargs)
        assert response.status_code == 200
        if lose_reply:
            lose_reply = False
            raise requests.Timeout('Synthetic lost HTTP acknowledgement')
        return response

    monkeypatch.setattr(requests, 'post', save_then_timeout)
    before = set(cases(api_bridge))
    app = AppTest.from_file(UI).run()
    continue_to_question(app)
    app.text_area[0].set_value('How does PTO accrue?')
    run_button(app).click().run()
    assert app.error and not app.exception
    first_id = (set(cases(api_bridge)) - before).pop()
    if directory_outage:
        original_get = requests.get

        def unavailable(*args, **kwargs):
            raise requests.ConnectionError('Synthetic temporary outage')

        monkeypatch.setattr(requests, 'get', unavailable)
        app.run()
        assert not app.text_area and not app.exception
        monkeypatch.setattr(requests, 'get', original_get)
        app.run()
        continue_to_question(app)
        assert app.text_area[0].value == 'How does PTO accrue?'
    run_button(app).click().run()
    assert not app.exception
    assert sent_keys[0] and sent_keys[0] == sent_keys[1]
    assert set(cases(api_bridge)) - before == {first_id}
    assert app.session_state['last_case_id'] == first_id
    saved = api_bridge.get(f'/cases/{first_id}', headers={'X-Employee-ID': 'E10234'}).json()
    assert len(saved['case_notes']) == 1 and len(saved['steps']) == 5


def test_ui_has_no_new_submission_button_and_unchanged_retry_reuses_case(api_bridge):
    before = set(cases(api_bridge))
    app = AppTest.from_file(UI).run()
    continue_to_question(app)
    assert 'New submission' not in [button.label for button in app.button]
    assert not any('New submission' in caption.value for caption in app.caption)
    app.text_area[0].set_value('How does PTO accrue?')
    run_button(app).click().run()
    first = app.session_state['last_case_id']
    run_button(app).click().run()
    assert app.session_state['last_case_id'] == first
    assert not app.exception
    assert set(cases(api_bridge)) - before == {first}


def test_changed_ui_question_creates_new_submission_and_employee_switch_clears_key(api_bridge, monkeypatch):
    original = requests.post
    keys = []

    def capture(*args, **kwargs):
        keys.append(kwargs['json']['idempotency_key'])
        return original(*args, **kwargs)

    monkeypatch.setattr(requests, 'post', capture)
    app = AppTest.from_file(UI).run()
    continue_to_question(app)
    run_button(app).click().run()
    first = app.session_state['last_case_id']
    app.text_area[0].set_value('Can I relocate to another state while working remotely?')
    run_button(app).click().run()
    assert app.session_state['last_case_id'] != first
    next(button for button in app.button if button.label == 'Change employee').click().run()
    continue_to_question(app, 'E10897')
    run_button(app).click().run()
    assert all(keys) and len(set(keys)) == 3


def test_hidden_automatic_key_retains_backend_conflict_protection(api_bridge):
    from app.storage.database import SessionLocal
    from app.storage.models import CaseNote
    app = AppTest.from_file(UI).run()
    continue_to_question(app)
    app.text_area[0].set_value('How does PTO accrue?')
    run_button(app).click().run()
    first = app.session_state['last_case_id']
    before = set(cases(api_bridge))
    with SessionLocal() as db:
        before_notes = db.query(CaseNote).count()
    assert not app.text_input and not app.number_input and not app.selectbox
    pending = app.session_state["submission"]
    conflict = api_bridge.post("/cases", json={
        "inputs": {**pending["inputs"], "case_transcript": "How many PTO hours can I carry over?"},
        "idempotency_key": pending["key"]})
    assert conflict.status_code == 409
    assert set(cases(api_bridge)) == before
    with SessionLocal() as db:
        assert db.query(CaseNote).count() == before_notes


def test_employee_password_required_and_wrong_password_blocks_questions(api_bridge):
    app = AppTest.from_file(UI).run()
    app.selectbox[0].select("E10234").run()
    password = next(w for w in app.text_input if w.label == "Password")
    from streamlit.proto.TextInput_pb2 import TextInput
    assert password.proto.type == TextInput.PASSWORD
    assert next(b for b in app.button if b.label == "Continue").disabled
    password.set_value("wrong").run()
    next(b for b in app.button if b.label == "Continue").click().run()
    assert not app.text_area and not app.dataframe
    assert any("Incorrect password" in e.value for e in app.error)
    next(w for w in app.text_input if w.label == "Password").set_value("samsara").run()
    next(b for b in app.button if b.label == "Continue").click().run()
    assert app.text_area and not app.exception
    assert "employee_password" not in app.session_state
    next(b for b in app.button if b.label == "Change employee").click().run()
    app.selectbox[0].select("E10897").run()
    assert next(w for w in app.text_input if w.label == "Password").value == ""
    assert next(b for b in app.button if b.label == "Continue").disabled


def test_existing_ui_session_must_pass_new_password_gate(api_bridge):
    app = AppTest.from_file(UI)
    app.session_state["selected_employee"] = "E10234"
    app.session_state["case_result"] = {"status": "completed", "outputs": {"body": "old draft"}}
    app.run()
    assert not app.exception and not app.text_area and not app.dataframe
    assert any(w.label == "Password" for w in app.text_input)
    assert "case_result" not in app.session_state


def test_blank_question_does_not_create_submission(api_bridge, monkeypatch):
    app = AppTest.from_file(UI).run()
    continue_to_question(app)
    def forbidden(*args, **kwargs):
        pytest.fail('Blank question must not reach API')
    monkeypatch.setattr(requests, 'post', forbidden)
    app.text_area[0].set_value('   ')
    run_button(app).click().run()
    assert not app.exception
    assert any('Write a question' in e.value for e in app.error)


def test_header_motion_can_be_disabled(api_bridge):
    app = AppTest.from_file(UI).run()
    assert app.toggle(key='reduce_motion').value is False
    app.toggle(key='reduce_motion').set_value(True).run()
    assert app.toggle(key='reduce_motion').value is True


def test_policy_library_denies_unverified_access(monkeypatch):
    def denied(*args, **kwargs):
        response = requests.Response()
        response.status_code = 401
        return response
    monkeypatch.setattr(requests, 'get', denied)
    page = str(Path(UI).parent / 'app_pages' / 'policies.py')
    app = AppTest.from_file(page).run()
    assert not app.exception
    assert app.error and not app.expander


def test_long_draft_retains_plain_text_subject_and_citation():
    def screen():
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path.cwd() / 'ui'))
        from presentation import render_answer
        render_answer({'subject': '<script>unsafe()</script>',
                       'body': 'Long policy explanation. ' * 1000,
                       'cited_snippets': ['<img src="https://example.invalid/track">']})
    app = AppTest.from_function(screen).run()
    assert not app.exception
    assert any('<script>unsafe()' in t.value for t in app.text)
    assert any('<img src=' in t.value for t in app.text)
    assert any('Long policy explanation.' in m.value for m in app.markdown)
