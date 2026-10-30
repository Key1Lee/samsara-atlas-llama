import json
import os
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
import requests
import streamlit as st
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from presentation import apply_presentation, render_answer, render_logistics_header, render_masthead, render_workspace_intro

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_BASE = "http://localhost:8000"


def employee_headers(employee_id=None):
    selected = employee_id or st.session_state.get("selected_employee")
    headers = {"X-Employee-ID": selected} if selected else {}
    token = st.session_state.get("access_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def run_workflow(employee_id: str, case_transcript: str, tone: str, idempotency_key: str | None = None):
    resp = requests.post(
        f"{API_BASE}/cases",
        json={
            "inputs": {
                "employee_id": employee_id,
                "case_transcript": case_transcript,
                "tone": tone,
            },
            "idempotency_key": idempotency_key,
        },
        headers=employee_headers(employee_id),
        timeout=120,
    )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        err = requests.exceptions.HTTPError(f"{resp.status_code}: {detail}", response=resp)
        raise err
    return resp.json()


def list_cases():
    resp = requests.get(f"{API_BASE}/cases", headers=employee_headers(), timeout=10)
    if resp.status_code == 404:
        return {"case_ids": []}
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Atlas · Employee guidance", page_icon=":material/spa:", layout="centered", initial_sidebar_state="collapsed")
apply_presentation()
render_masthead()
render_logistics_header()
st.html('<div class="atlas-kicker">YOUR SAMSARAN HUB</div>')
st.title("Increase the safety, efficiency, and sustainability of the operations that power the global economy.")
st.html('<p class="atlas-lead"><strong>Welcome to your Samsaran Hub for Time Off, Employee Benefits, and Work-Life Resources.</strong></p>')
render_workspace_intro()

def clear_employee_view():
    for key in ("selected_employee", "employee_choice", "case_result", "loaded_case",
                "last_case_id", "question", "response_tone", "retry_key", "explorer_case_id", "submission",
                "employee_password", "password_employee"):
        st.session_state.pop(key, None)


with st.sidebar:
    if os.getenv("ATLAS_EMPLOYEE_TOKEN_HASHES") or os.getenv("ATLAS_ENV", "production").strip().lower() != "development":
        st.text_input("Employee access token", type="password", key="access_token",
                      on_change=clear_employee_view)
    else:
        st.caption("Local demo: employee selection is not authentication.")
    try:
        info = requests.get(f"{API_BASE}/info", headers=employee_headers(), timeout=2).json()
        provider = info.get("provider", "?")
        st.caption(f"Configured provider: {provider}")
        if info.get("mode") == "demo" or provider in {"stub", "local"}:
            st.warning("Demo provider: simulated model output, not a live-model answer.")
        elif info.get("mode") == "configuration_error":
            st.error("Provider configuration error. Select stub or ollama and restart the API.")
    except Exception:
        st.warning("API not reachable")
    st.toggle("Reduce motion", value=False, key="reduce_motion",
              help="Disable nonessential animation. Your system’s reduced-motion preference is always respected.")
    with st.expander("Connection details"):
        st.caption("Local Atlas API · workflow and model configuration are managed by your administrator.")
    st.caption("All employee data shown here is synthetic fixture data.")

def render_question_page():
    try:
        employees_resp = requests.get(f"{API_BASE}/employees", headers=employee_headers(), timeout=2)
        if employees_resp.status_code == 429:
            st.error("Daily limit of 200 requests reached for this employee. Try again after midnight UTC.")
            if st.button("Change employee"):
                st.session_state.clear()
                st.rerun()
            st.stop()
        employees = employees_resp.json().get("employees", []) if employees_resp.ok else []
    except Exception:
        employees = []
    if not isinstance(employees, list) or any(
        not isinstance(employee, str) or not employee.strip() for employee in employees
    ):
        employees = []
    if not employees:
        st.error("No employees available. Check the API connection and seed the database before submitting a case.")

    # Keep navigation in this existing Streamlit app; no new page framework needed.
    conversation_keys = (
        "selected_employee", "case_result", "loaded_case", "last_case_id",
        "question", "response_tone", "retry_key", "explorer_case_id", "submission",
        "password_employee",
    )
    selected_employee = st.session_state.get("selected_employee")
    if (selected_employee not in employees or
            (selected_employee and st.session_state.get("password_employee") != selected_employee)):
        for key in conversation_keys:
            if key != "submission":
                st.session_state.pop(key, None)

    if "selected_employee" not in st.session_state:
        st.subheader("Your space, your questions")
        st.write("Choose your synthetic demo profile to get started. Your question history stays with this profile.")
        employee_id = st.selectbox(
            "Employee ID", options=employees, index=None,
            placeholder="Select an employee", disabled=not employees,
            key="employee_choice",
        )
        password = st.text_input("Password", type="password", key="employee_password",
                                 disabled=not employees)
        if st.button("Continue", disabled=employee_id not in employees or not password):
            if password == "samsara":
                st.session_state["selected_employee"] = employee_id
                st.session_state["password_employee"] = employee_id
                pending = st.session_state.get("submission")
                if pending and pending["inputs"]["employee_id"] == employee_id:
                    st.session_state["question"] = pending["inputs"]["case_transcript"]
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        st.stop()

    st.session_state.pop("employee_password", None)
    employee_id = st.session_state["selected_employee"]
    st.caption(f"YOUR WORKSPACE · {employee_id}")
    if st.button("Change employee"):
        for key in (*conversation_keys, "employee_choice", "employee_password"):
            st.session_state.pop(key, None)
        st.rerun()

    pending = st.session_state.get("submission")
    if "question" not in st.session_state and pending and pending["inputs"]["employee_id"] == employee_id:
        st.session_state["question"] = pending["inputs"]["case_transcript"]
    st.subheader("What’s on your mind?")
    st.caption("Try a question about time off, benefits, or working remotely. Avoid adding sensitive personal details.")
    with st.form("workflow_form"):
        case_transcript = st.text_area(
            "Case question",
            value="Asked how PTO carryover works going into next year. Has about 30 unused hours.",
            height=150, key="question",
        )
        submitted = st.form_submit_button("Ask Atlas", type="primary", width="stretch")

    if submitted and not case_transcript.strip():
        st.error("Write a question before asking Atlas.")
    if submitted and case_transcript.strip() and employee_id in employees:
        tone = "empathetic"
        inputs = {"employee_id": employee_id, "case_transcript": case_transcript, "tone": tone}
        submission = st.session_state.get("submission")
        if submission is None or submission["inputs"] != inputs:
            submission = {"inputs": inputs, "key": uuid4().hex}
            st.session_state["submission"] = submission
        idempotency_key = submission["key"]
        loading = st.empty()
        loading.info("Preparing your answer. Keep this page open; repeating the request will reuse the same case.")
        with st.container():
            try:
                result = run_workflow(
                    employee_id=employee_id,
                    case_transcript=case_transcript,
                    tone=tone,
                    idempotency_key=idempotency_key,
                )
                if result["status"] == "completed":
                    st.success(f"Case completed: {result['case_id']}")
                else:
                    st.info(f"Case {result['case_id']} is {result['status']}. Check the employee case list below for its status.")
                if result.get('recovery') == 'manual_reconciliation_required':
                    st.warning('Manual reconciliation is required. Ask an authorized HR representative to inspect this case before any write is repeated.')
                st.session_state["last_case_id"] = result["case_id"]
                st.session_state["case_result"] = result
            except requests.exceptions.RequestException as e:
                st.session_state.pop("case_result", None)
                st.error(f"Error: {e}")
                st.caption("You can retry by asking again with the same question. Atlas will reuse the submission key.")
            finally:
                loading.empty()

    if "case_result" in st.session_state and st.session_state["case_result"].get("status") == "completed":
        render_answer(st.session_state["case_result"].get("outputs", {}))

    st.divider()
    try:
        st.subheader("Your question history")
        records = list_cases().get("cases", [])
        if records:
            st.dataframe([{"Case ID": row["case_id"], "Question": row["question"],
                           "Status": row["status"], "Submitted (UTC)": row["submitted_at"]}
                          for row in records], hide_index=True, width="stretch")
        else:
            st.info("No cases for this employee yet.")
    except requests.exceptions.RequestException:
        st.warning("Could not load cases. Check the API connection.")


page = st.navigation([
    st.Page(render_question_page, title="Ask Atlas", icon=":material/chat:", default=True),
    st.Page("app_pages/policies.py", title="Policy library", icon=":material/menu_book:"),
    st.Page("app_pages/benchmark.py", title="Benchmark", icon=":material/bar_chart:"),
], position="top")
page.run()
