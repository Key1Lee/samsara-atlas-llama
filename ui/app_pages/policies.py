"""Read the supplied policy corpus; never implies policy approval or eligibility."""
from pathlib import Path
import os
import streamlit as st
import requests

headers = {}
if st.session_state.get('access_token'):
    headers['Authorization'] = f"Bearer {st.session_state['access_token']}"
if st.session_state.get('selected_employee'):
    headers['X-Employee-ID'] = st.session_state['selected_employee']
try:
    access = requests.get('http://localhost:8000/employees', headers=headers, timeout=2)
    if access.status_code == 429:
        st.warning('Daily request limit reached. Try again after midnight UTC.')
        st.stop()
    if access.status_code != 200:
        st.error('Policy access requires a valid employee session. Check your access token or try again later.')
        st.stop()
except requests.exceptions.RequestException:
    st.error('Policy access could not be verified. Please try again later.')
    st.stop()

st.subheader('A place to find your bearings')
st.write('Browse the policy text behind Atlas’s guidance.')
st.caption('These are supplied assessment documents. Approval, effective dates, and personal applicability require HR verification.')
category = st.selectbox('Explore a topic', ['Time off', 'Benefits', 'Workplace', 'All policies'])
allowed = {
    'Time off': ['pto-policy', 'parental-leave-policy', 'leave-of-absence-policy'],
    'Benefits': ['benefits-overview'],
    'Workplace': ['remote-work-policy', 'expense-policy', 'workplace-accommodation-policy', 'code-of-conduct', 'harassment-reporting-policy'],
}
root = Path(os.getenv('DOCS_DIR', str(Path(__file__).resolve().parents[2] / 'data/docs'))).resolve()
try:
    documents = [p for p in sorted(root.glob('*.md')) if p.resolve().is_relative_to(root) and
                 (category == 'All policies' or p.stem in allowed[category])]
    for path in documents:
        text = path.read_text(encoding='utf-8')
        with st.expander(text.splitlines()[0].lstrip('# ') if text else path.stem):
            st.caption(f'Source: {path.name}')
            # Plain text prevents policy-authored HTML, links or instructions becoming UI controls.
            st.text(text)
    if not documents:
        st.info('No policy text is available. Ask HR for the current approved policy.')
except (OSError, UnicodeError):
    st.error('Policy text could not be loaded. Ask HR for the current approved policy.')
