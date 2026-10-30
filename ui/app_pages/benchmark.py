"""Operator-authorized, aggregate-only application benchmarks."""
import requests
import streamlit as st

st.subheader('Application benchmark')
st.caption('All employees · active benchmark window or all-time history')
st.text_input('Analytics access token', type='password', key='analytics_access_token',
              help='Separate operator credential. Employee tokens and the demo password do not grant analytics access.')
if not st.session_state.get('analytics_access_token'):
    st.info('Enter an authorized analytics token to view application-wide aggregate results.')
    st.stop()


@st.fragment(run_every='15s')
def show_benchmark():
    include_history = st.toggle('Include earlier submissions', value=False,
                                help='Show all-time totals without changing or deleting any cases.')
    st.button('Refresh benchmark')
    try:
        response = requests.get('http://localhost:8000/benchmarks/latest',
                                params={'include_history': str(include_history).lower()},
                                headers={'Authorization': f"Bearer {st.session_state['analytics_access_token']}"}, timeout=10)
        if response.status_code in (401, 403):
            st.error('Analytics access denied. Enter a valid analytics token.')
            return
        if response.status_code == 503:
            st.error('Analytics unavailable. Check the API database and analytics authentication configuration.')
            return
        response.raise_for_status()
        benchmark = response.json()
    except requests.exceptions.RequestException:
        st.error('Could not load the benchmark. Check the API connection.')
        return

    summary = benchmark['summary']
    window = benchmark.get('benchmark_window')
    st.subheader('Current benchmark submissions' if window else 'All recorded application submissions')
    if window:
        st.caption(f"{window['label']} · Reset at {window['reset_at']} · Earlier cases and audit history are preserved.")
    left, right = st.columns(2)
    left.metric('Total submissions', str(summary['total']))
    right.metric('Recorded retry attempts', str(summary['retry_attempts_recorded']))
    st.caption('One saved HR-question submission counts once. Retries reuse it; page loads and other non-question requests are excluded.')
    for column, (category, count) in zip(st.columns(5), summary['counts'].items()):
        column.metric(category, str(count))
        column.caption(f"{summary['percentages'][category]}% of all submissions")
    st.write(f"**Benchmark score: {summary['appropriate_outcomes']}/{summary['total']} ({summary['appropriate_outcome_percent']}%)** — reviewed correct answers plus reviewed appropriate no-information responses.")
    st.caption(f"Reviewed answer coverage: {summary['reviewed_answers']}/{summary['total']} ({summary['review_coverage_percent']}%). Awaiting review and workflow errors remain in the denominator. In progress: {summary['in_progress']}.")
    if not summary['total']:
        st.info('No recorded HR-question submissions yet.')
    if not benchmark['review_evidence_available']:
        st.warning('Documented review evidence is unavailable or policy files changed. Completed answers remain awaiting review.')
    st.subheader('Questions by HR topic')
    topic_rows = [{'Topic': topic, 'Submissions': count} for topic, count in benchmark['topic_counts'].items()]
    st.bar_chart(topic_rows, x='Topic', y='Submissions', horizontal=True)
    st.dataframe(topic_rows, hide_index=True, width='stretch')
    st.caption('A question can count in several topics, but only once per topic. Topic counts can add up to more than total submissions.')
    with st.expander('Scoring, topic rules and limitations'):
        st.write(benchmark['scoring_method'])
        st.write(benchmark['topic_method'])
        for limitation in benchmark['limitations']:
            st.write(f'• {limitation}')
        st.write(f"Excluded invalid historical records: {summary['excluded_invalid_records']}")
    st.caption(f"Updated: {benchmark['updated_at']} · Refreshes every 15 seconds while this page is open.")


show_benchmark()
