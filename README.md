# Samsara Atlas

A lightweight GenAI workflow orchestration platform for People (HR) teams.

## Quick Start

### 1. Install Dependencies

```bash
make setup
```

Or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Seed Data

```bash
make seed
```

### 3. Run Demo

Start the API and UI in two terminals:

**Terminal 1:**
```bash
ATLAS_ENV=development make api
```

**Terminal 2:**
```bash
ATLAS_ENV=development make ui
```

Or use `ATLAS_ENV=development make demo` to start both (API in background, then UI).

- API: http://localhost:8000
- UI: http://localhost:8501

### 4. Run Tests

```bash
make test
```

### Core workflow behavior

`make test` uses synthetic employee fixtures, a temporary SQLite database, and a
deterministic stub provider. It exercises the Streamlit form through FastAPI,
workflow steps, and HRIS persistence without requiring Ollama.

Use a nonempty `idempotency_key` when retrying a submission. The same key and
inputs return the existing run; a failed run resumes its unfinished step.
Reusing a key with different inputs returns HTTP 409. Submissions without a key
and explicit replays create independent cases. Transient LLM transport failures
retry up to three times; invalid model output fails without posting a case note.

Startup adds unique indexes for case keys and HRIS notes to clean existing
SQLite databases. Historical duplicate keys or notes require manual review
before startup; records are never deleted automatically. Guarded interrupted
local runs can be recovered as described below. Historical snapshot replay remains
outside this minimal workflow validation.

### 5. Run the Local Reliability Probe

With the synthetic database seeded and the API running (`make api` in another
terminal), use the same employee access configuration as the UI. In authenticated
mode, export `ATLAS_EMPLOYEE_TOKEN` with that employee's bearer token:

```bash
make load
```

This bounded correctness check sends eight attempts using one fresh idempotency
key. The default question is "How does PTO accrue?" and expects one completed
case and one local case note. It checks response IDs, employee history, stored
inputs/outputs and note counts against the API's SQLite file, opened read-only.
Any unexpected HTTP failure, exception or integrity mismatch exits nonzero.
The default general workflow uses the configured model; this is not a model
quality evaluation or capacity benchmark. Requests still consume the daily quota.

Use `LOAD_ARGS` for another employee/database or an explicit scenario. The file
must be the one used by the API; `DATABASE_URL` or `data/atlas.db` is the default.
`--independent` means separate intentional submissions even for identical questions.
Custom questions require an explicit local-note expectation, since deterministic
routes skip note creation. For example, this three-attempt probe needs no LLM:

```bash
make load LOAD_ARGS='--requests 3 --question "How much PTO can I carry over to next year?" --expected-notes 0'
```

`--expect-rejection 422` (or 409/429) explicitly requires every POST to return
that rejection and verifies zero case rows for the fresh keys. Unlisted or
unexpected failures never count as successful results. The probe preserves
existing records and leaves its synthetic submissions available for inspection.

## Architecture

- **app/engine/** - Workflow engine (steps, context, runner)
- **app/providers/** - Model provider abstraction + stub provider
- **app/tools/** - Tool registry (HRIS, policy retrieval)
- **app/storage/** - DB models (cases, steps, case notes, employees)
- **app/api/** - FastAPI routes
- **ui/** - Streamlit UI

## What this does

An HR partner (or an employee, via an intake form) submits a short description of
an HR case - a PTO question, a leave-of-absence request, a remote-work exception,
an expense dispute, etc. - along with the employee's ID and a desired response tone.
Atlas retrieves the employee's HRIS context, retrieves relevant policy excerpts,
summarizes the case and extracts action items, drafts a structured response, and
logs a case note back to the HRIS.

## API Endpoints

- `POST /cases` - Create a case run (inputs + optional idempotency_key)
- `GET /cases` - List case IDs
- `GET /cases/{case_id}` - Get case details, steps, case notes
- `POST /cases/{case_id}/replay` - Replay a previous case
- `GET /employees` - List seeded employee records
- `GET /info` - Current provider type

## Real LLM (Ollama)

To use a local LLM instead of the stub:

1. **Install Ollama** from [ollama.ai](https://ollama.ai) and pull a model:
   ```bash
   ollama pull llama3.2
   ```

2. **Configure the provider** - either:
   - Create a `.env` file with `PROVIDER_TYPE=ollama`, then run `make api`
   - Or run `make api-ollama` (sets env for that command only)

3. **Restart the API** - stop any running API (Ctrl+C) and start fresh so it picks up the config.

4. **Verify** - the sidebar shows the active provider, or visit http://localhost:8000/info

Environment variables:
- `OLLAMA_BASE_URL` - default `http://localhost:11434`
- `OLLAMA_MODEL` - default `llama3.2`

## Limits

Key limits: case lists include all cases for the selected employee; retrieval top-k=5; LLM retries=3. See `.env.example` and code for configurable values.

Policy retrieval keeps BM25 ranking and a source-authored lexical topic gate.
Narrow vocabulary normalization covers PTO/vacation and remote-work phrasing;
scores establish a candidate topic, not approval, entitlement or legal applicability.
The supplied Markdown files have no authoritative approval/version registry.
Run `.venv/bin/python scripts/evaluate_retrieval.py --require-all` for the fixed
23-question source/clause evaluation. It currently reports two failing unsupported
overlap questions (pet-insurance benefits and relocation reimbursement).

Model calls preserve complete selected policy documents. The assembled prompt
has a 6,000 UTF-8-byte application bound. If needed, whole supplementary
`case_history`/`last_activities` collections are omitted with an explicit
`context_omissions` marker; saved history and employee metadata are preserved.
If the prompt still cannot fit, the case fails with inspectable step error
`ContextBudgetExceeded`, without sending or silently clipping policy evidence.
Large unrelated sections can therefore make an entire policy unusable; there
is no section dependency parser. Ollama requests `num_ctx=8192` and
`num_predict=1024`. The byte bound is not a tokenizer or a guarantee that any
configured model/runtime honors that context window; live limits remain unverified.

Known PTO carryover drafts are checked independently of model prose: Atlas uses
the complete Accrual section, including its local-law exception and attached
qualifications, and asks HR to verify jurisdiction/eligibility. Distinct same-title
policy revisions or mixed PTO/relocation questions require HR clarification.
Deterministic relocation guidance also retains Eligible Roles and Location Changes.
These narrow checks do not establish general legal or exception completeness.

Citation evidence keeps the selected content fingerprint and exact quote span.
`source_verification=matched` means one current file in `DOCS_DIR` exactly matches
that selected snapshot; missing/changed files are `unavailable`, identical copies
are `ambiguous`, and neither case invents a filename. Model-generated paths or
versions never resolve files. Policy symlinks outside the configured directory
are excluded. Approval remains `unknown`, applicability `unverified`, and missing
policy ID/version/effective date remain null. Lexical relevance and exact linkage
do not prove a generated claim is supported. Retrieved text is untrusted data;
known application-control directives cannot become deterministic policy guidance
or citations. This limited check is not general prompt-injection protection.

### Answer safety and operation evidence

Only explicit `stub` or `local` demo configuration selects simulated output;
unknown providers and an unavailable optional local adapter fail without fallback.
`/info` identifies configured demo/live mode; it is not a model health check.
Ollama timeouts, unavailable service, rate limits, rejections and unusable output
have fixed safe diagnostic categories in audit events. Transport/unavailability
retries remain bounded at three attempts; rate limits and rejections fail without
automatic retry. Failed generation leaves the case failed and does not save a note.

Case submission/results and case details expose `review_status: null`, consistent
with the audit endpoint. No verified HR review decision store or reviewer policy
exists. Processing `completed` and model `quality_status` cannot grant approval;
Streamlit always labels completed responses as drafts requiring HR review. No
approval endpoint or authority was invented, and historical outputs are preserved.

Employee context includes an application-owned `pto_balance` availability record.
The current fixture store has no approved balance source, so its status is
`unavailable` and value, units, source and as-of date remain null. Arbitrary fixture
metadata, employee statements and historical drafts cannot verify a balance.
Personal PTO balance responses refer to HR and identify reported amounts as
unverified; narrow output checks also block known model-generated official balance
claims. These checks do not establish semantic safety for every paraphrase.
No freshness threshold or production HRIS integration is defined. Matching,
conflicting or zero authoritative balances require that missing source contract.

Drafts include application-owned `answerability` (`supported`,
`clarification_needed`, or `hr_review_required`) and `recommendation_review`.
The latter always requires review, grants no authorization, and marks employee
facts unverified. `supported` describes the narrow, source-checked carryover or
relocation guidance; policy approval and personal applicability remain unverified.
Other model drafts require HR review. Known unsupported questions receive a fixed
HR referral; vague questions request clarification. Known unsafe recommendations
and additional completion claims are replaced with a review response. Existing
invalid-output checks still fail the step. These bounded text checks do not prove
arbitrary model prose safe or establish a trusted policy approval registry.

`operation_status` is separate from model wording. During generation, no operation
is confirmed. At completion, a local `case_note` receipt requires an actual SQLite
note matching the case, employee and saved draft. Failed writes, missing evidence
and uncertain outcomes cannot supply a success receipt. Authorized case details
also expose this independently checked status for failed/interrupted runs.
`execution_receipts` remains the empty external-action list: Atlas has no email,
notification, meeting, balance-change or human-approval executor. A saved local
draft note proves only that note was stored; workflow completion is not HR approval.

Run `.venv/bin/python scripts/evaluate_answer_safety.py --require-all` for the
fixed 14-question controlled-output evaluation. It uses real retrieval and
adversarial JSON fixtures, without a live model or case-note writes. Separate
regression tests exercise actual disposable SQLite writes, failures and retries.

The API allows 200 attributed HTTP requests per selected employee per UTC
calendar day. A SQLite counter persists across API restarts and updates atomically.
Streamlit sends `X-Employee-ID` on API calls after employee selection; other API
clients should send that header too. Case submissions also derive the ID from
their body. Case reads and replays require an employee scope.
All attributed methods share the counter, including reads, retries and failures.
Requests above the limit return 429 with `Retry-After` and `X-RateLimit-*` headers.
The UI's automatic employee, provider and case-list reads also consume requests.
Unattributed requests before selection and CORS preflight requests are not counted.
In authenticated mode the counter uses the authenticated employee identity;
headers cannot override it. In local demo mode selected IDs can be impersonated.

## Environment

- `DATABASE_URL` - SQLite path (default: `sqlite:///./data/atlas.db`)
- `DOCS_DIR` - Path to markdown policy docs for retrieval (default: `data/docs`)
- `PROVIDER_TYPE` - `stub` (default), `ollama`, or `local` for alternate provider

## Data notice

All employee records and case content in this repo are synthetic fixture data
generated for this assessment. Do not use real employee data, real case content,
or real credentials with this codebase.

## Employee case access

The employee-selection page has a masked shared UI password field below the ID
selector. Enter `samsara` to continue. Changing employee or API credential requires
entering it again. This synthetic-demo gate does not establish employee identity
or replace the backend bearer-token authorization described below.

`GET /cases` returns all independent cases for one employee, newest submission
first (Case ID breaks date ties), with Case ID, question, status and UTC submission
date. Existing JSON inputs determine ownership through a SQLite query; no data
migration is required. The Streamlit list refreshes after submissions and employee
changes. Reads and replays of another employee's case return 404 without revealing
whether the case exists. A repeated submission using the same idempotency key
returns the original case; separate keys remain independent cases.

Workflow employee context also reads this authoritative `cases` table into
`case_history`, including cases without HRIS notes. Each record contains the
original question, processing status, submission date and `source=cases`; it does
not turn generated recommendations into verified employee facts. The active case
is excluded from its own workflow history. Existing notes and seeded activities
remain supplementary context. Fresh reads observe committed records; rolled-back
inserts are absent, while committed failed/running cases retain those statuses.
Workflow context is a snapshot when its step runs; completed steps reused on a
retry retain their original snapshot. No new cache or history store is used.

The default profile requires configured employee credentials and denies access
when authentication is missing or misconfigured. To intentionally run the local
synthetic-data demo, set `ATLAS_ENV=development` on both API and Streamlit, then
send `X-Employee-ID` or the
`employee_id` query parameter on case reads/replays. Selection is not authentication.
Unscoped case reads are rejected. Do not expose demo mode for production access.

For authenticated access, use the default or set `ATLAS_ENV=production` on both API and Streamlit.
Set `ATLAS_EMPLOYEE_TOKEN_HASHES` on the API to a JSON object mapping employee IDs
to SHA-256 hashes of distinct, securely generated bearer tokens. Configuring this
mapping also enables authentication in development. Issue each employee only their
own token. Enter the token in Streamlit's password field or send
`Authorization: Bearer <token>` to the API. Generate tokens with Python's
`secrets.token_urlsafe(32)` and hash with `hashlib.sha256(token.encode()).hexdigest()`;
keep tokens private and out of source control and logs. Do not use employee IDs as
tokens. The API derives the authorized employee from the credential and rejects
conflicting header, query or submission IDs. The employee selector lists only that
identity. Production without a valid mapping fails closed with 503; missing/invalid
tokens receive 401. Both processes should receive consistent configuration. Only an explicit
`ATLAS_ENV=development` with no credential mapping enables anonymous demo access;
empty or unrecognized environment names never disable authentication.

This minimal credential mechanism has no SSO, credential enrollment, expiry or
self-service revocation; revoke/rotate credentials through server configuration and
restart the processes. Use HTTPS and protect configuration in a deployment.
Production identity-provider integration and deployment controls remain unverified.

The existing minimal access policy is employee self-access only: the server-mapped
bearer credential identifies the requester; a selected/header/body ID is the target,
and the saved case input determines case ownership. No HR-wide role, delegated
access, or administrator permission is defined. The UI token field only transports
a credential to FastAPI; it does not authenticate a Streamlit user by itself.


## Case audit foundation

Every handled API request receives a server-generated `X-Request-ID` response
header; client-supplied request IDs are ignored. Each retry has its own request
ID while retaining the same case ID. This is HTTP correlation, not a distributed
trace ID. Trusted direct workflow/tool calls have no HTTP request ID.

`GET /cases/{case_id}/audit` uses the same employee authentication and case-owner
filter as case details. It returns ordered `case_audit_events`, the current
`Case.status`, saved note IDs and safe policy fingerprints. Events contain only
internal case/request/note references, UTC timestamps, event/outcome names,
step/attempt identifiers and bounded error codes. Questions, employee IDs,
credentials, raw model output and policy text are excluded from this response.
The existing case-details path remains the authorized source for full records.

Case creation/resumption, step starts/results, local note creation and final
case outcomes add events to their corresponding SQLite transactions. Failure to
persist a required event fails/rolls back that transition. A confirmed note can
therefore remain visible even if a later acknowledgement or case write fails;
it does not prove that an external HR notification or other action occurred.
Optional failure logging is best effort and never replaces the durable outcome.
The additive table is created by existing `init_db()` startup, with no changes
or invented events for historical cases. The app does not expose event mutation.

Policy fingerprints identify cited or retrieved content, not approved policy
versions or proof of applicability. Citation filenames are retained when exact
selected content resolves uniquely to a current configured file; uncited retrieval
fingerprints still have no filename. Unknown `policy_id` and
`policy_version` remain null. `review_status` remains null because no human
approval transaction exists; `draft_quality_status=needs_hr_review` is distinct
from processing completion.

This is a local audit foundation. Events cover admitted workflow operations,
not every rejected/read HTTP request or provider invocation. Guarded local
interrupted runs can now be reconciled through the recovery path below.
`Case.status` remains authoritative; history is ordered by event ID. Policy
references describe stored outputs/retrieval snapshots, not an immutable
per-attempt policy archive. A complete database outage may prevent both a
transition and its failure event from being saved; no success is claimed.
Retention/deletion rules, audit-reader governance, tamper resistance, backups,
production identity/deployment controls and distributed tracing remain undefined
or unverified. Explicit development mode still permits synthetic selection and
must not be used for production records.


## Submission retries and interrupted cases

Streamlit manages submission keys automatically without displaying a key input.
Unchanged retries retain that key after timeouts, temporary API
outages, running results or completion. Changing the question starts a new
submission. Employee/token changes clear pending state. A new browser session does not
retain Streamlit state. The question form uses the default empathetic tone and
shows the selected employee's cases below, without tone or case-ID inputs.
The second **Benchmark** page queries live application-wide aggregates across all
employees, independent of the selected employee. Each persisted HR-question case
counts once, including failed and unfinished cases; idempotent retries do not
increase the denominator. Page loads, health checks and rejected requests without
a saved case are excluded. Audited HTTP reuse/resume request IDs provide a separate
recorded-retry count; earlier attempts without audit records cannot be reconstructed.
Each category percentage is its count divided by total submissions. The benchmark
score is reviewed correct answers plus reviewed appropriate no-information responses,
divided by all submissions. HTTP 200 and model confidence never establish accuracy.
Unreviewed/changed answers await review; failed workflows are counted separately.
The existing synthetic run file is only documented review evidence, matched by full
input, submission-key and output fingerprints with unchanged policy files. It does
not supply totals and does not automatically grade new employee answers.

Global analytics uses a **separate operator bearer token**, even in development.
Configure `ATLAS_ANALYTICS_TOKEN_HASH` with its SHA-256 hex digest on the API and
enter the corresponding token in the Benchmark page's masked **Analytics access
token** field. Employee tokens and the shared demo password do not grant this access.
Keep the private token outside tracked files. An absent/invalid configuration fails
closed with HTTP 503; missing/incorrect credentials return 401/403.
`GET /benchmarks/latest` returns only aggregate metrics and rules, never employee IDs,
question text, drafts or case IDs. It does not use employee scope or consume an
employee's daily quota. Access to case endpoints remains separately employee-authorized.

The stage table separates the **100-question / 10 requests-per-minute baseline**,
**350-question / 30 requests-per-minute expansion**, and subsequent **stress test**.
Unrun stages display **Not run** and no invented scores. Actual stage reports in
`data/benchmarks/stages.json` include workflow success, BM25 source/clause checks,
automated answer-contract checks, persistence consistency, p50/p95 latency,
schedule lag, completion throughput and a generation/application failure breakdown.
The hardware, model digest and unchanged retrieval/provider source fingerprints
are recorded. Sustainable capacity remains unknown; competing local workloads are
uncontrolled. A completed test run is not a claim that all test questions passed.

The initial fixture, `tests/fixtures/hr_baseline_100.json`, has 100 unique synthetic
questions spanning 25 fact/topic groups, four wording variants each. Expected
clauses are checked verbatim against all nine supplied policy files; this is not
HR approval or 100 independent policy facts. Four personal-balance probes have
no BM25 source-selection expectation, so retrieval uses 96 eligible questions.
Answer checks require authored facts/patterns and exact supporting citation clauses;
these are conservative mechanical contracts, separate from reviewed answer accuracy.
A separate standalone BM25 check is used even for deterministic routes that skip
the model/retriever in the end-to-end workflow. BM25 and Llama are not modified.

Run the first paced batch against the existing synthetic deployment with:
```
make benchmark BENCHMARK_ARGS="--stage baseline --output-dir /absolute/private/report-directory"
```
This takes about ten minutes, creates 100 independent synthetic cases and consumes
normal employee request quotas. `--database` must identify the exact API SQLite
file (verified read-only); `--api-base` and `--employees` select the deployment and
authorized synthetic employee IDs. Production-style access requires a matching
`ATLAS_EMPLOYEE_TOKEN`; one token cannot authorize three different employees.
Use one authorized employee or an explicitly configured synthetic demo environment.
The scheduler bounds in-flight requests (default eight), records schedule lag, and
reports observed throughput rather than assuming the target rate was achieved.
The 350-question stage requires a separately source-checked 350-question fixture;
stress runs require explicit `--requests` and `--rpm`. Investigate baseline failures
before escalating. Per-question reports remain in the specified local directory;
the authorized application endpoint exposes only aggregate stage results.
Before expansion, check the normal 200-per-day employee quotas, including other
HTTP traffic: a quota rejection is distinct from a model or hardware capacity limit.

The page refreshes every 15 seconds, on navigation, or using **Refresh benchmark**.
Its bar chart uses documented case-insensitive keyword/phrase rules for PTO, remote
work, benefits, expenses, leave and other/unknown. A question counts once in each
matching topic; mixed-topic questions may make the chart sum exceed the submission
total. This is approximate categorization, not an LLM or embedding relevance score.

API callers must retain their own key, since an unkeyed POST remains an intentional
new case. Changed inputs with a manually reused key still return409.

Retry the same keyed POST to recover an interrupted guarded case. Authorized
`POST /cases/{case_id}/recover` also preserves case identity for runs created
without a key. Both paths revalidate the employee and use the existing case
ownership boundary; explicit replay still creates a new case.

On a single host with file-backed SQLite, a nonblocking kernel file lock protects
each entire execution, including note and status commits. A live owner retains
the lock; process death releases it without relying on a timeout or PID guess.
A retry that cannot acquire the lock observes the current case instead of
starting competing work. The lock directory is `<resolved-database>.case-locks`;
keep it beside the database and never delete/replace its files while Atlas runs.
All processes using that database must use this execution protocol and one
canonical database path. This is not a distributed lease or network-filesystem
guarantee. Unsupported/unavailable guards require manual reconciliation.

Completed steps and the already-started workflow route are reused. Before the
note step starts, its backend and stable case-ID operation reference are committed
in existing case artifacts. A matching committed local note (same employee and
exact saved draft text) supplies the existing receipt; the write is not invoked
again. A missing note can be retried automatically only for the recorded,
queryable local SQLite backend. Unknown write outcomes, mismatched receipts,
backend changes, and legacy running cases without a verified guard return
`recovery=manual_reconciliation_required` with a safe reason, preserving their
actual running/failed status. Inspect the case/audit and reconcile the original
system before any further write; the API provides no force-replay override.

Recovery appends audit events and keeps the original Case ID and note history.
It is request-triggered, not a startup scanner or background worker. There is no
exactly-once claim for a future remote HRIS adapter: that adapter needs a verified
receipt lookup/idempotency contract before automatic retries are allowed. Policy,
prompt/model version changes, full historical snapshot replay, disk failure and
production deployment remain outside the verified local recovery contract.
