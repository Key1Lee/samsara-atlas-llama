# Atlas repository instructions

## Project purpose
Atlas is a local HR case-assistance assessment using synthetic employee data.
It supports intake, policy guidance, response drafts and case history; do not assume a complete production HR system.

## Architecture and navigation
Read the components affected by the task; this map is not a mandatory reading list.
- [ui/streamlit_app.py](ui/streamlit_app.py): employee selection, question form, case list/explorer and review warnings.
- [app/main.py](app/main.py), [app/api/routes.py](app/api/routes.py), [app/schemas.py](app/schemas.py): API startup, endpoints and input/output contracts.
- [app/api/access.py](app/api/access.py), [app/middleware/rate_limit.py](app/middleware/rate_limit.py): employee authorization and persistent daily request quotas.
- [app/engine/runner.py](app/engine/runner.py), [app/engine/steps.py](app/engine/steps.py): case/step persistence, retries and response routing.
- [app/tools/hris.py](app/tools/hris.py): local employee/history reads and case-note writes, not an external HRIS transaction service.
- [app/tools/retrieval.py](app/tools/retrieval.py), [data/docs/](data/docs/): BM25 policy retrieval from Markdown; relevance scores are not proof of applicability.
- [app/providers/](app/providers/), [app/prompts/](app/prompts/): model adapters (including stub/Ollama) and prompts.
- [app/storage/models.py](app/storage/models.py), [app/storage/database.py](app/storage/database.py): SQLAlchemy/SQLite records; `DATABASE_URL` selects storage.
- [data/seed.py](data/seed.py), [data/employee_fixture.json](data/employee_fixture.json): explicit fixture seeding, separate from question submission and test expectations.

The general workflow reads employee context and policies, summarizes, drafts, then saves a local case note.
Recognized PTO carryover/remote-relocation intents and questions with no retrieved policy take a deterministic response path that skips the model and HRIS tools.
Case ownership is currently stored in input JSON; backend queries filter it deterministically.
Production mode requires employee-mapped bearer credentials; local demo selection is not authenticated identity.
There is no HR approval transaction or outbound employee-message executor; action-claim checks are incomplete.

## Relevant documentation
- Read [README.md](README.md) for setup, API behavior, retry semantics, quotas and employee-access configuration.
- Read [CANDIDATE_PROMPT.md](CANDIDATE_PROMPT.md) for assessment scope and deliverables, not proof of implemented behavior.
- Read [DESIGN_NOTE.md](DESIGN_NOTE.md) when investigating recorded risks; it is historical and must be checked against current code/tests.
- Read [Makefile](Makefile), [requirements.txt](requirements.txt) and [.env.example](.env.example) for commands, dependencies and configuration changes.
- For Streamlit changes, consult the existing [.agents/skills/developing-with-streamlit/SKILL.md](.agents/skills/developing-with-streamlit/SKILL.md) when available after setup.
- No repository document currently defines BR-01 through BR-18; obtain the relevant definition/acceptance criteria from the task rather than inventing them.

## Engineering invariants
These are required correctness boundaries, not claims that every current path already enforces them.
- Authorize employee and case access in the backend; selected IDs and client headers are not proof of identity. Keep production access fail-closed.
- Reject invalid identities or report lookup failures explicitly; never silently substitute a sample employee.
- Keep employee claims, old drafts and fixture metadata distinct from authoritative HRIS facts.
- Use applicable, approved policy evidence and preserve qualifications/exceptions. An exact citation alone does not establish that a conclusion is supported.
- Missing or uncertain information requires clarification or HR review, never invented facts, policy decisions or eligibility.
- Show proposed actions as proposals. Claim completion only with a successful, relevant, verified execution record; a saved case note does not prove an email, notification or balance update.
- Keep workflow completion distinct from HR approval and permission to share an answer.
- Preserve retry/idempotency contracts: the same key and inputs reuse/resume the case; changed inputs conflict; explicit replay is a new case.
- Prevent duplicate business operations after failures/retries; preserve consistent case/step/note history and report unverified crash-recovery limitations.
- Keep employee content, model payloads and credentials out of diagnostic logs and public errors.

## Task execution: Understand → Verify → Change → Test → Review → Complete
1. **Understand:** Locate the relevant code and requirement; read only necessary context. Give each task one independently verifiable goal, allowing multiple files when needed.
2. **Verify:** Establish current behavior, reproduce defects where feasible and define the expected outcome. Separate proposed requirements, observed defects and historical reports.
3. **Change:** Make the smallest coherent fix to the root cause; preserve established contracts and unrelated user changes. For complex work, use a brief plan with measurable completion criteria; implement small tasks directly.
4. **Test:** Exercise the affected behavior and relevant failures, then broaden regression checks for shared contracts or changes spanning components.
5. **Review:** Inspect the diff for correctness, privacy, side effects and unnecessary complexity; use `git diff --check` for whitespace issues.
6. **Complete:** Finish authorized local work and fix failures introduced by it until acceptance criteria are met or a concrete blocker is established; stop at the task boundary.

## Risk prioritization
User-supplied planning priorities (BR definitions are not present in this repository):
- **P0:** BR-01, BR-02, BR-04, BR-14, BR-18.
- **P1:** BR-05, BR-07, BR-09, BR-11, BR-13, BR-15, BR-16.
- **P2:** BR-03, BR-06, BR-08, BR-12.
- **P3:** BR-10.
BR-17 has no supplied priority; do not infer one.
Sequence by critical safety/correctness dependencies, verified business impact, shared prerequisites, independent testability, then smallest coherent scope.
These groups guide planning, not a rigid order or permission to violate release safety. Do not automatically start at BR-18 or implement unrelated requirements.

## Testing and verification
Run commands from the repository root; [Makefile](Makefile) owns their definitions.
- Setup: `make setup`; seed synthetic demo employees only when needed with `make seed`.
- Run locally: `make api` (or `make api-ollama`) and `make ui` in separate terminals; check existing listeners before starting duplicates.
- Focused example: `.venv/bin/pytest tests/test_core.py -q`; full regression: `make test`.
- [tests/conftest.py](tests/conftest.py) selects a temporary SQLite database and deterministic stub; ordinary tests do not require Ollama or external credentials.
- The test database is shared within a run; quotas reset per test. Use isolated synthetic setup/unique IDs and avoid dependencies on test order or leftover cases.
- Prioritize independently runnable tests. Use concurrency tests to verify race behavior; parallel execution is a secondary optimization.
- Route checks by behavior: [tests/test_core.py](tests/test_core.py) for retries/persistence, [tests/test_employee_cases.py](tests/test_employee_cases.py) for ownership/auth, [tests/test_rate_limit.py](tests/test_rate_limit.py) for quotas, [tests/test_ui.py](tests/test_ui.py) for Streamlit/API integration.
- Use [tests/test_policy_intents.py](tests/test_policy_intents.py) and [tests/fixtures/policy_intents.json](tests/fixtures/policy_intents.json) for deterministic routing; [tests/test_policy_grounding.py](tests/test_policy_grounding.py) and [tests/test_generation_review.py](tests/test_generation_review.py) for evidence/action safeguards.
- Stub tests establish contracts, not live-model accuracy. Validate actual model behavior separately when the task requires it.
- `make load` runs [scripts/load_test.py](scripts/load_test.py), a bounded eight-attempt same-key correctness probe, not a capacity benchmark. It uses employee scope/`ATLAS_EMPLOYEE_TOKEN` and checks the matching SQLite file read-only; see README for `LOAD_ARGS` and explicit case/note/rejection expectations.

## Safety and scope boundaries
Investigate, make authorized scoped edits, run safe disposable local tests and repair introduced failures without repeated permission requests.
Obtain explicit authorization before production-data access, destructive migrations, deployment or irreversible external operations.
Preserve existing data and secrets; use synthetic fixtures and disposable databases for experiments.
Prefer the current architecture; add services, database systems, cloud infrastructure, agent frameworks, embeddings or abstractions only for a concrete requirement.
Avoid unrelated refactoring, unnecessary orchestration and extensive new documentation/skills.

## Completion criteria
Report what changed, tests actually executed and their results, acceptance criteria verified, and remaining blockers/security limitations.
Do not present planned work, historical test results or unavailable verification as current proof of success.
Keep this file a concise, durable map; link task-specific detail rather than duplicating it.

Guidance sources: [OpenAI AGENTS.md guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md), [context efficiency](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra), [durable review invariants](https://developers.openai.com/blog/custom-code-review-rules-for-codex).
