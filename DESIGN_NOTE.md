# Samsara Atlas Design Note

This note records identified risks in the existing Samsara Atlas application and the smallest proposed changes needed to address them. The application uses Streamlit for case intake, FastAPI and a workflow runner for processing, and a local SQLite database for synthetic employee records and case results.

## Identified Risks

### 1. Hardcoded employee fallback can hide employee lookup failures

**Severity:** Medium  
**Status:** Open

If the API is unavailable or returns no employees, Streamlit displays the same three employee IDs as a hardcoded fallback: `E10234`, `E10897`, and `E11342`. Seeing the dropdown alone therefore does not prove that the database connection is working.

**Evidence:** In `ui/streamlit_app.py`, the employee lookup requests `GET /employees`. A failed request, an unsuccessful HTTP response, or an empty employee list leads to the hardcoded fallback before the employee dropdown is displayed.

**Impact:** The dropdown can appear populated even when employee retrieval is failing or the database has not been seeded. Users may select an ID that has not been confirmed to exist in the connected database. A separate sidebar connectivity warning may appear, but the employee selector itself still looks usable.

**Proposed minimal change:** Remove the silent fallback. Show an explicit employee lookup error when the request fails, and a separate empty-state message when no employee records are returned. Disable case submission until employee records have been successfully loaded. Keep fixture data creation in `make seed`.

**Verification needed:** Check API connection failure, an unsuccessful employee endpoint response, an empty employee list, and a successful response. Confirm that only successfully retrieved employee IDs populate the selector and that submission is disabled in the failure and empty states.

## Important Engineering Requirements

The most important requirements are system controls and do not depend on which foundation model Atlas uses. The controls below describe the target design; listing them here does not mean they are all implemented in the current application.

| Requirement | Engineering control |
| --- | --- |
| Protect employee data | Corporate SSO, server-side authorization, and least-privilege IAM. |
| Avoid duplicate HR notes | Database uniqueness constraints and idempotent processing. |
| Handle model failures | Timeouts, bounded retries, explicit error states, and a review queue. |
| Prevent fabricated policy statements | Validate answers against supporting evidence and require human review for sensitive outcomes. |
| Control costs | Per-case token budgets, metering, alerts, and request quotas. |
| Scale safely | Fargate autoscaling, queue backpressure, and load testing. |
| Support audits | Versioned policies, trace IDs, case history, and privacy-safe logs. |

## Recommended Tool Stacks by Function

The following are provider-specific implementation options for Atlas capabilities. The AWS and Databricks columns are recommendations, not requirements or claims that these services are currently deployed. The underlying retrieval-augmented generation (RAG) architecture can remain the same regardless of provider.

| Atlas capability | AWS recommendation | Databricks recommendation |
| --- | --- | --- |
| Employee web interface | React / Next.js | React / Next.js or Databricks Apps |
| Backend APIs | FastAPI on ECS Fargate | FastAPI on Databricks Apps or AWS ECS |
| Employee authentication | Cognito or corporate SSO | Corporate SSO + Unity Catalog governance |
| Document storage | Amazon S3 | Delta Lake / Unity Catalog volumes |
| Document ingestion | Python workers + S3 events | Lakeflow pipelines / Spark |
| Keyword retrieval | OpenSearch BM25 | AI Search full-text |
| Semantic retrieval | OpenSearch vector search | AI Search ANN |
| Hybrid retrieval | OpenSearch hybrid queries | AI Search hybrid |
| Reranking | Bedrock Rerank | AI Search reranking |
| LLM inference | Amazon Bedrock | Mosaic AI Model Serving |
| Model experiments | SageMaker AI / MLflow | Managed MLflow |
| Logs and monitoring | CloudWatch + OpenTelemetry | MLflow traces + system tables |
| Load testing | k6 + CloudWatch | k6 + endpoint telemetry |
| Deployment automation | Terraform + GitHub Actions | Terraform + Databricks Asset Bundles |
| Secrets and encryption | Secrets Manager + KMS | AWS KMS + Databricks secrets/governance |

## Change Scope

This note documents the risk and proposed mitigation. The hardcoded employee fallback has not been changed as part of this documentation request.
