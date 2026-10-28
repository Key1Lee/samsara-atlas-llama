#!/usr/bin/env python3
"""Fixed adversarial JSON drafts test application controls, not live-model quality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine.context import ExecutionContext
from app.engine import steps
from app.providers.base import CompletionResponse
from app.tools.retrieval import retrieve_policy_snippets


def evaluate(fixture_path, docs_dir):
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    documents = {path.stem: path.read_text(encoding="utf-8")
                 for path in sorted(docs_dir.glob("*.md"))}
    for item in fixture["questions"]:
        for passage in item["applicable_evidence"]:
            if passage["text"] not in documents[passage["doc_id"]]:
                raise ValueError(f"Expected source passage drifted: {item['id']}")
    employee = json.loads((ROOT / "data/employee_fixture.json").read_text())["employees"][0]
    results = []
    for item in fixture["questions"]:
        calls = []

        class ControlledProvider:
            def complete(self, prompt, **kwargs):
                calls.append({"prompt_bytes": len(prompt.encode("utf-8")), "settings": kwargs})
                return CompletionResponse(json.dumps(item["controlled_model_response"]))

        with tempfile.TemporaryDirectory(prefix="atlas-answer-safety-") as directory:
            corpus = Path(directory)
            for path in sorted(docs_dir.glob("*.md")):
                shutil.copyfile(path, corpus / path.name)
            if item["corpus"] == "missing_pto":
                (corpus / "pto-policy.md").unlink()
            elif item["corpus"] == "conflicting_pto":
                source = (corpus / "pto-policy.md").read_text()
                (corpus / "pto-policy-conflict.md").write_text(source.replace("40 hours", "80 hours"))
            elif item["corpus"] != "default":
                raise ValueError("Unknown evaluation corpus variant")
            with patch.dict(os.environ, {"DOCS_DIR": str(corpus)}), \
                    patch.object(steps, "get_provider", lambda: ControlledProvider()):
                sources = retrieve_policy_snippets(item["question"])
                ctx = ExecutionContext(case_id=0, inputs={
                    "employee_id": employee["employee_id"], "case_transcript": item["question"], "tone": "direct",
                }, artifacts={}, step_outputs={
                    "retrieve_employee_context": {**employee, "found": True, "fact_verification": "unverified"},
                    "retrieve_policy_snippets": sources,
                    "summarize_and_extract_actions": {
                        "summary": "Synthetic policy question", "action_items": [], "next_steps": [],
                    },
                })
                output, error = {}, None
                try:
                    output = steps.generate_structured_response_step(ctx)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
        answerability = output.get("answerability")
        status = answerability.get("status") if isinstance(answerability, dict) else answerability
        text = output.get("subject", "") + "\n" + output.get("body", "")
        unsafe = {name: [pattern for pattern in item[field] if re.search(pattern, text, re.IGNORECASE)]
                  for name, field in (
                      ("answers", "disallowed_answer_patterns"),
                      ("recommendations", "disallowed_recommendation_patterns"),
                      ("completed_actions", "disallowed_completed_action_patterns"))}
        evidence = output.get("citation_evidence", [])
        missing = [passage for passage in item["applicable_evidence"]
                   if item["expected_answerability"] == "supported" and not any(
                       passage["text"] in entry.get("quote", "") and
                       entry.get("source_file") == passage["doc_id"] + ".md" for entry in evidence)]
        receipts_valid = output.get("execution_receipts", []) == item["expected_execution_receipts"]
        answerability_correct = status == item["expected_answerability"]
        results.append({
            "id": item["id"], "expected_answerability": item["expected_answerability"],
            "actual_answerability": answerability, "answerability_correct": answerability_correct,
            "unsafe_statements": unsafe, "unexpected_execution_receipt": not receipts_valid,
            "missing_supported_passages": missing, "provider_calls": len(calls),
            "provider_call_details": calls, "error": error,
            "blocked_invalid_output": bool(error), "output": output,
            "passed": answerability_correct and not any(unsafe.values()) and receipts_valid and not missing and not error,
        })

    def rate(field, name):
        challenged = [(item, result) for item, result in zip(fixture["questions"], results) if item[field]]
        return {"unsafe": sum(bool(result["unsafe_statements"][name]) for _, result in challenged),
                "total": len(challenged)}

    ambiguous = [result for item, result in zip(fixture["questions"], results)
                 if item["expected_answerability"] == "clarification_needed"]
    return {
        "fixture_revision": fixture["revision"],
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in documents.items()},
        "provider_configuration": fixture["provider_configuration"],
        "limitations": fixture["limitations"] + [
            "Direct response-step probes use real retrieval and synthetic employee context, without case persistence or HRIS writes.",
            "Review flags alone do not neutralize unsafe body text; explicit expected forbidden statements are checked independently.",
            "An invalid-output exception is safe blocking for these drafts, but does not earn correct answerability or overall success.",
        ],
        "metrics": {
            "questions": len(results), "passed": sum(result["passed"] for result in results),
            "failed": sum(not result["passed"] for result in results),
            "correct_answerability": {"correct": sum(result["answerability_correct"] for result in results), "total": len(results)},
            "unsupported_answer_rate": rate("disallowed_answer_patterns", "answers"),
            "unsupported_recommendation_rate": rate("disallowed_recommendation_patterns", "recommendations"),
            "false_completed_action_claim_rate": rate("disallowed_completed_action_patterns", "completed_actions"),
            "correct_ambiguity": {"correct": sum(result["answerability_correct"] for result in ambiguous), "total": len(ambiguous)},
            "invalid_output_rejections": sum(result["blocked_invalid_output"] for result in results),
            "provider_calls": sum(result["provider_calls"] for result in results),
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/answer_safety.json")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "data/docs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.fixture, args.docs_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 1 if args.require_all and report["metrics"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
