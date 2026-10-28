#!/usr/bin/env python3
"""Evaluate policy retrieval against independently authored synthetic expectations.

This small, fixed corpus check measures source selection and exact required-clause
coverage, not answer quality, legal applicability, or model accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools.retrieval import retrieve_policy_snippets


def evaluate(fixture_path, docs_dir, top_k=5):
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    documents = {path.stem: path.read_text(encoding="utf-8")
                 for path in sorted(docs_dir.glob("*.md"))}
    # Check the authored expected clauses against source text before measuring.
    # A fixture/source drift must not quietly become a retrieval regression.
    for item in fixture["questions"]:
        for passage in item["required_passages"]:
            if passage["text"] not in documents[passage["doc_id"]]:
                raise ValueError(f"Expected passage missing from source: {item['id']}")

    previous_docs = os.environ.get("DOCS_DIR")
    os.environ["DOCS_DIR"] = str(docs_dir)
    results = []
    try:
        for item in fixture["questions"]:
            started = time.perf_counter()
            snippets = retrieve_policy_snippets(item["question"], top_k=top_k)
            latency_ms = (time.perf_counter() - started) * 1000
            # The current contract is list[str], with complete sources or exact
            # source excerpts. Unknown/transformed text remains unidentified.
            retrieved = []
            for snippet in snippets:
                matches = [name for name, source in documents.items()
                           if snippet and snippet in source]
                retrieved.append(matches[0] if len(matches) == 1 else "<unidentified>")
            expected = item["expected_doc_ids"]
            acceptable = item.get("acceptable_doc_ids", expected)
            found = sorted(set(expected) & set(retrieved))
            missing_passages = [passage for passage in item["required_passages"]
                                if not any(passage["text"] in snippet and
                                           retrieved[index] == passage["doc_id"]
                                           for index, snippet in enumerate(snippets))]
            top1_correct = bool(retrieved and retrieved[0] in expected) if expected else not retrieved
            irrelevant = [name for name in retrieved if name not in acceptable]
            results.append({
                "id": item["id"], "question": item["question"],
                "category": item["category"], "expected_doc_ids": expected,
                "retrieved_doc_ids": retrieved, "relevant_sources_found": found,
                "top1_correct": top1_correct,
                "required_passage_count": len(item["required_passages"]),
                "missing_passages": missing_passages,
                "irrelevant_doc_ids": irrelevant,
                "passed": set(expected) <= set(retrieved) and not missing_passages and not irrelevant,
                "latency_ms": round(latency_ms, 3),
            })
    finally:
        if previous_docs is None:
            os.environ.pop("DOCS_DIR", None)
        else:
            os.environ["DOCS_DIR"] = previous_docs

    supported = [result for result in results if result["expected_doc_ids"]]
    unsupported = [result for result in results if not result["expected_doc_ids"]]
    expected_count = sum(len(result["expected_doc_ids"]) for result in results)
    found_count = sum(len(result["relevant_sources_found"]) for result in results)
    passage_count = sum(result["required_passage_count"] for result in results)
    missing_count = sum(len(result["missing_passages"]) for result in results)
    retrieved_count = sum(len(result["retrieved_doc_ids"]) for result in results)
    irrelevant_count = sum(len(result["irrelevant_doc_ids"]) for result in results)
    latencies = [result["latency_ms"] for result in results]
    return {
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest()
                          for name, source in documents.items()},
        "top_k": top_k,
        "limitations": [
            "Fixed synthetic examples; no statistical superiority claim.",
            "Sources have no authoritative approval, effective-date, or jurisdiction registry.",
            "Exact clause coverage measures retrieval output, not downstream prompt or answer quality.",
            "Latency is one local sequential pass, not a load benchmark.",
        ],
        "metrics": {
            "questions": len(results), "passed": sum(result["passed"] for result in results),
            "failed": sum(not result["passed"] for result in results),
            "supported_questions": len(supported), "unsupported_questions": len(unsupported),
            "relevant_document_recall_at_k": {"found": found_count, "expected": expected_count,
                                               "ratio": found_count / expected_count if expected_count else 1},
            "top1_accuracy_supported": {"correct": sum(result["top1_correct"] for result in supported),
                                         "total": len(supported)},
            "unsupported_abstention": {"correct": sum(not result["retrieved_doc_ids"] for result in unsupported),
                                         "total": len(unsupported)},
            "required_passage_coverage": {"found": passage_count - missing_count, "expected": passage_count},
            "irrelevant_retrieval": {"irrelevant": irrelevant_count, "retrieved": retrieved_count},
            "questions_missing_required_conditions": sum(bool(result["missing_passages"]) for result in supported),
            "latency_ms": {"mean": round(statistics.mean(latencies), 3),
                             "median": round(statistics.median(latencies), 3), "max": max(latencies)},
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/policy_retrieval.json")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "data/docs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true", help="Exit 1 if any fixed expectation fails")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be positive")
    report = evaluate(args.fixture, args.docs_dir, args.top_k)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 1 if args.require_all and report["metrics"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
