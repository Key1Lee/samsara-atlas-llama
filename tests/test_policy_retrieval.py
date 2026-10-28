"""Source-selection regressions, independent of models, employees and case state."""
import json
from pathlib import Path

import pytest

from app.tools.retrieval import retrieve_policy_snippets


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data/docs"
FIXTURE = json.loads((ROOT / "tests/fixtures/policy_retrieval.json").read_text())
# The full evaluation retains pet-benefit and relocation-reimbursement questions
# as visible answerability gaps. This slice covers source selection, vocabulary,
# unrelated generic headings, and the existing unsupported-pet contract.
SOURCE_SELECTION_CASES = [
    item for item in FIXTURE["questions"]
    if item["expected_doc_ids"] or item["id"] in {"unknown_iguana", "unknown_bereavement"}
]


@pytest.mark.parametrize("item", SOURCE_SELECTION_CASES, ids=lambda item: item["id"])
def test_policy_sources_and_complete_conditions(item, monkeypatch):
    monkeypatch.setenv("DOCS_DIR", str(DOCS))
    documents = {path.stem: path.read_text() for path in DOCS.glob("*.md")}
    # Expectations were authored from the Markdown, not retrieval results.
    for passage in item["required_passages"]:
        assert passage["text"] in documents[passage["doc_id"]]

    snippets = retrieve_policy_snippets(item["question"])
    retrieved = []
    for snippet in snippets:
        matching_ids = [name for name, source in documents.items() if snippet and snippet in source]
        assert len(matching_ids) == 1, "Returned evidence must identify one exact policy source"
        retrieved.append(matching_ids[0])

    assert set(item["expected_doc_ids"]) <= set(retrieved)
    assert set(retrieved) <= set(item.get("acceptable_doc_ids", item["expected_doc_ids"]))
    for passage in item["required_passages"]:
        assert any(passage["text"] in snippet and retrieved[index] == passage["doc_id"]
                   for index, snippet in enumerate(snippets)), "Required condition was dropped"


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_depth_returns_no_sources(top_k, monkeypatch):
    monkeypatch.setenv("DOCS_DIR", str(DOCS))
    assert retrieve_policy_snippets("PTO carryover", top_k=top_k) == []


def test_missing_policy_directory_returns_no_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "missing"))
    assert retrieve_policy_snippets("PTO carryover") == []


def test_existing_location_change_heading_still_routes_to_remote_policy(monkeypatch):
    monkeypatch.setenv("DOCS_DIR", str(DOCS))
    assert retrieve_policy_snippets("How do I request a location change?") == [(DOCS / "remote-work-policy.md").read_text()]


def test_equal_scores_have_stable_source_order_and_respect_depth(tmp_path, monkeypatch):
    # Numeric differences are irrelevant to the existing tokenizer, so these
    # sources have equal BM25 scores. Creation order must not determine ranking.
    second = "# Travel Policy\n- Travel allowance is 222.\n"
    first = "# Travel Policy\n- Travel allowance is 111.\n"
    (tmp_path / "z-policy.md").write_text(second)
    (tmp_path / "a-policy.md").write_text(first)
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    assert retrieve_policy_snippets("Travel allowance") == [first, second]
    assert retrieve_policy_snippets("Travel allowance", top_k=1) == [first]
    assert retrieve_policy_snippets("Travel allowance") == [first, second]
