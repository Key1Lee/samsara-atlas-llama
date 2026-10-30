"""Regression tests for the read-only policy lookup route."""
import pytest

from app.rag.fast_lookup import lookup_policy_passage


@pytest.fixture
def policies(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "pto-policy.md").write_text(
        "# Paid Time Off (PTO) Policy\n\nSamsarans accrue PTO based on tenure.\n"
        "\n## Requesting\nManager approval is required.\n", encoding="utf-8"
    )
    (docs / "expense-policy.md").write_text(
        "# Expense Reimbursement Policy\n\nEligible business expenses can be reimbursed.\n"
        "\n## Receipts\nFollow approved receipt rules.\n", encoding="utf-8"
    )
    monkeypatch.setenv("DOCS_DIR", str(docs))
    monkeypatch.delenv("ATLAS_POLICY_METADATA", raising=False)
    return docs


@pytest.mark.parametrize("question,expected", [
    ("What is the Expense Reimbursement Policy?", "expense-policy.md"),
    ("What is my paid time policy?", "pto-policy.md"),
    ("What is Paid Time Off (PTO) Policy?", "pto-policy.md"),
])
def test_direct_lookup_is_extractive(policies, question, expected):
    result = lookup_policy_passage(question, {"employee_id": "EMP-1"})
    assert result["status"] == "matched"
    assert result["source_file"] == expected
    assert result["llm_called"] is False
    original = (policies / expected).read_text(encoding="utf-8")
    assert original[result["citation"]["start"]:result["citation"]["end"]] == result["passage"]


def test_unknown_topic_abstains(policies):
    result = lookup_policy_passage("What is the moon travel policy?")
    assert result["status"] == "no_match"
    assert result["passage"] is None
