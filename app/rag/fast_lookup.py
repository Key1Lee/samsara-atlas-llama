"""Fast, extractive policy overview. Does not invoke Ollama or mutate cases."""
from __future__ import annotations

import re
from time import perf_counter

from app.rag.config import Settings
from app.rag.documents import allowed, load_sources, sections, verify_span
from app.rag.errors import SourceUnauthorized
from app.tools.retrieval import _tokenize, contains_policy_override


def _terms(value: str) -> set[str]:
    return set(_tokenize(value))


def lookup_policy_passage(question: str, user_context: dict | None = None) -> dict:
    """Return a verified verbatim policy passage only for an unambiguous title match.

    Unknown or ambiguous questions are deliberately not answered by guessing.
    Authorization is checked before returning any source metadata or passage.
    """
    started = perf_counter()
    settings = Settings()
    question_terms = _terms(question)
    if not question_terms:
        return {"status": "no_match", "passage": None,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2)}
    candidates = []
    for doc in load_sources(settings):
        title = doc.metadata["document_title"]
        title_terms = _terms(title)
        if not title_terms:
            continue
        shared = question_terms & title_terms
        # Two-word policy names require meaningful overlap; short names like
        # PTO must match their distinctive term. Never guess from 'policy'.
        coverage = len(shared) / len(title_terms)
        if (len(title_terms) == 1 and coverage == 1) or (
            len(title_terms) > 1 and len(shared) >= 2 and coverage >= 0.66
        ):
            candidates.append((coverage, doc))
    if not candidates:
        return {"status": "no_match", "passage": None,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2)}
    candidates.sort(key=lambda item: -item[0])
    best = candidates[0][0]
    winners = [doc for score, doc in candidates if score == best]
    # Ties can reflect multiple policies, and must not be arbitrarily resolved.
    if len(winners) != 1:
        return {"status": "ambiguous", "passage": None,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2)}
    doc = winners[0]
    if not allowed(doc.metadata, user_context):
        raise SourceUnauthorized()
    spans = sections(doc)
    # A general "what is this policy?" request is answered with the original
    # text, not a generated interpretation. No inference of entitlement.
    passage = doc.page_content
    metadata = {**doc.metadata, "span_start": 0, "span_end": len(passage)}
    verify_span(settings, metadata, passage, user_context)
    if contains_policy_override(passage):
        # Do not surface or execute suspected application-control instructions.
        return {"status": "unsafe_source", "passage": None,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2)}
    return {
        "status": "matched",
        "title": doc.metadata["document_title"],
        "passage": passage,
        "source_file": doc.metadata["source_path"],
        "policy_id": doc.metadata["document_id"],
        "policy_version": doc.metadata.get("policy_version"),
        "effective_date": doc.metadata.get("effective_date"),
        "approval_status": doc.metadata.get("approval_status", "unknown"),
        "citation": {"source_file": doc.metadata["source_path"],
                     "start": 0, "end": len(passage)},
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        "answer_type": "verbatim_policy",
        "llm_called": False,
    }
