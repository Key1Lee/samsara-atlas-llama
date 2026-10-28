import os
import re
import hashlib
from pathlib import Path

from rank_bm25 import BM25Okapi

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Topic routing only: these common words cannot establish policy applicability.
# Missing/ambiguous language can abstain; HR still reviews legal applicability.
_COMMON = frozenset("a an and are as at be been by can company employee employees for from has have how hr i in is it me my of on or our policy policies samsara samsarans the their them they this to us we what when where which who why will with would your all overview eligibility process duration coordination other leave time off work day days question please".split())

# These source-heading words describe generic workflow/grammar, not a policy
# category. For example, 'do' in 'Do Not Self-Determine' cannot route bereavement.
_TOPIC_COMMON = frozenset("do not non self determine requesting eligible workplace".split())
_ALIASES = {"remotely": "remote", "vacation": "pto"}


def _load_docs(docs_dir: str) -> list[tuple[str, str]]:
    docs_path = Path(docs_dir).resolve()
    if not docs_path.exists():
        return []
    pairs = []
    for path in sorted(docs_path.glob("*.md")):
        try:
            if not path.resolve().is_relative_to(docs_path):
                continue
            pairs.append((path.stem, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError):
            continue
    return pairs


def policy_source_reference(source: str) -> dict:
    """Link an exact selected snapshot to current configured files, not model IDs."""
    docs_dir = os.getenv("DOCS_DIR", str(_PROJECT_ROOT / "data" / "docs"))
    names = [name + ".md" for name, text in _load_docs(docs_dir) if text == source]
    return {
        "source_id": "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_file": names[0] if len(names) == 1 else None,
        "source_verification": "matched" if len(names) == 1 else "ambiguous" if names else "unavailable",
        "policy_id": None, "policy_version": None, "effective_date": None,
        "approval_status": "unknown", "applicability": "unverified",
    }


def policy_topic_matches(query: str, source: str) -> bool:
    """Lexical relevance only; this cannot establish legal applicability/support."""
    return bool(set(_tokenize(query)) & _topic_tokens(source))


def contains_policy_override(text: str) -> bool:
    """Recognize explicit application-control instructions, not general policy rules.

    This narrow rejection check is not a general prompt-injection detector.
    """
    text = " ".join(text.split())
    return bool(re.search(
        r"\b(?:ignore|override|disregard)\b.{0,80}\b(?:instructions?|rules?|prompts?|system|developer)\b"
        r"|\b(?:disable|bypass)\b.{0,60}\b(?:authentication|authorization|access controls?)\b"
        r"|\b(?:set|mark)\b.{0,60}\bquality_status\b",
        text, flags=re.IGNORECASE,
    ))


def _tokenize(text: str) -> list[str]:
    # Normalize punctuation and ordinary plurals, without introducing embeddings.
    text = re.sub(r"\bpaid\s+time\s+off\b", "pto", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwork(?:ing)?\s+from\s+home\b", "remote", text, flags=re.IGNORECASE)
    # 'Medical documentation' alone cannot establish a medical-leave topic.
    text = re.sub(r"\b(medical|personal)[\s-]+leave\b", lambda m: m[1].lower() + "leave", text, flags=re.IGNORECASE)
    # Relocation in a state/location context is distinct from moving expenses.
    text = re.sub(r"\b(?:mov(?:e|ing)|relocat(?:e|ing|ion))\b(?=.{0,40}\bstates?\b)", "remote", text, flags=re.IGNORECASE)
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_ALIASES.get(word, word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 4 else word)
            for word in words if word not in _COMMON and not word.isdigit()]


def _topic_tokens(content: str) -> set[str]:
    # Use source-authored headings and its introduction as a conservative topic
    # gate. A shared word deep in an unrelated provision must not route a case.
    introduction = content.split("\n## ", 1)[0]
    headings = "\n".join(line for line in content.splitlines() if line.startswith("#"))
    return set(_tokenize(introduction + "\n" + headings)) - _TOPIC_COMMON


def retrieve_policy_snippets(query: str, top_k: int = 5) -> list[str]:
    """Return complete small policy sources, gated by lexical topic evidence.

    BM25 ranks candidates; it does not prove entitlement or legal applicability.
    No topic match returns an empty list for explicit HR review. Sources are read
    fresh so edits and deterministic tie ordering are reflected in every request.
    """
    if top_k <= 0:
        return []
    docs_dir = os.getenv("DOCS_DIR", str(_PROJECT_ROOT / "data" / "docs"))
    pairs = _load_docs(docs_dir)
    if not pairs:
        return []
    query_tokens = _tokenize(query)
    query_terms = set(query_tokens)
    if not query_terms:
        return []
    corpus = [_tokenize(content) for _, content in pairs]
    # Rank in the existing architecture. The topic gate, rather than a positive
    # BM25 score alone, excludes common-word matches from unrelated sources.
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    candidates = [(float(score), name, content)
                  for score, (name, content) in zip(scores, pairs)
                  if query_terms & _topic_tokens(content)]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    # Metadata is collected only inside an explicitly enabled observation context.
    from app.evaluation.observation import record_retrieval
    record_retrieval([{'source_id': 'sha256:' + hashlib.sha256(content.encode()).hexdigest(),
                      'chunk_id': 'sha256:' + hashlib.sha256(content.encode()).hexdigest() + ':whole',
                      'rank': rank, 'score': score, 'span_start': 0, 'span_end': len(content)}
                     for rank, (score, _, content) in enumerate(candidates[:top_k], 1)])
    return [content for _, _, content in candidates[:top_k]]
