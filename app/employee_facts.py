"""The local fixture store has no approved PTO balance source contract."""
import re


def personal_pto_question(question: str) -> bool:
    text = question.lower()
    return bool((re.search(r"\b(?:pto|paid time off)\b", text) and
                 re.search(r"\b(?:balance|remaining|left|i have|do i have)\b", text)) or
                (re.search(r"\b\d+(?:\.\d+)? hours?\b", text) and
                 re.search(r"\b(?:remaining|left|carry|roll)\b", text)))


def personal_balance_claim(text: str) -> bool:
    """Bounded known personal-balance wording; not a general semantic verifier."""
    text = " ".join(text.lower().split())
    return bool(re.search(r"\b(?:pto|paid time off)\b", text) and re.search(
        r"\b(?:your|verified|official|current)\b.{0,60}\bbalance\b.{0,30}\d"
        r"|\byou have\b.{0,40}\d+(?:\.\d+)? (?:hours?|days?)\b", text))


def unavailable_pto_balance(employee_id: str) -> dict:
    return {"status": "unavailable", "reason": "authoritative_source_not_configured",
            "employee_id": employee_id, "value": None, "units": None,
            "as_of": None, "source": None}
