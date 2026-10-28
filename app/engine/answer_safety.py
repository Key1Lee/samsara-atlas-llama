"""Small application-owned answer decisions, separate from model assertions."""
import re


def answerability(response: dict, status: str, reason: str) -> dict:
    return {**response, "answerability": {"status": status, "reason": reason},
            "recommendation_review": recommendation_review(), "operation_status": no_operations()}


def no_operations() -> dict:
    return {"case_note": {"status": "not_attempted", "attempted": False, "receipt": None},
            "external_actions": {"status": "not_attempted", "supported_operations": []}}


def unverified_completion_claim(text: str) -> bool:
    """Additional known passive/elided claims; not a semantic proof for prose."""
    claim = re.compile(
        r"\b(?:email|message|notification|meeting|appointment|request|balance|hr|manager|you|case note)\b"
        r"[^.!?]{0,70}\b(?:is|are|was|were|has been|have been|has|have) "
        r"(?:sent|notified|scheduled|booked|approved|processed|updated|increased|decreased|adjusted|changed|saved|cleared)\b"
        r"|\b(?:pto|balance)\b[^.!?]{0,40}\b(?:increased|decreased|updated|changed)\b"
        r"|\b(?:hr|manager|team)\b[^.!?]{0,40}\b(?:received|knows|is aware)\b"
        r"|\b(?:notification|email|message)\b[^.!?]{0,40}\b(?:went|reached|arrived)\b"
        r"|^(?:successfully |already )?(?:booked|scheduled|sent|notified|updated|approved|processed|saved)\b"
    )
    for sentence in re.split(r"[.!?\n]+", text.lower()):
        sentence = " ".join(sentence.split())
        # A condition is not evidence of completion. Still check its consequence
        # and all following sentences for independent completion claims.
        if sentence.startswith("if ") and "," in sentence:
            sentence = sentence.split(",", 1)[1].strip()
        if (re.fullmatch(r"ask (?:hr|your hr partner) whether [^,;]+", sentence) and
                not re.search(r"\b(?:and|then|but)\b", sentence)):
            continue
        if claim.search(sentence):
            return True
    return False


def recommendation_review() -> dict:
    return {"status": "proposed", "review": "required", "authorization": "not_granted",
            "employee_facts": "unverified"}


def unsafe_recommendation(text: str) -> bool:
    """Reject known unsupported permissions/facts, not all free-text semantics."""
    text = " ".join(text.lower().split())
    return bool(re.search(
        r"\b(?:increase|decrease|adjust|update|change|set|add|remove|credit)\w*\b.{0,100}\b(?:pto|balance)\b"
        r"|\b(?:pto|balance)\b.{0,80}\b(?:adjustment|increase|change)\b.{0,30}\bauthorized\b"
        r"|\b(?:skip|bypass|ignore)\b.{0,80}\b(?:interactive process|approval|hr review|manager review)\b"
        r"|\b(?:hr|manager) review is not (?:needed|required)\b"
        r"|\byou (?:are|qualify as) (?:authorized|approved|eligible)\b"
        r"|\byour\b.{0,100}\b(?:is|confirms)\b.{0,30}\b(?:authorized|approved|eligible)\b"
        r"|^(?:automatically )?(?:approve|authorize)\b.{0,100}\b(?:accommodation|leave|request)\b",
        text,
    ))


def reviewed_summary(data: dict) -> dict:
    proposal = "Ask an authorized HR partner to verify employee facts, applicable policy, eligibility and required approvals before proposing any change."
    result = {**data, "recommendation_review": recommendation_review()}
    for field in ("action_items", "next_steps"):
        result[field] = [proposal if unsafe_recommendation(item) else item for item in data[field]]
    if unsafe_recommendation(data["summary"]):
        result["summary"] = "The employee's reported request needs HR verification; employee facts, eligibility and approval are unverified."
    return result


def question_review(question: str) -> tuple[str, str] | None:
    """Negative coverage/clarification checks for the current small policy set.

    These checks can withhold an answer; they never establish positive support.
    Supported decisions require separately validated source clauses. The supplied
    policies do not establish pets or relocation reimbursement entitlements.
    """
    text = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    pet = re.search(r"\b(?:pets?|veterinary|iguana|snake|bees|dogs?|cats?)\b", text)
    if pet and re.search(r"\b(?:bring|keep|allow\w*|permit\w*|insurance|benefits?|cover\w*)\b", text):
        return "hr_review_required", "unsupported_policy_topic"
    if (re.search(r"\b(?:relocat\w*|moving|move)\b", text) and
            re.search(r"\b(?:expenses?|costs?|reimburs\w*|pay|paid|payment)\b", text)):
        return "hr_review_required", "insufficient_policy_evidence"
    words = set(text.split())
    if re.fullmatch(r"(?:(?:can|may|should|could) (?:i|we) (?:do|take|use|have|get) (?:it|this|that)"
                    r"|is (?:it|this|that) (?:allowed|approved|possible))(?: (?:now|today|please))?", text):
        return "clarification_needed", "ambiguous_question"
    if (words & {"pto", "vacation", "benefits"} and
            not re.search(r"\b(?:accru\w*|carry\w*|roll\w*|balanc\w*|payout|request\w*|approv\w*|"
                          r"expir\w*|forfeit\w*|hours?|unused|advance|health|enroll\w*|retire\w*|match|coverage)\b", text)):
        return "clarification_needed", "ambiguous_question"
    return None
