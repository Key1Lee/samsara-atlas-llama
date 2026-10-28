import json
import logging
import os
import re
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.diagnostics import diagnosed, OutputRuleError, completion_status
from app.providers.factory import get_provider
from app.providers.base import ContextBudgetExceeded, require_prompt_budget
from app.providers.errors import ProviderInvalidOutput
from app.tools.registry import get_tool_registry
from app.tools.retrieval import _load_docs, policy_source_reference, policy_topic_matches, contains_policy_override
from app.schemas import StructuredResponse, ExtractActionsOutput
from app.engine.answer_safety import (answerability, question_review, recommendation_review, unsafe_recommendation,
                                      reviewed_summary, no_operations, unverified_completion_claim)

from app.employee_facts import unavailable_pto_balance, personal_pto_question, personal_balance_claim

logger = logging.getLogger(__name__)

def classify_policy_intent(question: str) -> str | None:
    """Conservative lexical routing for carryover and relocation policy only."""
    words = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    pto = bool(re.search(r"\b(?:pto|paid time off)\b", words) and
               re.search(r"\b(?:carryover|carry over|carries over|rollover|roll over)\b", words))
    moving = bool(re.search(r"\b(?:relocate|relocating|relocation|move|moving)\b", words))
    remote = moving and bool(re.search(r"\b(?:remote|remotely|state|states)\b", words))
    if pto == remote:
        return None
    return "PTO_POLICY" if pto else "REMOTE_WORK"


def extract_policy_statements(intent: str, source: str) -> list[str]:
    """Verify known rules, then preserve their whole governing sections."""
    bullets = re.findall(r"(?m)^- (.+(?:\n[ \t]+[^\n]+)*)", source)
    if intent == "PTO_POLICY":
        rules = [r"unused pto.*carr(?:y|ies) over.*\d+(?:\.\d+)? hours",
                 r"carryover.*forfeit.*unless.*local law.*payout"]
    elif intent == "REMOTE_WORK":
        rules = [r"roles designated remote.eligible.*approved us zone",
                 r"notify hr before relocating.*payroll tax withholding.*benefits eligibility.*compensation zone"]
    else:
        raise ValueError("Unsupported policy intent")
    selected = []
    for rule in rules:
        matches = [b for b in bullets if re.search(rule, " ".join(b.split()), re.IGNORECASE)]
        # Ambiguous or incomplete provisions cannot establish a verified rule.
        if len(matches) != 1:
            return []
        selected.append(matches[0])
    headings = {"PTO_POLICY": ["Accrual"], "REMOTE_WORK": ["Eligible Roles", "Location Changes"]}[intent]
    sections = []
    for heading in headings:
        matches = re.findall(r"^##[ \t]+" + re.escape(heading) + r"[ \t]*\r?\n(.*?)(?=^#{1,2}[ \t]|\Z)",
                             source, flags=re.MULTILINE | re.DOTALL)
        if len(matches) != 1 or not matches[0].strip():
            return []
        sections.append(matches[0].strip())
    if not all(any(bullet in section for section in sections) for bullet in selected):
        return []
    return sections


def _policy_review(tone: str, reason: str) -> dict:
    return {**unsupported_policy_response(tone), "subject": "Policy needs HR verification",
            "answerability": {"status": "hr_review_required", "reason": "policy_evidence_uncertain"},
            "body": "The governing policy could not be verified. " + reason +
                    " Ask HR to clarify the applicable policy and jurisdiction before making a decision."}


def deterministic_policy_response(intent: str, tone: str, *, selected_source: str | None = None) -> dict:
    filename = {"PTO_POLICY": "pto-policy.md", "REMOTE_WORK": "remote-work-policy.md"}[intent]
    docs_dir = Path(os.getenv("DOCS_DIR", str(Path(__file__).resolve().parents[2] / "data/docs")))
    try:
        documents = _load_docs(str(docs_dir))
        source = selected_source if selected_source is not None else next(
            (text for name, text in documents if name == Path(filename).stem), "")
        if contains_policy_override(source):
            return _policy_review(tone, "The source contains application-control instructions and needs verification.")
        # With no authoritative version/effective-date registry, filenames must
        # not silently choose between distinct revisions of the same policy.
        title = source.splitlines()[0].strip() if source.splitlines() else ""
        siblings = {text for _, text in documents
                    if text.splitlines() and text.splitlines()[0].strip() == title}
        if len(siblings | {source}) > 1:
            return _policy_review(tone, "Conflicting or changed policy sources require review.")
        quotes = extract_policy_statements(intent, source)
    except (OSError, UnicodeError):
        source, quotes = "", []
    if not quotes:
        body = "The requested policy rule could not be verified from the designated source. HR review is needed to clarify the rule."
    else:
        lead = {"empathetic": "Here is the policy guidance for your question: ",
                "direct": "The policy states: ", "formal": "The applicable source states: "}[tone]
        body = lead + ". ".join(" ".join(q.split()).rstrip(".") for q in quotes) + "."
        if intent == "PTO_POLICY":
            body += " This does not verify your personal PTO balance or determine which local law applies. Ask HR to verify jurisdiction and eligibility."
        else:
            body += " Notifying HR does not automatically authorize relocation; ask HR to confirm the applicable requirements."
    response = StructuredResponse(subject={"PTO_POLICY": "PTO carryover policy", "REMOTE_WORK": "Remote work relocation policy"}[intent],
                                  body=body, tone=tone, cited_snippets=quotes)
    reference = policy_source_reference(source) if quotes else {}
    evidence = [{**reference, "quote": q, "start": source.index(q),
                 "end": source.index(q) + len(q)} for q in quotes]
    result = {**response.model_dump(), "citation_evidence": evidence,
              "quality_status": "needs_hr_review", "execution_receipts": []}
    if quotes and reference.get("source_verification") == "matched":
        return answerability(result, "supported", "policy_evidence_available")
    return answerability(result, "hr_review_required", "missing_policy_evidence")


def _preserve_carryover_conditions(ctx, response: dict) -> dict:
    """Known PTO rule only; never infer legal applicability from model prose."""
    question = " ".join(re.findall(r"[a-z0-9]+", ctx.inputs.get("case_transcript", "").lower()))
    pto = re.search(r"\b(?:pto|paid time off|vacation)\b", question)
    carryover = re.search(r"\b(?:carry over|carries over|carryover|roll over|rollover|forfeit|expire|next year|year end)\b", question)
    draft_rule = re.search(r"\b(?:carryover|carry over|carries over|rollover|forfeit\w*)\b",
                           response["subject"] + " " + response["body"], re.IGNORECASE)
    sources = [source for source in ctx.step_outputs.get("retrieve_policy_snippets", [])
               if source.splitlines() and source.splitlines()[0].strip() == "# Paid Time Off (PTO) Policy"]
    if not ((pto and carryover) or (sources and draft_rule)):
        return response
    if re.search(r"\b(?:remote|remotely|relocat\w*|moving states?)\b", question):
        return _policy_review(ctx.inputs["tone"], "This question spans multiple policy topics; HR must review all applicable conditions.")
    if len(set(sources)) != 1:
        return _policy_review(ctx.inputs["tone"], "Missing or conflicting PTO evidence requires review.")
    # Keep the existing general workflow, but replace this known legal-sensitive
    # rule with complete conditional source guidance, not an LLM determination.
    return deterministic_policy_response("PTO_POLICY", ctx.inputs["tone"], selected_source=sources[0])


def unsupported_policy_response(tone: str) -> dict:
    """Give a fixed HR referral when no available policy matches the question."""
    return {
        "subject": "Please check with an HR Representative",
        "body": (
            "We don't have verified information about this question in the available HR policies. "
            "Please reach out to an HR Representative, especially for edge cases."
        ),
        "tone": tone,
        "cited_snippets": [],
        "citation_evidence": [],
        "quality_status": "needs_hr_review",
        "execution_receipts": [],
        "answerability": {"status": "hr_review_required", "reason": "unsupported_policy_topic"},
        "recommendation_review": recommendation_review(),
        "operation_status": no_operations(),
    }


def _load_prompt(name: str) -> str:
    """Load prompt from file - no content hash; replay uses current file (prompt drift, see P010)."""
    path = Path(__file__).parent.parent / "prompts" / f"{name}.txt"
    return path.read_text()


def _parse_json_from_text(text: str) -> dict:
    from app.evaluation.observation import record_parse
    try:
        value = _parse_json_impl(text)
    except (ValueError, TypeError):
        record_parse(success=False)
        raise
    record_parse(value, success=True)
    return value


def _parse_json_impl(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        lines = text.split("\n")
        in_block = False
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if not in_block:
                    in_block = True
                    start_idx = i + 1
                else:
                    text = "\n".join(lines[start_idx:i])
                    break
        else:
            text = "\n".join(lines[start_idx:])
    try:
        return _json_object(json.loads(text))
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return _json_object(json.loads(text[start:end]))
        raise


def _json_object(data):
    if not isinstance(data, dict):
        raise OutputRuleError("json_object_required", "Expected a JSON object")
    return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
)
def _retryable_llm_call(prompt: str, **kwargs):
    require_prompt_budget(prompt)
    provider = get_provider()
    completion_status.set("requested")
    schema = StructuredResponse if "tone" in kwargs else ExtractActionsOutput
    resp = provider.complete(prompt, output_schema=schema.model_json_schema(), **kwargs)
    completion_status.set(getattr(resp, "completion_status", "unknown"))
    if not isinstance(resp.text, str) or not resp.text.strip():
        raise ProviderInvalidOutput()
    return resp.text


def retrieve_employee_context_step(ctx):
    registry = get_tool_registry()
    out = registry.execute("retrieve_employee_context", employee_id=ctx.inputs["employee_id"],
                           exclude_case_id=ctx.case_id)
    if not out.get("found"):
        raise ValueError("Employee not found in HRIS")
    if out.get("employee_id") != ctx.inputs["employee_id"]:
        raise ValueError("Employee context identity mismatch")
    # The fixture has no authoritative balance/provenance contract.
    return {**out, "fact_verification": "unverified",
            "pto_balance": unavailable_pto_balance(ctx.inputs["employee_id"])}


def retrieve_policy_snippets_step(ctx):
    registry = get_tool_registry()
    out = registry.execute(
        "retrieve_policy_snippets",
        query=ctx.inputs["case_transcript"],
    )
    return out


def _normalize_summary(data: dict) -> dict:
    """Normalize LLM output keys (action_items vs actionItems, etc)."""
    return {
        "summary": data.get("summary", data.get("Summary")),
        "action_items": data.get("action_items", data.get("actionItems")),
        "next_steps": data.get("next_steps", data.get("nextSteps")),
    }


def _bounded_prompt(name: str, employee_context, **values) -> str:
    template = _load_prompt(name)
    prompt = template.format(employee_context=json.dumps(employee_context, indent=2), **values)
    try:
        require_prompt_budget(prompt)
    except ContextBudgetExceeded:
        # Prior cases/notes are supplementary, unverified context. Omit whole
        # collections explicitly before sacrificing any current policy clause.
        fields = [key for key in ("case_history", "last_activities")
                  if isinstance(employee_context, dict) and employee_context.get(key)]
        if not fields:
            raise
        compact = {key: value for key, value in employee_context.items() if key not in fields}
        compact["context_omissions"] = {
            "reason": "context_budget", "fields": fields,
            "guidance": "Prior history was omitted. Ask HR if that history is needed; do not infer its contents.",
        }
        prompt = template.format(employee_context=json.dumps(compact, indent=2), **values)
        require_prompt_budget(prompt)
    return prompt


@diagnosed("extract_actions.v1")
def summarize_and_extract_actions_step(ctx):
    prompt = _bounded_prompt(
        "summarize", ctx.step_outputs.get("retrieve_employee_context", {}),
        employee_id=ctx.inputs["employee_id"],
        case_transcript=ctx.inputs["case_transcript"],
        policy_snippets="\n\n---\n\n".join(ctx.step_outputs.get("retrieve_policy_snippets", [])) or "(none retrieved)",
    )
    text = _retryable_llm_call(prompt)
    data = _parse_json_from_text(text)
    return reviewed_summary(ExtractActionsOutput.model_validate(_normalize_summary(data)).model_dump())


def _pto_balance_review(ctx):
    introductions = {
        "empathetic": "I understand you'd like to check your PTO balance. ",
        "direct": "Your current PTO balance is unavailable in Atlas. ",
        "formal": "Your request requires verification of your current PTO balance. ",
    }
    return {
        "subject": "Your PTO balance needs verification",
        "body": introductions[ctx.inputs["tone"]] + (
            "Atlas does not have a verified current PTO balance for you, so it cannot confirm how many hours you have. "
            "Any balance mentioned in your question is employee-reported and unverified; case history cannot verify it. "
            "Please ask your HR partner to verify the balance in the authoritative HR system."
        ),
        "tone": ctx.inputs["tone"], "cited_snippets": [],
        "citation_evidence": [], "quality_status": "needs_hr_review",
        "execution_receipts": [],
        "answerability": {"status": "hr_review_required", "reason": "unverified_employee_facts"},
        "recommendation_review": recommendation_review(),
        "operation_status": no_operations(),
        "pto_balance": unavailable_pto_balance(ctx.inputs["employee_id"]),
    }


@diagnosed("structured_response.v1")
def generate_structured_response_step(ctx):
    review = question_review(ctx.inputs.get("case_transcript", ""))
    if review:
        status, reason = review
        result = unsupported_policy_response(ctx.inputs["tone"])
        if status == "clarification_needed":
            result.update(subject="Please clarify your HR question",
                          body="Please clarify which HR policy, request, or situation you mean. Ask your HR partner if you need help providing the necessary details.")
        return answerability(result, status, reason)
    intent = classify_policy_intent(ctx.inputs.get("case_transcript", ""))
    if intent:
        result = deterministic_policy_response(intent, ctx.inputs["tone"])
        if intent == "PTO_POLICY":
            result["pto_balance"] = unavailable_pto_balance(ctx.inputs["employee_id"])
            if personal_pto_question(ctx.inputs.get("case_transcript", "")):
                result["body"] += " Any balance you mention is employee-reported and unverified; ask HR to confirm the balance in the authoritative HR system."
        return result
    employee = ctx.step_outputs.get("retrieve_employee_context", {})
    question = ctx.inputs.get("case_transcript", "").strip().rstrip("?.!")
    # The assessment has no authoritative balance adapter. For a direct balance
    # lookup, old draft notes and a model cannot supply the missing payroll fact.
    balance_lookup = re.fullmatch(
        r"(?:how many (?:hours of pto|pto hours) do i have|"
        r"what(?:'s| is) my (?:current )?pto balance)", question, re.IGNORECASE,
    )
    if balance_lookup or personal_pto_question(ctx.inputs.get("case_transcript", "")):
        return _pto_balance_review(ctx)
    snippets = ctx.step_outputs.get("retrieve_policy_snippets", [])
    if not snippets:
        return unsupported_policy_response(ctx.inputs["tone"])
    actions = ctx.step_outputs.get("summarize_and_extract_actions", {})
    actions_str = json.dumps(actions, indent=2) if isinstance(actions, dict) else str(actions)
    snippets_str = "\n\n---\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets)) if snippets else "(none retrieved)"

    prompt = _bounded_prompt(
        "generate_response", employee,
        employee_id=ctx.inputs["employee_id"],
        case_transcript=ctx.inputs.get("case_transcript", ""),
        policy_snippets=snippets_str,
        actions=actions_str,
        tone=ctx.inputs["tone"],
    )
    text = _retryable_llm_call(prompt, tone=ctx.inputs["tone"])
    data = _parse_json_from_text(text)
    normalized = {
        "subject": data.get("subject", data.get("Subject")),
        "body": data.get("body", data.get("Body")),
        "tone": data.get("tone", data.get("Tone")),
        "cited_snippets": data.get("cited_snippets", data.get("citedSnippets")),
    }
    response = StructuredResponse(**normalized)
    if response.tone != ctx.inputs["tone"]:
        raise OutputRuleError("tone_mismatch", "Response tone does not match requested tone")
    # No external-operation executor exists in this workflow. This deliberately
    # catches known completion language only; free-text semantics still need HR.
    claim = re.compile(
        r"\b(?:i|we)(?:['’](?:ve|d)|\s+(?:have|had))?\s+"
        r"(?:already\s+|successfully\s+)?"
        r"(?:scheduled|sent|updated|notified|emailed|booked|approved|processed|submitted|changed)\b"
        r"|\b(?:meeting|email|balance|notification|request)\b[^.!?\n]{0,60}"
        r"\b(?:has been|was|is now)\s+(?:scheduled|sent|updated|approved|processed)\b",
        re.IGNORECASE,
    )
    if claim.search(response.subject) or claim.search(response.body):
        raise OutputRuleError("unsupported_execution_claim", "External action completion claim has no execution receipt")

    if personal_balance_claim(response.subject + " " + response.body):
        return _pto_balance_review(ctx)

    evidence = []
    for quote in response.cited_snippets:
        sources = list(dict.fromkeys(source for source in snippets if quote in source))
        if not sources:
            raise OutputRuleError("citation_absent", "Citation is absent from selected policy sources")
        if contains_policy_override(quote):
            raise OutputRuleError("citation_instruction", "Citation contains application-control instructions")
        if ctx.inputs.get("case_transcript"):
            sources = [source for source in sources if policy_topic_matches(ctx.inputs["case_transcript"], source)]
            if not sources:
                raise OutputRuleError("citation_unrelated", "Citation source is unrelated to the question")
        if len(sources) != 1:
            raise OutputRuleError("citation_ambiguous", "Citation maps to ambiguous selected policy sources")
        source = sources[0]
        start = source.index(quote)
        evidence.append({**policy_source_reference(source),
                         "quote": quote, "start": start, "end": start + len(quote)})
    if unsafe_recommendation(response.subject) or unsafe_recommendation(response.body):
        return answerability(_policy_review(ctx.inputs["tone"],
            "The proposed recommendation requires verification and does not grant permission for an HR operation."),
            "hr_review_required", "recommendation_requires_review")
    if unverified_completion_claim(response.subject) or unverified_completion_claim(response.body):
        return answerability(_policy_review(ctx.inputs["tone"],
            "Atlas has no verified execution receipt for the described action; no external action is confirmed."),
            "hr_review_required", "unverified_execution_claim")
    return _preserve_carryover_conditions(ctx, {
        **response.model_dump(),
        "citation_evidence": evidence,
        "quality_status": "needs_hr_review",
        "execution_receipts": [],
        "answerability": {"status": "hr_review_required", "reason": "unverified_applicability"},
        "recommendation_review": recommendation_review(),
        "operation_status": no_operations(),
    })


def post_case_note_to_hris_step(ctx):
    response = ctx.step_outputs.get("generate_structured_response", {})
    note = f"Unapproved response draft (needs HR review): {response.get('subject', '')} - {response.get('body', '')[:200]}"
    registry = get_tool_registry()
    out = registry.execute(
        "post_case_note_to_hris",
        case_id=ctx.case_id,
        employee_id=ctx.inputs["employee_id"],
        note=note,
    )
    return out


STEPS = [
    ("retrieve_employee_context", retrieve_employee_context_step),
    ("retrieve_policy_snippets", retrieve_policy_snippets_step),
    ("summarize_and_extract_actions", summarize_and_extract_actions_step),
    ("generate_structured_response", generate_structured_response_step),
    ("post_case_note_to_hris", post_case_note_to_hris_step),
]
