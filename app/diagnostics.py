"""Allowlisted diagnostics. Never serialize exception messages, inputs or completions."""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import traceback
import json
from pydantic import ValidationError
from app.audit import current_request_id
from app.providers.base import ContextBudgetExceeded

completion_status = ContextVar('completion_status', default='not_called')

class OutputRuleError(ValueError):
    def __init__(self, rule, message):
        super().__init__(message)
        self.rule = rule

RULES = frozenset({'json_object_required', 'tone_mismatch', 'unsupported_execution_claim',
    'citation_absent', 'citation_instruction', 'citation_unrelated', 'citation_ambiguous',
    'citation_required', 'citation_unverified',
    'json_syntax', 'schema_fields', 'context_budget', 'dependency_import', 'unclassified'})

def failure_diagnostic(exc, schema='unknown'):
    from tenacity import RetryError
    if isinstance(exc, RetryError):
        exc = exc.last_attempt.exception() or exc
    rule = getattr(exc, 'rule', 'unclassified')
    if rule not in RULES: rule = 'unclassified'
    subtype = 'application_error'
    if isinstance(exc, ValidationError): rule, subtype = 'schema_fields', 'schema_validation'
    elif isinstance(exc, json.JSONDecodeError): rule, subtype = 'json_syntax', 'json_parse'
    elif isinstance(exc, ContextBudgetExceeded): rule, subtype = 'context_budget', 'prompt_budget'
    elif isinstance(exc, ImportError): rule, subtype = 'dependency_import', 'dependency'
    elif isinstance(exc, OutputRuleError):
        subtype = 'format_validation' if rule == 'json_object_required' else 'semantic_validation'
    elif getattr(exc, 'code', '').startswith('provider_'): subtype = 'provider_failure'
    elif getattr(exc, 'code', '').startswith('rag_'): subtype = 'retrieval_failure'
    result = {'error_subtype': subtype, 'validation_rule': rule,
              'model_completion_status': completion_status.get(),
              'schema_version': schema, 'request_id': current_request_id.get(),
              'timeout': getattr(exc, 'code', '') == 'provider_timeout',
              'truncated': completion_status.get() == 'truncated'}
    if isinstance(exc, ValidationError):
        result['schema_error_types'] = sorted({e['type'] for e in exc.errors(include_input=False, include_context=False)
            if e['type'] in {'missing','string_type','string_too_short','list_type','literal_error','model_type'}})
    if isinstance(exc, ImportError):
        # Module names may be supplied by untrusted plugins: only known dependencies.
        result['dependency_module'] = exc.name if exc.name in {'requests','app.providers.ollama','app.providers.local_stub','app.providers.base'} else 'unknown'
        result['dependency_symbol'] = next((name for name in ('CompletionResponse','ModelProvider','OllamaProvider','LocalStubProvider')
            if f"cannot import name '{name}'" in str(exc)), 'unknown')
    safe_messages = {'json_syntax':'Model output is not valid JSON', 'schema_fields':'Model output violates the required schema',
        'tone_mismatch':'Response tone differs from requested tone', 'citation_absent':'Citation is absent from selected sources',
        'citation_instruction':'Citation contains application-control instructions', 'citation_unrelated':'Citation is unrelated to question',
        'citation_ambiguous':'Citation matches multiple sources', 'unsupported_execution_claim':'No execution receipt supports the claim',
        'context_budget':'Complete prompt exceeds context budget', 'dependency_import':'A Python import failed',
        'json_object_required':'Expected a JSON object'}
    result['sanitized_message'] = safe_messages.get(rule, 'Inspect the recorded error subtype')
    from app.rag.errors import RAGError
    if isinstance(exc, RAGError):
        result['retrieval_error_code'] = exc.code
    result['error_type'] = type(exc).__name__ if type(exc).__name__ in {'ValueError','OutputRuleError','JSONDecodeError','ValidationError','ImportError','ModuleNotFoundError','ContextBudgetExceeded','ProviderInvalidOutput','ProviderTimeout'} else 'other'
    root=Path(__file__).resolve().parent.parent
    frames=[]
    for frame in traceback.extract_tb(exc.__traceback__):
        path=Path(frame.filename)
        try:relative=str(path.relative_to(root))
        except ValueError:continue
        if relative.startswith('app/'):
            frames.append({'file':relative,'function':frame.name,'line':frame.lineno})
    result['stack_trace'] = frames[-12:]
    result['failing_code_location'] = frames[-1] if frames else None
    if isinstance(exc, json.JSONDecodeError):result['json_location']={'line':exc.lineno,'column':exc.colno,'offset':exc.pos}
    if isinstance(exc, ValidationError):
        allowed={'subject','body','tone','cited_snippets','summary','action_items','next_steps'}
        result['schema_errors']=[{'path':'/'+str(e['loc'][0]) if e['loc'] and e['loc'][0] in allowed else '/',
            'code':e['type'] if e['type'] in result['schema_error_types'] else 'invalid'}
            for e in exc.errors(include_input=False,include_context=False)]
    return result

def diagnosed(schema):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            token = completion_status.set('not_called')
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                exc.atlas_diagnostic = failure_diagnostic(exc, schema)
                raise
            finally:
                completion_status.reset(token)
        return wrapped
    return decorate
