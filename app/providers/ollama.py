"""Ollama provider - calls local Ollama API. No API key required."""

import os
import requests
import time
import hashlib
import json
from app.evaluation.observation import record_model

from app.providers.base import ModelProvider, CompletionResponse, require_prompt_budget
from app.providers.errors import (ProviderTimeout, ProviderUnavailable,
    ProviderRateLimited, ProviderRejected, ProviderInvalidOutput)


class OllamaProvider(ModelProvider):
    """Calls local Ollama API at http://localhost:11434."""

    def __init__(self, base_url: str | None = None, model: str = "llama3.2"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", model)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        require_prompt_budget(prompt)
        self._call_started = time.monotonic()
        try:
            return self._complete(prompt, kwargs.get("output_schema"))
        except requests.Timeout:
            from app.diagnostics import completion_status
            completion_status.set("timeout")
            record_model(prompt, self.model, "timeout", elapsed_seconds=time.monotonic()-self._call_started)
            raise ProviderTimeout() from None
        except requests.ConnectionError:
            from app.diagnostics import completion_status
            completion_status.set("unavailable")
            record_model(prompt, self.model, "unavailable", elapsed_seconds=time.monotonic()-self._call_started)
            raise ProviderUnavailable() from None
        except requests.RequestException:
            raise ProviderUnavailable() from None

    def _complete(self, prompt, output_schema=None):
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **({"format": output_schema} if output_schema else {}),
                "options": {"num_ctx": int(os.getenv('OLLAMA_CONTEXT_TOKENS', '8192')),
                            "num_predict": int(os.getenv('OLLAMA_OUTPUT_TOKENS', '1024'))},
            },
            timeout=120,
        )
        if resp.status_code == 429:
            raise ProviderRateLimited()
        if resp.status_code >= 500:
            raise ProviderUnavailable()
        if not 200 <= resp.status_code < 300:
            raise ProviderRejected()
        try:
            data = resp.json()
        except ValueError:
            raise ProviderInvalidOutput() from None
        from app.diagnostics import completion_status
        status = 'unknown'
        if isinstance(data, dict):
            status = 'truncated' if data.get('done_reason') == 'length' else ('completed' if data.get('done') is True else 'unknown')
            if data.get('refusal') or data.get('done_reason') == 'refusal': status = 'refused'
        completion_status.set(status)
        response_text=data.get('response') if isinstance(data, dict) else None
        record_model(prompt, self.model, status, elapsed_seconds=time.monotonic()-self._call_started,
            response_sha256=hashlib.sha256(response_text.encode()).hexdigest() if isinstance(response_text,str) else None,
            response_bytes=len(response_text.encode()) if isinstance(response_text,str) else None,
            model_version=None, schema_sha256=hashlib.sha256(json.dumps(output_schema,sort_keys=True).encode()).hexdigest() if output_schema else None,
            **{k:data.get(k) for k in ('prompt_eval_count','eval_count','total_duration','load_duration','eval_duration') if isinstance(data,dict)})
        if status in {'truncated', 'refused'}:
            raise ProviderInvalidOutput()
        text = data.get("response") if isinstance(data, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ProviderInvalidOutput()
        raw = {"response": text, "model": self.model}
        result = CompletionResponse(text=text, raw=raw)
        result.completion_status = status
        return result
