from abc import ABC, abstractmethod
from typing import Any


MAX_PROMPT_BYTES = 6000


class ContextBudgetExceeded(ValueError):
    """Complete evidence cannot fit; no mandatory text has been truncated."""


def require_prompt_budget(prompt: str) -> None:
    # This is a conservative application byte bound, not a tokenizer or proof
    # of an arbitrary configured model's effective context capacity.
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ContextBudgetExceeded(
            "Complete prompt exceeds the 6000-byte context budget; request HR review or reduce context."
        )


class CompletionResponse:
    """Normalized response - use .text for provider-agnostic access."""

    def __init__(self, text: str, raw: Any = None):
        self.text = text
        self.raw = raw


class ModelProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        pass
