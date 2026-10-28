import os

from app.providers.base import ModelProvider
from app.providers.stub import StubProvider
from app.providers.errors import ProviderConfigurationError


def get_provider() -> ModelProvider:
    provider_type = os.getenv("PROVIDER_TYPE", "stub").lower()
    if provider_type == "stub":
        return StubProvider()
    if provider_type == "ollama":
        from app.providers.ollama import OllamaProvider
        return OllamaProvider()
    if provider_type == "local":
        try:
            from app.providers.local_stub import LocalStubProvider
            return LocalStubProvider()
        except ImportError:
            raise ProviderConfigurationError() from None
    raise ProviderConfigurationError()
