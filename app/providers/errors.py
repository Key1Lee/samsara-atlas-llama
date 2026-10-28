"""Fixed provider categories; diagnostics never contain remote payloads."""


class ProviderError(Exception):
    code = "provider_error"

    def __init__(self):
        super().__init__(self.code)


class ProviderConfigurationError(ProviderError):
    code = "provider_configuration_error"

    def __init__(self):
        Exception.__init__(self, "Provider configuration error; select stub or ollama")


class ProviderTimeout(ProviderError, TimeoutError):
    code = "provider_timeout"


class ProviderUnavailable(ProviderError, ConnectionError):
    code = "provider_unavailable"


class ProviderRateLimited(ProviderError):
    code = "provider_rate_limited"


class ProviderRejected(ProviderError):
    code = "provider_rejected"


class ProviderInvalidOutput(ProviderError, ValueError):
    code = "provider_invalid_output"
