"""Provider-agnostic classifier interface.

Nothing here depends on a vendor SDK. Each provider adapter converts its own
responses and errors into ``ClassificationResult`` objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .prompts import PROMPT_VERSION, prompt_fingerprint
from .schemas import ClassificationResult, EmailInput


class ProviderError(Exception):
    """Base class for provider errors."""


class ProviderNotConfiguredError(ProviderError):
    """The provider cannot run in this environment (missing credentials, SDK, or adapter)."""


class UnknownProviderError(ProviderError):
    """No provider is registered under the requested name."""


# Error types recorded on failed ClassificationResults. Keeping them in one
# place makes error-rate breakdowns comparable across providers.
class ErrorType:
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    RATE_LIMIT = "rate_limit_error"
    TIMEOUT = "timeout_error"
    CONNECTION = "connection_error"
    SERVER = "server_error"
    BAD_REQUEST = "bad_request_error"
    NOT_FOUND = "not_found_error"
    REFUSAL = "model_refusal"
    TRUNCATED = "truncated_output"
    INVALID_OUTPUT = "invalid_output"
    UNEXPECTED = "unexpected_error"

    #: Errors that mean the provider configuration is broken (bad key, no model
    #: access, wrong model id). Every further call would fail the same way.
    FATAL = frozenset({AUTHENTICATION, PERMISSION, NOT_FOUND})
    #: Errors caused by one particular email's content rather than the setup.
    EMAIL_SPECIFIC = frozenset({REFUSAL, TRUNCATED, INVALID_OUTPUT})


class EmailClassifierProvider(ABC):
    """Common interface for every classification provider."""

    #: Short registry name, e.g. ``"sonnet"`` or ``"jev"``.
    name: str

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the underlying model, as configured."""

    @abstractmethod
    def is_configured(self) -> tuple[bool, str]:
        """Return (ready, human-readable explanation)."""

    @abstractmethod
    def classify(self, email: EmailInput) -> ClassificationResult:
        """Classify one email.

        Must not raise for per-email failures (API errors, invalid output);
        those are returned as a ``ClassificationResult`` with ``error_type``
        set so evaluation can count them. May raise
        ``ProviderNotConfiguredError`` when the provider cannot run at all.
        """

    def config_fingerprint(self) -> dict[str, str]:
        """Settings that can change this provider's output.

        Used to key the local result cache, so a cached answer is only reused
        when it was produced under the same configuration. Providers should
        extend this with any output-affecting settings.
        """
        return {"provider": self.name, "model": self.model_id}

    def prompt_info(self) -> tuple[str, str]:
        """(version, fingerprint) of the instructions this provider sends.

        Defaults to the shared system prompt. Providers whose API takes a
        different input format (e.g. typed questions) override this so reports
        and the result cache reflect what was actually sent.
        """
        return PROMPT_VERSION, prompt_fingerprint()

    def ensure_configured(self) -> None:
        ready, message = self.is_configured()
        if not ready:
            raise ProviderNotConfiguredError(f"Provider '{self.name}' is not configured: {message}")
