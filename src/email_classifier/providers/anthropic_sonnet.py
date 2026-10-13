"""Claude Sonnet provider using the official Anthropic Python SDK.

All Anthropic-specific request/response handling lives in this module. The
same request is sent either to the Anthropic API (``anthropic.Anthropic``) or
to Amazon Bedrock, selected by ``CLAUDE_PLATFORM`` or inferred from which
credential is set. On Bedrock, ``BEDROCK_ENDPOINT`` chooses the
``bedrock-runtime`` InvokeModel endpoint (``anthropic.AnthropicBedrock``, the
default, which supports cross-region inference profiles such as
``eu.anthropic.claude-sonnet-5-5``) or the ``bedrock-mantle`` Messages API
(``anthropic.AnthropicBedrockMantle``). Both take the same request body.

Request design:
- Structured outputs (``output_config.format``) constrain the response to the
  taxonomy JSON schema; Pydantic still validates every response.
- ``output_config.effort`` (default ``low``) keeps thinking short. Sonnet 5.5
  rejects non-default ``temperature``, so effort plus a constrained schema is
  the low-variance configuration.
- The shared system prompt is marked for prompt caching, so evaluation runs
  pay full price for it only once per cache window.
- Transient failures (429, 5xx, timeouts, connection errors) are retried by
  the SDK up to ``max_retries``. Invalid model output gets at most
  ``max_repair_attempts`` additional requests that include the validation
  error. Refusals are not retried and are never rescued by another model,
  so every successful prediction is attributable to the configured model.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import anthropic
from pydantic import ValidationError

from ..classifier_interface import EmailClassifierProvider, ErrorType
from ..config import (
    BEDROCK_KEY_VARS,
    PLATFORM_BEDROCK,
    AnthropicSettings,
    estimate_cost_usd,
    load_anthropic_settings,
)
from ..prompts import (
    PROMPT_VERSION,
    build_repair_note,
    build_system_prompt,
    build_user_message,
    output_json_schema,
)
from ..schemas import (
    ClassificationResult,
    EmailClassification,
    EmailInput,
    ModelClassificationOutput,
    UsageInfo,
    evidence_is_grounded,
)

logger = logging.getLogger(__name__)


def _summarize_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        location = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{location}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _classify_api_error(exc: anthropic.APIError) -> tuple[str, str]:
    """Map an SDK exception to (error_type, safe message). Most specific first."""
    if isinstance(exc, anthropic.AuthenticationError):
        return ErrorType.AUTHENTICATION, "Authentication failed; check the API key (ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK)."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return ErrorType.PERMISSION, f"Permission denied (on Bedrock, check model access in this region): {exc.message}"
    if isinstance(exc, anthropic.NotFoundError):
        return ErrorType.NOT_FOUND, (
            f"Not found (check ANTHROPIC_MODEL; on Bedrock the region may require an inference profile "
            f"id such as 'eu.anthropic.claude-sonnet-5-5'): {exc.message}"
        )
    if isinstance(exc, anthropic.RateLimitError):
        return ErrorType.RATE_LIMIT, "Rate limited after SDK retries were exhausted."
    if isinstance(exc, anthropic.BadRequestError):
        return ErrorType.BAD_REQUEST, f"Bad request: {exc.message}"
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return ErrorType.SERVER, f"Server error {exc.status_code} after SDK retries."
        return ErrorType.UNEXPECTED, f"API error {exc.status_code}: {exc.message}"
    # APITimeoutError subclasses APIConnectionError, so check it first.
    if isinstance(exc, anthropic.APITimeoutError):
        return ErrorType.TIMEOUT, "Request timed out after SDK retries."
    if isinstance(exc, anthropic.APIConnectionError):
        return ErrorType.CONNECTION, "Could not connect to the Anthropic API."
    return ErrorType.UNEXPECTED, f"{type(exc).__name__}: {exc}"


def _usage_from_response(response: Any) -> UsageInfo:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageInfo()
    return UsageInfo(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None),
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
    )


def _response_text(response: Any) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


class AnthropicSonnetProvider(EmailClassifierProvider):
    name = "sonnet"

    def __init__(self, settings: Optional[AnthropicSettings] = None, client: Optional[anthropic.Anthropic] = None):
        self._settings = settings or load_anthropic_settings()
        self._client = client
        self._system_prompt = build_system_prompt()
        self._schema = output_json_schema()

    @property
    def model_id(self) -> str:
        return self._settings.model

    @property
    def settings(self) -> AnthropicSettings:
        return self._settings

    @property
    def platform(self) -> str:
        return self._settings.platform

    def config_fingerprint(self) -> dict[str, str]:
        return {
            **super().config_fingerprint(),
            "platform": self.platform,
            **({"bedrock_endpoint": self._settings.bedrock_endpoint} if self.platform == PLATFORM_BEDROCK else {}),
            "effort": self._settings.effort,
            "prompt_version": PROMPT_VERSION,
        }

    def is_configured(self) -> tuple[bool, str]:
        if self._client is not None:
            return True, "Using an injected Anthropic client."
        if self.platform == PLATFORM_BEDROCK:
            if not self._settings.api_key_present:
                return False, "Bedrock selected but AWS_BEARER_TOKEN_BEDROCK is not set. Add your Bedrock API key to the environment or .env."
            if not self._settings.aws_region:
                return False, "Bedrock selected but no region is set. Set AWS_REGION (for example us-east-1)."
            return True, (
                f"Amazon Bedrock API key found; region '{self._settings.aws_region}', "
                f"endpoint '{self._settings.bedrock_endpoint}', model '{self.model_id}'."
            )
        if not self._settings.api_key_present:
            return False, (
                "ANTHROPIC_API_KEY is not set. Add it to your environment or a local .env file "
                "(or set AWS_BEARER_TOKEN_BEDROCK and AWS_REGION to use Amazon Bedrock)."
            )
        return True, f"Anthropic API key found; model '{self.model_id}'."

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self.ensure_configured()
            if self.platform == PLATFORM_BEDROCK:
                # Passing the key explicitly selects bearer-token auth (no SigV4 / boto3 needed).
                api_key = next(os.environ[v].strip() for v in BEDROCK_KEY_VARS if os.environ.get(v, "").strip())
                client_cls = (
                    anthropic.AnthropicBedrockMantle
                    if self._settings.bedrock_endpoint == "mantle"
                    else anthropic.AnthropicBedrock
                )
                self._client = client_cls(
                    api_key=api_key,
                    aws_region=self._settings.aws_region,
                    timeout=self._settings.timeout_seconds,
                    max_retries=self._settings.max_retries,
                )
            else:
                self._client = anthropic.Anthropic(
                    timeout=self._settings.timeout_seconds,
                    max_retries=self._settings.max_retries,
                )
        return self._client

    def _request(self, user_content: str) -> Any:
        return self._get_client().messages.create(
            model=self._settings.model,
            max_tokens=self._settings.max_tokens,
            system=[{"type": "text", "text": self._system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "effort": self._settings.effort,
                "format": {"type": "json_schema", "schema": self._schema},
            },
        )

    def classify(self, email: EmailInput) -> ClassificationResult:
        self._get_client()  # raises ProviderNotConfiguredError before any work if unusable

        base_message = build_user_message(email)
        user_content = base_message
        usage_total = UsageInfo()
        attempts = 0
        request_id: Optional[str] = None
        served_model: Optional[str] = None
        error_type, error_message = ErrorType.UNEXPECTED, "No attempt was made."
        started = time.perf_counter()

        def _finish(**kwargs: Any) -> ClassificationResult:
            p = self._settings.pricing
            return ClassificationResult(
                email_id=email.email_id,
                provider=self.name,
                model=self.model_id,
                served_model=served_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                attempts=attempts,
                usage=usage_total,
                estimated_cost_usd=estimate_cost_usd(
                    p,
                    usage_total.input_tokens,
                    usage_total.output_tokens,
                    usage_total.cache_creation_input_tokens,
                    usage_total.cache_read_input_tokens,
                ),
                request_id=request_id,
                **kwargs,
            )

        def _fail(kind: str, message: str) -> ClassificationResult:
            logger.warning(
                "Classification failed: email_id=%s provider=%s error_type=%s request_id=%s attempts=%d",
                email.email_id, self.name, kind, request_id, attempts,
            )
            return _finish(error_type=kind, error_message=message)

        for _ in range(1 + self._settings.max_repair_attempts):
            attempts += 1
            try:
                response = self._request(user_content)
            except anthropic.APIError as exc:
                return _fail(*_classify_api_error(exc))

            usage_total = usage_total.add(_usage_from_response(response))
            request_id = getattr(response, "_request_id", None) or request_id
            served_model = getattr(response, "model", None) or served_model
            stop_reason = getattr(response, "stop_reason", None)

            if stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) if details else None
                return _fail(ErrorType.REFUSAL, f"Model declined to classify (category: {category or 'unspecified'}).")

            if stop_reason == "max_tokens":
                error_type = ErrorType.TRUNCATED
                error_message = f"Output hit max_tokens={self._settings.max_tokens} before completing."
                user_content = base_message + build_repair_note("Previous output was truncated; keep rationale and evidence short.")
                continue

            try:
                output = ModelClassificationOutput.model_validate_json(_response_text(response))
            except ValidationError as exc:
                error_type = ErrorType.INVALID_OUTPUT
                error_message = _summarize_validation_error(exc)
                user_content = base_message + build_repair_note(error_message)
                continue

            classification = EmailClassification.from_model_output(email.email_id, output, model_confidence=None)
            return _finish(
                classification=classification,
                evidence_grounded=evidence_is_grounded(classification.evidence, email),
            )

        return _fail(error_type, f"{error_message} (after {attempts} attempt(s))")
