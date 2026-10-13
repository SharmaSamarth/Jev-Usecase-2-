"""Shared test fixtures: a fake Anthropic client and helpers. No network access."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from email_classifier.config import ANTHROPIC_PRICING, AnthropicSettings
from email_classifier.providers.anthropic_sonnet import AnthropicSonnetProvider
from email_classifier.schemas import EmailInput

API_URL = "https://api.anthropic.com/v1/messages"

VALID_OUTPUT = {
    "tone": "frustrated",
    "urgency": "high",
    "department": "billing_payments",
    "intent": "refund_request",
    "recommended_action": "respond_today",
    "human_review_required": False,
    "review_reason": None,
    "rationale": "Duplicate charge with a same-day refund request.",
    "evidence": "Please refund the duplicate payment today.",
}


def make_response(
    payload: Any = None,
    *,
    text: str | None = None,
    stop_reason: str = "end_turn",
    stop_details: Any = None,
    model: str = "claude-sonnet-5-5",
    input_tokens: int = 300,
    output_tokens: int = 120,
    cache_read: int = 0,
    cache_write: int = 0,
    request_id: str = "req_test_001",
) -> SimpleNamespace:
    body = text if text is not None else json.dumps(payload if payload is not None else VALID_OUTPUT)
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=body)]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        ),
        _request_id=request_id,
    )


def api_status_error(cls: type[anthropic.APIStatusError], status: int, message: str = "error") -> anthropic.APIStatusError:
    request = httpx2.Request("POST", API_URL)
    return cls(message, response=httpx2.Response(status, request=request), body=None)


def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=httpx2.Request("POST", API_URL))


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx2.Request("POST", API_URL))


class FakeMessages:
    def __init__(self, outcomes: list[Any]):
        self._outcomes = deque(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("FakeMessages.create called more times than expected")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeAnthropic:
    def __init__(self, outcomes: list[Any]):
        self.messages = FakeMessages(outcomes)


@pytest.fixture
def settings() -> AnthropicSettings:
    return AnthropicSettings(
        api_key_present=True,
        model="claude-sonnet-5-5",
        effort="low",
        max_tokens=4096,
        timeout_seconds=30.0,
        max_retries=2,
        max_repair_attempts=1,
        pricing=ANTHROPIC_PRICING["claude-sonnet-5-5"],
    )


@pytest.fixture
def make_provider(settings):
    def _make(outcomes: list[Any], **overrides: Any) -> tuple[AnthropicSonnetProvider, FakeAnthropic]:
        client = FakeAnthropic(outcomes)
        provider = AnthropicSonnetProvider(settings=replace(settings, **overrides), client=client)
        return provider, client

    return _make


@pytest.fixture
def email() -> EmailInput:
    return EmailInput(
        email_id="EMAIL-001",
        subject="Charged twice",
        body="I was charged twice this month. Please refund the duplicate payment today.",
    )
