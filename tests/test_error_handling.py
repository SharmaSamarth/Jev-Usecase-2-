import json
from types import SimpleNamespace

import anthropic
import pytest

from conftest import VALID_OUTPUT, api_status_error, connection_error, make_response, timeout_error
from email_classifier.classifier_interface import ErrorType


@pytest.mark.parametrize(
    "exc, expected_type",
    [
        (api_status_error(anthropic.RateLimitError, 429), ErrorType.RATE_LIMIT),
        (api_status_error(anthropic.AuthenticationError, 401), ErrorType.AUTHENTICATION),
        (api_status_error(anthropic.PermissionDeniedError, 403), ErrorType.PERMISSION),
        (api_status_error(anthropic.NotFoundError, 404), ErrorType.NOT_FOUND),
        (api_status_error(anthropic.BadRequestError, 400), ErrorType.BAD_REQUEST),
        (api_status_error(anthropic.InternalServerError, 500), ErrorType.SERVER),
        (api_status_error(anthropic.APIStatusError, 529), ErrorType.SERVER),
        (timeout_error(), ErrorType.TIMEOUT),
        (connection_error(), ErrorType.CONNECTION),
    ],
)
def test_api_errors_become_failed_results(make_provider, email, exc, expected_type):
    provider, client = make_provider([exc])
    result = provider.classify(email)
    assert not result.success
    assert result.error_type == expected_type
    assert result.email_id == email.email_id
    assert result.attempts == 1
    # API errors are retried by the SDK (max_retries), not by the repair loop.
    assert len(client.messages.calls) == 1
    assert email.body not in (result.error_message or "")


def test_malformed_json_is_repaired(make_provider, email):
    provider, client = make_provider([make_response(text="{not json"), make_response()])
    result = provider.classify(email)
    assert result.success
    assert result.attempts == 2
    second_prompt = client.messages.calls[1]["messages"][0]["content"]
    assert "failed validation" in second_prompt
    # Usage from both attempts is counted.
    assert result.usage.input_tokens == 600


def test_invalid_label_fails_after_bounded_attempts(make_provider, email):
    bad = make_response({**VALID_OUTPUT, "department": "finance"})
    provider, client = make_provider([bad, bad])
    result = provider.classify(email)
    assert not result.success
    assert result.error_type == ErrorType.INVALID_OUTPUT
    assert "department" in result.error_message
    assert result.attempts == 2 and len(client.messages.calls) == 2


def test_repair_attempts_can_be_disabled(make_provider, email):
    provider, client = make_provider([make_response(text="[]")], max_repair_attempts=0)
    result = provider.classify(email)
    assert result.error_type == ErrorType.INVALID_OUTPUT
    assert len(client.messages.calls) == 1


def test_missing_review_reason_triggers_repair(make_provider, email):
    flagged = {**VALID_OUTPUT, "human_review_required": True, "review_reason": None, "recommended_action": "human_review"}
    fixed = {**flagged, "review_reason": "Ambiguous request."}
    provider, _ = make_provider([make_response(flagged), make_response(fixed)])
    result = provider.classify(email)
    assert result.success and result.classification.review_reason == "Ambiguous request."


def test_refusal_is_reported_and_not_retried(make_provider, email):
    refusal = make_response(text="", stop_reason="refusal", stop_details=SimpleNamespace(category="general_harms"))
    provider, client = make_provider([refusal])
    result = provider.classify(email)
    assert result.error_type == ErrorType.REFUSAL
    assert "general_harms" in result.error_message
    assert len(client.messages.calls) == 1


def test_truncated_output_is_retried_then_reported(make_provider, email):
    truncated = make_response(text=json.dumps(VALID_OUTPUT)[:40], stop_reason="max_tokens")
    provider, client = make_provider([truncated, truncated])
    result = provider.classify(email)
    assert result.error_type == ErrorType.TRUNCATED
    assert len(client.messages.calls) == 2


def test_truncation_then_success(make_provider, email):
    truncated = make_response(text="{", stop_reason="max_tokens")
    provider, _ = make_provider([truncated, make_response()])
    assert provider.classify(email).success


def test_api_error_during_repair_is_reported(make_provider, email):
    provider, _ = make_provider([make_response(text="oops"), api_status_error(anthropic.RateLimitError, 429)])
    result = provider.classify(email)
    assert result.error_type == ErrorType.RATE_LIMIT
    assert result.attempts == 2
