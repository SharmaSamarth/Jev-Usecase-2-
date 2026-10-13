import pytest

from conftest import VALID_OUTPUT, make_response
from email_classifier.classifier_interface import ProviderNotConfiguredError, UnknownProviderError
from email_classifier.config import ANTHROPIC_PRICING, estimate_cost_usd, load_anthropic_settings
from email_classifier.prompts import PROMPT_VERSION, build_system_prompt, output_json_schema
from email_classifier.providers import AnthropicSonnetProvider, available_providers, get_provider
from email_classifier.providers.jev import JevProvider
from email_classifier.taxonomy import Urgency


def test_successful_classification(make_provider, email):
    provider, client = make_provider([make_response(input_tokens=400, output_tokens=150, cache_read=1800)])
    result = provider.classify(email)

    assert result.success
    assert result.email_id == "EMAIL-001"
    assert result.classification.email_id == "EMAIL-001"
    assert result.classification.urgency is Urgency.HIGH
    assert result.classification.model_confidence is None
    assert result.provider == "sonnet"
    assert result.model == result.served_model == "claude-sonnet-5-5"
    assert result.prompt_version == PROMPT_VERSION
    assert result.attempts == 1
    assert result.request_id == "req_test_001"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.usage.input_tokens == 400 and result.usage.cache_read_input_tokens == 1800
    expected_cost = (400 * 2.00 + 150 * 10.00 + 1800 * 0.20) / 1_000_000
    assert result.estimated_cost_usd == pytest.approx(expected_cost)
    assert result.evidence_grounded is True
    assert len(client.messages.calls) == 1


def test_request_shape(make_provider, email):
    provider, client = make_provider([make_response()])
    provider.classify(email)
    call = client.messages.calls[0]

    assert call["model"] == "claude-sonnet-5-5"
    assert call["max_tokens"] == 4096
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": output_json_schema()}
    assert call["system"][0]["text"] == build_system_prompt()
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Sampling parameters are rejected by Sonnet 5.5; thinking is left at the adaptive default.
    assert "temperature" not in call and "top_p" not in call and "thinking" not in call
    user = call["messages"][0]
    assert user["role"] == "user"
    assert "<email>" in user["content"] and email.body in user["content"]
    assert email.email_id not in user["content"]


def test_ungrounded_evidence_is_flagged_not_rejected(make_provider, email):
    provider, _ = make_provider([make_response({**VALID_OUTPUT, "evidence": "text that is not in the email"})])
    result = provider.classify(email)
    assert result.success and result.evidence_grounded is False


def test_served_model_recorded_separately(make_provider, email):
    provider, _ = make_provider([make_response(model="claude-sonnet-5-5-other")])
    result = provider.classify(email)
    assert result.model == "claude-sonnet-5-5"
    assert result.served_model == "claude-sonnet-5-5-other"


def test_missing_api_key_is_explicit_error(settings, email):
    from dataclasses import replace

    provider = AnthropicSonnetProvider(settings=replace(settings, api_key_present=False))
    ready, message = provider.is_configured()
    assert not ready and "ANTHROPIC_API_KEY" in message
    with pytest.raises(ProviderNotConfiguredError, match="ANTHROPIC_API_KEY"):
        provider.classify(email)


def test_settings_from_environment():
    s = load_anthropic_settings({"ANTHROPIC_API_KEY": "x", "ANTHROPIC_EFFORT": "MEDIUM", "ANTHROPIC_MAX_RETRIES": "5"})
    assert s.api_key_present and s.model == "claude-sonnet-5-5" and s.effort == "medium" and s.max_retries == 5
    assert s.pricing == ANTHROPIC_PRICING["claude-sonnet-5-5"]

    unknown = load_anthropic_settings({"ANTHROPIC_MODEL": "some-future-model"})
    assert unknown.pricing is None and not unknown.api_key_present

    priced = load_anthropic_settings({"ANTHROPIC_MODEL": "some-future-model",
                                      "ANTHROPIC_PRICE_INPUT_PER_MTOK": "1", "ANTHROPIC_PRICE_OUTPUT_PER_MTOK": "5"})
    assert priced.pricing.input_per_mtok == 1 and priced.pricing.source == "environment override"

    with pytest.raises(ValueError):
        load_anthropic_settings({"ANTHROPIC_EFFORT": "extreme"})
    with pytest.raises(ValueError):
        load_anthropic_settings({"ANTHROPIC_MAX_TOKENS": "lots"})


def test_cost_unavailable_without_pricing_or_usage():
    assert estimate_cost_usd(None, 10, 10) is None
    assert estimate_cost_usd(ANTHROPIC_PRICING["claude-sonnet-5-5"], None, 10) is None


# --- registry and Jev stub --------------------------------------------------------

def test_registry_lists_both_providers():
    assert available_providers() == ["jev", "sonnet"]


def test_unknown_provider_is_explicit_error():
    with pytest.raises(UnknownProviderError, match="Available: jev, sonnet"):
        get_provider("gpt")


def test_jev_without_key_is_explicitly_not_configured(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    provider = get_provider("jev", require_configured=False)
    assert isinstance(provider, JevProvider)
    ready, message = provider.is_configured()
    assert not ready and ("TYPESAFE_API_KEY" in message or "pip install" in message)
    with pytest.raises(ProviderNotConfiguredError):
        get_provider("jev")
    from email_classifier.schemas import EmailInput

    with pytest.raises(ProviderNotConfiguredError):
        provider.classify(EmailInput(email_id="E", body="x"))


def test_no_silent_fallback_when_sonnet_unconfigured(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_AWS_API_KEY", "CLAUDE_PLATFORM"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ProviderNotConfiguredError, match="sonnet"):
        get_provider("sonnet")


def test_real_sdk_round_trip_over_mock_transport(settings, email):
    """Exercise the real Anthropic SDK (serialization, response parsing) without network access."""
    import json

    import anthropic
    import httpx2

    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            headers={"request-id": "req_mock_123"},
            json={
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": json.dumps(VALID_OUTPUT)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1200, "output_tokens": 90,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            },
        )

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    result = AnthropicSonnetProvider(settings=settings, client=client).classify(email)

    assert result.success, result.error_message
    assert result.request_id == "req_mock_123"
    assert result.usage.input_tokens == 1200
    assert captured["url"].endswith("/v1/messages")
    body = captured["body"]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


# --- Amazon Bedrock -------------------------------------------------------------

def test_bedrock_inferred_from_bedrock_key():
    s = load_anthropic_settings({"AWS_BEARER_TOKEN_BEDROCK": "bk", "AWS_REGION": "us-east-1"})
    assert s.platform == "bedrock"
    assert s.model == "anthropic.claude-sonnet-5-5"
    assert s.api_key_present and s.aws_region == "us-east-1"
    # Anthropic list prices are not applied to Bedrock.
    assert s.pricing is None
    priced = load_anthropic_settings({"AWS_BEARER_TOKEN_BEDROCK": "bk", "AWS_REGION": "us-east-1",
                                      "ANTHROPIC_PRICE_INPUT_PER_MTOK": "3", "ANTHROPIC_PRICE_OUTPUT_PER_MTOK": "15"})
    assert priced.pricing.input_per_mtok == 3


def test_platform_selection_rules():
    both = {"ANTHROPIC_API_KEY": "a", "AWS_BEARER_TOKEN_BEDROCK": "b"}
    assert load_anthropic_settings(both).platform == "anthropic"
    assert load_anthropic_settings({**both, "CLAUDE_PLATFORM": "bedrock", "AWS_REGION": "eu-west-1"}).platform == "bedrock"
    assert load_anthropic_settings({}).platform == "anthropic"
    with pytest.raises(ValueError, match="CLAUDE_PLATFORM"):
        load_anthropic_settings({"CLAUDE_PLATFORM": "azure"})
    with pytest.raises(ValueError, match="not a Bedrock model id"):
        load_anthropic_settings({"CLAUDE_PLATFORM": "bedrock", "ANTHROPIC_MODEL": "claude-sonnet-5-5"})
    for profile in ("eu.anthropic.claude-sonnet-5-5", "global.anthropic.claude-sonnet-5-5", "anthropic.claude-sonnet-5-5"):
        assert load_anthropic_settings({"CLAUDE_PLATFORM": "bedrock", "ANTHROPIC_MODEL": profile}).model == profile


def test_bedrock_configuration_messages():
    no_region = AnthropicSonnetProvider(settings=load_anthropic_settings({"AWS_BEARER_TOKEN_BEDROCK": "bk"}))
    ready, message = no_region.is_configured()
    assert not ready and "AWS_REGION" in message
    no_key = AnthropicSonnetProvider(settings=load_anthropic_settings({"CLAUDE_PLATFORM": "bedrock", "AWS_REGION": "us-east-1"}))
    ready, message = no_key.is_configured()
    assert not ready and "AWS_BEARER_TOKEN_BEDROCK" in message
    with pytest.raises(ProviderNotConfiguredError):
        no_key.classify(__import__("email_classifier.schemas", fromlist=["EmailInput"]).EmailInput(email_id="E", body="x"))


def test_bedrock_client_round_trip_over_mock_transport(monkeypatch, email):
    """Real AnthropicBedrockMantle client: bearer-token auth, anthropic.-prefixed model id, Mantle endpoint."""
    import json

    import anthropic
    import httpx2

    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_mock", "type": "message", "role": "assistant", "model": "anthropic.claude-sonnet-5-5",
            "content": [{"type": "text", "text": json.dumps(VALID_OUTPUT)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 80},
        })

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-test-key")
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_PLATFORM", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_ENDPOINT", "mantle")

    real_init = anthropic.AnthropicBedrockMantle.__init__

    def init_with_mock_transport(self, **kwargs):
        real_init(self, http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(anthropic.AnthropicBedrockMantle, "__init__", init_with_mock_transport)

    provider = AnthropicSonnetProvider()
    assert provider.platform == "bedrock"
    result = provider.classify(email)

    assert result.success, result.error_message
    assert result.served_model == "anthropic.claude-sonnet-5-5"
    assert result.estimated_cost_usd is None  # no Bedrock prices configured
    assert captured["url"] == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
    assert captured["auth"] == "Bearer bedrock-test-key"
    assert captured["body"]["model"] == "anthropic.claude-sonnet-5-5"
    assert captured["body"]["output_config"]["format"]["type"] == "json_schema"


def test_bedrock_runtime_endpoint_is_default_and_supports_inference_profiles(monkeypatch, email):
    """Real AnthropicBedrock client: InvokeModel URL with the profile id, bearer auth, same request body."""
    import json

    import anthropic
    import httpx2

    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_mock", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": json.dumps(VALID_OUTPUT)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 80},
        })

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-test-key")
    monkeypatch.setenv("AWS_REGION", "eu-north-1")
    monkeypatch.setenv("ANTHROPIC_MODEL", "eu.anthropic.claude-sonnet-5-5")
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_PLATFORM", "BEDROCK_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)

    real_init = anthropic.AnthropicBedrock.__init__

    def init_with_mock_transport(self, **kwargs):
        real_init(self, http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(anthropic.AnthropicBedrock, "__init__", init_with_mock_transport)

    provider = AnthropicSonnetProvider()
    assert provider.settings.bedrock_endpoint == "runtime"
    result = provider.classify(email)

    assert result.success, result.error_message
    assert captured["url"] == "https://bedrock-runtime.eu-north-1.amazonaws.com/model/eu.anthropic.claude-sonnet-5-5/invoke"
    assert captured["auth"] == "Bearer bedrock-test-key"
    assert captured["body"]["output_config"]["format"]["type"] == "json_schema"
    assert captured["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_invalid_bedrock_endpoint_rejected():
    with pytest.raises(ValueError, match="BEDROCK_ENDPOINT"):
        load_anthropic_settings({"BEDROCK_ENDPOINT": "classic"})
