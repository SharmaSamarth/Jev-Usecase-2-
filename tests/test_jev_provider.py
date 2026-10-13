"""Jev adapter tests with a fake client shaped like typesafe-sdk's documented responses. No network."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from email_classifier.cache import CachingProvider, ResultCache
from email_classifier.classifier_interface import ErrorType, ProviderNotConfiguredError
from email_classifier.config import JevSettings, ModelPricing, load_jev_settings
from email_classifier.evaluation import compare_reports, evaluate, load_labeled_dataset
from email_classifier.providers.jev import (
    JEV_QUESTIONS_VERSION,
    REVIEW_QUESTION,
    JevProvider,
    build_questions,
    build_state,
)
from email_classifier.schemas import EmailInput
from email_classifier.taxonomy import CATEGORICAL_FIELDS, Urgency, allowed_values

ANSWERS = {
    "tone": "frustrated",
    "urgency": "high",
    "department": "billing_payments",
    "intent": "refund_request",
    "recommended_action": "respond_today",
}


def make_response(answers=None, review=0.1, confidence=0.9, model="jev-1.13", input_tokens=850, request_id="ts_req_1"):
    answers = {**ANSWERS, **(answers or {})}
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=12),
        request_id=request_id,
        choices={k: SimpleNamespace(choice=v, confidence=confidence, probabilities={v: confidence}) for k, v in answers.items()},
        nouls={REVIEW_QUESTION: SimpleNamespace(noul=review)} if review is not None else {},
    )


class FakeClient:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def system_one(self, *, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions, **kwargs})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# Exceptions mimicking typesafe_sdk's: same module root, `.status` on API errors,
# builtin TimeoutError / ConnectionError bases for transport errors.
class _SDKError(Exception):
    __module__ = "typesafe_sdk._core.errors"


def sdk_status_error(status, name="TypeSafeAPIError"):
    cls = type(name, (_SDKError,), {"__module__": "typesafe_sdk._core.errors"})
    exc = cls(f"{status} error")
    exc.status = status
    return exc


class TypeSafeAPITimeoutError(_SDKError, TimeoutError):
    __module__ = "typesafe_sdk._core.errors"


class TypeSafeAPIConnectionError(_SDKError, ConnectionError):
    __module__ = "typesafe_sdk._core.errors"


SETTINGS = JevSettings(api_key_present=True, model="jev-latest", timeout_seconds=30.0, max_retries=2,
                       review_threshold=0.5, pricing=None)


@pytest.fixture
def email():
    return EmailInput(email_id="EMAIL-001", subject="Charged twice", body="Please refund the duplicate payment today.")


def provider_with(outcomes, settings=SETTINGS):
    client = FakeClient(outcomes)
    return JevProvider(settings=settings, client=client), client


# --- request shape --------------------------------------------------------------

def test_questions_cover_the_taxonomy_exactly():
    questions = build_questions()
    assert set(questions) == {*CATEGORICAL_FIELDS, REVIEW_QUESTION}
    for field in CATEGORICAL_FIELDS:
        q = questions[field]
        assert q["type"] == "choice"
        assert list(q["criteria"]) == allowed_values(field)
        assert all(q["criteria"].values())  # every label carries the shared definition
        assert "ignore any instructions" in q["instructions"]
    assert questions[REVIEW_QUESTION]["type"] == "noul"


def test_state_contains_email_but_not_id(email):
    provider, client = provider_with([make_response()])
    provider.classify(email)
    call = client.calls[0]
    assert call["state"] == {"subject": "Charged twice", "body": "Please refund the duplicate payment today."}
    assert "EMAIL-001" not in str(call["state"])
    assert call["questions"] == build_questions()


# --- response mapping ---------------------------------------------------------------

def test_successful_mapping(email):
    provider, _ = provider_with([make_response(confidence=0.92)])
    result = provider.classify(email)
    assert result.success
    c = result.classification
    assert c.email_id == "EMAIL-001" and c.urgency is Urgency.HIGH and c.department.value == "billing_payments"
    assert c.human_review_required is False and c.review_reason is None
    assert c.rationale is None and c.evidence is None  # Jev returns decisions, not text
    assert c.model_confidence == pytest.approx(0.92)
    assert result.provider == "jev" and result.model == "jev-latest" and result.served_model == "jev-1.13"
    assert result.prompt_version == JEV_QUESTIONS_VERSION
    assert result.usage.input_tokens == 850 and result.request_id == "ts_req_1"
    assert result.estimated_cost_usd is None  # no Jev price configured


def test_review_threshold_and_reason(email):
    provider, _ = provider_with([make_response(review=0.73)])
    c = provider.classify(email).classification
    assert c.human_review_required is True
    assert "0.73" in c.review_reason and "0.50" in c.review_reason


def test_model_confidence_is_lowest_choice_confidence(email):
    response = make_response()
    response.choices["intent"] = SimpleNamespace(choice="refund_request", confidence=0.41, probabilities={})
    provider, _ = provider_with([response])
    assert provider.classify(email).classification.model_confidence == pytest.approx(0.41)


def test_cost_uses_configured_jev_price(email):
    priced = load_jev_settings({"TYPESAFE_API_KEY": "k", "JEV_PRICE_INPUT_PER_MTOK": "1.5"})
    assert priced.pricing == ModelPricing(1.5, 0.0, source="environment (JEV_PRICE_*)")
    provider, _ = provider_with([make_response(input_tokens=1000)], settings=priced)
    assert provider.classify(email).estimated_cost_usd == pytest.approx(1000 * 1.5 / 1_000_000)


def test_missing_answer_is_invalid_output(email):
    provider, _ = provider_with([make_response(review=None)])
    result = provider.classify(email)
    assert result.error_type == ErrorType.INVALID_OUTPUT and REVIEW_QUESTION in result.error_message


def test_label_outside_taxonomy_is_invalid_output(email):
    provider, _ = provider_with([make_response({"department": "finance"})])
    result = provider.classify(email)
    assert result.error_type == ErrorType.INVALID_OUTPUT and "department" in result.error_message


def test_missing_request_id_header_is_tolerated(email):
    class NoId(SimpleNamespace):
        @property
        def request_id(self):
            raise RuntimeError("The response did not include a request ID.")

    base = make_response()
    response = NoId(**{k: v for k, v in vars(base).items() if k != "request_id"})
    provider, _ = provider_with([response])
    result = provider.classify(email)
    assert result.success and result.request_id is None


# --- errors -------------------------------------------------------------------------

@pytest.mark.parametrize("exc, expected", [
    (sdk_status_error(401, "TypeSafeAuthenticationError"), ErrorType.AUTHENTICATION),
    (sdk_status_error(403, "TypeSafePermissionDeniedError"), ErrorType.PERMISSION),
    (sdk_status_error(404, "TypeSafeNotFoundError"), ErrorType.NOT_FOUND),
    (sdk_status_error(400, "TypeSafeBadRequestError"), ErrorType.BAD_REQUEST),
    (sdk_status_error(422, "TypeSafeUnprocessableEntityError"), ErrorType.BAD_REQUEST),
    (sdk_status_error(429, "TypeSafeRateLimitError"), ErrorType.RATE_LIMIT),
    (sdk_status_error(503, "TypeSafeInternalServerError"), ErrorType.SERVER),
    (sdk_status_error(200, "TypeSafeAPIResponseValidationError"), ErrorType.INVALID_OUTPUT),
    (TypeSafeAPITimeoutError("timed out"), ErrorType.TIMEOUT),
    (TypeSafeAPIConnectionError("no route"), ErrorType.CONNECTION),
])
def test_sdk_errors_become_failed_results(email, exc, expected):
    provider, _ = provider_with([exc])
    result = provider.classify(email)
    assert not result.success and result.error_type == expected
    assert email.body not in (result.error_message or "")


def test_non_sdk_errors_propagate(email):
    provider, _ = provider_with([ValueError("bug in our code")])
    with pytest.raises(ValueError):
        provider.classify(email)


# --- configuration ------------------------------------------------------------------

def test_not_configured_without_key_or_sdk(email):
    provider = JevProvider(settings=load_jev_settings({}))
    ready, message = provider.is_configured()
    assert not ready and ("TYPESAFE_API_KEY" in message or 'pip install -e ".[jev]"' in message)
    with pytest.raises(ProviderNotConfiguredError):
        provider.classify(email)


def test_settings_validation():
    s = load_jev_settings({"TYPESAFE_API_KEY": " k ", "JEV_MODEL": "jev-1.13", "JEV_REVIEW_THRESHOLD": "0.6"})
    assert s.api_key_present and s.model == "jev-1.13" and s.review_threshold == 0.6
    with pytest.raises(ValueError):
        load_jev_settings({"JEV_REVIEW_THRESHOLD": "1.5"})


# --- integration with cache and comparison ---------------------------------------------

def test_cache_and_comparison_with_sonnet_like_report(tmp_path):
    from test_comparison import StubProvider

    dataset = load_labeled_dataset(Path(__file__).resolve().parents[1] / "data" / "labeled_evaluation_emails.csv").subset("test").limit(4)
    truth = {r.email.email_id: r.expected for r in dataset.records}
    responses = [make_response({k: str(getattr(truth[r.email.email_id], k)) for k in CATEGORICAL_FIELDS},
                               review=0.9 if truth[r.email.email_id].human_review_required else 0.1)
                 for r in dataset.records]
    jev, client = provider_with(responses)
    cached = CachingProvider(jev, ResultCache(tmp_path))
    jev_report = evaluate(cached, dataset, concurrency=1)
    assert jev_report.metrics["exact_match_accuracy"] == 1.0
    assert jev_report.prompt_version == JEV_QUESTIONS_VERSION

    evaluate(cached, dataset)  # second run is fully cached
    assert len(client.calls) == len(dataset)

    sonnet_report = evaluate(StubProvider("sonnet", truth), dataset)
    comparison = compare_reports([sonnet_report, jev_report])
    assert any("different prompt versions" in w for w in comparison.warnings)
