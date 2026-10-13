"""Result cache, warm-up/fail-fast behaviour, and dry runs: everything that limits API calls."""

from pathlib import Path

import pytest

from conftest import make_response
from email_classifier.cache import CachingProvider, ResultCache
from email_classifier.classifier_interface import ErrorType
from email_classifier.evaluation import evaluate, load_labeled_dataset
from email_classifier.evaluation.runner import EvaluationAborted
from email_classifier.schemas import ClassificationResult, EmailInput
from test_comparison import StubProvider

DATASET = Path(__file__).resolve().parents[1] / "data" / "labeled_evaluation_emails.csv"


@pytest.fixture(scope="module")
def dataset():
    return load_labeled_dataset(DATASET).subset("test").limit(8)


@pytest.fixture(scope="module")
def truth(dataset):
    return {r.email.email_id: r.expected for r in dataset.records}


class ErrorStub(StubProvider):
    """Stub that returns a given error type for selected emails."""

    def __init__(self, name, truth_by_id, errors):
        super().__init__(name, truth_by_id)
        self._errors = errors

    def classify(self, email):
        if email.email_id in self._errors:
            self.calls += 1
            return ClassificationResult(email_id=email.email_id, provider=self.name,
                                        error_type=self._errors[email.email_id], error_message="x", attempts=1)
        return super().classify(email)


# --- result cache ------------------------------------------------------------------

def test_second_run_makes_no_api_calls(tmp_path, dataset, truth):
    inner = StubProvider("sonnet", truth)
    cached = CachingProvider(inner, ResultCache(tmp_path))
    first = evaluate(cached, dataset)
    assert inner.calls == len(dataset)
    assert first.metrics["this_run"] == {"api_calls": len(dataset), "results_from_cache": 0,
                                         "new_spend_usd": pytest.approx(0.0005 * len(dataset))}

    second = evaluate(cached, dataset)
    assert inner.calls == len(dataset)  # unchanged: everything served from cache
    assert all(r.result.from_cache for r in second.records)
    assert second.metrics["this_run"] == {"api_calls": 0, "results_from_cache": len(dataset), "new_spend_usd": 0.0}
    # Accuracy, latency, and cost of producing the predictions are unchanged.
    assert second.metrics["exact_match_accuracy"] == first.metrics["exact_match_accuracy"]
    assert second.metrics["cost"]["total_usd"] == first.metrics["cost"]["total_usd"]


def test_cache_is_keyed_on_content_not_email_id(tmp_path, truth):
    inner = StubProvider("sonnet", {"A": truth["EMAIL-T001"], "B": truth["EMAIL-T001"]})
    cached = CachingProvider(inner, ResultCache(tmp_path))
    cached.classify(EmailInput(email_id="A", subject="s", body="same body"))
    hit = cached.classify(EmailInput(email_id="B", subject="s", body="same body"))
    assert inner.calls == 1
    assert hit.email_id == "B" and hit.classification.email_id == "B" and hit.from_cache


def test_failures_are_not_cached(tmp_path, dataset, truth):
    failing_id = dataset.records[1].email.email_id
    inner = StubProvider("sonnet", truth, fail=[failing_id])
    cached = CachingProvider(inner, ResultCache(tmp_path))
    evaluate(cached, dataset)
    calls_after_first = inner.calls
    evaluate(cached, dataset)
    assert inner.calls == calls_after_first + 1  # only the failed email is retried


def test_config_change_misses_cache(tmp_path, make_provider, email):
    cache = ResultCache(tmp_path)
    low, low_client = make_provider([make_response()])
    CachingProvider(low, cache).classify(email)
    high, high_client = make_provider([make_response()], effort="high")
    CachingProvider(high, cache).classify(email)
    assert len(low_client.messages.calls) == 1 and len(high_client.messages.calls) == 1
    again, again_client = make_provider([])
    assert CachingProvider(again, cache).classify(email).from_cache
    assert again_client.messages.calls == []


def test_refresh_bypasses_cache(tmp_path, truth):
    inner = StubProvider("sonnet", truth)
    email = EmailInput(email_id="EMAIL-T001", body="hello")
    CachingProvider(inner, ResultCache(tmp_path)).classify(email)
    CachingProvider(inner, ResultCache(tmp_path), refresh=True).classify(email)
    assert inner.calls == 2


def test_corrupt_cache_entry_is_ignored(tmp_path, truth):
    inner = StubProvider("sonnet", truth)
    cache = ResultCache(tmp_path)
    cached = CachingProvider(inner, cache)
    email = EmailInput(email_id="EMAIL-T001", body="hello")
    cached.classify(email)
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{broken", encoding="utf-8")
    assert not cached.classify(email).from_cache
    assert inner.calls == 2


# --- warm-up and fail-fast ------------------------------------------------------------

@pytest.mark.parametrize("error", [ErrorType.AUTHENTICATION, ErrorType.PERMISSION, ErrorType.NOT_FOUND,
                                   ErrorType.BAD_REQUEST])
def test_configuration_error_on_first_call_stops_the_run(dataset, truth, error):
    first_id = dataset.records[0].email.email_id
    provider = ErrorStub("sonnet", truth, {first_id: error})
    with pytest.raises(EvaluationAborted, match=error):
        evaluate(provider, dataset, concurrency=4)
    assert provider.calls == 1


@pytest.mark.parametrize("error", [ErrorType.RATE_LIMIT, ErrorType.TIMEOUT, ErrorType.CONNECTION, ErrorType.SERVER])
def test_two_consecutive_transient_failures_stop_the_run(dataset, truth, error):
    ids = [r.email.email_id for r in dataset.records[:2]]
    provider = ErrorStub("sonnet", truth, {i: error for i in ids})
    with pytest.raises(EvaluationAborted, match="2 consecutive"):
        evaluate(provider, dataset, concurrency=4)
    assert provider.calls == 2


def test_single_transient_failure_is_tolerated(dataset, truth):
    provider = ErrorStub("sonnet", truth, {dataset.records[0].email.email_id: ErrorType.TIMEOUT})
    report = evaluate(provider, dataset)
    assert provider.calls == len(dataset)
    assert report.metrics["errors_by_type"] == {ErrorType.TIMEOUT: 1}


def test_email_specific_first_failure_does_not_stop_the_run(dataset, truth):
    first_id = dataset.records[0].email.email_id
    provider = ErrorStub("sonnet", truth, {first_id: ErrorType.INVALID_OUTPUT})
    report = evaluate(provider, dataset)
    assert provider.calls == len(dataset)
    assert report.metrics["errors_by_type"] == {ErrorType.INVALID_OUTPUT: 1}


def test_fatal_error_mid_run_cancels_remaining_calls(dataset, truth):
    third_id = dataset.records[2].email.email_id
    provider = ErrorStub("sonnet", truth, {third_id: ErrorType.AUTHENTICATION})
    with pytest.raises(EvaluationAborted, match="authentication_error"):
        evaluate(provider, dataset, concurrency=1)
    assert provider.calls == 3


def test_fully_cached_run_skips_warmup_and_calls(tmp_path, dataset, truth):
    inner = StubProvider("sonnet", truth)
    cached = CachingProvider(inner, ResultCache(tmp_path))
    evaluate(cached, dataset)
    inner.calls = 0
    evaluate(cached, dataset)
    assert inner.calls == 0


# --- CLI --------------------------------------------------------------------------

def test_dry_run_makes_no_calls(tmp_path, monkeypatch, capsys):
    from email_classifier.cli import EXIT_OK, main

    monkeypatch.setenv("CLASSIFIER_CACHE_DIR", str(tmp_path))
    for var in ("ANTHROPIC_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "CLAUDE_PLATFORM"):
        monkeypatch.delenv(var, raising=False)
    assert main(["--env-file", "", "evaluate", "--provider", "sonnet", "--split", "test", "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "0 cached, 31 API call(s) expected" in out and "NOT CONFIGURED" in out
