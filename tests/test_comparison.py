import csv
from pathlib import Path

import pytest

from email_classifier.classifier_interface import EmailClassifierProvider, ErrorType, ProviderNotConfiguredError
from email_classifier.evaluation import (
    ComparisonError,
    compare_providers,
    compare_reports,
    evaluate,
    load_labeled_dataset,
)
from email_classifier.providers.jev import JevProvider
from email_classifier.reporting import (
    load_evaluation_report,
    render_comparison_markdown,
    render_evaluation_markdown,
    save_comparison_report,
    save_evaluation_report,
)
from email_classifier.schemas import ClassificationResult, EmailClassification, EmailInput, ModelClassificationOutput

DATASET = Path(__file__).resolve().parents[1] / "data" / "labeled_evaluation_emails.csv"


class StubProvider(EmailClassifierProvider):
    """Returns the ground truth, except for configured mistakes or failures."""

    def __init__(self, name, truth_by_id, wrong=None, fail=(), crash=()):
        self.name = name
        self._truth = truth_by_id
        self._wrong = wrong or {}
        self._fail = set(fail)
        self._crash = set(crash)
        self.calls = 0

    @property
    def model_id(self):
        return f"{self.name}-model"

    def is_configured(self):
        return True, "stub"

    def classify(self, email: EmailInput) -> ClassificationResult:
        self.calls += 1
        if email.email_id in self._crash:
            raise RuntimeError("boom")
        if email.email_id in self._fail:
            return ClassificationResult(email_id=email.email_id, provider=self.name, error_type=ErrorType.TIMEOUT, latency_ms=10.0, attempts=1)
        labels = self._truth[email.email_id].model_dump()
        labels.update(self._wrong.get(email.email_id, {}))
        output = ModelClassificationOutput(
            **labels,
            review_reason="needs review" if labels["human_review_required"] else None,
            rationale="stub", evidence=email.body[:20] or "x",
        )
        return ClassificationResult(
            email_id=email.email_id, provider=self.name, model=self.model_id,
            classification=EmailClassification.from_model_output(email.email_id, output),
            latency_ms=50.0, attempts=1, estimated_cost_usd=0.0005,
        )


@pytest.fixture(scope="module")
def dataset():
    return load_labeled_dataset(DATASET).subset("test")


@pytest.fixture(scope="module")
def truth(dataset):
    return {r.email.email_id: r.expected for r in dataset.records}


def test_evaluate_preserves_order_and_accounts_for_every_record(dataset, truth):
    provider = StubProvider("a", truth, fail=["EMAIL-T005"], crash=["EMAIL-T006"])
    report = evaluate(provider, dataset, concurrency=4)
    assert [r.email_id for r in report.records] == [r.email.email_id for r in dataset.records]
    assert report.metrics["n_records"] == len(dataset)
    assert report.metrics["n_failed"] == 2
    assert report.metrics["errors_by_type"] == {ErrorType.TIMEOUT: 1, ErrorType.UNEXPECTED: 1}
    assert report.dataset_fingerprint == dataset.fingerprint
    assert report.recompute_metrics() == report.metrics


def test_comparison_of_two_providers(dataset, truth):
    a = StubProvider("alpha", truth)
    b = StubProvider("beta", truth, wrong={"EMAIL-T001": {"urgency": "medium", "recommended_action": "standard_queue"}},
                     fail=["EMAIL-T002"])
    comparison, reports = compare_providers([a, b], dataset, concurrency=2)

    assert comparison.providers == ["alpha", "beta"]
    assert comparison.summary["alpha"]["exact_match_accuracy"] == 1.0
    assert comparison.summary["beta"]["critical_missed"] == ["EMAIL-T001"]
    assert comparison.field_accuracy["urgency"]["alpha"] == 1.0
    assert comparison.field_accuracy["urgency"]["beta"] == pytest.approx((len(dataset) - 2) / len(dataset))

    exact = comparison.pairwise[0]["fields"]["exact_match"]
    assert exact["only_alpha_correct"] == 2 and exact["only_beta_correct"] == 0
    assert exact["mcnemar_p_value"] == pytest.approx(0.5)

    disagreements = {d["email_id"]: d for d in comparison.disagreements}
    assert set(disagreements) == {"EMAIL-T001", "EMAIL-T002"}
    assert disagreements["EMAIL-T001"]["fields"] == ["urgency", "recommended_action"]
    assert disagreements["EMAIL-T001"]["correct"] == {"alpha": ["urgency", "recommended_action"], "beta": []}
    assert disagreements["EMAIL-T002"]["predictions"]["beta"]["error_type"] == ErrorType.TIMEOUT
    assert any("not statistically meaningful" in w for w in comparison.warnings)


def test_compare_refuses_different_datasets(dataset, truth):
    full = load_labeled_dataset(DATASET)
    full_truth = {r.email.email_id: r.expected for r in full.records}
    r1 = evaluate(StubProvider("a", truth), dataset)
    r2 = evaluate(StubProvider("b", full_truth), full)
    with pytest.raises(ComparisonError, match="different datasets"):
        compare_reports([r1, r2])
    with pytest.raises(ComparisonError, match="At least two"):
        compare_reports([r1])
    with pytest.raises(ComparisonError, match="Duplicate provider"):
        compare_reports([r1, evaluate(StubProvider("a", truth), dataset)])


def test_compare_fails_fast_before_spending_on_other_providers(dataset, truth):
    from email_classifier.config import load_jev_settings

    sonnet_like = StubProvider("sonnet", truth)
    unconfigured_jev = JevProvider(settings=load_jev_settings({}))  # no TYPESAFE_API_KEY
    with pytest.raises(ProviderNotConfiguredError, match="jev"):
        compare_providers([sonnet_like, unconfigured_jev], dataset)
    assert sonnet_like.calls == 0


def test_reports_round_trip_and_render(tmp_path, dataset, truth):
    r1 = evaluate(StubProvider("alpha", truth), dataset)
    r2 = evaluate(StubProvider("beta", truth, wrong={"EMAIL-T003": {"tone": "angry"}}), dataset)
    paths = save_evaluation_report(r1, tmp_path)
    assert all(p.exists() for p in paths.values())
    reloaded = load_evaluation_report(paths["json"])
    assert reloaded == r1

    # Comparing saved reports (e.g. Sonnet from one machine, Jev from another).
    comparison = compare_reports([reloaded, r2])
    md = render_comparison_markdown(comparison)
    assert "alpha vs. beta" in md and "EMAIL-T003" in md and "angry" in md
    saved = save_comparison_report(comparison, tmp_path)
    assert saved["markdown"].read_text(encoding="utf-8").startswith("# Provider comparison")

    eval_md = render_evaluation_markdown(r2)
    assert "Exact-match accuracy" in eval_md and "## Confusion matrices" in eval_md
    assert "| EMAIL-T003 | tone | frustrated | angry |" in eval_md
    with paths["csv"].open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:4] == ["email_id", "split", "success", "error_type"]
    assert len(rows) == len(dataset) + 1
