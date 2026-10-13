import pytest

from conftest import VALID_OUTPUT
from email_classifier.evaluation.metrics import (
    FAILED,
    accuracy,
    binary_metrics,
    compute_metrics,
    confusion_matrix,
    macro_f1,
    mcnemar_exact_p,
    per_class_metrics,
    percentile,
    weighted_f1,
)
from email_classifier.schemas import (
    ClassificationResult,
    EmailClassification,
    GroundTruth,
    ModelClassificationOutput,
    UsageInfo,
)

Y_TRUE = ["a", "a", "b", "b", "c"]
Y_PRED = ["a", "b", "b", "b", "a"]


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.6)
    assert accuracy([], []) is None
    with pytest.raises(ValueError):
        accuracy(["a"], [])


def test_per_class_and_f1_match_hand_computation():
    per = per_class_metrics(Y_TRUE, Y_PRED, ["a", "b", "c"])
    assert per["a"] == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2})
    assert per["b"]["f1"] == pytest.approx(0.8)
    assert per["c"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert macro_f1(Y_TRUE, Y_PRED) == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert weighted_f1(Y_TRUE, Y_PRED) == pytest.approx(0.52)


def test_failed_predictions_lower_recall_but_not_label_set():
    y_true, y_pred = ["a", "b"], ["a", FAILED]
    assert accuracy(y_true, y_pred) == 0.5
    assert macro_f1(y_true, y_pred) == pytest.approx(0.5)  # a: 1.0, b: 0.0
    cm = confusion_matrix(y_true, y_pred, ["a", "b"])
    assert cm == {"a": {"a": 1, "b": 0, FAILED: 0}, "b": {"a": 0, "b": 0, FAILED: 1}}


def test_binary_metrics():
    m = binary_metrics([True, True, False, False, True], [True, False, True, False, None])
    assert (m["tp"], m["fp"], m["fn"], m["tn"], m["failed"]) == (1, 1, 2, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(0.4)
    assert binary_metrics([False], [False])["recall"] is None


def test_percentile_and_mcnemar():
    assert percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)
    assert percentile([], 50) is None
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact_p(3, 3) == 1.0
    assert mcnemar_exact_p(0, 0) is None


# --- end-to-end metric computation -------------------------------------------------

def truth(**kw) -> GroundTruth:
    base = {k: VALID_OUTPUT[k] for k in ("tone", "urgency", "department", "intent", "recommended_action", "human_review_required")}
    return GroundTruth(**{**base, **kw})


def ok(email_id: str, latency: float = 100.0, cost: float = 0.001, **kw) -> ClassificationResult:
    output = ModelClassificationOutput.model_validate({**VALID_OUTPUT, **kw})
    return ClassificationResult(
        email_id=email_id, provider="p", classification=EmailClassification.from_model_output(email_id, output),
        latency_ms=latency, attempts=1, estimated_cost_usd=cost, evidence_grounded=True,
        usage=UsageInfo(input_tokens=100, output_tokens=50),
    )


def failed(email_id: str) -> ClassificationResult:
    return ClassificationResult(email_id=email_id, provider="p", error_type="timeout_error", latency_ms=500.0, attempts=1)


def test_compute_metrics_counts_failures_as_wrong():
    pairs = [
        (truth(), ok("E1")),                                    # exact match
        (truth(urgency="critical", recommended_action="immediate_escalation"), ok("E2")),  # critical under-triaged
        (truth(urgency="critical", recommended_action="immediate_escalation"), failed("E3")),
        (truth(urgency="low", recommended_action="standard_queue"), ok("E4", latency=300)),  # over-triaged
    ]
    m = compute_metrics(pairs)
    assert m["n_records"] == 4 and m["n_success"] == 3 and m["n_failed"] == 1
    assert m["coverage"] == 0.75 and m["errors_by_type"] == {"timeout_error": 1}
    assert m["exact_match_accuracy"] == pytest.approx(0.25)
    assert m["exact_match_accuracy_on_success"] == pytest.approx(1 / 3)
    assert m["fields"]["urgency"]["accuracy"] == pytest.approx(0.25)
    assert m["fields"]["department"]["accuracy"] == pytest.approx(0.75)
    assert m["fields"]["department"]["accuracy_on_success"] == 1.0
    ut = m["urgency_triage"]
    assert ut["critical_total"] == 2
    assert ut["critical_under_triaged"] == ["E2"] and ut["critical_failed"] == ["E3"]
    assert ut["critical_miss_rate"] == 1.0
    assert ut["under_triaged"] == ["E2"] and ut["over_triaged"] == ["E4"]
    assert m["latency_ms"]["n"] == 4 and m["latency_ms"]["max"] == 500.0
    assert m["latency_ms_success_only"]["n"] == 3
    assert m["cost"]["available"] is False and m["cost"]["total_usd"] == pytest.approx(0.003)
    assert m["cost"]["tokens"]["input_tokens"] == 300
    assert m["human_review"]["failed"] == 1
