"""Evaluation metrics, implemented without third-party dependencies.

Failed classifications are never dropped. In label-level metrics they appear
as the pseudo-label ``FAILED`` - always wrong, lowering recall for the true
class - so a provider cannot look more accurate by failing on hard emails.
Accuracy on successful predictions only is reported separately.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

from ..schemas import ClassificationResult, GroundTruth
from ..taxonomy import CATEGORICAL_FIELDS, URGENCY_SEVERITY, Urgency

FAILED = "__failed__"
EVALUATED_FIELDS: tuple[str, ...] = (*CATEGORICAL_FIELDS, "human_review_required")


# ---------------------------------------------------------------------------
# Generic label metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> Optional[float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return None
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_class_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1, and support per label (0.0 when undefined, as in scikit-learn's zero_division=0)."""
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def _metric_labels(y_true: Sequence[str], y_pred: Sequence[str]) -> list[str]:
    """Labels that occur in truth or predictions, excluding the FAILED pseudo-label."""
    return sorted({*y_true, *y_pred} - {FAILED})


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> Optional[float]:
    labels = _metric_labels(y_true, y_pred)
    if not labels:
        return None
    per = per_class_metrics(y_true, y_pred, labels)
    return sum(m["f1"] for m in per.values()) / len(labels)


def weighted_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> Optional[float]:
    labels = _metric_labels(y_true, y_pred)
    per = per_class_metrics(y_true, y_pred, labels)
    total = sum(m["support"] for m in per.values())
    if not total:
        return None
    return sum(m["f1"] * m["support"] for m in per.values()) / total


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """Nested dict: matrix[true_label][predicted_label] = count. Includes a FAILED column when needed."""
    columns = list(labels) + ([FAILED] if FAILED in y_pred else [])
    matrix = {t: {p: 0 for p in columns} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix.setdefault(t, {c: 0 for c in columns})
        matrix[t][p] = matrix[t].get(p, 0) + 1
    return matrix


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[Optional[bool]]) -> dict[str, Any]:
    """Metrics for a boolean target with True as the positive class. None predictions count as wrong."""
    tp = sum(t and p is True for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p is True for t, p in zip(y_true, y_pred))
    fn = sum(t and p is not True for t, p in zip(y_true, y_pred))
    tn = sum((not t) and p is False for t, p in zip(y_true, y_pred))
    failed = sum(p is None for p in y_pred)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    n = len(y_true)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "failed": failed,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision, "recall": recall, "f1": f1,
        "positives_expected": tp + fn,
    }


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (same method as numpy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def latency_summary(latencies_ms: Iterable[Optional[float]]) -> dict[str, Optional[float]]:
    values = [v for v in latencies_ms if v is not None]
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def mcnemar_exact_p(only_a_correct: int, only_b_correct: int) -> Optional[float]:
    """Two-sided exact McNemar test on discordant pairs. None when there are no discordant pairs."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return None
    k = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Classification-specific metrics
# ---------------------------------------------------------------------------

def predicted_label(result: ClassificationResult, field: str) -> str:
    if result.classification is None:
        return FAILED
    value = getattr(result.classification, field)
    return str(value).lower() if isinstance(value, bool) else str(value)


def expected_label(truth: GroundTruth, field: str) -> str:
    value = getattr(truth, field)
    return str(value).lower() if isinstance(value, bool) else str(value)


def is_field_correct(truth: GroundTruth, result: ClassificationResult, field: str) -> bool:
    return result.classification is not None and expected_label(truth, field) == predicted_label(result, field)


def is_exact_match(truth: GroundTruth, result: ClassificationResult) -> bool:
    return all(is_field_correct(truth, result, f) for f in EVALUATED_FIELDS)


def compute_metrics(pairs: Sequence[tuple[GroundTruth, ClassificationResult]]) -> dict[str, Any]:
    """Compute the full metric set for (ground truth, result) pairs from one provider run."""
    n = len(pairs)
    successes = [(t, r) for t, r in pairs if r.success]
    failures = [r for _, r in pairs if not r.success]

    fields: dict[str, Any] = {}
    for field in EVALUATED_FIELDS:
        y_true = [expected_label(t, field) for t, _ in pairs]
        y_pred = [predicted_label(r, field) for _, r in pairs]
        labels = _metric_labels(y_true, y_pred)
        fields[field] = {
            "accuracy": accuracy(y_true, y_pred),
            "accuracy_on_success": accuracy(
                [expected_label(t, field) for t, _ in successes],
                [predicted_label(r, field) for _, r in successes],
            ),
            "macro_f1": macro_f1(y_true, y_pred),
            "weighted_f1": weighted_f1(y_true, y_pred),
            "labels": labels,
            "per_class": per_class_metrics(y_true, y_pred, labels),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
        }

    exact = [is_exact_match(t, r) for t, r in pairs]
    exact_success = [is_exact_match(t, r) for t, r in successes]

    review = binary_metrics(
        [t.human_review_required for t, _ in pairs],
        [r.classification.human_review_required if r.classification else None for _, r in pairs],
    )

    costs = [r.estimated_cost_usd for _, r in pairs if r.estimated_cost_usd is not None]
    tokens = Counter()
    for _, r in pairs:
        if r.usage:
            for key, value in r.usage.model_dump().items():
                tokens[key] += value or 0
    grounded = [r.evidence_grounded for _, r in successes if r.evidence_grounded is not None]

    return {
        "n_records": n,
        "n_success": len(successes),
        "n_failed": len(failures),
        "coverage": len(successes) / n if n else None,
        "error_rate": len(failures) / n if n else None,
        "errors_by_type": dict(sorted(Counter(r.error_type for r in failures).items())),
        "exact_match_accuracy": sum(exact) / n if n else None,
        "exact_match_accuracy_on_success": sum(exact_success) / len(successes) if successes else None,
        "fields": fields,
        "human_review": review,
        "urgency_triage": urgency_triage(pairs),
        "latency_ms": latency_summary(r.latency_ms for _, r in pairs),
        "latency_ms_success_only": latency_summary(r.latency_ms for _, r in successes),
        "cost": {
            "available": len(costs) == n and n > 0,
            "records_with_cost": len(costs),
            "total_usd": sum(costs) if costs else None,
            "mean_per_email_usd": sum(costs) / len(costs) if costs else None,
            "tokens": dict(tokens),
        },
        "this_run": this_run_usage([r for _, r in pairs]),
        "mean_attempts": sum(r.attempts for _, r in pairs) / n if n else None,
        "evidence_grounded_rate": sum(grounded) / len(grounded) if grounded else None,
    }


def this_run_usage(results: Sequence[ClassificationResult]) -> dict[str, Any]:
    """API calls and spend incurred by this run (cached results cost nothing now).

    ``api_calls`` counts provider requests (attempts). Transport-level retries
    performed inside a vendor SDK (e.g. on 429/5xx) are not visible here.
    """
    fresh = [r for r in results if not r.from_cache]
    costs = [r.estimated_cost_usd for r in fresh if r.estimated_cost_usd is not None]
    return {
        "api_calls": sum(r.attempts for r in fresh),
        "results_from_cache": len(results) - len(fresh),
        "new_spend_usd": sum(costs) if costs else (0.0 if not fresh else None),
    }


def urgency_triage(pairs: Sequence[tuple[GroundTruth, ClassificationResult]]) -> dict[str, Any]:
    """Under-/over-triage analysis, with special attention to critical emails."""
    critical_ids, under_critical, failed_critical = [], [], []
    under_any, over_any = [], []
    for truth, result in pairs:
        email_id = result.email_id
        if truth.urgency is Urgency.CRITICAL:
            critical_ids.append(email_id)
            if result.classification is None:
                failed_critical.append(email_id)
            elif result.classification.urgency is not Urgency.CRITICAL:
                under_critical.append(email_id)
        if result.classification is None:
            continue
        diff = URGENCY_SEVERITY[result.classification.urgency] - URGENCY_SEVERITY[truth.urgency]
        if diff < 0:
            under_any.append(email_id)
        elif diff > 0:
            over_any.append(email_id)
    n_critical = len(critical_ids)
    return {
        "critical_total": n_critical,
        "critical_under_triaged": under_critical,
        "critical_failed": failed_critical,
        "critical_miss_rate": (len(under_critical) + len(failed_critical)) / n_critical if n_critical else None,
        "under_triaged": under_any,
        "over_triaged": over_any,
    }
