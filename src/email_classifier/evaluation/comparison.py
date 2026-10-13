"""Compare providers on the same frozen dataset.

Comparisons can be built from live runs (``compare_providers``) or from saved
evaluation reports (``compare_reports``), for example a Sonnet report produced
on one machine and a Jev report produced on another. Reports are only
compared when they were produced from the identical dataset fingerprint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from ..classifier_interface import EmailClassifierProvider
from .dataset_loader import Dataset
from .metrics import EVALUATED_FIELDS, expected_label, is_exact_match, is_field_correct, mcnemar_exact_p, predicted_label
from .runner import EvaluationReport, ProgressCallback, evaluate


class ComparisonError(ValueError):
    """Raised when reports cannot be compared fairly."""


class ComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str
    dataset_fingerprint: str
    dataset_name: str
    dataset_split: str
    n_records: int
    providers: list[str]
    run_ids: dict[str, str]
    models: dict[str, str]
    prompt_versions: dict[str, str]
    warnings: list[str]
    summary: dict[str, dict[str, Any]]
    field_accuracy: dict[str, dict[str, Optional[float]]]
    pairwise: list[dict[str, Any]]
    disagreements: list[dict[str, Any]]


def _summary_row(report: EvaluationReport) -> dict[str, Any]:
    m = report.metrics
    return {
        "model": report.model,
        "exact_match_accuracy": m["exact_match_accuracy"],
        "exact_match_accuracy_on_success": m["exact_match_accuracy_on_success"],
        "coverage": m["coverage"],
        "error_rate": m["error_rate"],
        "errors_by_type": m["errors_by_type"],
        "macro_f1": {f: m["fields"][f]["macro_f1"] for f in EVALUATED_FIELDS},
        "weighted_f1": {f: m["fields"][f]["weighted_f1"] for f in EVALUATED_FIELDS},
        "human_review": {k: m["human_review"][k] for k in ("precision", "recall", "f1", "accuracy")},
        "critical_total": m["urgency_triage"]["critical_total"],
        "critical_missed": m["urgency_triage"]["critical_under_triaged"] + m["urgency_triage"]["critical_failed"],
        "critical_miss_rate": m["urgency_triage"]["critical_miss_rate"],
        "latency_ms": m["latency_ms"],
        "cost_total_usd": m["cost"]["total_usd"],
        "cost_mean_per_email_usd": m["cost"]["mean_per_email_usd"],
        "cost_available": m["cost"]["available"],
    }


def _validate_comparable(reports: Sequence[EvaluationReport]) -> list[str]:
    if len(reports) < 2:
        raise ComparisonError("At least two evaluation reports are required for a comparison.")
    fingerprints = {r.dataset_fingerprint for r in reports}
    if len(fingerprints) != 1:
        details = ", ".join(f"{r.provider}={r.dataset_fingerprint[:12]}" for r in reports)
        raise ComparisonError(f"Reports were produced from different datasets ({details}); refusing to compare.")
    ids = [tuple(rec.email_id for rec in r.records) for r in reports]
    if len(set(ids)) != 1:
        raise ComparisonError("Reports do not contain the same email_ids in the same order.")
    names = [r.provider for r in reports]
    if len(set(names)) != len(names):
        raise ComparisonError(f"Duplicate provider names in comparison: {names}")

    warnings = []
    if len({r.prompt_fingerprint for r in reports}) > 1:
        versions = ", ".join(f"{r.provider}={r.prompt_version}/{r.prompt_fingerprint}" for r in reports)
        warnings.append(f"Providers used different prompt versions ({versions}); differences may reflect the prompt, not the model.")
    n = len(reports[0].records)
    if n < 100:
        warnings.append(
            f"Only {n} evaluation records. Differences of a few emails are not statistically meaningful; "
            "see the McNemar p-values and treat results as indicative, not production accuracy."
        )
    return warnings


def _pairwise(a: EvaluationReport, b: EvaluationReport) -> dict[str, Any]:
    out: dict[str, Any] = {"a": a.provider, "b": b.provider, "fields": {}}
    checks = {**{f: (lambda t, r, f=f: is_field_correct(t, r, f)) for f in EVALUATED_FIELDS}, "exact_match": is_exact_match}
    for name, check in checks.items():
        both = only_a = only_b = neither = 0
        for ra, rb in zip(a.records, b.records):
            ca, cb = check(ra.expected, ra.result), check(rb.expected, rb.result)
            both += ca and cb
            only_a += ca and not cb
            only_b += cb and not ca
            neither += not ca and not cb
        out["fields"][name] = {
            "both_correct": both,
            f"only_{a.provider}_correct": only_a,
            f"only_{b.provider}_correct": only_b,
            "both_wrong": neither,
            "mcnemar_p_value": mcnemar_exact_p(only_a, only_b),
        }
    return out


def _disagreements(reports: Sequence[EvaluationReport]) -> list[dict[str, Any]]:
    rows = []
    for idx, first in enumerate(reports[0].records):
        per_provider = {r.provider: r.records[idx] for r in reports}
        differing = [
            f for f in EVALUATED_FIELDS
            if len({predicted_label(rec.result, f) for rec in per_provider.values()}) > 1
        ]
        if not differing:
            continue
        rows.append({
            "email_id": first.email_id,
            "fields": differing,
            "expected": {f: expected_label(first.expected, f) for f in differing},
            "predictions": {
                name: {
                    f: predicted_label(rec.result, f) for f in differing
                } | ({"error_type": rec.result.error_type} if not rec.result.success else {})
                for name, rec in per_provider.items()
            },
            "correct": {
                name: [f for f in differing if is_field_correct(rec.expected, rec.result, f)]
                for name, rec in per_provider.items()
            },
        })
    return rows


def compare_reports(reports: Sequence[EvaluationReport]) -> ComparisonReport:
    warnings = _validate_comparable(reports)
    first = reports[0]
    return ComparisonReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=first.dataset_fingerprint,
        dataset_name=first.dataset_name,
        dataset_split=first.dataset_split,
        n_records=len(first.records),
        providers=[r.provider for r in reports],
        run_ids={r.provider: r.run_id for r in reports},
        models={r.provider: r.model for r in reports},
        prompt_versions={r.provider: r.prompt_version for r in reports},
        warnings=warnings,
        summary={r.provider: _summary_row(r) for r in reports},
        field_accuracy={
            f: {r.provider: r.metrics["fields"][f]["accuracy"] for r in reports}
            for f in EVALUATED_FIELDS
        } | {"exact_match": {r.provider: r.metrics["exact_match_accuracy"] for r in reports}},
        pairwise=[_pairwise(a, b) for a, b in combinations(reports, 2)],
        disagreements=_disagreements(reports),
    )


def compare_providers(
    providers: Sequence[EmailClassifierProvider],
    dataset: Dataset,
    *,
    concurrency: int = 4,
    progress: Optional[ProgressCallback] = None,
) -> tuple[ComparisonReport, list[EvaluationReport]]:
    """Run every provider on the same dataset, then compare.

    All providers are checked for configuration before any API call is made,
    so a missing provider fails fast instead of after spending on the others.
    """
    for provider in providers:
        provider.ensure_configured()
    reports = [evaluate(p, dataset, concurrency=concurrency, progress=progress) for p in providers]
    return compare_reports(reports), reports
