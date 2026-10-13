"""Write evaluation and comparison reports as JSON, Markdown, and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .evaluation.comparison import ComparisonReport
from .evaluation.metrics import EVALUATED_FIELDS, expected_label, is_exact_match, predicted_label
from .evaluation.runner import EvaluationReport

SMALL_DATASET_CAVEAT = (
    "Results come from a small synthetic dataset. They are useful for comparing providers under identical "
    "conditions and for finding systematic mistakes, but they do not establish production-level accuracy."
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


def _usd(value: Optional[float]) -> str:
    return "n/a" if value is None else f"${value:,.5f}"


def _table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def render_evaluation_markdown(report: EvaluationReport) -> str:
    m = report.metrics
    n = m["n_records"]
    exact_n = sum(is_exact_match(r.expected, r.result) for r in report.records)
    out = [
        f"# Email classification evaluation: {report.provider} ({report.model})",
        "",
        _table(["Item", "Value"], [
            ["Run id", report.run_id],
            ["Dataset", f"{report.dataset_name} (split: {report.dataset_split}, {n} records)"],
            ["Dataset fingerprint", report.dataset_fingerprint[:16]],
            ["Prompt version", f"{report.prompt_version} ({report.prompt_fingerprint})"],
            ["Started / finished (UTC)", f"{report.started_at} / {report.finished_at}"],
            *([k, v] for k, v in report.run_settings.items()),
        ]),
        "",
        f"> {SMALL_DATASET_CAVEAT}",
        "",
        "## Headline",
        "",
        f"- **Exact-match accuracy (all six fields correct; failures count as wrong): {_pct(m['exact_match_accuracy'])}** ({exact_n}/{n})",
        f"- Exact-match accuracy on successful classifications only: {_pct(m['exact_match_accuracy_on_success'])}",
        f"- Coverage: {_pct(m['coverage'])} ({m['n_success']}/{n}); error rate: {_pct(m['error_rate'])}"
        + (f"; errors: {m['errors_by_type']}" if m["errors_by_type"] else ""),
        f"- Critical emails missed (classified lower or failed): {len(m['urgency_triage']['critical_under_triaged']) + len(m['urgency_triage']['critical_failed'])}"
        f" of {m['urgency_triage']['critical_total']}",
        f"- Evidence excerpts found verbatim in the email: {_pct(m['evidence_grounded_rate'])}",
        "",
        "## Per-field metrics",
        "",
        _table(
            ["Field", "Accuracy", "Accuracy (successful only)", "Macro F1", "Weighted F1"],
            [[f, _pct(fm["accuracy"]), _pct(fm["accuracy_on_success"]), _num(fm["macro_f1"]), _num(fm["weighted_f1"])]
             for f, fm in m["fields"].items()],
        ),
        "",
        "## Urgency triage",
        "",
    ]
    ut = m["urgency_triage"]
    out += [
        f"- Critical emails: {ut['critical_total']}; missed rate: {_pct(ut['critical_miss_rate'])}",
        f"- Critical classified lower: {', '.join(ut['critical_under_triaged']) or 'none'}",
        f"- Critical failed: {', '.join(ut['critical_failed']) or 'none'}",
        f"- Any under-triage (urgency predicted lower than expected): {len(ut['under_triaged'])} ({', '.join(ut['under_triaged']) or 'none'})",
        f"- Any over-triage (urgency predicted higher than expected): {len(ut['over_triaged'])} ({', '.join(ut['over_triaged']) or 'none'})",
        "",
        "## Human-review flag",
        "",
    ]
    hr = m["human_review"]
    out += [
        _table(["Precision", "Recall", "F1", "Accuracy", "TP", "FP", "FN", "TN", "Failed"],
               [[_pct(hr["precision"]), _pct(hr["recall"]), _num(hr["f1"]), _pct(hr["accuracy"]),
                 hr["tp"], hr["fp"], hr["fn"], hr["tn"], hr["failed"]]]),
        "",
        "## Latency (per email, including retries)",
        "",
    ]
    lat = m["latency_ms"]
    out += [
        _table(["n", "Mean", "p50", "p90", "p95", "p99", "Max"],
               [[lat["n"], _ms(lat["mean"]), _ms(lat["p50"]), _ms(lat["p90"]), _ms(lat["p95"]), _ms(lat["p99"]), _ms(lat["max"])]]),
        "",
        "## Cost",
        "",
    ]
    cost = m["cost"]
    out += [
        f"- Estimated total: {_usd(cost['total_usd'])}; mean per email: {_usd(cost['mean_per_email_usd'])}"
        + ("" if cost["available"] else f" (cost available for {cost['records_with_cost']}/{n} records)"),
        f"- Tokens: {cost['tokens'] or 'not reported'}",
        f"- Mean attempts per email: {_num(m['mean_attempts'], 2)}",
        f"- This run: {m['this_run']['api_calls']} API call(s), {m['this_run']['results_from_cache']} result(s) "
        f"reused from the local cache, new spend {_usd(m['this_run']['new_spend_usd'])}",
        "- Costs are estimates from reported token usage and the configured price table; verify against your invoice.",
        "",
        "## Confusion matrices",
        "",
        "Rows are expected labels, columns are predicted labels.",
        "",
    ]
    for field, fm in m["fields"].items():
        cm = fm["confusion_matrix"]
        if not cm:
            continue
        columns = list(next(iter(cm.values())).keys())
        out += [f"### {field}", "", _table(["expected \\ predicted", *columns], [[t, *(cm[t].get(c, 0) for c in columns)] for t in cm]), ""]

    failures = [r for r in report.records if not r.result.success]
    if failures:
        out += ["## Failed classifications", "",
                _table(["email_id", "error_type", "message", "request_id"],
                       [[r.email_id, r.result.error_type, r.result.error_message, r.result.request_id or ""] for r in failures]), ""]

    wrong_rows = []
    for r in report.records:
        if not r.result.success:
            continue
        for f in EVALUATED_FIELDS:
            exp, pred = expected_label(r.expected, f), predicted_label(r.result, f)
            if exp != pred:
                wrong_rows.append([r.email_id, f, exp, pred])
    out += ["## Misclassified fields", ""]
    out += [_table(["email_id", "field", "expected", "predicted"], wrong_rows) if wrong_rows else "None.", ""]
    return "\n".join(out)


def write_predictions_csv(report: EvaluationReport, path: Path) -> None:
    headers = ["email_id", "split", "success", "error_type"]
    for f in EVALUATED_FIELDS:
        headers += [f"expected_{f}", f"predicted_{f}", f"correct_{f}"]
    headers += ["exact_match", "review_reason", "rationale", "evidence", "evidence_grounded",
                "latency_ms", "attempts", "input_tokens", "output_tokens", "estimated_cost_usd", "request_id"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for r in report.records:
            res, cls = r.result, r.result.classification
            row: list[Any] = [r.email_id, r.split, res.success, res.error_type or ""]
            for f in EVALUATED_FIELDS:
                exp, pred = expected_label(r.expected, f), predicted_label(res, f)
                row += [exp, pred, exp == pred]
            row += [
                is_exact_match(r.expected, res),
                (cls.review_reason or "") if cls else "",
                cls.rationale if cls else "",
                cls.evidence if cls else "",
                res.evidence_grounded if res.evidence_grounded is not None else "",
                res.latency_ms if res.latency_ms is not None else "",
                res.attempts,
                res.usage.input_tokens if res.usage and res.usage.input_tokens is not None else "",
                res.usage.output_tokens if res.usage and res.usage.output_tokens is not None else "",
                res.estimated_cost_usd if res.estimated_cost_usd is not None else "",
                res.request_id or "",
            ]
            writer.writerow(row)


def save_evaluation_report(report: EvaluationReport, output_dir: Path | str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"evaluation_{report.run_id}"
    paths = {
        "json": output_dir / f"{stem}.json",
        "markdown": output_dir / f"{stem}.md",
        "csv": output_dir / f"{stem}_predictions.csv",
    }
    paths["json"].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_evaluation_markdown(report), encoding="utf-8")
    write_predictions_csv(report, paths["csv"])
    return paths


def load_evaluation_report(path: Path | str) -> EvaluationReport:
    return EvaluationReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def render_comparison_markdown(report: ComparisonReport, max_disagreements: int = 100) -> str:
    providers = report.providers
    s = report.summary
    out = [
        f"# Provider comparison: {' vs. '.join(providers)}",
        "",
        _table(["Item", "Value"], [
            ["Dataset", f"{report.dataset_name} (split: {report.dataset_split}, {report.n_records} records)"],
            ["Dataset fingerprint", report.dataset_fingerprint[:16]],
            *([f"{p} run", f"{report.run_ids[p]} - model {report.models[p]}, prompt {report.prompt_versions[p]}"] for p in providers),
            ["Created (UTC)", report.created_at],
        ]),
        "",
        f"> {SMALL_DATASET_CAVEAT}",
        "",
    ]
    if report.warnings:
        out += ["## Warnings", "", *(f"- {w}" for w in report.warnings), ""]

    def row(label: str, fn) -> list[Any]:
        return [label, *(fn(s[p]) for p in providers)]

    out += [
        "## Summary",
        "",
        _table(["Metric", *providers], [
            row("Exact-match accuracy (failures = wrong)", lambda x: _pct(x["exact_match_accuracy"])),
            row("Exact-match accuracy (successful only)", lambda x: _pct(x["exact_match_accuracy_on_success"])),
            row("Coverage", lambda x: _pct(x["coverage"])),
            row("Error rate", lambda x: _pct(x["error_rate"])),
            row("Critical emails missed", lambda x: f"{len(x['critical_missed'])}/{x['critical_total']}"),
            row("Human-review precision", lambda x: _pct(x["human_review"]["precision"])),
            row("Human-review recall", lambda x: _pct(x["human_review"]["recall"])),
            row("Human-review F1", lambda x: _num(x["human_review"]["f1"])),
            row("Latency mean", lambda x: _ms(x["latency_ms"]["mean"])),
            row("Latency p50", lambda x: _ms(x["latency_ms"]["p50"])),
            row("Latency p95", lambda x: _ms(x["latency_ms"]["p95"])),
            row("Est. cost per email", lambda x: _usd(x["cost_mean_per_email_usd"])),
            row("Est. cost, whole dataset", lambda x: _usd(x["cost_total_usd"]) + ("" if x["cost_available"] else " (partial)")),
        ]),
        "",
        "## Accuracy by field",
        "",
        _table(["Field", *providers], [[f, *(_pct(v.get(p)) for p in providers)] for f, v in report.field_accuracy.items()]),
        "",
        "## Macro / weighted F1 by field",
        "",
        _table(["Field", *providers],
               [[f, *(f"{_num(s[p]['macro_f1'][f])} / {_num(s[p]['weighted_f1'][f])}" for p in providers)]
                for f in EVALUATED_FIELDS]),
        "",
        "## Paired comparison",
        "",
        "Counts are per email. The exact McNemar p-value tests whether the two providers' error rates differ "
        "on the emails where exactly one of them was correct; p >= 0.05 means the difference is not significant.",
        "",
    ]
    for pair in report.pairwise:
        a, b = pair["a"], pair["b"]
        out += [
            f"### {a} vs. {b}",
            "",
            _table(["Field", "Both correct", f"Only {a}", f"Only {b}", "Both wrong", "McNemar p"],
                   [[f, d["both_correct"], d[f"only_{a}_correct"], d[f"only_{b}_correct"], d["both_wrong"], _num(d["mcnemar_p_value"])]
                    for f, d in pair["fields"].items()]),
            "",
        ]

    out += [f"## Disagreements ({len(report.disagreements)} emails)", ""]
    if report.disagreements:
        rows = []
        for d in report.disagreements[:max_disagreements]:
            for f in d["fields"]:
                rows.append([d["email_id"], f, d["expected"][f],
                             *(d["predictions"][p][f] + (" ✓" if f in d["correct"][p] else "") for p in providers)])
        out.append(_table(["email_id", "field", "expected", *providers], rows))
        if len(report.disagreements) > max_disagreements:
            out.append(f"\n_{len(report.disagreements) - max_disagreements} more in the JSON report._")
    else:
        out.append("None: all providers produced identical labels.")
    out.append("")
    return "\n".join(out)


def save_comparison_report(report: ComparisonReport, output_dir: Path | str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"comparison_{'_vs_'.join(report.providers)}_{report.created_at[:19].replace(':', '').replace('-', '')}"
    paths = {"json": output_dir / f"{stem}.json", "markdown": output_dir / f"{stem}.md"}
    paths["json"].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_comparison_markdown(report), encoding="utf-8")
    return paths


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)
