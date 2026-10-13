"""Command-line interface.

Examples:
    email-classifier providers
    email-classifier classify --subject "Charged twice" --body "Please refund the duplicate payment today."
    email-classifier classify --input data/sample_emails.csv
    email-classifier validate-dataset
    email-classifier evaluate --provider sonnet --split test --dry-run
    email-classifier evaluate --provider sonnet --split test
    email-classifier compare --providers sonnet,jev --split test
    email-classifier compare --reports reports/evaluation_A.json reports/evaluation_B.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .cache import CachingProvider, ResultCache
from .classifier_interface import EmailClassifierProvider, ProviderError
from .config import load_anthropic_settings, load_env_file, load_evaluation_settings, load_jev_settings
from .evaluation.comparison import ComparisonError, compare_providers, compare_reports
from .evaluation.dataset_loader import DatasetError, load_labeled_dataset, load_unlabeled_emails
from .evaluation.runner import EvaluationAborted, evaluate
from .providers import available_providers, get_provider
from .reporting import (
    load_evaluation_report,
    save_comparison_report,
    save_evaluation_report,
    to_pretty_json,
)
from .schemas import ClassificationResult, EmailInput

DEFAULT_DATASET = Path("data/labeled_evaluation_emails.csv")

EXIT_OK, EXIT_FAILURE, EXIT_CONFIG = 0, 1, 2


def _result_payload(result: ClassificationResult) -> dict:
    if result.classification is not None:
        payload = {"status": "ok", "classification": result.classification.model_dump(mode="json")}
    else:
        payload = {"status": "error", "email_id": result.email_id, "error_type": result.error_type, "error_message": result.error_message}
    payload["meta"] = {
        "provider": result.provider,
        "model": result.model,
        "served_model": result.served_model,
        "prompt_version": result.prompt_version,
        "latency_ms": result.latency_ms,
        "attempts": result.attempts,
        "usage": result.usage.model_dump() if result.usage else None,
        "estimated_cost_usd": result.estimated_cost_usd,
        "evidence_grounded": result.evidence_grounded,
        "request_id": result.request_id,
    }
    return payload


def _progress(done: int, total: int, result: ClassificationResult) -> None:
    status = "ok" if result.success else f"FAILED ({result.error_type})"
    print(f"  [{done}/{total}] {result.provider} {result.email_id}: {status}", file=sys.stderr)


def cmd_providers(_: argparse.Namespace) -> int:
    for name in available_providers():
        provider = get_provider(name, require_configured=False)
        ready, message = provider.is_configured()
        print(f"{name:8s} {'READY' if ready else 'NOT CONFIGURED':15s} model={provider.model_id}  {message}")
    return EXIT_OK


def _provider(name: str, args: argparse.Namespace, *, require_configured: bool = True) -> EmailClassifierProvider:
    """Instantiate a provider, wrapped in the result cache unless caching is disabled."""
    provider = get_provider(name, require_configured=require_configured)
    settings = load_evaluation_settings()
    if settings.cache_enabled and not getattr(args, "no_cache", False):
        provider = CachingProvider(provider, ResultCache(settings.cache_dir), refresh=getattr(args, "refresh_cache", False))
    return provider


def _print_plan(providers: list[EmailClassifierProvider], dataset) -> None:
    """Dry run: how many API calls a run would make, without making any."""
    print(f"Dataset: {len(dataset)} records [split={dataset.split}, fingerprint={dataset.fingerprint[:12]}]")
    for provider in providers:
        is_cached = getattr(provider, "is_cached", None)
        cached = sum(1 for e in dataset.emails() if is_cached and is_cached(e))
        needed = len(dataset) - cached
        # Only the Sonnet provider makes repair calls; Jev answers are typed and never repaired.
        repairs = load_anthropic_settings().max_repair_attempts if provider.name == "sonnet" else 0
        ready, message = provider.is_configured()
        print(f"{provider.name}: {cached} cached, {needed} API call(s) expected "
              f"(at most {needed * (1 + repairs)} if every output needed a repair; SDK retries on throttling not included)"
              f"  [{'READY' if ready else 'NOT CONFIGURED: ' + message}]")


def cmd_classify(args: argparse.Namespace) -> int:
    if args.input:
        emails = load_unlabeled_emails(args.input)
    else:
        body = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else args.body
        if not body:
            print("Provide --body, --body-file, or --input.", file=sys.stderr)
            return EXIT_FAILURE
        emails = [EmailInput(email_id=args.email_id, subject=args.subject or "", body=body)]

    provider = _provider(args.provider, args)
    payloads = [_result_payload(provider.classify(email)) for email in emails]
    print(to_pretty_json(payloads if len(payloads) > 1 else payloads[0]))
    return EXIT_OK if all(p["status"] == "ok" for p in payloads) else EXIT_FAILURE


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    dataset = load_labeled_dataset(args.dataset)
    print(f"Dataset OK: {dataset.path}")
    print(f"Records: {len(dataset)}  fingerprint: {dataset.fingerprint}")
    for field, counts in dataset.label_distribution().items():
        print(f"  {field}: {counts}")
    return EXIT_OK


def _load_split(args: argparse.Namespace):
    return load_labeled_dataset(args.dataset).subset(args.split).limit(args.limit)


def _run_settings_for(provider_name: str) -> dict:
    if provider_name == "jev":
        j = load_jev_settings()
        return {
            "model_requested": j.model,
            "timeout_seconds": j.timeout_seconds,
            "sdk_max_retries": j.max_retries,
            "review_threshold": j.review_threshold,
            "result_cache": "on" if load_evaluation_settings().cache_enabled else "off",
            "pricing_source": j.pricing.source if j.pricing else "unavailable (set JEV_PRICE_INPUT_PER_MTOK)",
        }
    if provider_name != "sonnet":
        return {}
    s = load_anthropic_settings()
    return {
        "platform": s.platform,
        **({"aws_region": s.aws_region, "bedrock_endpoint": s.bedrock_endpoint} if s.platform == "bedrock" else {}),
        "effort": s.effort,
        "max_tokens": s.max_tokens,
        "timeout_seconds": s.timeout_seconds,
        "sdk_max_retries": s.max_retries,
        "max_repair_attempts": s.max_repair_attempts,
        "result_cache": "on" if load_evaluation_settings().cache_enabled else "off",
        "pricing_source": s.pricing.source if s.pricing else (
            "unavailable (set ANTHROPIC_PRICE_* from Bedrock pricing)" if s.platform == "bedrock" else "unavailable"
        ),
    }


def cmd_evaluate(args: argparse.Namespace) -> int:
    dataset = _load_split(args)
    if args.dry_run:
        _print_plan([_provider(args.provider, args, require_configured=False)], dataset)
        return EXIT_OK
    provider = _provider(args.provider, args)
    eval_settings = load_evaluation_settings()
    concurrency = args.concurrency or eval_settings.concurrency
    print(f"Evaluating {provider.name} ({provider.model_id}) on {len(dataset)} records "
          f"[split={dataset.split}, fingerprint={dataset.fingerprint[:12]}]", file=sys.stderr)
    report = evaluate(provider, dataset, concurrency=concurrency, progress=_progress,
                      run_settings={"concurrency": concurrency, **_run_settings_for(provider.name)})
    paths = save_evaluation_report(report, args.output_dir or eval_settings.reports_dir)
    m = report.metrics
    print(f"API calls this run: {m['this_run']['api_calls']}  reused from cache: {m['this_run']['results_from_cache']}")
    print(f"\nExact-match accuracy: {m['exact_match_accuracy']:.1%}  coverage: {m['coverage']:.1%}  "
          f"errors: {m['errors_by_type'] or 'none'}")
    for field, fm in m["fields"].items():
        print(f"  {field:24s} accuracy {fm['accuracy']:.1%}  macro-F1 {fm['macro_f1']:.3f}")
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    eval_settings = load_evaluation_settings()
    output_dir = args.output_dir or eval_settings.reports_dir
    if args.reports:
        reports = [load_evaluation_report(p) for p in args.reports]
        comparison = compare_reports(reports)
    else:
        names = [n.strip() for n in args.providers.split(",") if n.strip()]
        dataset = _load_split(args)
        if args.dry_run:
            _print_plan([_provider(n, args, require_configured=False) for n in names], dataset)
            return EXIT_OK
        providers = [_provider(n, args) for n in names]  # fails fast on any unconfigured provider
        comparison, reports = compare_providers(providers, dataset, concurrency=args.concurrency or eval_settings.concurrency, progress=_progress)
        for r in reports:
            save_evaluation_report(r, output_dir)
    paths = save_comparison_report(comparison, output_dir)
    for warning in comparison.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for provider, row in comparison.summary.items():
        print(f"{provider:8s} exact-match {row['exact_match_accuracy']:.1%}  coverage {row['coverage']:.1%}")
    print(f"Disagreements: {len(comparison.disagreements)} emails")
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="email-classifier", description="Email triage classification and provider evaluation.")
    parser.add_argument("--env-file", default=".env", help="Local .env file to load (default: .env). Existing variables take precedence.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging (never logs email bodies).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="List providers and whether they are configured.").set_defaults(func=cmd_providers)

    def add_cache_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--no-cache", action="store_true", help="Do not read or write the local result cache.")
        sp.add_argument("--refresh-cache", action="store_true",
                        help="Ignore cached results and call the API again (e.g. to measure run-to-run variance).")

    p = sub.add_parser("classify", help="Classify one email or a file of emails.")
    p.add_argument("--provider", default="sonnet", choices=available_providers())
    p.add_argument("--subject", default="")
    p.add_argument("--body")
    p.add_argument("--body-file", help="Read the email body from a text file.")
    p.add_argument("--email-id", default="ADHOC-001")
    p.add_argument("--input", help="CSV/JSONL with email_id, subject, body columns.")
    add_cache_args(p)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("validate-dataset", help="Validate a labeled dataset and print its fingerprint.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.set_defaults(func=cmd_validate_dataset)

    def add_dataset_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset", default=str(DEFAULT_DATASET))
        sp.add_argument("--split", default="test", choices=["test", "dev", "all"],
                        help="Use 'dev' for prompt iteration and 'test' for final numbers (default: test).")
        sp.add_argument("--limit", type=int, help="Only use the first N records (smoke tests).")
        sp.add_argument("--concurrency", type=int, help="Parallel requests (default: EVAL_CONCURRENCY or 4).")
        sp.add_argument("--output-dir", help="Report directory (default: REPORTS_DIR or ./reports).")
        sp.add_argument("--dry-run", action="store_true", help="Show how many API calls would be made, then exit without calling.")
        add_cache_args(sp)

    p = sub.add_parser("evaluate", help="Evaluate one provider against the labeled dataset.")
    p.add_argument("--provider", default="sonnet", choices=available_providers())
    add_dataset_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="Compare providers live, or compare saved evaluation reports.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--providers", help="Comma-separated provider names, e.g. sonnet,jev")
    group.add_argument("--reports", nargs="+", help="Saved evaluation report JSON files to compare.")
    add_dataset_args(p)
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows consoles often use a legacy code page; never crash on non-ASCII email text.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # SDK/HTTP debug logs include full request bodies (email content); keep them quiet.
    for noisy in ("anthropic", "httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.env_file and Path(args.env_file).is_file():
        load_env_file(args.env_file)
    try:
        return args.func(args)
    except ProviderError as exc:
        print(f"Provider error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except EvaluationAborted as exc:
        print(f"Evaluation aborted: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except (DatasetError, ComparisonError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
