"""Provider-agnostic evaluation runner.

``evaluate`` sends every email in a frozen dataset to one provider and
produces an ``EvaluationReport``. The report stores each prediction next to
its ground truth plus run metadata (dataset fingerprint, prompt version), so
it can be saved, moved to another machine, and compared later.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..classifier_interface import EmailClassifierProvider, ErrorType, ProviderNotConfiguredError
from ..schemas import ClassificationResult, EmailInput, GroundTruth
from .dataset_loader import Dataset
from .metrics import compute_metrics

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, ClassificationResult], None]


class RecordOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email_id: str
    split: str
    expected: GroundTruth
    result: ClassificationResult


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    provider: str
    model: str
    prompt_version: str
    prompt_fingerprint: str
    dataset_name: str
    dataset_path: str
    dataset_split: str
    dataset_fingerprint: str
    started_at: str
    finished_at: str
    run_settings: dict[str, Any] = Field(default_factory=dict)
    records: list[RecordOutcome]
    metrics: dict[str, Any]

    def recompute_metrics(self) -> dict[str, Any]:
        return compute_metrics([(r.expected, r.result) for r in self.records])


def _safe_classify(provider: EmailClassifierProvider, email: EmailInput) -> ClassificationResult:
    """Classify, converting unexpected exceptions into failed results so one bad email cannot abort a run."""
    try:
        result = provider.classify(email)
    except ProviderNotConfiguredError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad; recorded as a failure
        logger.exception("Unexpected error classifying email_id=%s with provider=%s", email.email_id, provider.name)
        return ClassificationResult(
            email_id=email.email_id,
            provider=provider.name,
            model=provider.model_id,
            error_type=ErrorType.UNEXPECTED,
            error_message=f"{type(exc).__name__}: {exc}"[:500],
        )
    if result.email_id != email.email_id:
        return ClassificationResult(
            email_id=email.email_id,
            provider=provider.name,
            model=provider.model_id,
            error_type=ErrorType.UNEXPECTED,
            error_message=f"Provider returned mismatched email_id '{result.email_id}'.",
        )
    return result


_ABORT_IMMEDIATELY = ErrorType.FATAL | {ErrorType.BAD_REQUEST, ErrorType.UNEXPECTED}


class EvaluationAborted(RuntimeError):
    """Raised when a configuration error makes further API calls pointless.

    No report is produced for an aborted run. Successful results obtained
    before the abort are kept in the result cache (when enabled), so they are
    not paid for again on the next run.
    """


_TRANSIENT_WARMUP_LIMIT = 2


def _needs_call(provider: EmailClassifierProvider, email: EmailInput) -> bool:
    is_cached = getattr(provider, "is_cached", None)
    return is_cached is None or not is_cached(email)


def evaluate(
    provider: EmailClassifierProvider,
    dataset: Dataset,
    *,
    concurrency: int = 4,
    progress: Optional[ProgressCallback] = None,
    run_settings: Optional[dict[str, Any]] = None,
) -> EvaluationReport:
    """Evaluate one provider on every record of ``dataset`` (order preserved).

    To avoid wasted API calls:
    - Warm-up: emails that need a real call are sent one at a time until one
      succeeds (or fails for a reason specific to that email), which proves the
      setup works and writes the prompt cache before parallel requests start.
      A configuration error (bad key, no model access, model not found, bad
      request) stops the run immediately; two consecutive transient failures
      (timeouts, throttling, connection or server errors after SDK retries)
      stop it too.
    - After warm-up, a configuration error on any email stops every request
      that has not started yet.
    """
    provider.ensure_configured()
    if not len(dataset):
        raise ValueError("Cannot evaluate an empty dataset")

    started = datetime.now(timezone.utc)
    emails = dataset.emails()
    results: list[Optional[ClassificationResult]] = [None] * len(emails)
    done = 0
    lock = threading.Lock()

    def record(index: int, result: ClassificationResult) -> None:
        nonlocal done
        with lock:
            results[index] = result
            done += 1
            if progress:
                progress(done, len(emails), result)

    # Warm-up: sequential until the setup is proven to work.
    transient_failures = 0
    for i, email in enumerate(emails):
        if not _needs_call(provider, email):
            continue
        result = _safe_classify(provider, email)
        record(i, result)
        if result.success or result.error_type in ErrorType.EMAIL_SPECIFIC:
            break
        if result.error_type in _ABORT_IMMEDIATELY:
            raise EvaluationAborted(
                f"Stopped after an API call failed with a configuration error ({result.error_type}: "
                f"{result.error_message}). No further calls were made."
            )
        transient_failures += 1
        if transient_failures >= _TRANSIENT_WARMUP_LIMIT:
            raise EvaluationAborted(
                f"Stopped after {transient_failures} consecutive failed API calls (last: {result.error_type}: "
                f"{result.error_message}). The service looks unavailable, so no further calls were made."
            )

    stop = threading.Event()
    fatal: list[ClassificationResult] = []

    def task(i: int) -> None:
        if stop.is_set():
            return
        result = _safe_classify(provider, emails[i])
        if result.error_type in ErrorType.FATAL:
            stop.set()
            fatal.append(result)
        record(i, result)

    remaining = [i for i in range(len(emails)) if results[i] is None]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for future in [pool.submit(task, i) for i in remaining]:
            future.result()
    if fatal:
        skipped = sum(r is None for r in results)
        raise EvaluationAborted(
            f"Stopped after a configuration error ({fatal[0].error_type}: {fatal[0].error_message}); "
            f"{skipped} email(s) were not sent."
        )

    records = [
        RecordOutcome(email_id=rec.email.email_id, split=rec.split, expected=rec.expected, result=res)
        for rec, res in zip(dataset.records, results)
        if res is not None
    ]
    prompt_version, prompt_fp = provider.prompt_info()
    return EvaluationReport(
        run_id=f"{started.strftime('%Y%m%dT%H%M%SZ')}-{provider.name}-{uuid.uuid4().hex[:6]}",
        provider=provider.name,
        model=provider.model_id,
        prompt_version=prompt_version,
        prompt_fingerprint=prompt_fp,
        dataset_name=dataset.name,
        dataset_path=dataset.path,
        dataset_split=dataset.split,
        dataset_fingerprint=dataset.fingerprint,
        started_at=started.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        run_settings=run_settings or {},
        records=records,
        metrics=compute_metrics([(r.expected, r.result) for r in records]),
    )
