"""Load and validate labeled evaluation datasets (CSV or JSONL).

Each record is split into an ``EmailInput`` (the only thing providers see)
and a ``GroundTruth`` (kept by the evaluator). Datasets are fingerprinted so
reports can prove that providers were evaluated on identical frozen data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from ..schemas import EmailInput, GroundTruth, LabeledEmail
from ..taxonomy import Action, Urgency

REQUIRED_COLUMNS = (
    "email_id",
    "subject",
    "body",
    "expected_tone",
    "expected_urgency",
    "expected_department",
    "expected_intent",
    "expected_action",
    "expected_human_review_required",
)
OPTIONAL_COLUMNS = ("split", "notes")
VALID_SPLITS = {"dev", "test"}

_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}


class DatasetError(ValueError):
    """Raised when a dataset file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    split: str
    records: tuple[LabeledEmail, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", fingerprint_records(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def emails(self) -> list[EmailInput]:
        return [r.email for r in self.records]

    def subset(self, split: str) -> "Dataset":
        """Return records from one split (``dev``/``test``), or all for ``all``."""
        if split == "all":
            return Dataset(self.name, self.path, "all", self.records)
        if split not in VALID_SPLITS:
            raise DatasetError(f"Unknown split '{split}'. Use one of: all, {', '.join(sorted(VALID_SPLITS))}")
        return Dataset(self.name, self.path, split, tuple(r for r in self.records if r.split == split))

    def limit(self, n: Optional[int]) -> "Dataset":
        if n is None or n >= len(self.records):
            return self
        return Dataset(self.name, self.path, f"{self.split}[:{n}]", self.records[:n])

    def label_distribution(self) -> dict[str, dict[str, int]]:
        dist: dict[str, dict[str, int]] = {}
        for name in GroundTruth.model_fields:
            counts = Counter(str(getattr(r.expected, name)).lower() for r in self.records)
            dist[name] = dict(sorted(counts.items()))
        dist["split"] = dict(sorted(Counter(r.split for r in self.records).items()))
        return dist


def fingerprint_records(records: Iterable[LabeledEmail]) -> str:
    """SHA-256 over the canonical JSON of the records (inputs and labels), in order."""
    canonical = json.dumps([r.model_dump(mode="json") for r in records], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"expected a boolean (true/false), got {value!r}")


def label_consistency_issues(record: LabeledEmail) -> list[str]:
    """Check a ground-truth record against the labeling rules stated in the prompt."""
    exp = record.expected
    issues: list[str] = []
    if exp.urgency is Urgency.CRITICAL:
        allowed = {Action.IMMEDIATE_ESCALATION}
    elif exp.human_review_required:
        allowed = {Action.HUMAN_REVIEW}
    elif exp.urgency is Urgency.HIGH:
        allowed = {Action.RESPOND_TODAY}
    elif exp.urgency is Urgency.MEDIUM:
        allowed = {Action.STANDARD_QUEUE}
    else:
        allowed = {Action.STANDARD_QUEUE, Action.NO_ACTION}
    if exp.recommended_action not in allowed:
        issues.append(
            f"expected_action '{exp.recommended_action}' is inconsistent with urgency '{exp.urgency}' "
            f"and human_review_required={exp.human_review_required}; allowed: {sorted(a.value for a in allowed)}"
        )
    return issues


def _row_to_record(row: dict[str, Any]) -> LabeledEmail:
    email = EmailInput(email_id=str(row["email_id"]), subject=str(row.get("subject") or ""), body=str(row["body"]))
    expected = GroundTruth(
        tone=str(row["expected_tone"]).strip(),
        urgency=str(row["expected_urgency"]).strip(),
        department=str(row["expected_department"]).strip(),
        intent=str(row["expected_intent"]).strip(),
        recommended_action=str(row["expected_action"]).strip(),
        human_review_required=parse_bool(row["expected_human_review_required"]),
    )
    split = str(row.get("split") or "test").strip().lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
    return LabeledEmail(email=email, expected=expected, split=split, notes=str(row.get("notes") or ""))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise DatasetError(f"{path}: missing required columns: {', '.join(missing)}")
            return list(reader)
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
        return rows
    raise DatasetError(f"{path}: unsupported dataset format '{suffix}' (use .csv or .jsonl)")


def load_labeled_dataset(path: str | Path, *, check_consistency: bool = True) -> Dataset:
    """Load, validate, and fingerprint a labeled dataset.

    All problems are collected and reported together. Nothing is silently
    skipped: any invalid row makes the load fail.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")

    problems: list[str] = []
    records: list[LabeledEmail] = []
    seen: set[str] = set()
    for index, row in enumerate(_read_rows(path), start=2 if path.suffix.lower() == ".csv" else 1):
        where = f"row {index}"
        missing = [c for c in REQUIRED_COLUMNS if c not in row or row[c] is None]
        if missing:
            problems.append(f"{where}: missing values for {', '.join(missing)}")
            continue
        try:
            record = _row_to_record(row)
        except (ValidationError, ValueError) as exc:
            problems.append(f"{where} ({row.get('email_id', '?')}): {exc}")
            continue
        if record.email.email_id in seen:
            problems.append(f"{where}: duplicate email_id '{record.email.email_id}'")
            continue
        seen.add(record.email.email_id)
        if check_consistency:
            problems.extend(f"{where} ({record.email.email_id}): {issue}" for issue in label_consistency_issues(record))
        records.append(record)

    if problems:
        raise DatasetError(f"{path}: {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))
    if not records:
        raise DatasetError(f"{path}: dataset is empty")
    return Dataset(name=path.stem, path=str(path), split="all", records=tuple(records))


def load_unlabeled_emails(path: str | Path) -> list[EmailInput]:
    """Load emails (email_id, subject, body) for ad-hoc classification. Any extra columns are ignored."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"File not found: {path}")
    rows = _read_unlabeled_rows(path)
    emails, problems = [], []
    for index, row in enumerate(rows, start=1):
        try:
            emails.append(EmailInput(email_id=str(row.get("email_id", "")), subject=str(row.get("subject") or ""), body=str(row.get("body", ""))))
        except ValidationError as exc:
            problems.append(f"record {index}: {exc.errors()[0]['msg']}")
    if problems:
        raise DatasetError(f"{path}: " + "; ".join(problems))
    return emails


def _read_unlabeled_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    raise DatasetError(f"{path}: unsupported format (use .csv or .jsonl)")
