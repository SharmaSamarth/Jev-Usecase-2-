import csv
import json
from pathlib import Path

import pytest

from email_classifier.evaluation.dataset_loader import (
    REQUIRED_COLUMNS,
    DatasetError,
    load_labeled_dataset,
    load_unlabeled_emails,
    parse_bool,
)
from email_classifier.schemas import EmailInput

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "data" / "labeled_evaluation_emails.csv"

ROW = {
    "email_id": "E1", "split": "test", "subject": "Charged twice", "body": "Please refund the duplicate.",
    "expected_tone": "frustrated", "expected_urgency": "high", "expected_department": "billing_payments",
    "expected_intent": "refund_request", "expected_action": "respond_today",
    "expected_human_review_required": "false", "notes": "",
}


def write_csv(path: Path, rows: list[dict], columns=None) -> Path:
    columns = columns or list(ROW)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_bundled_dataset_is_valid_and_consistent():
    ds = load_labeled_dataset(BUNDLED)
    assert len(ds) >= 40
    assert {r.split for r in ds.records} == {"dev", "test"}
    assert len(ds.subset("dev")) + len(ds.subset("test")) == len(ds)
    # Every label value of every field is covered somewhere in the dataset.
    dist = ds.label_distribution()
    assert len(dist["tone"]) == 10 and len(dist["department"]) == 10 and len(dist["intent"]) == 10
    assert len(dist["urgency"]) == 4 and len(dist["recommended_action"]) == 5


def test_bundled_dataset_contains_required_scenarios():
    ds = load_labeled_dataset(BUNDLED).subset("test")
    notes = " ".join(r.notes.lower() for r in ds.records)
    for scenario in ("polite but critical", "angry tone, low urgency", "prompt injection", "misleading urgency", "ambiguous"):
        assert scenario in notes, scenario


def test_providers_only_receive_email_content():
    ds = load_labeled_dataset(BUNDLED)
    for email in ds.emails():
        assert isinstance(email, EmailInput)
        assert set(email.model_dump()) == {"email_id", "subject", "body"}


def test_fingerprint_is_stable_and_label_sensitive(tmp_path):
    a = load_labeled_dataset(write_csv(tmp_path / "a.csv", [ROW]))
    b = load_labeled_dataset(write_csv(tmp_path / "b.csv", [ROW]))
    c = load_labeled_dataset(write_csv(tmp_path / "c.csv", [{**ROW, "expected_tone": "angry"}]))
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.subset("test").fingerprint == a.fingerprint


def test_missing_column(tmp_path):
    cols = [c for c in ROW if c != "expected_intent"]
    with pytest.raises(DatasetError, match="missing required columns: expected_intent"):
        load_labeled_dataset(write_csv(tmp_path / "x.csv", [ROW], columns=cols))


def test_invalid_label_and_duplicate_reported_together(tmp_path):
    rows = [ROW, {**ROW}, {**ROW, "email_id": "E3", "expected_department": "finance"}]
    with pytest.raises(DatasetError) as info:
        load_labeled_dataset(write_csv(tmp_path / "x.csv", rows))
    message = str(info.value)
    assert "duplicate email_id 'E1'" in message
    assert "E3" in message and "2 problem(s)" in message


def test_inconsistent_action_is_flagged(tmp_path):
    row = {**ROW, "expected_urgency": "critical", "expected_action": "respond_today"}
    with pytest.raises(DatasetError, match="inconsistent with urgency 'critical'"):
        load_labeled_dataset(write_csv(tmp_path / "x.csv", [row]))
    # The check can be disabled for datasets that intentionally use another policy.
    assert len(load_labeled_dataset(tmp_path / "x.csv", check_consistency=False)) == 1


def test_invalid_split(tmp_path):
    with pytest.raises(DatasetError, match="split"):
        load_labeled_dataset(write_csv(tmp_path / "x.csv", [{**ROW, "split": "train"}]))
    ds = load_labeled_dataset(write_csv(tmp_path / "y.csv", [ROW]))
    with pytest.raises(DatasetError):
        ds.subset("train")


def test_jsonl_format(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({**ROW, "expected_human_review_required": False}) + "\n\n", encoding="utf-8")
    ds = load_labeled_dataset(path)
    assert ds.records[0].expected.human_review_required is False


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_labeled_dataset(tmp_path / "nope.csv")
    with pytest.raises(DatasetError, match="empty"):
        load_labeled_dataset(write_csv(tmp_path / "empty.csv", []))
    with pytest.raises(DatasetError, match="unsupported"):
        (tmp_path / "x.txt").write_text("hi")
        load_labeled_dataset(tmp_path / "x.txt")


def test_limit_keeps_order():
    ds = load_labeled_dataset(BUNDLED).subset("test").limit(3)
    assert [r.email.email_id for r in ds.records] == ["EMAIL-T001", "EMAIL-T002", "EMAIL-T003"]


@pytest.mark.parametrize("raw, value", [("true", True), ("FALSE", False), ("1", True), ("no", False), (True, True)])
def test_parse_bool(raw, value):
    assert parse_bool(raw) is value


def test_parse_bool_rejects_garbage():
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_unlabeled_sample_file():
    emails = load_unlabeled_emails(ROOT / "data" / "sample_emails.csv")
    assert len(emails) >= 3 and all(e.body for e in emails)


def test_required_columns_documented():
    assert set(REQUIRED_COLUMNS) <= set(ROW)
