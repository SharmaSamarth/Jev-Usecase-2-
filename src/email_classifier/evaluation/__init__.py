"""Provider-agnostic evaluation and comparison."""

from .comparison import ComparisonError, ComparisonReport, compare_providers, compare_reports
from .dataset_loader import Dataset, DatasetError, load_labeled_dataset, load_unlabeled_emails
from .metrics import compute_metrics
from .runner import EvaluationReport, RecordOutcome, evaluate

__all__ = [
    "ComparisonError",
    "ComparisonReport",
    "Dataset",
    "DatasetError",
    "EvaluationReport",
    "RecordOutcome",
    "compare_providers",
    "compare_reports",
    "compute_metrics",
    "evaluate",
    "load_labeled_dataset",
    "load_unlabeled_emails",
]
