"""AI-powered email triage classification with provider-agnostic evaluation."""

from .classifier_interface import (
    EmailClassifierProvider,
    ProviderError,
    ProviderNotConfiguredError,
    UnknownProviderError,
)
from .schemas import ClassificationResult, EmailClassification, EmailInput

__version__ = "0.1.0"

__all__ = [
    "ClassificationResult",
    "EmailClassification",
    "EmailClassifierProvider",
    "EmailInput",
    "ProviderError",
    "ProviderNotConfiguredError",
    "UnknownProviderError",
]
