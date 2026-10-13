"""Local result cache that avoids paying twice for the same classification.

A result is reused only when the email content, the provider configuration
(model, platform, effort, prompt version), and the shared prompt fingerprint
are all identical. Only successful results are stored, so failed emails are
retried on the next run. Cached results keep the latency, usage, and cost of
the original call, so evaluation metrics describe how the prediction was
produced, while ``from_cache`` shows that no API call was made in this run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .classifier_interface import EmailClassifierProvider
from .schemas import ClassificationResult, EmailInput

logger = logging.getLogger(__name__)


class ResultCache:
    """One JSON file per cached result under ``directory``. Safe for concurrent threads."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def key_for(self, provider: EmailClassifierProvider, email: EmailInput) -> str:
        payload = {
            "config": provider.config_fingerprint(),
            "prompt_fingerprint": provider.prompt_info()[1],
            "subject": email.subject,
            "body": email.body,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[ClassificationResult]:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return ClassificationResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            logger.warning("Ignoring unreadable cache entry %s", path.name)
            return None

    def put(self, key: str, result: ClassificationResult) -> None:
        if not result.success:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically so a crash or concurrent writer never leaves a partial file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(result.model_dump_json())
            os.replace(tmp, path)
        except OSError:
            logger.warning("Could not write cache entry %s", path.name)
            Path(tmp).unlink(missing_ok=True)

    def contains(self, key: str) -> bool:
        return self._path(key).is_file()


class CachingProvider(EmailClassifierProvider):
    """Wraps any provider; serves repeated classifications from ``ResultCache``."""

    def __init__(self, inner: EmailClassifierProvider, cache: ResultCache, *, refresh: bool = False):
        self.inner = inner
        self.cache = cache
        self.refresh = refresh
        self.name = inner.name

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def is_configured(self) -> tuple[bool, str]:
        return self.inner.is_configured()

    def config_fingerprint(self) -> dict[str, str]:
        return self.inner.config_fingerprint()

    def prompt_info(self) -> tuple[str, str]:
        return self.inner.prompt_info()

    def is_cached(self, email: EmailInput) -> bool:
        return not self.refresh and self.cache.contains(self.cache.key_for(self.inner, email))

    def classify(self, email: EmailInput) -> ClassificationResult:
        key = self.cache.key_for(self.inner, email)
        if not self.refresh:
            hit = self.cache.get(key)
            if hit is not None and hit.classification is not None:
                return hit.model_copy(update={
                    "email_id": email.email_id,
                    "classification": hit.classification.model_copy(update={"email_id": email.email_id}),
                    "from_cache": True,
                })
        result = self.inner.classify(email)
        self.cache.put(key, result)
        return result
