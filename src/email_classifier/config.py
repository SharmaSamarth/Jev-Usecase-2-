"""Runtime configuration read from environment variables.

``.env`` files are loaded only by the CLI entry point (``load_env_file``), so
library imports and tests never pick up local secrets implicitly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
# Amazon Bedrock model ids carry an "anthropic." provider prefix. In regions
# where a model is only offered through a cross-region inference profile, the
# id also has a geography prefix, e.g. "eu.anthropic.claude-sonnet-5-5" (stays
# within EU regions) or "global.anthropic.claude-sonnet-5-5".
DEFAULT_BEDROCK_MODEL = "anthropic.claude-sonnet-5-5"
_BEDROCK_MODEL_ID = re.compile(r"^(?:[a-z]{2,8}\.)?anthropic\.claude-")

# Bedrock serving endpoints:
# - "runtime": bedrock-runtime InvokeModel (AnthropicBedrock). Works with
#   cross-region inference profiles; the default.
# - "mantle":  bedrock-mantle Messages API (AnthropicBedrockMantle). Use when
#   the model is offered there in your region.
BEDROCK_ENDPOINTS = {"runtime", "mantle"}

PLATFORM_ANTHROPIC = "anthropic"
PLATFORM_BEDROCK = "bedrock"
_PLATFORMS = {PLATFORM_ANTHROPIC, PLATFORM_BEDROCK}

# Credential variables per platform. Bedrock API keys are read by the SDK's
# Bedrock client from AWS_BEARER_TOKEN_BEDROCK (or ANTHROPIC_AWS_API_KEY).
ANTHROPIC_KEY_VARS = ("ANTHROPIC_API_KEY",)
BEDROCK_KEY_VARS = ("AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_AWS_API_KEY")


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: Optional[float] = None
    cache_read_per_mtok: Optional[float] = None
    source: str = ""


# Anthropic first-party list prices, recorded 2026-10. Verify against
# https://www.anthropic.com/pricing before relying on cost figures, or override
# with the ANTHROPIC_PRICE_* environment variables. These are NOT applied on
# Amazon Bedrock, which has its own pricing: set ANTHROPIC_PRICE_* from the
# Bedrock pricing page to get cost estimates there.
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-5-5": ModelPricing(2.00, 10.00, 2.50, 0.20, "Anthropic list price, recorded 2026-10"),
}


@dataclass(frozen=True)
class AnthropicSettings:
    api_key_present: bool
    model: str
    effort: str
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    max_repair_attempts: int
    pricing: Optional[ModelPricing]
    platform: str = PLATFORM_ANTHROPIC
    aws_region: Optional[str] = None
    bedrock_endpoint: str = "runtime"


@dataclass(frozen=True)
class EvaluationSettings:
    concurrency: int
    reports_dir: Path
    cache_enabled: bool = True
    cache_dir: Path = Path(".cache/classifications")


_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: Optional[float]) -> Optional[float]:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _pricing_for(model: str, env: Mapping[str, str], platform: str = PLATFORM_ANTHROPIC) -> Optional[ModelPricing]:
    base = ANTHROPIC_PRICING.get(model) if platform == PLATFORM_ANTHROPIC else None
    input_price = _float(env, "ANTHROPIC_PRICE_INPUT_PER_MTOK", base.input_per_mtok if base else None)
    output_price = _float(env, "ANTHROPIC_PRICE_OUTPUT_PER_MTOK", base.output_per_mtok if base else None)
    if input_price is None or output_price is None:
        return None
    overridden = base is None or input_price != base.input_per_mtok or output_price != base.output_per_mtok
    return ModelPricing(
        input_per_mtok=input_price,
        output_per_mtok=output_price,
        cache_write_per_mtok=_float(env, "ANTHROPIC_PRICE_CACHE_WRITE_PER_MTOK", base.cache_write_per_mtok if base else None),
        cache_read_per_mtok=_float(env, "ANTHROPIC_PRICE_CACHE_READ_PER_MTOK", base.cache_read_per_mtok if base else None),
        source="environment override" if overridden else base.source,
    )


def _has_any(env: Mapping[str, str], names: tuple[str, ...]) -> bool:
    return any(env.get(n, "").strip() for n in names)


def resolve_platform(env: Mapping[str, str]) -> str:
    """CLAUDE_PLATFORM if set; otherwise inferred from which credential is present.

    With only a Bedrock API key set, Bedrock is used; in every other case the
    Anthropic API is used (and reports a clear error if its key is missing).
    """
    explicit = env.get("CLAUDE_PLATFORM", "").strip().lower()
    if explicit:
        if explicit not in _PLATFORMS:
            raise ValueError(f"CLAUDE_PLATFORM must be one of {sorted(_PLATFORMS)}, got {explicit!r}")
        return explicit
    if _has_any(env, BEDROCK_KEY_VARS) and not _has_any(env, ANTHROPIC_KEY_VARS):
        return PLATFORM_BEDROCK
    return PLATFORM_ANTHROPIC


def load_anthropic_settings(env: Optional[Mapping[str, str]] = None) -> AnthropicSettings:
    env = os.environ if env is None else env
    platform = resolve_platform(env)
    is_bedrock = platform == PLATFORM_BEDROCK
    model = env.get("ANTHROPIC_MODEL", "").strip() or (DEFAULT_BEDROCK_MODEL if is_bedrock else DEFAULT_ANTHROPIC_MODEL)
    bedrock_endpoint = env.get("BEDROCK_ENDPOINT", "").strip().lower() or "runtime"
    if bedrock_endpoint not in BEDROCK_ENDPOINTS:
        raise ValueError(f"BEDROCK_ENDPOINT must be one of {sorted(BEDROCK_ENDPOINTS)}, got {bedrock_endpoint!r}")
    if is_bedrock and not _BEDROCK_MODEL_ID.match(model):
        raise ValueError(
            f"ANTHROPIC_MODEL={model!r} is not a Bedrock model id. On Bedrock use the "
            f"'anthropic.'-prefixed id, e.g. {DEFAULT_BEDROCK_MODEL!r}, or an inference profile "
            "such as 'eu.anthropic.claude-sonnet-5-5'."
        )
    effort = env.get("ANTHROPIC_EFFORT", "").strip().lower() or "low"
    if effort not in _VALID_EFFORTS:
        raise ValueError(f"ANTHROPIC_EFFORT must be one of {sorted(_VALID_EFFORTS)}, got {effort!r}")
    return AnthropicSettings(
        api_key_present=_has_any(env, BEDROCK_KEY_VARS if is_bedrock else ANTHROPIC_KEY_VARS),
        model=model,
        effort=effort,
        max_tokens=_int(env, "ANTHROPIC_MAX_TOKENS", 4096, minimum=256),
        timeout_seconds=float(_float(env, "ANTHROPIC_TIMEOUT_SECONDS", 60.0)),
        max_retries=_int(env, "ANTHROPIC_MAX_RETRIES", 2),
        max_repair_attempts=_int(env, "CLASSIFIER_MAX_REPAIR_ATTEMPTS", 1),
        pricing=_pricing_for(model, env, platform),
        platform=platform,
        aws_region=(env.get("AWS_REGION", "").strip() or env.get("AWS_DEFAULT_REGION", "").strip() or None) if is_bedrock else None,
        bedrock_endpoint=bedrock_endpoint,
    )


def load_evaluation_settings(env: Optional[Mapping[str, str]] = None) -> EvaluationSettings:
    env = os.environ if env is None else env
    return EvaluationSettings(
        concurrency=_int(env, "EVAL_CONCURRENCY", 4, minimum=1),
        reports_dir=Path(env.get("REPORTS_DIR", "").strip() or "reports"),
        cache_enabled=env.get("CLASSIFIER_CACHE", "on").strip().lower() not in {"off", "0", "false", "no"},
        cache_dir=Path(env.get("CLASSIFIER_CACHE_DIR", "").strip() or ".cache/classifications"),
    )


def load_env_file(path: str | os.PathLike[str] = ".env") -> bool:
    """Load a local .env file for development. Existing variables win."""
    from dotenv import load_dotenv

    return load_dotenv(path, override=False)


def estimate_cost_usd(
    pricing: Optional[ModelPricing],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cache_creation_input_tokens: Optional[int] = None,
    cache_read_input_tokens: Optional[int] = None,
) -> Optional[float]:
    """Estimate request cost. Returns None when pricing or usage is unavailable."""
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    cost = input_tokens * pricing.input_per_mtok + output_tokens * pricing.output_per_mtok
    if cache_creation_input_tokens:
        if pricing.cache_write_per_mtok is None:
            return None
        cost += cache_creation_input_tokens * pricing.cache_write_per_mtok
    if cache_read_input_tokens:
        if pricing.cache_read_per_mtok is None:
            return None
        cost += cache_read_input_tokens * pricing.cache_read_per_mtok
    return cost / 1_000_000


# ---------------------------------------------------------------------------
# Jev (TypeSafe AI)
# ---------------------------------------------------------------------------
# Based on TypeSafe's official Python SDK (typesafe-sdk, github.com/typesafe-ai/
# typesafe-sdk-python): the SDK reads the key from TYPESAFE_API_KEY, defaults to
# https://api.typesafe.ai and the model alias "jev-latest", and reports billable
# input tokens and output tokens (documented as currently free of charge).

DEFAULT_JEV_MODEL = "jev-latest"


@dataclass(frozen=True)
class JevSettings:
    api_key_present: bool
    model: str
    timeout_seconds: float
    max_retries: int
    review_threshold: float
    pricing: Optional[ModelPricing]


def load_jev_settings(env: Optional[Mapping[str, str]] = None) -> JevSettings:
    env = os.environ if env is None else env
    threshold = _float(env, "JEV_REVIEW_THRESHOLD", 0.5)
    if not 0 < float(threshold) < 1:
        raise ValueError(f"JEV_REVIEW_THRESHOLD must be between 0 and 1, got {threshold}")
    input_price = _float(env, "JEV_PRICE_INPUT_PER_MTOK", None)
    pricing = None
    if input_price is not None:
        raw_output = env.get("JEV_PRICE_OUTPUT_PER_MTOK", "").strip()
        output_price = float(raw_output) if raw_output else 0.0
        if output_price < 0:
            raise ValueError("JEV_PRICE_OUTPUT_PER_MTOK must be >= 0")
        pricing = ModelPricing(input_price, output_price, source="environment (JEV_PRICE_*)")
    return JevSettings(
        api_key_present=bool(env.get("TYPESAFE_API_KEY", "").strip()),
        model=env.get("JEV_MODEL", "").strip() or DEFAULT_JEV_MODEL,
        timeout_seconds=float(_float(env, "JEV_TIMEOUT_SECONDS", 30.0)),
        max_retries=_int(env, "JEV_MAX_RETRIES", 2),
        review_threshold=float(threshold),
        pricing=pricing,
    )
