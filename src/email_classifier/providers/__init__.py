"""Provider registry.

Providers are selected explicitly by name. A missing or unconfigured provider
raises a clear error; there is never a silent fallback to another model.
"""

from __future__ import annotations

from typing import Callable

from ..classifier_interface import EmailClassifierProvider, UnknownProviderError
from .anthropic_sonnet import AnthropicSonnetProvider
from .jev import JevProvider

_REGISTRY: dict[str, Callable[[], EmailClassifierProvider]] = {
    AnthropicSonnetProvider.name: AnthropicSonnetProvider,
    JevProvider.name: JevProvider,
}


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def register_provider(name: str, factory: Callable[[], EmailClassifierProvider]) -> None:
    _REGISTRY[name] = factory


def get_provider(name: str, *, require_configured: bool = True) -> EmailClassifierProvider:
    """Instantiate a provider by name.

    Raises ``UnknownProviderError`` for unknown names and
    ``ProviderNotConfiguredError`` when ``require_configured`` is set and the
    provider cannot run.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise UnknownProviderError(f"Unknown provider '{name}'. Available: {', '.join(available_providers())}")
    provider = _REGISTRY[key]()
    if require_configured:
        provider.ensure_configured()
    return provider


__all__ = [
    "AnthropicSonnetProvider",
    "JevProvider",
    "available_providers",
    "get_provider",
    "register_provider",
]
