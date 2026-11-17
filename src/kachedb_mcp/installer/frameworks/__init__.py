"""Registry of workflow framework adapters for ``kachedb-mcp install``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .aidlc import AidlcAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from .base import FrameworkAdapter

FRAMEWORK_CHOICES: tuple[str, ...] = ("aidlc", "auto", "all")

_ADAPTERS: dict[str, FrameworkAdapter] = {
    "aidlc": AidlcAdapter(),
}


def list_adapters() -> tuple[FrameworkAdapter, ...]:
    """Return all registered workflow framework adapters."""
    return tuple(_ADAPTERS.values())


def get_adapter(name: str) -> FrameworkAdapter:
    """Look up a single adapter by name or raise ``ValueError``."""
    adapter = _ADAPTERS.get(name)
    if adapter is None:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported framework '{name}' (registered: {supported})")
    return adapter


def detect_frameworks(root: Path) -> list[FrameworkAdapter]:
    """Return all registered adapters detected in ``root``."""
    return [adapter for adapter in _ADAPTERS.values() if adapter.detect(root)]


__all__ = [
    "FRAMEWORK_CHOICES",
    "detect_frameworks",
    "get_adapter",
    "list_adapters",
]
