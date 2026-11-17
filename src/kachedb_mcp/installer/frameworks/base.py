"""Framework adapter protocol and shared configuration options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from ..engine import InstallPlan

InstallProfile = Literal["preserve", "balanced", "minimal", "full"]
PROFILE_CHOICES: tuple[InstallProfile, ...] = ("preserve", "balanced", "minimal", "full")


@dataclass(frozen=True)
class InstallOptions:
    """Resolved installation parameters passed to a framework adapter."""

    root: Path
    profile: InstallProfile | None = None
    space: str = "default"
    patch_agents: bool = False
    dry_run: bool = False
    yes: bool = False
    affirmed_date: str | None = None


class FrameworkAdapter(Protocol):
    """Contract implemented by workflow framework configurators (AI-DLC, etc.)."""

    name: str

    def detect(self, root: Path) -> bool:
        """Return True if this framework is initialized in ``root``."""
        ...

    def configure(self, opts: InstallOptions) -> InstallPlan:
        """Apply KacheDB bounded retrieval rules and settings for this framework."""
        ...
