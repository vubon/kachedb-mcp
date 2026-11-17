"""Backward-compatible re-export shim for AWS AI-DLC installer APIs."""

from __future__ import annotations

from .frameworks.aidlc import (
    AIDLC_AGENT_TARGETS,
    PROFILE_SETTINGS,
    AidlcAdapter,
    configure_aidlc,
    resolve_aidlc_profile,
)

__all__ = [
    "AIDLC_AGENT_TARGETS",
    "PROFILE_SETTINGS",
    "AidlcAdapter",
    "configure_aidlc",
    "resolve_aidlc_profile",
]
