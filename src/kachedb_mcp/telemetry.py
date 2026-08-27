"""
Telemetry and Token Savings Tracking for KacheDB MCP Server.

Measures cache hit ratios, cumulative tokens saved by avoiding redundant LLM context,
and total latency reduction.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetryTracker:
    """Thread-safe telemetry and savings tracker."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    tokens_saved: int = 0
    latency_saved_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record_hit(self, cached_content_chars: int, elapsed_us: float) -> None:
        """Record a successful cache hit and compute saved tokens and latency."""
        with self._lock:
            self.hits += 1
            # Approximate standard OpenAI / Claude token ratio: ~4 characters per token
            saved_tokens = max(1, cached_content_chars // 4)
            self.tokens_saved += saved_tokens
            # Typical avoided LLM TTFT + generation latency: ~400 ms
            saved_ms = max(0.0, 400.0 - (elapsed_us / 1000.0))
            self.latency_saved_ms += saved_ms

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1

    def record_write(self) -> None:
        """Record a cache set or context save."""
        with self._lock:
            self.writes += 1

    def summary(self) -> dict[str, Any]:
        """Return a structured telemetry snapshot."""
        with self._lock:
            total_lookups = self.hits + self.misses
            hit_ratio = (self.hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
            # Estimated dollar savings assuming $5.00 / 1M prompt+completion tokens
            est_usd_saved = (self.tokens_saved / 1_000_000.0) * 5.0

            return {
                "total_lookups": total_lookups,
                "cache_hits": self.hits,
                "cache_misses": self.misses,
                "hit_ratio_percent": round(hit_ratio, 2),
                "total_writes": self.writes,
                "tokens_saved": self.tokens_saved,
                "latency_saved_seconds": round(self.latency_saved_ms / 1000.0, 2),
                "estimated_usd_saved": f"${est_usd_saved:.4f}",
            }


tracker = TelemetryTracker()
