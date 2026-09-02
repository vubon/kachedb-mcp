"""
Telemetry and Token Savings Tracking for KacheDB MCP Server.

Dogfoods KacheDB itself: persists cumulative token savings, avoided latency,
and cache hit ratios directly into atomic KacheDB system keys with zero external DB dependencies.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kachedb import KacheClient

KEY_HITS = "kachedb:telemetry:hits"
KEY_MISSES = "kachedb:telemetry:misses"
KEY_WRITES = "kachedb:telemetry:writes"
KEY_TOKENS = "kachedb:telemetry:tokens_saved"
KEY_LATENCY_MS = "kachedb:telemetry:latency_saved_ms"


@dataclass
class TelemetryTracker:
    """Thread-safe telemetry and savings tracker dogfooding KacheDB engine."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    tokens_saved: int = 0
    latency_saved_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record_hit(
        self,
        cached_content_chars: int,
        elapsed_us: float,
        client: KacheClient | None = None,
    ) -> None:
        """Record a successful cache hit and compute saved tokens and latency."""
        with self._lock:
            self.hits += 1
            saved_tokens = max(1, cached_content_chars // 4)
            self.tokens_saved += saved_tokens
            saved_ms = max(0.0, 400.0 - (elapsed_us / 1000.0))
            self.latency_saved_ms += saved_ms

        if client is not None:
            with contextlib.suppress(Exception):
                client.incr(KEY_HITS)
                client.incrby(KEY_TOKENS, saved_tokens)
                client.incrby(KEY_LATENCY_MS, int(saved_ms))

    def record_miss(self, client: KacheClient | None = None) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1

        if client is not None:
            with contextlib.suppress(Exception):
                client.incr(KEY_MISSES)

    def record_write(self, client: KacheClient | None = None) -> None:
        """Record a cache set or context save."""
        with self._lock:
            self.writes += 1

        if client is not None:
            with contextlib.suppress(Exception):
                client.incr(KEY_WRITES)

    def summary(self, client: KacheClient | None = None) -> dict[str, Any]:
        """Return a structured telemetry snapshot dogfooding KacheDB keys when connected."""
        hits = self.hits
        misses = self.misses
        writes = self.writes
        tokens = self.tokens_saved
        latency_ms = self.latency_saved_ms

        if client is not None:
            with contextlib.suppress(Exception):
                raw_hits = client.get(KEY_HITS)
                raw_misses = client.get(KEY_MISSES)
                raw_writes = client.get(KEY_WRITES)
                raw_tokens = client.get(KEY_TOKENS)
                raw_lat = client.get(KEY_LATENCY_MS)

                if raw_hits is not None:
                    h_val = raw_hits.decode() if isinstance(raw_hits, bytes) else raw_hits
                    hits = max(hits, int(h_val))
                if raw_misses is not None:
                    m_val = raw_misses.decode() if isinstance(raw_misses, bytes) else raw_misses
                    misses = max(misses, int(m_val))
                if raw_writes is not None:
                    w_val = raw_writes.decode() if isinstance(raw_writes, bytes) else raw_writes
                    writes = max(writes, int(w_val))
                if raw_tokens is not None:
                    t_val = raw_tokens.decode() if isinstance(raw_tokens, bytes) else raw_tokens
                    tokens = max(tokens, int(t_val))
                if raw_lat is not None:
                    l_val = raw_lat.decode() if isinstance(raw_lat, bytes) else raw_lat
                    latency_ms = max(latency_ms, float(l_val))

        total_lookups = hits + misses
        hit_ratio = (hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
        est_usd_saved = (tokens / 1_000_000.0) * 5.0

        return {
            "total_lookups": total_lookups,
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_ratio_percent": round(hit_ratio, 2),
            "total_writes": writes,
            "tokens_saved": tokens,
            "latency_saved_seconds": round(latency_ms / 1000.0, 2),
            "estimated_usd_saved": f"${est_usd_saved:.4f}",
            "persistence": "kachedb_atomic_keys",
        }


tracker = TelemetryTracker()
