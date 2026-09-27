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
KEY_EXACT_HITS = "kachedb:telemetry:exact_hits"
KEY_VECTOR_HITS = "kachedb:telemetry:vector_hits"
KEY_EMBED_TIME_SAVED_MS = "kachedb:telemetry:embed_time_saved_ms"

# Per-operation avoided latency baselines (ms).
#
# Represents the typical wall-clock time the AI agent would have spent WITHOUT
# the cache for each operation type:
#   - "semantic_search": replaces embed + 3 file reads + LLM processing (~6,200 ms)
#   - "exact_search":    replaces embed + 3 file reads + LLM processing (~6,200 ms)
#   - "kv_get":          replaces a single view_file tool call (~150 ms)
#   - "parent_doc":      replaces viewing a large source or doc file (~150 ms)
#
# Override via KACHEDB_BASELINE_SEMANTIC_MS / KACHEDB_BASELINE_KV_MS env vars.
OP_BASELINE_MS: dict[str, float] = {
    "semantic_search": 6_200.0,
    "exact_search": 6_200.0,
    "kv_get": 150.0,
    "parent_doc": 150.0,
}


def _get_baseline_ms(op_type: str) -> float:
    """Return the avoided-latency baseline (ms) for the given operation type."""
    import os

    env_key = f"KACHEDB_BASELINE_{op_type.upper()}_MS"
    env_val = os.getenv(env_key)
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass
    return OP_BASELINE_MS.get(op_type, 400.0)


@dataclass
class TelemetryTracker:
    """Thread-safe telemetry and savings tracker dogfooding KacheDB engine."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    tokens_saved: int = 0
    latency_saved_ms: float = 0.0
    exact_hits: int = 0
    vector_hits: int = 0
    embed_time_saved_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record_hit(
        self,
        cached_content_chars: int,
        elapsed_us: float,
        *,
        op_type: str = "semantic_search",
        avoided_source_chars: int | None = None,
        client: KacheClient | None = None,
        is_exact: bool = False,
        avoided_embedding_ms: float = 15.0,
    ) -> None:
        """Record a successful cache hit and compute saved tokens and latency.

        Args:
            cached_content_chars: Character length of the content returned from cache.
            elapsed_us: Actual KacheDB retrieval time in microseconds.
            op_type: Operation type for latency baseline selection.
                     Use ``"semantic_search"`` for vector hits, ``"exact_search"`` for
                     exact-symbol shortcuts, ``"kv_get"`` for ``kache_get`` hits.
            avoided_source_chars: Character length of the original source content that
                was avoided (e.g. the source file that would have been read).
                When provided, token savings are calculated from this instead of
                ``cached_content_chars``, giving a more accurate avoided-work estimate.
                Falls back to ``cached_content_chars`` if not provided.
            client: Optional live ``KacheClient`` for atomic persistence into KacheDB.
            is_exact: True if resolved via exact key shortcut without embedding model.
            avoided_embedding_ms: Estimated neural embedding inference time saved (ms).
        """
        # Use avoided_source_chars when available so token savings
        # reflect the original source size (what was avoided), not the response size.
        source_chars = (
            avoided_source_chars if avoided_source_chars is not None else cached_content_chars
        )
        saved_tokens = max(1, source_chars // 4)

        # Use per-operation baseline instead of a single hardcoded 400 ms.
        baseline_ms = _get_baseline_ms(op_type)
        saved_ms = max(0.0, baseline_ms - (elapsed_us / 1000.0))

        with self._lock:
            self.hits += 1
            self.tokens_saved += saved_tokens
            self.latency_saved_ms += saved_ms
            if is_exact:
                self.exact_hits += 1
                self.embed_time_saved_ms += avoided_embedding_ms
            elif op_type in ("semantic_search", "exact_search"):
                self.vector_hits += 1

        if client is not None:
            with contextlib.suppress(Exception):
                client.incr(KEY_HITS)
                client.incrby(KEY_TOKENS, saved_tokens)
                client.incrby(KEY_LATENCY_MS, int(saved_ms))
                if is_exact:
                    client.incr(KEY_EXACT_HITS)
                    client.incrby(KEY_EMBED_TIME_SAVED_MS, int(avoided_embedding_ms))
                elif op_type in ("semantic_search", "exact_search"):
                    client.incr(KEY_VECTOR_HITS)

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
        """Return a structured telemetry snapshot dogfooding KacheDB keys when connected.

        Reconciliation strategy: KacheDB atomic keys are the source of truth for
        cross-session persistence (previous sessions' hits survive server restarts).
        In-memory values cover hits recorded during network partitions where
        ``client.incr()`` was silently suppressed. ``max()`` handles both cases
        correctly without double-counting.
        """
        from .config import settings

        hits = self.hits
        misses = self.misses
        writes = self.writes
        tokens = self.tokens_saved
        latency_ms = self.latency_saved_ms
        exact_hits = self.exact_hits
        vector_hits = self.vector_hits
        embed_saved_ms = self.embed_time_saved_ms

        if client is not None:
            with contextlib.suppress(Exception):
                raw_hits = client.get(KEY_HITS)
                raw_misses = client.get(KEY_MISSES)
                raw_writes = client.get(KEY_WRITES)
                raw_tokens = client.get(KEY_TOKENS)
                raw_lat = client.get(KEY_LATENCY_MS)
                raw_exact = client.get(KEY_EXACT_HITS)
                raw_vector = client.get(KEY_VECTOR_HITS)
                raw_embed = client.get(KEY_EMBED_TIME_SAVED_MS)

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
                if raw_exact is not None:
                    e_val = raw_exact.decode() if isinstance(raw_exact, bytes) else raw_exact
                    exact_hits = max(exact_hits, int(e_val))
                if raw_vector is not None:
                    v_val = raw_vector.decode() if isinstance(raw_vector, bytes) else raw_vector
                    vector_hits = max(vector_hits, int(v_val))
                if raw_embed is not None:
                    em_val = raw_embed.decode() if isinstance(raw_embed, bytes) else raw_embed
                    embed_saved_ms = max(embed_saved_ms, float(em_val))

        total_lookups = hits + misses
        hit_ratio = (hits / total_lookups * 100.0) if total_lookups > 0 else 0.0

        # Use configurable per-model token cost instead of hardcoded $5.00/M.
        est_usd_saved = (tokens / 1_000_000.0) * settings.token_cost_per_million

        return {
            "total_lookups": total_lookups,
            "cache_hits": hits,
            "exact_hits": exact_hits,
            "vector_hits": vector_hits,
            "cache_misses": misses,
            "hit_ratio_percent": round(hit_ratio, 2),
            "total_writes": writes,
            "tokens_saved": tokens,
            "latency_saved_seconds": round(latency_ms / 1000.0, 2),
            "embed_time_saved_ms": round(embed_saved_ms, 2),
            "estimated_usd_saved": f"${est_usd_saved:.4f}",
            "token_cost_per_million": settings.token_cost_per_million,
            "persistence": "kachedb_atomic_keys",
        }


tracker = TelemetryTracker()
