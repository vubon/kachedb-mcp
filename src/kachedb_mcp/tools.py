"""
Tool definitions and implementations for KacheDB MCP Server.

Provides exact key-value caching, SIMD semantic vector search, and token telemetry
to AI coding agents (Antigravity IDE, Claude Desktop, Cursor).
"""

from __future__ import annotations

import json
import time

from kachedb import KacheClient, SemanticCache
from kachedb.semantic.embedders import (
    EmbeddingAdapter,
    FastEmbedAdapter,
    MockEmbedder,
    OpenAIAdapter,
    SentenceTransformersAdapter,
    TransformersEmbedder,
)

from .config import settings
from .telemetry import tracker

_client: KacheClient | None = None
_semantic_cache: SemanticCache | None = None


def get_client() -> KacheClient:
    """Lazily initialize connection to KacheDB daemon."""
    global _client
    if _client is None:
        _client = KacheClient(host=settings.host, port=settings.port)
        _client.connect()
    return _client


def _resolve_embedder() -> EmbeddingAdapter:
    """Instantiate the configured embedding backend."""
    provider = settings.embedder_provider.lower()

    if provider == "fastembed":
        return FastEmbedAdapter()
    if provider == "transformers":
        return TransformersEmbedder()
    if provider == "sentencetransformers":
        return SentenceTransformersAdapter()
    if provider == "openai":
        return OpenAIAdapter(api_key=settings.openai_api_key)
    if provider == "mock":
        return MockEmbedder()

    # Auto mode: try Transformers -> FastEmbed -> SentenceTransformers -> MockEmbedder
    for adapter_cls in (TransformersEmbedder, FastEmbedAdapter, SentenceTransformersAdapter):
        try:
            return adapter_cls()
        except Exception:
            pass

    return MockEmbedder()


def get_semantic_cache() -> SemanticCache:
    """Lazily initialize the SemanticCache engine."""
    global _semantic_cache
    if _semantic_cache is None:
        client = get_client()
        embedder = _resolve_embedder()
        _semantic_cache = SemanticCache(
            client=client,
            index_name=settings.index_name,
            similarity_threshold=settings.similarity_threshold,
            ttl_seconds=settings.default_ttl_seconds,
            embedder=embedder,
        )
    return _semantic_cache


# ── MCP Tool Implementations ──────────────────────────────────────────────────


def kache_get(key: str) -> str:
    """Retrieve cached file contents, AST chunk, analysis, or prompt memory from KacheDB.

    Args:
        key: The exact lookup key (e.g. 'ast:file.py', 'summary:auth_module').

    Returns:
        The cached text content or a NOT_FOUND notification.
    """
    t0 = time.perf_counter()
    try:
        client = get_client()
        val = client.get(key)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0

        if val is not None:
            text = val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
            tracker.record_hit(len(text), elapsed_us)
            return text

        tracker.record_miss()
        return f"[MISS] Key '{key}' not found in KacheDB."
    except Exception as e:
        return f"[ERROR] Failed to read from KacheDB: {e}"


def kache_set(key: str, value: str, ttl_seconds: int = 0) -> str:
    """Store raw text, code analysis, or intermediate tool outputs in KacheDB with optional TTL.

    Args:
        key: The unique storage key.
        value: The string content to cache in memory.
        ttl_seconds: Optional expiration time in seconds (0 for default/persistent).

    Returns:
        Confirmation status.
    """
    try:
        client = get_client()
        ex = ttl_seconds if ttl_seconds > 0 else None
        ok = client.set(key, value, ex=ex)
        if ok:
            tracker.record_write()
            ttl_msg = f" (TTL: {ttl_seconds}s)" if ttl_seconds > 0 else " (Persistent)"
            return f"OK: Cached {len(value)} characters under key '{key}'{ttl_msg}."
        return f"[ERROR] Server refused SET for key '{key}'."
    except Exception as e:
        return f"[ERROR] Failed to write to KacheDB: {e}"


def kache_save_context(topic: str, content: str, ttl_seconds: int = 86400) -> str:
    """Store an architectural insight, PR review, bug fix, or codebase knowledge in KacheDB
    with SIMD vector embeddings for future semantic retrieval.

    Args:
        topic: The topic, question, or search anchor (e.g. 'S3-FIFO cache eviction bug').
        content: The detailed knowledge, explanation, or code snippet to store.
        ttl_seconds: Lifetime in seconds (default: 86400 / 24 hours).

    Returns:
        Confirmation status.
    """
    try:
        cache = get_semantic_cache()
        ok = cache.set(topic, content, ttl_seconds=ttl_seconds)
        if ok:
            tracker.record_write()
            return (
                f"OK: Saved semantic memory for '{topic}' "
                f"({len(content)} chars, TTL: {ttl_seconds}s)."
            )
        return f"[ERROR] Failed to save semantic context for '{topic}'."
    except Exception as e:
        return f"[ERROR] Semantic save error: {e}"


def kache_semantic_search(query: str, top_k: int = 3, threshold: float = 0.75) -> str:
    """Search KacheDB's semantic vector cache for relevant insights, code context,
    and past decisions matching the natural language intent of your query.

    Args:
        query: Natural language query (e.g. 'How is memory allocated for tensors?').
        top_k: Maximum number of matches to return (default: 3).
        threshold: Minimum cosine similarity score 0.0 to 1.0 (default: 0.75).

    Returns:
        Formatted matches with similarity scores and cached content.
    """
    t0 = time.perf_counter()
    try:
        cache = get_semantic_cache()
        client = get_client()
        query_vec = cache.embedder.encode(query)

        matches = client.vsearch(
            index=cache.index_name,
            query_vector=query_vec,
            top_k=top_k,
            threshold=threshold,
        )

        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0

        if not matches:
            tracker.record_miss()
            return f"[NO_MATCH] No semantic matches found for '{query}' (threshold >= {threshold})."

        total_chars = 0
        output_lines = [
            f"🧠 KacheDB Semantic Matches for '{query}' ({elapsed_us:.1f} µs):",
            "-" * 60,
        ]

        for idx, (item_id, score, payload) in enumerate(matches, 1):
            key_str = item_id.decode("utf-8") if isinstance(item_id, bytes) else str(item_id)
            val_str = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload or "")
            total_chars += len(val_str)
            output_lines.append(f"[{idx}] Topic: {key_str} (Similarity: {score:.3f})")
            output_lines.append(f"    Content: {val_str}\n")

        tracker.record_hit(total_chars, elapsed_us)
        return "\n".join(output_lines)
    except Exception as e:
        return f"[ERROR] Semantic search error: {e}"


def kache_delete(key: str) -> str:
    """Delete a key or semantic vector entry from KacheDB.

    Args:
        key: The key or topic to delete.

    Returns:
        Confirmation status.
    """
    try:
        client = get_client()
        d1 = client.delete(key)
        d2 = client.vdel(settings.index_name, key)
        if d1 > 0 or d2:
            return f"OK: Removed '{key}' from KacheDB."
        return f"[NOTICE] Key '{key}' did not exist in KacheDB."
    except Exception as e:
        return f"[ERROR] Delete error: {e}"


def kache_stats() -> str:
    """Retrieve real-time memory footprint, active vector count, and connection health."""
    try:
        client = get_client()
        pong = client.ping()
        vstats = client.vstats(settings.index_name) or {}

        report = {
            "status": "HEALTHY" if pong == "PONG" else "DEGRADED",
            "server": f"{settings.host}:{settings.port}",
            "vector_index": settings.index_name,
            "vector_metrics": vstats,
            "telemetry": tracker.summary(),
        }
        return json.dumps(report, indent=2)
    except Exception as e:
        return json.dumps({"status": "OFFLINE", "error": str(e)}, indent=2)


def kache_telemetry() -> str:
    """Retrieve cumulative token savings, avoided latency, and cache hit metrics."""
    return json.dumps(tracker.summary(), indent=2)
