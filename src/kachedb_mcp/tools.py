"""
Tool definitions and implementations for KacheDB MCP Server v0.2.0.

Provides exact key-value caching, SIMD semantic vector search, multi-workspace memory isolation,
and token telemetry dogfooding to AI coding agents (Antigravity IDE, Claude Desktop, Cursor).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from kachedb import KacheClient, SemanticCache
from kachedb.semantic.embedders import (
    EmbeddingAdapter,
    FastEmbedAdapter,
    MockEmbedder,
    OpenAIAdapter,
    SentenceTransformersAdapter,
    TransformersEmbedder,
)

try:
    from kachedb.semantic.chunker import MarkdownChunker
except ImportError:
    from .chunker import MarkdownChunker  # type: ignore[assignment]

try:
    from kachedb.tags import tags_to_bitmask
except ImportError:
    from .tags import tags_to_bitmask

from .config import settings
from .telemetry import tracker

logger = logging.getLogger(__name__)

_client: KacheClient | None = None
_semantic_cache: SemanticCache | None = None


def get_client() -> KacheClient:
    """Lazily initialize connection to KacheDB daemon."""
    global _client
    if _client is None or _client.host != settings.host or _client.port != settings.port:
        _client = KacheClient(host=settings.host, port=settings.port)
    return _client


def _client_vadd(
    client: Any,
    *,
    index: str | bytes,
    item_id: str | bytes,
    vector: Any,
    payload: str | bytes | None = None,
    ex: int | None = None,
    tag_mask: int = 0,
    tags: list[str] | None = None,
    parent_key: str | bytes | None = None,
) -> bool:
    """Invoke vadd with backward-compatibility for kachedb < 0.2.0."""
    try:
        res = client.vadd(
            index=index,
            item_id=item_id,
            vector=vector,
            payload=payload,
            ex=ex,
            tag_mask=tag_mask,
            tags=tags,
            parent_key=parent_key,
        )
        return bool(res)
    except TypeError:
        res = client.vadd(
            index=index,
            item_id=item_id,
            vector=vector,
            payload=payload,
            ex=ex,
        )
        return bool(res)


def _client_vsearch(
    client: Any,
    *,
    index: str | bytes,
    query_vector: Any,
    top_k: int = 1,
    threshold: float = 0.0,
    filter_tags: list[str] | None = None,
) -> list[Any]:
    """Invoke vsearch with backward-compatibility for kachedb < 0.2.0."""
    try:
        res = client.vsearch(
            index=index,
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
            filter_tags=filter_tags,
        )
        return list(res)
    except TypeError:
        res = client.vsearch(
            index=index,
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
        )
        return list(res)


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


def _get_target_index(workspace_id: str = "") -> str:
    """Resolve index name with workspace isolation."""
    clean_ws = workspace_id.strip()
    if clean_ws:
        return f"{settings.index_name}:{clean_ws}"
    return settings.index_name


# ── MCP Tool Implementations ──────────────────────────────────────────────────


def kache_get(key: str) -> str:
    """Retrieve cached file contents, AST chunk, analysis, or prompt memory from KacheDB.

    Args:
        key: The exact lookup key (e.g. 'ast:file.py', 'summary:auth_module').

    Returns:
        The cached text content or a NOT_FOUND notification.
    """
    try:
        client = get_client()
        t0 = time.perf_counter()
        val = client.get(key)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0

        if val is not None:
            text = val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
            # For exact KV hits, the cached text IS the source that would have been
            # read (e.g. a file chunk), so avoided_source_chars == len(text).
            tracker.record_hit(
                len(text),
                elapsed_us,
                op_type="kv_get",
                avoided_source_chars=len(text),
                client=client,
            )
            return text

        tracker.record_miss(client=client)
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
            tracker.record_write(client=client)
            ttl_msg = f" (TTL: {ttl_seconds}s)" if ttl_seconds > 0 else " (Persistent)"
            return f"OK: Cached {len(value)} characters under key '{key}'{ttl_msg}."
        return f"[ERROR] Server refused SET for key '{key}'."
    except Exception as e:
        return f"[ERROR] Failed to write to KacheDB: {e}"


def kache_save_context(
    topic: str,
    content: str,
    ttl_seconds: int = 86400,
    workspace_id: str = "",
    tags: list[str] | None = None,
    auto_chunk: bool = True,
    prompt_alias: str | None = None,
) -> str:
    """Store an architectural insight, PR review, bug fix, or codebase knowledge in KacheDB
    with SIMD vector embeddings, automatic hierarchical chunking, and tag bitmask filtering.

    Args:
        topic: The topic, question, or search anchor (e.g. 'S3-FIFO cache eviction bug').
        content: The detailed knowledge, explanation, or code snippet to store.
        ttl_seconds: Lifetime in seconds (default: 86400 / 24 hours).
        workspace_id: Optional workspace ID for multi-tenant isolation.
        tags: Optional metadata tags (e.g. ['arch', 'simd', 'redis']) encoded into 64-bit mask.
        auto_chunk: If True, documents >1500 chars are chunked by Markdown headings
            with parent pointers.
        prompt_alias: Optional natural language question or user prompt alias to link,
            enabling instantaneous (<50ns) exact-symbol hits for identical user queries.

    Returns:
        Confirmation status.
    """
    try:
        cache = get_semantic_cache()
        client = get_client()
        target_index = _get_target_index(workspace_id)
        tag_mask = tags_to_bitmask(tags) if tags else 0
        ex_val = ttl_seconds if ttl_seconds > 0 else None
        clean_ws = workspace_id.strip()

        # Decide whether to chunk hierarchically
        should_chunk = auto_chunk and len(content) > 1500
        parent_key = f"doc:{clean_ws}:{topic}" if clean_ws else f"doc:{topic}"

        # Helper to register exact prompt alias keys pointing to parent_key
        def _register_prompt_aliases(target_ref: str) -> None:
            aliases_to_add: list[str] = []
            if prompt_alias and prompt_alias.strip():
                p = prompt_alias.strip()
                aliases_to_add.append(p)
                p_canon = p.lower().rstrip("?.!").strip()
                if p_canon and p_canon != p:
                    aliases_to_add.append(p_canon)

            clean_top = topic.strip()
            if " " in clean_top or clean_top.endswith("?"):
                aliases_to_add.append(clean_top)
                top_canon = clean_top.lower().rstrip("?.!").strip()
                if top_canon and top_canon != clean_top:
                    aliases_to_add.append(top_canon)

            for a in set(aliases_to_add):
                client.set(f"query:{a}", target_ref, ex=ex_val)
                if clean_ws:
                    client.set(f"query:{clean_ws}:{a}", target_ref, ex=ex_val)

        if should_chunk:
            # 1. Store full unabridged document in KV under parent_key
            client.set(parent_key, content, ex=ex_val)
            if clean_ws:
                client.set(f"{clean_ws}:{topic}", content, ex=ex_val)

            _register_prompt_aliases(parent_key)

            # 2. Chunk document into semantic segments
            chunker = MarkdownChunker(target_chunk_size=300, overlap=40)
            chunks = chunker.chunk_document(parent_key=parent_key, markdown_text=content)

            saved_chunks = 0
            for chunk in chunks:
                chunk_id = chunk.chunk_id
                embed_text = (
                    f"{topic} > {chunk.section_title}\n{chunk.content}"
                    if chunk.section_title and chunk.section_title != "Root"
                    else f"{topic}\n{chunk.content}"
                )
                chunk_vec = cache.embedder.encode(embed_text)

                ok = _client_vadd(
                    client,
                    index=target_index,
                    item_id=chunk_id,
                    vector=chunk_vec,
                    payload=chunk.content,
                    ex=ex_val,
                    tag_mask=tag_mask,
                    tags=tags,
                    parent_key=parent_key,
                )
                if ok:
                    saved_chunks += 1

            # Also index prompt_alias directly as an anchor vector pointing to parent_key
            if prompt_alias and prompt_alias.strip():
                anchor_vec = cache.embedder.encode(f"{topic}\n{prompt_alias.strip()}")
                _client_vadd(
                    client,
                    index=target_index,
                    item_id=f"{topic}#prompt_anchor",
                    vector=anchor_vec,
                    payload=content[:400],
                    ex=ex_val,
                    tag_mask=tag_mask,
                    tags=tags,
                    parent_key=parent_key,
                )

            tracker.record_write(client=client)
            ws_tag = f" [Workspace: {workspace_id}]" if workspace_id else ""
            tags_tag = f" [Tags: {', '.join(tags)}]" if tags else ""
            alias_tag = f" [Prompt: '{prompt_alias}']" if prompt_alias else ""
            summary_info = (
                f"({saved_chunks}/{len(chunks)} chunks, parent: '{parent_key}', "
                f"{len(content)} chars, TTL: {ttl_seconds}s)."
            )
            msg = f"OK: Saved hierarchical context for '{topic}'{ws_tag}{tags_tag}{alias_tag}"
            return f"{msg} {summary_info}"
        else:
            # Single-item storage: embed topic, prompt_alias, and content snippet
            embed_components = [topic]
            if prompt_alias and prompt_alias.strip():
                embed_components.append(prompt_alias.strip())
            content_snippet = content[:1500].strip()
            if content_snippet:
                embed_components.append(content_snippet)
            embed_text = "\n\n".join(embed_components)

            vector = cache.embedder.encode(embed_text)
            client.set(parent_key, content, ex=ex_val)
            if clean_ws:
                client.set(f"{clean_ws}:{topic}", content, ex=ex_val)

            _register_prompt_aliases(parent_key)

            ok = _client_vadd(
                client,
                index=target_index,
                item_id=topic,
                vector=vector,
                payload=content,
                ex=ex_val,
                tag_mask=tag_mask,
                tags=tags,
            )

            # Also index prompt_alias directly as an anchor vector pointing to parent_key
            if prompt_alias and prompt_alias.strip():
                anchor_vec = cache.embedder.encode(prompt_alias.strip())
                _client_vadd(
                    client,
                    index=target_index,
                    item_id=f"{topic}#prompt_anchor",
                    vector=anchor_vec,
                    payload=content[:400],
                    ex=ex_val,
                    tag_mask=tag_mask,
                    tags=tags,
                    parent_key=parent_key,
                )

            if ok:
                tracker.record_write(client=client)
                ws_tag = f" [Workspace: {workspace_id}]" if workspace_id else ""
                tags_tag = f" [Tags: {', '.join(tags)}]" if tags else ""
                alias_tag = f" [Prompt: '{prompt_alias}']" if prompt_alias else ""
                return (
                    f"OK: Saved semantic memory for '{topic}'{ws_tag}{tags_tag}{alias_tag} "
                    f"({len(content)} chars, TTL: {ttl_seconds}s)."
                )
            return f"[ERROR] Failed to save semantic context for '{topic}'."
    except Exception as e:
        return f"[ERROR] Semantic save error: {e}"


def kache_semantic_search(
    query: str,
    top_k: int = 3,
    threshold: float = 0.75,
    workspace_id: str = "",
    tags: list[str] | None = None,
    exact_first: bool = True,
    resolve_parent: bool = False,
) -> str:
    """Search KacheDB's semantic vector cache for relevant insights, code context,
    and past decisions matching natural language intent, with exact symbol shortcut
    and tag bitmask filtering.

    Args:
        query: Natural language query (e.g. 'How is memory allocated for tensors?').
        top_k: Maximum number of matches to return (default: 3).
        threshold: Minimum cosine similarity score 0.0 to 1.0 (default: 0.75).
        workspace_id: Optional workspace ID for multi-tenant isolation.
        tags: Optional metadata tags to pre-filter vectors before SIMD cosine math.
        exact_first: If True, checks direct KV store (<50ns) before executing neural embedding.
        resolve_parent: If True, automatically resolves and appends full parent document for chunks.

    Returns:
        Formatted matches with similarity scores, parent keys, and cached content.
    """
    try:
        cache = get_semantic_cache()
        client = get_client()
        target_index = _get_target_index(workspace_id)
        clean_ws = workspace_id.strip()
        clean_q = query.strip()
        canonical_q = clean_q.lower().rstrip("?.!").strip()

        # 1. Exact-Symbol & Query Shortcut (< 50 ns, skips neural embedder entirely)
        if exact_first:
            candidates: list[str] = []
            if clean_ws:
                candidates.extend(
                    [
                        f"query:{clean_ws}:{clean_q}",
                        f"query:{clean_ws}:{canonical_q}",
                        f"sym:{clean_ws}:{clean_q}",
                        f"sym:{clean_ws}:{canonical_q}",
                        f"doc:{clean_ws}:{clean_q}",
                        f"doc:{clean_ws}:{canonical_q}",
                        f"{clean_ws}:{clean_q}",
                        f"{clean_ws}:{canonical_q}",
                    ]
                )
            candidates.extend(
                [
                    f"query:{clean_q}",
                    f"query:{canonical_q}",
                    f"sym:{clean_q}",
                    f"sym:{canonical_q}",
                    f"doc:{clean_q}",
                    f"doc:{canonical_q}",
                    clean_q,
                    canonical_q,
                ]
            )

            seen: set[str] = set()
            deduped_candidates: list[str] = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    deduped_candidates.append(c)

            t_exact_start = time.perf_counter()
            for cand_key in deduped_candidates:
                exact_val = client.get(cand_key)
                if exact_val is not None and isinstance(exact_val, (bytes, str)):
                    content_str = (
                        exact_val.decode("utf-8", errors="replace")
                        if isinstance(exact_val, bytes)
                        else str(exact_val)
                    )
                    # If this is an alias pointing to a doc: or query: key, dereference it
                    deref_info = ""
                    if (
                        (content_str.startswith("doc:") or content_str.startswith("query:"))
                        and "\n" not in content_str
                        and len(content_str) < 120
                    ):
                        resolved_doc = client.get(content_str)
                        if resolved_doc is not None and isinstance(resolved_doc, (bytes, str)):
                            deref_info = f" -> {content_str}"
                            content_str = (
                                resolved_doc.decode("utf-8", errors="replace")
                                if isinstance(resolved_doc, bytes)
                                else str(resolved_doc)
                            )

                    exact_elapsed_us = (time.perf_counter() - t_exact_start) * 1_000_000.0
                    tracker.record_hit(
                        len(content_str),
                        exact_elapsed_us,
                        op_type="exact_search",
                        client=client,
                        is_exact=True,
                    )
                    ws_header = f" (Workspace: {workspace_id})" if clean_ws else ""
                    return (
                        f"⚡ KacheDB Exact Match for '{query}'{ws_header} "
                        f"({exact_elapsed_us:.1f} µs, 0 ms embedding):\n"
                        f"{'-' * 60}\n"
                        f"[1] Key: {cand_key}{deref_info} (Score: 1.000 [EXACT])\n"
                        f"    Content: {content_str}\n"
                    )

        # 2. Dense Vector Semantic Search (SIMD accelerated)
        query_vec = cache.embedder.encode(query)

        t0 = time.perf_counter()
        matches = _client_vsearch(
            client,
            index=target_index,
            query_vector=query_vec,
            top_k=top_k,
            threshold=threshold,
            filter_tags=tags,
        )
        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0

        # Adaptive fallback: if no matches found with strict threshold or tags
        is_adaptive = False
        adaptive_note = ""
        if not matches:
            # 1. Try relaxing tags if tags were provided
            if tags:
                matches = _client_vsearch(
                    client,
                    index=target_index,
                    query_vector=query_vec,
                    top_k=top_k,
                    threshold=threshold,
                )
                if matches:
                    is_adaptive = True
                    adaptive_note = " [Adaptive: relaxed tag filter]"

            # 2. If still no matches and threshold > 0.55, try relaxed threshold (step 1: -0.20)
            if not matches and threshold > 0.55:
                fallback_threshold = max(0.50, threshold - 0.20)
                matches = _client_vsearch(
                    client,
                    index=target_index,
                    query_vector=query_vec,
                    top_k=top_k,
                    threshold=fallback_threshold,
                    filter_tags=tags,
                )
                if matches:
                    is_adaptive = True
                    adaptive_note = (
                        f" [Adaptive: threshold relaxed from {threshold:.2f} "
                        f"to {fallback_threshold:.2f}]"
                    )
                elif tags:
                    matches = _client_vsearch(
                        client,
                        index=target_index,
                        query_vector=query_vec,
                        top_k=top_k,
                        threshold=fallback_threshold,
                    )
                    if matches:
                        is_adaptive = True
                        adaptive_note = (
                            f" [Adaptive: relaxed tags and threshold to {fallback_threshold:.2f}]"
                        )

            # 3. Stage 2: Deep relaxation for code symbols (down to 0.30)
            if not matches and threshold > 0.35:
                code_threshold = 0.30
                matches = _client_vsearch(
                    client,
                    index=target_index,
                    query_vector=query_vec,
                    top_k=top_k,
                    threshold=code_threshold,
                    filter_tags=tags,
                )
                if matches:
                    is_adaptive = True
                    adaptive_note = (
                        f" [Adaptive: threshold relaxed from {threshold:.2f} "
                        f"to {code_threshold:.2f}]"
                    )

        # If no matches found and workspace specified: check if workspace needs lazy auto-indexing
        if (
            not matches
            and clean_ws
            and settings.auto_index
            and _ensure_workspace_indexed(workspace_id=clean_ws)
        ):
            matches = _client_vsearch(
                client,
                index=target_index,
                query_vector=query_vec,
                top_k=top_k,
                threshold=threshold,
                filter_tags=tags,
            )
            if not matches and threshold > 0.35:
                fallback_threshold = max(0.30, threshold - 0.40)
                matches = _client_vsearch(
                    client,
                    index=target_index,
                    query_vector=query_vec,
                    top_k=top_k,
                    threshold=fallback_threshold,
                    filter_tags=tags,
                )
                if matches:
                    is_adaptive = True
                    adaptive_note = f" [Adaptive: threshold relaxed to {fallback_threshold:.2f}]"

        if not matches:
            tracker.record_miss(client=client)
            ws_info = f" in workspace '{workspace_id}'" if workspace_id else ""
            tag_info = f" with tags {tags}" if tags else ""
            msg = f"[NO_MATCH] No semantic matches found for '{query}'{ws_info}{tag_info}"
            return f"{msg} (threshold >= {threshold})."

        total_chars = 0
        ws_header = f" (Workspace: {workspace_id})" if workspace_id else ""
        tags_header = f" [Filter: {', '.join(tags)}]" if tags else ""
        adaptive_header = adaptive_note if is_adaptive else ""
        match_title = (
            f"🧠 KacheDB Semantic Matches for '{query}'"
            f"{ws_header}{tags_header}{adaptive_header} ({elapsed_us:.1f} µs):"
        )
        output_lines = [
            match_title,
            "-" * 60,
        ]

        for idx, match in enumerate(matches, 1):
            item_id = match[0]
            score = match[1]
            payload = match[2]
            parent_key = getattr(match, "parent_key", None)
            if parent_key is None and len(match) > 3:
                parent_key = match[3]

            key_str = (
                item_id.decode("utf-8", errors="replace")
                if isinstance(item_id, bytes)
                else str(item_id)
            )
            val_str = (
                payload.decode("utf-8", errors="replace")
                if isinstance(payload, bytes)
                else str(payload or "")
            )
            parent_str = ""
            if parent_key:
                parent_str = (
                    parent_key.decode("utf-8", errors="replace")
                    if isinstance(parent_key, bytes)
                    else str(parent_key)
                )

            should_resolve = (resolve_parent or key_str.endswith("#prompt_anchor")) and bool(
                parent_str
            )
            if should_resolve:
                parent_content = client.get(parent_str)
                if parent_content is not None and isinstance(parent_content, (bytes, str)):
                    p_text = (
                        parent_content.decode("utf-8", errors="replace")
                        if isinstance(parent_content, bytes)
                        else str(parent_content)
                    )
                    if key_str.endswith("#prompt_anchor"):
                        val_str = p_text
                    else:
                        val_str = (
                            f"{val_str}\n\n--- [FULL PARENT DOCUMENT: {parent_str}] ---\n{p_text}"
                        )

            total_chars += len(val_str)
            parent_info = f", Parent: {parent_str}" if parent_str else ""
            output_lines.append(f"[{idx}] Topic: {key_str} (Similarity: {score:.3f}{parent_info})")
            output_lines.append(f"    Content: {val_str}\n")

        tracker.record_hit(
            total_chars,
            elapsed_us,
            op_type="semantic_search",
            client=client,
            is_exact=False,
        )
        return "\n".join(output_lines)
    except Exception as e:
        return f"[ERROR] Semantic search error: {e}"


def kache_get_parent_document(parent_key: str) -> str:
    """Fetch the full unabridged parent document for a chunked context from KacheDB.

    Args:
        parent_key: The parent document key (e.g. 'doc:workspace:topic' or 'doc:topic').

    Returns:
        The full parent document text or a NOT_FOUND notification.
    """
    try:
        client = get_client()
        t0 = time.perf_counter()
        val = client.get(parent_key)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0

        if val is not None and isinstance(val, (bytes, str)):
            text = val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
            tracker.record_hit(
                len(text),
                elapsed_us,
                op_type="parent_doc",
                avoided_source_chars=len(text),
                client=client,
            )
            return (
                f"📄 Parent Document '{parent_key}' ({elapsed_us:.1f} µs, {len(text)} chars):\n"
                f"{'-' * 60}\n"
                f"{text}"
            )

        tracker.record_miss(client=client)
        return f"[NOT_FOUND] Parent document '{parent_key}' not found in KacheDB."
    except Exception as e:
        return f"[ERROR] Failed to fetch parent document: {e}"


def kache_delete(key: str, workspace_id: str = "") -> str:
    """Delete a key or semantic vector entry from KacheDB.

    Args:
        key: The key or topic to delete.
        workspace_id: Optional workspace ID namespace.

    Returns:
        Confirmation status.
    """
    try:
        client = get_client()
        target_index = _get_target_index(workspace_id)
        d1 = client.delete(key)
        d2 = client.vdel(target_index, key)
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
            "telemetry": tracker.summary(client=client),
        }
        return json.dumps(report, indent=2)
    except Exception as e:
        return json.dumps({"status": "OFFLINE", "error": str(e)}, indent=2)


def kache_telemetry() -> str:
    """Retrieve cumulative token savings, avoided latency, and cache hit metrics
    dogfooding KacheDB.
    """
    try:
        client = get_client()
        return json.dumps(tracker.summary(client=client), indent=2)
    except Exception:
        return json.dumps(tracker.summary(), indent=2)


# ── Sprint 3: Codebase Indexing Tools ────────────────────────────────────────


def _ensure_workspace_indexed(
    workspace_id: str = "",
    path: str = ".",
    force: bool = False,
) -> bool:
    """Check if workspace watermark exists in SwissTable; if not, index it lazily on demand.

    Runs zero-overhead check (<50 ns). If the watermark is absent (e.g. fresh boot or server
    restart), automatically runs `kache_index_workspace` for the workspace.

    Returns:
        True if indexing was executed, False if already up-to-date or skipped.
    """
    if not settings.auto_index:
        return False

    from .indexer.discovery import WorkspaceDiscovery, resolve_workspace_path

    try:
        client = get_client()
        try:
            if client.ping() != "PONG":
                return False
        except Exception:
            return False

        target_dir = resolve_workspace_path(
            workspace_id=workspace_id, candidate_path=path, kache_client=client
        )
        if target_dir is None:
            return False
        resolved_path = str(target_dir)

        discovery = WorkspaceDiscovery(resolved_path)
        projects = discovery.discover()
        if not projects:
            return False

        clean_ws = (
            workspace_id.strip().lower().replace(" ", "-").replace("_", "-") if workspace_id else ""
        )

        needs_index = False
        for project in projects:
            ws = clean_ws or project.workspace_id
            meta_key = f"kache:meta:{ws}:commit"
            try:
                stored = client.get(meta_key)
                if not stored:
                    needs_index = True
                    break
                if force:
                    needs_index = True
                    break
                if project.commit_sha:
                    sha_str = (
                        stored.decode("utf-8", errors="replace")
                        if isinstance(stored, bytes)
                        else str(stored)
                    )
                    if sha_str.strip() != project.commit_sha.strip():
                        needs_index = True
                        break
            except Exception:
                needs_index = True
                break

        if needs_index:
            logger.info(
                "Lazy autonomous indexing triggered for '%s' (workspace='%s')",
                resolved_path,
                clean_ws or "auto",
            )
            kache_index_workspace(path=resolved_path, workspace_id=clean_ws, force=force)
            return True

        return False
    except Exception as e:
        logger.debug("Lazy auto-indexing check skipped: %s", e)
        return False


def kache_index_workspace(
    path: str = ".",
    workspace_id: str = "",
    force: bool = False,
) -> str:
    """Scan and index all codebase symbols into KacheDB SwissTable and vector index.

    Discovers eligible projects inside `path`, parses AST symbols across supported
    languages (Python, Rust, Go, TypeScript/JS), and persists them into KacheDB as:
    - Exact SwissTable keys: `sym:<workspace>:<symbol>` (multi-definition arrays).
    - Fully qualified keys: `sym:<workspace>:<file_rel_path>:<symbol>`.
    - Watermark: `kache:meta:<workspace>:commit` for incremental sync.

    Honors .gitignore and skips build/vendor directories automatically.

    Args:
        path: Root directory to index. Defaults to current working directory.
        workspace_id: Optional workspace ID override. Auto-derived from project name if empty.
        force: If True, re-indexes even when the git commit watermark is unchanged.

    Returns:
        JSON status report with indexed symbol counts, languages, and elapsed time.
    """
    import os
    import time as _time

    from .indexer.discovery import GitEngine, WorkspaceDiscovery, resolve_workspace_path
    from .indexer.parser import CodeParser, SymbolRegistry

    t_start = _time.perf_counter()

    # Explicit path check
    if path and path != ".":
        expanded = os.path.realpath(os.path.expanduser(path))
        if not os.path.exists(expanded):
            return json.dumps({"status": "ERROR", "error": f"Path does not exist: {path}"})
        from pathlib import Path

        p_obj = Path(expanded).resolve()
        if p_obj in (Path.home().resolve(), Path("/").resolve(), Path("/Users").resolve()):
            return json.dumps(
                {
                    "status": "ERROR",
                    "error": f"Refusing to index user home directory or filesystem root: {path}",
                }
            )
        resolved_path = str(p_obj)
    else:
        target_dir = resolve_workspace_path(workspace_id=workspace_id, candidate_path=path)
        if target_dir is None:
            target_desc = workspace_id or path
            return json.dumps(
                {
                    "status": "ERROR",
                    "error": (
                        f"Could not resolve a valid project workspace path for '{target_desc}'. "
                        "Refusing to index user home directory or filesystem root. "
                        "Please provide a valid workspace_id or explicit path."
                    ),
                }
            )
        resolved_path = str(target_dir)

    try:
        client = get_client()
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": f"KacheDB connection failed: {e}"})

    # 1. Discover projects
    discovery = WorkspaceDiscovery(resolved_path)
    projects = discovery.discover()

    if not projects:
        return json.dumps(
            {
                "status": "SKIPPED",
                "reason": "No eligible codebase projects found.",
                "path": resolved_path,
            }
        )

    results: list[dict[str, object]] = []
    total_symbols = 0
    total_files = 0

    parser = CodeParser()

    for project in projects:
        ws_id = workspace_id.strip() or project.workspace_id
        meta_key = f"kache:meta:{ws_id}:commit"

        # 2. Incremental: skip if commit watermark is unchanged
        if not force and project.commit_sha:
            try:
                stored_sha = client.get(meta_key)
                if stored_sha:
                    sha_str = (
                        stored_sha.decode("utf-8", errors="replace")
                        if isinstance(stored_sha, bytes)
                        else str(stored_sha)
                    )
                    if sha_str.strip() == project.commit_sha.strip():
                        results.append(
                            {
                                "project": project.name,
                                "workspace_id": ws_id,
                                "status": "UP_TO_DATE",
                                "commit": project.commit_sha[:8],
                            }
                        )
                        continue
            except Exception:
                pass

        # 3. Walk project source files and parse symbols
        registry = SymbolRegistry()
        project_files = 0
        git_engine = GitEngine(project.git_root) if project.is_git and project.git_root else None

        for root, dirs, files in os.walk(project.path):
            dirs[:] = [
                d
                for d in dirs
                if d not in discovery.ignored_dirs
                and not d.startswith(".")
                and not (
                    git_engine
                    and git_engine.is_ignored(
                        os.path.relpath(os.path.join(root, d), project.path).replace("\\", "/")
                    )
                )
            ]
            for fname in files:
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, project.path).replace("\\", "/")
                if git_engine and git_engine.is_ignored(rel_path):
                    continue
                symbols = parser.parse_file(abs_path, rel_path=rel_path)
                if symbols:
                    registry.add_all(symbols)
                    project_files += 1

        # 4. Write SwissTable KV payload
        payload = registry.to_swisstable_payload(ws_id)
        written_kv = 0
        for key, value in payload.items():
            try:
                client.set(key, value)
                written_kv += 1
            except Exception:
                pass

        # 5. Write vector records (signatures + docstrings) for semantic search
        vec_records = registry.to_vector_records(ws_id)
        try:
            cache = get_semantic_cache()
            target_index = _get_target_index(ws_id)
            for _vec_key, embed_text, meta in vec_records:
                try:
                    embedding = cache.embedder.encode(embed_text)
                    _client_vadd(
                        client,
                        index=target_index,
                        item_id=_vec_key,
                        vector=embedding,
                        payload=json.dumps(meta, ensure_ascii=False),
                    )
                except Exception:
                    pass
        except Exception:
            pass  # Vector indexing is best-effort; SwissTable exact lookup always works

        # 6. Persist commit watermark and workspace path
        import contextlib

        if project.commit_sha:
            with contextlib.suppress(Exception):
                client.set(meta_key, project.commit_sha)

        with contextlib.suppress(Exception):
            client.set(f"kache:meta:{ws_id}:path", str(project.path))

        sym_count = len(registry.symbols)
        total_symbols += sym_count
        total_files += project_files

        results.append(
            {
                "project": project.name,
                "workspace_id": ws_id,
                "path": str(project.path),
                "status": "INDEXED",
                "commit": project.commit_sha[:8] if project.commit_sha else None,
                "branch": project.branch,
                "manifest_type": project.manifest_type,
                "files_parsed": project_files,
                "symbols_indexed": sym_count,
                "kv_keys_written": written_kv,
            }
        )

    elapsed_ms = (_time.perf_counter() - t_start) * 1000.0

    return json.dumps(
        {
            "status": "OK",
            "projects": results,
            "total_symbols_indexed": total_symbols,
            "total_files_parsed": total_files,
            "elapsed_ms": round(elapsed_ms, 2),
        },
        indent=2,
    )


def kache_explore_symbol(
    symbol: str,
    workspace_id: str = "",
    file_path: str = "",
) -> str:
    """Instantaneous (< 50 ns) symbol inspection from KacheDB SwissTable.

    Returns all definition locations, signatures, file paths, line numbers, and
    docstrings for a given symbol name in a surgical < 300 token payload.
    No file reads required.

    Args:
        symbol: The symbol name to look up (e.g. "Config", "new", "fetch_data").
        workspace_id: Optional workspace ID to scope lookup. Searches globally if empty.
        file_path: Optional file path or hint to automatically anchor and index the codebase.

    Returns:
        JSON with all definition locations for the symbol, or a NOT_FOUND message.
    """
    t0 = _import_time()
    try:
        client = get_client()
        ws = workspace_id.strip().lower().replace(" ", "-").replace("_", "-")

        # If workspace_id is empty but file_path is provided, resolve root to derive workspace
        if not ws and file_path:
            from .indexer.discovery import WorkspaceDiscovery, find_root_from_path

            cand_root = find_root_from_path(file_path)
            if cand_root is not None:
                projects = WorkspaceDiscovery(cand_root).discover()
                if projects:
                    ws = projects[0].workspace_id

        # Build lookup keys to attempt
        keys_to_try: list[str] = []
        if ws:
            keys_to_try.append(f"sym:{ws}:{symbol}")
        # Fallback: scan without workspace prefix (best-effort for unnamed workspaces)
        keys_to_try.append(f"sym::{symbol}")
        keys_to_try.append(f"sym:{symbol}")

        for key in keys_to_try:
            try:
                raw = client.get(key)
                if raw is None:
                    continue
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                definitions = json.loads(text)
                elapsed_ns = (_import_time() - t0) * 1_000_000_000.0
                tracker.record_hit(
                    cached_content_chars=len(text),
                    elapsed_us=elapsed_ns / 1000.0,
                    op_type="kv_get",
                    client=client,
                )
                return json.dumps(
                    {
                        "symbol": symbol,
                        "workspace_id": ws or "(global)",
                        "definitions": definitions,
                        "definition_count": (
                            len(definitions) if isinstance(definitions, list) else 1
                        ),
                        "lookup_key": key,
                        "elapsed_ns": round(elapsed_ns, 1),
                    },
                    indent=2,
                )
            except Exception:
                continue

        # Missing from SwissTable on first pass: check if workspace needs lazy auto-indexing
        # (e.g. server was restarted, cache is cold, or file_path anchor provided)
        indexed = (
            _ensure_workspace_indexed(workspace_id=ws, path=file_path)
            if file_path
            else _ensure_workspace_indexed(workspace_id=ws)
        )
        if indexed:
            for key in keys_to_try:
                try:
                    raw = client.get(key)
                    if raw is None:
                        continue
                    text = (
                        raw.decode("utf-8", errors="replace")
                        if isinstance(raw, bytes)
                        else str(raw)
                    )
                    definitions = json.loads(text)
                    elapsed_ns = (_import_time() - t0) * 1_000_000_000.0
                    tracker.record_hit(
                        cached_content_chars=len(text),
                        elapsed_us=elapsed_ns / 1000.0,
                        op_type="kv_get",
                        client=client,
                    )
                    return json.dumps(
                        {
                            "symbol": symbol,
                            "workspace_id": ws or "(global)",
                            "definitions": definitions,
                            "definition_count": (
                                len(definitions) if isinstance(definitions, list) else 1
                            ),
                            "lookup_key": key,
                            "elapsed_ns": round(elapsed_ns, 1),
                            "auto_indexed": True,
                        },
                        indent=2,
                    )
                except Exception:
                    continue

        tracker.record_miss(client=client)
        return json.dumps(
            {
                "status": "NOT_FOUND",
                "symbol": symbol,
                "workspace_id": ws or "(global)",
                "hint": "Run kache_index_workspace first to populate the symbol registry.",
            }
        )
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def _import_time() -> float:
    """Return current perf_counter time for sub-microsecond elapsed calculations."""
    import time as _t

    return _t.perf_counter()


def kache_workspace_status(
    workspace_id: str = "",
    auto_index: bool = True,
) -> str:
    """Return indexing status, total indexed symbols, languages detected,
    last git commit indexed, and cache freshness for the workspace.

    Args:
        workspace_id: Optional workspace ID to inspect. Checks all workspaces if empty.
        auto_index: If True (default), lazily indexes the workspace if not yet indexed.

    Returns:
        JSON status report with symbol counts and git watermark info.
    """
    try:
        client = get_client()
        ws = workspace_id.strip().lower().replace(" ", "-").replace("_", "-")
        meta_key = f"kache:meta:{ws}:commit" if ws else None

        status: dict[str, object] = {
            "workspace_id": ws or "(global)",
            "server": f"{settings.host}:{settings.port}",
        }

        # Check commit watermark
        if meta_key:
            try:
                raw_commit = client.get(meta_key)
                if not raw_commit and auto_index and _ensure_workspace_indexed(workspace_id=ws):
                    status["auto_indexed"] = True
                    raw_commit = client.get(meta_key)

                if raw_commit:
                    commit_sha = (
                        raw_commit.decode("utf-8", errors="replace")
                        if isinstance(raw_commit, bytes)
                        else str(raw_commit)
                    )
                    status["last_indexed_commit"] = commit_sha.strip()
                    status["index_status"] = "INDEXED"
                else:
                    status["index_status"] = "NOT_INDEXED"
                    status["hint"] = "Run kache_index_workspace to populate the symbol registry."
            except Exception as e:
                status["index_status"] = "UNKNOWN"
                status["error"] = str(e)

            # Retrieve or resolve workspace directory path
            try:
                raw_path = client.get(f"kache:meta:{ws}:path")
                if raw_path:
                    status["workspace_path"] = (
                        raw_path.decode("utf-8", errors="replace")
                        if isinstance(raw_path, bytes)
                        else str(raw_path)
                    )
            except Exception:
                pass
            if "workspace_path" not in status:
                from .indexer.discovery import resolve_workspace_path

                resolved_p = resolve_workspace_path(ws, kache_client=client)
                if resolved_p:
                    status["workspace_path"] = str(resolved_p)
        else:
            status["index_status"] = "UNKNOWN"
            status["hint"] = "Provide workspace_id to check specific workspace status."

        # KacheDB health
        try:
            pong = client.ping()
            status["kachedb_status"] = "HEALTHY" if pong == "PONG" else "DEGRADED"
        except Exception as e:
            status["kachedb_status"] = "OFFLINE"
            status["kachedb_error"] = str(e)

        # Vector index stats
        try:
            target_index = _get_target_index(ws)
            vstats = client.vstats(target_index) or {}
            status["vector_index"] = target_index
            status["vector_stats"] = vstats
        except Exception:
            pass

        # Telemetry summary
        try:
            status["telemetry"] = tracker.summary(client=client)
        except Exception:
            status["telemetry"] = tracker.summary()

        return json.dumps(status, indent=2)
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})
