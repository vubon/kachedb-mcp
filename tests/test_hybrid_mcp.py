"""Unit tests for Phase 3 hybrid context engine features in kachedb-mcp."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kachedb import KacheClient
from kachedb.client import VectorMatch
from kachedb_mcp.telemetry import TelemetryTracker
from kachedb_mcp.tools import (
    kache_get_parent_document,
    kache_save_context,
    kache_semantic_search,
)


class TestHybridContextMCP:
    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_save_context_auto_chunking(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify large document (>1500 chars) triggers hierarchical chunking."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.05] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.set.return_value = True
        mock_client.vadd.return_value = True
        mock_get_client.return_value = mock_client

        # Document > 1500 chars with markdown sections
        section_1 = "## Architecture Overview\n" + ("KacheDB is built with Rust and SIMD. " * 30)
        section_2 = "## Memory Management\n" + ("Custom SwissTable and S3-FIFO eviction. " * 30)
        large_doc = f"# Deep Dive\n\n{section_1}\n\n{section_2}"
        assert len(large_doc) > 1500

        res = kache_save_context(
            topic="kachedb_arch",
            content=large_doc,
            workspace_id="test_ws",
            tags=["rust", "arch"],
            auto_chunk=True,
        )

        assert "OK: Saved hierarchical context for 'kachedb_arch'" in res
        assert "[Workspace: test_ws]" in res
        assert "[Tags: rust, arch]" in res
        assert "chunks" in res
        assert "parent: 'doc:test_ws:kachedb_arch'" in res

        # Verify parent document saved in KV
        mock_client.set.assert_any_call("doc:test_ws:kachedb_arch", large_doc, ex=86400)
        # Verify chunks added via vadd with parent_key
        assert mock_client.vadd.call_count >= 2
        call_kwargs = mock_client.vadd.call_args[1]
        assert call_kwargs["parent_key"] == "doc:test_ws:kachedb_arch"
        assert call_kwargs["tags"] == ["rust", "arch"]
        assert call_kwargs["tag_mask"] > 0

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_save_context_small_doc_no_chunk(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify short document does not chunk but stores exact KV key."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.set.return_value = True
        mock_client.vadd.return_value = True
        mock_get_client.return_value = mock_client

        res = kache_save_context(
            topic="quick_tip",
            content="Use NEON SIMD on Apple Silicon.",
            workspace_id="dev",
            tags=["tip"],
            auto_chunk=True,
        )

        assert "OK: Saved semantic memory for 'quick_tip'" in res
        assert mock_client.vadd.call_count == 1
        mock_client.set.assert_any_call(
            "doc:dev:quick_tip", "Use NEON SIMD on Apple Silicon.", ex=86400
        )

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_semantic_search_exact_hit(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify exact symbol match in KV returns instantly (<50ns) without embedding."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        # Client returns exact content on get()
        mock_client.get.return_value = b"def evict_expired_keys_batch(): pass"
        mock_get_client.return_value = mock_client

        res = kache_semantic_search(
            query="evict_expired_keys_batch",
            workspace_id="core",
            exact_first=True,
        )

        assert "⚡ KacheDB Exact Match for 'evict_expired_keys_batch'" in res
        assert "Score: 1.000 [EXACT]" in res
        assert "def evict_expired_keys_batch(): pass" in res
        # Verify embedding model was NOT invoked!
        assert not mock_cache.embedder.encode.called
        # Verify vsearch was NOT invoked!
        assert not mock_client.vsearch.called

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_semantic_search_vector_fallback_with_tags(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify vector search is used when exact match is missing, passing filter tags."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.2] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.get.return_value = None  # Exact miss
        mock_client.vsearch.return_value = [
            VectorMatch(("doc:core#chunk_1", 0.885, "SIMD vector dot product kernel", "doc:core"))
        ]
        mock_get_client.return_value = mock_client

        res = kache_semantic_search(
            query="how does vector dot product work?",
            workspace_id="core",
            tags=["simd", "kernel"],
            exact_first=True,
            resolve_parent=False,
        )

        assert "🧠 KacheDB Semantic Matches" in res
        assert "Filter: simd, kernel" in res
        assert "SIMD vector dot product kernel" in res
        assert "Parent: doc:core" in res
        assert mock_cache.embedder.encode.called
        assert mock_client.vsearch.called

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_semantic_search_resolve_parent(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify resolve_parent=True pulls the parent document from KV."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.2] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)

        def mock_get(key: str) -> bytes | None:
            if key == "doc:core:full":
                return b"# Complete Architecture Specification Document"
            return None

        mock_client.get.side_effect = mock_get
        mock_client.vsearch.return_value = [
            VectorMatch(("chunk_1", 0.92, "Chunk summary text", "doc:core:full"))
        ]
        mock_get_client.return_value = mock_client

        res = kache_semantic_search(
            query="architecture spec",
            exact_first=False,
            resolve_parent=True,
        )

        assert "🧠 KacheDB Semantic Matches" in res
        assert "FULL PARENT DOCUMENT: doc:core:full" in res
        assert "Complete Architecture Specification Document" in res

    @patch("kachedb_mcp.tools.get_client")
    def test_kache_get_parent_document_hit(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.get.return_value = b"# Full Parent Document Content"
        mock_get_client.return_value = mock_client

        res = kache_get_parent_document("doc:arch:simd")
        assert "📄 Parent Document 'doc:arch:simd'" in res
        assert "# Full Parent Document Content" in res

    @patch("kachedb_mcp.tools.get_client")
    def test_kache_get_parent_document_not_found(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        res = kache_get_parent_document("doc:nonexistent")
        assert "[NOT_FOUND]" in res

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_save_context_with_prompt_alias(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify prompt_alias registers SwissTable query keys and anchor vector."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.set.return_value = True
        mock_client.vadd.return_value = True
        mock_get_client.return_value = mock_client

        res = kache_save_context(
            topic="vector_search_arch",
            content="Vector search uses AVX-512 and NEON SIMD kernels for cosine similarity.",
            workspace_id="test_ws",
            prompt_alias="How does vector search work in kachedb?",
            tags=["simd", "vector"],
        )

        assert "OK: Saved semantic memory for 'vector_search_arch'" in res
        assert "[Prompt: 'How does vector search work in kachedb?']" in res

        # Verify query aliases registered in SwissTable pointing to parent doc
        mock_client.set.assert_any_call(
            "query:How does vector search work in kachedb?",
            "doc:test_ws:vector_search_arch",
            ex=86400,
        )
        mock_client.set.assert_any_call(
            "query:how does vector search work in kachedb",
            "doc:test_ws:vector_search_arch",
            ex=86400,
        )
        mock_client.set.assert_any_call(
            "query:test_ws:how does vector search work in kachedb",
            "doc:test_ws:vector_search_arch",
            ex=86400,
        )

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_semantic_search_exact_query_alias_dereference(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify natural language question hits exact query alias and dereferences parent doc."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)

        def mock_get(key: str) -> bytes | None:
            if key == "query:how does vector search work in kachedb":
                return b"doc:vector_search_arch"
            if key == "doc:vector_search_arch":
                return b"Vector search uses AVX-512 and NEON SIMD kernels for cosine similarity."
            return None

        mock_client.get.side_effect = mock_get
        mock_get_client.return_value = mock_client

        res = kache_semantic_search(
            query="How does vector search work in kachedb?",
            exact_first=True,
        )

        assert "⚡ KacheDB Exact Match for 'How does vector search work in kachedb?'" in res
        assert "Key: query:how does vector search work in kachedb -> doc:vector_search_arch" in res
        assert "Vector search uses AVX-512 and NEON SIMD kernels" in res
        # Verify neural embedder and vsearch were completely skipped!
        assert not mock_cache.embedder.encode.called
        assert not mock_client.vsearch.called

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_semantic_search_adaptive_fallback(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """Verify adaptive fallback relaxes threshold when strict threshold yields no match."""
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.15] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.get.return_value = None  # Exact miss

        # First call with threshold=0.85 returns empty;
        # second call with relaxed threshold returns match
        def mock_vsearch(
            index: str,
            query_vector: list[float],
            top_k: int,
            threshold: float,
            filter_tags: list[str] | None = None,
        ) -> list[VectorMatch]:
            if threshold >= 0.80:
                return []
            return [VectorMatch(("doc:arch#chunk_1", 0.68, "Adaptive match content", "doc:arch"))]

        mock_client.vsearch.side_effect = mock_vsearch
        mock_get_client.return_value = mock_client

        res = kache_semantic_search(
            query="Tell me about architecture",
            threshold=0.85,
            exact_first=True,
        )

        assert "🧠 KacheDB Semantic Matches" in res
        assert "[Adaptive: threshold relaxed from 0.85 to 0.65]" in res
        assert "Adaptive match content" in res


class TestTelemetryTrackerPhase3:
    def test_exact_hits_and_embedding_time_tracking(self) -> None:
        t = TelemetryTracker()
        # Record 1 exact hit (avoids 15 ms embedding)
        t.record_hit(
            cached_content_chars=500,
            elapsed_us=30.0,
            is_exact=True,
            avoided_embedding_ms=15.0,
        )
        # Record 1 vector hit
        t.record_hit(
            cached_content_chars=800,
            elapsed_us=120.0,
            is_exact=False,
            op_type="semantic_search",
        )

        assert t.hits == 2
        assert t.exact_hits == 1
        assert t.vector_hits == 1
        assert t.embed_time_saved_ms == 15.0

        summary = t.summary()
        assert summary["exact_hits"] == 1
        assert summary["vector_hits"] == 1
        assert summary["embed_time_saved_ms"] == 15.0
