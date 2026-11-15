"""Unit tests for kachedb-mcp tools, telemetry, and FastMCP registration."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kachedb import KacheClient
from kachedb.semantic.embedders import EmbeddingGemmaAdapter
from kachedb_mcp.server import mcp
from kachedb_mcp.telemetry import TelemetryTracker
from kachedb_mcp.tools import (
    _resolve_embedder,
    kache_delete,
    kache_get,
    kache_save_context,
    kache_semantic_search,
    kache_set,
    kache_stats,
    kache_telemetry,
)


class TestTelemetryTracker:
    def test_record_hit_semantic_search(self) -> None:
        """Verify baseline latency, avoided source chars, and USD rate for semantic search."""
        t = TelemetryTracker()
        # Simulate a semantic search hit: 400-char payload, 50 µs search time.
        t.record_hit(cached_content_chars=400, elapsed_us=50.0, op_type="semantic_search")
        assert t.hits == 1
        assert t.tokens_saved == 100  # 400 chars // 4
        # Baseline is 6200 ms for semantic_search, so latency_saved ~ 6199.95 ms
        assert t.latency_saved_ms > 6000.0

    def test_record_hit_kv_get(self) -> None:
        """Verify kv_get uses 150 ms baseline instead of legacy 400 ms."""
        t = TelemetryTracker()
        t.record_hit(cached_content_chars=400, elapsed_us=50.0, op_type="kv_get")
        assert t.hits == 1
        # Baseline is 150 ms for kv_get, so latency_saved ~ 149.95 ms
        assert 100.0 < t.latency_saved_ms < 200.0

    def test_record_hit_avoided_source_chars(self) -> None:
        """Verify token savings use avoided_source_chars when provided."""
        t = TelemetryTracker()
        # 200-char cached summary replaced a 10,000-char source file read.
        t.record_hit(
            cached_content_chars=200,
            elapsed_us=50.0,
            op_type="kv_get",
            avoided_source_chars=10_000,
        )
        assert t.tokens_saved == 2_500  # 10,000 // 4, not 200 // 4 = 50

    def test_record_miss(self) -> None:
        t = TelemetryTracker()
        t.record_miss()
        assert t.misses == 1

    def test_summary_fields_and_usd(self) -> None:
        """Verify USD savings use configurable rate, not hardcoded $5.00/M."""
        t = TelemetryTracker()
        t.record_hit(cached_content_chars=400, elapsed_us=50.0, op_type="semantic_search")
        t.record_miss()

        summary = t.summary()
        assert summary["total_lookups"] == 2
        assert summary["cache_hits"] == 1
        assert summary["cache_misses"] == 1
        assert summary["hit_ratio_percent"] == 50.0
        assert summary["tokens_saved"] == 100
        # Default rate is $3.00/M, so $0.0003 for 100 tokens
        assert summary["estimated_usd_saved"] == "$0.0003"
        assert "token_cost_per_million" in summary
        assert summary["token_cost_per_million"] == 3.0


class TestMCPToolsWithMocks:
    @patch("kachedb_mcp.tools.get_client")
    def test_kache_get_hit(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.get.return_value = b"def my_cached_function(): pass"
        mock_get_client.return_value = mock_client

        res = kache_get("ast:sample.py")
        assert res == "def my_cached_function(): pass"
        assert mock_client.get.called

    @patch("kachedb_mcp.tools.get_client")
    def test_kache_get_miss(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        res = kache_get("missing:key")
        assert "[MISS]" in res

    @patch("kachedb_mcp.tools.get_client")
    def test_kache_set(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.set.return_value = True
        mock_get_client.return_value = mock_client

        res = kache_set("key1", "val1", ttl_seconds=3600)
        assert "OK: Cached 4 characters" in res
        assert "TTL: 3600s" in res

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_kache_save_context(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        mock_cache = MagicMock()
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.vadd.return_value = True
        mock_get_client.return_value = mock_client

        res = kache_save_context(
            "S3-FIFO cache logic",
            "Uses small and main FIFO queues.",
            workspace_id="ws_123",
        )
        assert "OK: Saved semantic memory" in res
        assert "[Workspace: ws_123]" in res
        assert mock_client.vadd.called

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_kache_semantic_search_hit(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        mock_cache = MagicMock()
        mock_cache.index_name = "agent_semantic_memory"
        mock_cache.embedder = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 384
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.vsearch.return_value = [
            (b"S3-FIFO cache logic", 0.942, b"Uses small and main FIFO queues.")
        ]
        mock_get_client.return_value = mock_client

        res = kache_semantic_search("how does eviction work?", top_k=1, threshold=0.7)
        assert "🧠 KacheDB Semantic Matches" in res
        assert "S3-FIFO cache logic" in res
        assert "0.942" in res

    @patch("kachedb_mcp.tools.get_client")
    def test_kache_delete(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.delete.return_value = 1
        mock_client.vdel.return_value = True
        mock_get_client.return_value = mock_client

        res = kache_delete("old_key")
        assert "OK: Removed 'old_key'" in res

    @patch("kachedb_mcp.tools.get_client")
    def test_kache_stats(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.ping.return_value = "PONG"
        mock_client.vstats.return_value = {"dimension": 384, "total_vectors": 128}
        mock_get_client.return_value = mock_client

        stats_str = kache_stats()
        stats_json = json.loads(stats_str)
        assert stats_json["status"] == "HEALTHY"
        assert stats_json["vector_metrics"]["total_vectors"] == 128

    def test_kache_telemetry(self) -> None:
        telemetry_str = kache_telemetry()
        telemetry_json = json.loads(telemetry_str)
        assert "tokens_saved" in telemetry_json
        assert "hit_ratio_percent" in telemetry_json


class TestFastMCPRegistration:
    def test_all_tools_registered(self) -> None:
        # FastMCP stores tools in its internal registry
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        expected = [
            "kache_get",
            "kache_set",
            "kache_save_context",
            "kache_semantic_search",
            "kache_get_parent_document",
            "kache_delete",
            "kache_stats",
            "kache_telemetry",
        ]
        for name in expected:
            assert name in tool_names, f"Tool '{name}' not found in FastMCP registry"


class TestEmbeddingGemma2MCPIntegration:
    """Tests for EmbeddingGemma 2 configuration, resolution, and asymmetric task encoding in MCP."""

    @staticmethod
    def _build_synthetic_gemma(
        dimension: int = 384, recorded_prompts: list[str] | None = None
    ) -> EmbeddingGemmaAdapter:
        tok = MagicMock()

        def _encode(text: str, add_special_tokens: bool = True) -> MagicMock:
            assert add_special_tokens is True
            if recorded_prompts is not None:
                recorded_prompts.append(text)
            enc = MagicMock()
            enc.ids = [101] * max(1, min(8000, len(text) // 4))
            enc.attention_mask = [1] * len(enc.ids)
            return enc

        tok.encode.side_effect = _encode

        sess = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        sess.get_inputs.return_value = [inp]
        sess.run.return_value = [np.ones((1, 768), dtype=np.float32)]

        return EmbeddingGemmaAdapter(
            dimension=dimension,
            _session=sess,
            _tokenizer=tok,
        )

    def test_resolve_embedder_embeddinggemma_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KACHEDB_EMBEDDER", "embeddinggemma-2")
        monkeypatch.setenv("KACHEDB_EMBEDDING_DIM", "384")
        monkeypatch.setenv("KACHEDB_MODEL_PATH", "/models/embeddinggemma-2")

        fake_adapter = self._build_synthetic_gemma(dimension=384)
        with patch(
            "kachedb_mcp.tools.EmbeddingGemmaAdapter", return_value=fake_adapter
        ) as mock_cls:
            resolved = _resolve_embedder()
            assert resolved is fake_adapter
            mock_cls.assert_called_once_with(
                model_path="/models/embeddinggemma-2",
                dimension=384,
                search_prefix="task: search query | text: ",
                indexing_prefix="task: document indexing | text: ",
            )

    def test_resolve_embedder_auto_with_model_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KACHEDB_EMBEDDER", "auto")
        monkeypatch.setenv("EMBEDDING_MODEL_PATH", "/models/gemma2")
        monkeypatch.setenv("KACHEDB_EMBEDDING_DIM", "768")

        fake_adapter = self._build_synthetic_gemma(dimension=768)
        with patch(
            "kachedb_mcp.tools.EmbeddingGemmaAdapter", return_value=fake_adapter
        ) as mock_cls:
            resolved = _resolve_embedder()
            assert resolved is fake_adapter
            mock_cls.assert_called_once_with(
                model_path="/models/gemma2",
                dimension=768,
                search_prefix="task: search query | text: ",
                indexing_prefix="task: document indexing | text: ",
            )

    @patch("kachedb_mcp.tools.get_client")
    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_asymmetric_encoding_and_8k_context_in_mcp_tools(
        self, mock_get_cache: MagicMock, mock_get_client: MagicMock
    ) -> None:
        recorded_prompts: list[str] = []
        gemma_embedder = self._build_synthetic_gemma(
            dimension=384, recorded_prompts=recorded_prompts
        )

        mock_cache = MagicMock()
        mock_cache.index_name = "agent_semantic_memory"
        mock_cache.embedder = gemma_embedder
        mock_get_cache.return_value = mock_cache

        mock_client = MagicMock(spec=KacheClient)
        mock_client.get.return_value = None
        mock_client.vadd.return_value = True
        mock_client.vsearch.return_value = [
            (b"Large architecture doc", 0.915, b"Full 8K context payload")
        ]
        mock_get_client.return_value = mock_client

        # 1. Save a 5,000-char document with default auto_chunk=True (preserved as single 8K item)
        long_doc = "Architecture detail " * 250  # 5,000 chars (> 1500 legacy truncation)
        save_res = kache_save_context(
            topic="Large architecture doc",
            content=long_doc,
            workspace_id="kolom",
            auto_chunk=True,
            prompt_alias="How does the architecture scale?",
        )
        assert "OK: Saved semantic memory" in save_res
        assert len(recorded_prompts) == 2
        # First call is document indexing with full 5,000-char body preserved
        assert recorded_prompts[0].startswith("task: document indexing | text: ")
        assert len(recorded_prompts[0]) > 4500
        # Second call is prompt_alias anchor encoded with search query prefix
        assert recorded_prompts[1] == "task: search query | text: How does the architecture scale?"

        # 2. Semantic search uses search query task prefix and 384-d unit vector
        search_res = kache_semantic_search(
            "Explain how the architecture scales",
            workspace_id="kolom",
            exact_first=False,
        )
        assert "🧠 KacheDB Semantic Matches" in search_res
        assert (
            recorded_prompts[2] == "task: search query | text: Explain how the architecture scales"
        )
        vsearch_call = mock_client.vsearch.call_args
        q_vec = vsearch_call.kwargs["query_vector"]
        assert len(q_vec) == 384
        assert abs(math.sqrt(sum(x * x for x in q_vec)) - 1.0) < 1e-6

    def test_invalid_embedding_dim_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KACHEDB_EMBEDDING_DIM", "512")
        with pytest.raises(ValueError, match="Unsupported KACHEDB_EMBEDDING_DIM=512"):
            _resolve_embedder()
