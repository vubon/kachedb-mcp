"""Unit tests for kachedb-mcp tools, telemetry, and FastMCP registration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from kachedb import KacheClient, SemanticCache
from kachedb_mcp.server import mcp
from kachedb_mcp.telemetry import TelemetryTracker
from kachedb_mcp.tools import (
    kache_delete,
    kache_get,
    kache_save_context,
    kache_semantic_search,
    kache_set,
    kache_stats,
    kache_telemetry,
)


class TestTelemetryTracker:
    def test_record_hit_and_miss(self) -> None:
        t = TelemetryTracker()
        t.record_hit(cached_content_chars=400, elapsed_us=50.0)
        assert t.hits == 1
        assert t.tokens_saved == 100
        assert t.latency_saved_ms > 300.0

        t.record_miss()
        assert t.misses == 1

        summary = t.summary()
        assert summary["total_lookups"] == 2
        assert summary["cache_hits"] == 1
        assert summary["cache_misses"] == 1
        assert summary["hit_ratio_percent"] == 50.0
        assert summary["tokens_saved"] == 100


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

    @patch("kachedb_mcp.tools.get_semantic_cache")
    def test_kache_save_context(self, mock_get_cache: MagicMock) -> None:
        mock_cache = MagicMock(spec=SemanticCache)
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache

        res = kache_save_context("S3-FIFO cache logic", "Uses small and main FIFO queues.")
        assert "OK: Saved semantic memory" in res
        assert mock_cache.set.called

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
            "kache_delete",
            "kache_stats",
            "kache_telemetry",
        ]
        for name in expected:
            assert name in tool_names, f"Tool '{name}' not found in FastMCP registry"
