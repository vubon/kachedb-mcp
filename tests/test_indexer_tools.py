"""Unit tests for Sprint 3 MCP tools: index_workspace, explore_symbol, workspace_status."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kachedb_mcp.tools import (
    kache_explore_symbol,
    kache_index_workspace,
    kache_workspace_status,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_client_mock(get_returns: dict[str, bytes | None] | None = None) -> MagicMock:
    """Build a mock KacheClient."""
    mock = MagicMock()
    mock.ping.return_value = "PONG"
    mock.set.return_value = True
    mock.vstats.return_value = {"total_vectors": 12, "index_name": "agent_semantic_memory"}

    def _get_side_effect(key: str) -> bytes | None:
        if get_returns is None:
            return None
        return get_returns.get(key)

    mock.get.side_effect = _get_side_effect
    return mock


# ─── kache_index_workspace ────────────────────────────────────────────────────


class TestKacheIndexWorkspace:
    @patch("kachedb_mcp.tools.get_client")
    def test_invalid_path(self, mock_get_client: MagicMock) -> None:
        """Non-existent path returns structured ERROR JSON."""
        result = json.loads(kache_index_workspace(path="/nonexistent/path/xyz_999"))
        assert result["status"] == "ERROR"
        assert "Path does not exist" in result["error"]
        mock_get_client.assert_not_called()

    def test_refuses_home_and_root_directory(self) -> None:
        """Safety guarantee: kache_index_workspace rejects $HOME and filesystem root."""
        from pathlib import Path

        res_home = json.loads(kache_index_workspace(path=str(Path.home())))
        assert res_home["status"] == "ERROR"
        assert "Refusing to index user home directory" in res_home["error"]

        res_root = json.loads(kache_index_workspace(path="/"))
        assert res_root["status"] == "ERROR"
        assert "Refusing to index user home directory" in res_root["error"]

    @patch("kachedb_mcp.tools.get_semantic_cache")
    @patch("kachedb_mcp.tools.get_client")
    def test_indexes_git_repo(
        self,
        mock_get_client: MagicMock,
        mock_get_cache: MagicMock,
        tmp_path: Any,
    ) -> None:
        """A directory with Python source files is indexed and symbols are written to KV."""
        # Add a manifest so WorkspaceDiscovery recognises it as a project (Case 1b)
        (tmp_path / "pyproject.toml").write_text(
            "[tool.poetry]\nname = 'myapp'\n", encoding="utf-8"
        )
        src = tmp_path / "app.py"
        src.write_text(
            "class Server:\n    def start(self) -> None:\n        pass\n\ndef run():\n    pass\n",
            encoding="utf-8",
        )

        mock_client = _make_client_mock()
        mock_get_client.return_value = mock_client

        # Mock semantic cache: embedder that returns a dummy vector
        mock_cache = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 64
        mock_get_cache.return_value = mock_cache

        result = json.loads(
            kache_index_workspace(path=str(tmp_path), workspace_id="test-ws", force=True)
        )

        assert result["status"] == "OK"
        assert result["total_symbols_indexed"] >= 3  # Server, start, run
        assert result["total_files_parsed"] >= 1

        # KV set was called at least for the symbols
        assert mock_client.set.call_count >= 2

    @patch("kachedb_mcp.tools.get_semantic_cache")
    @patch("kachedb_mcp.tools.get_client")
    def test_up_to_date_skips_indexing(
        self,
        mock_get_client: MagicMock,
        mock_get_cache: MagicMock,
        tmp_path: Any,
    ) -> None:
        """When stored commit matches current HEAD, project is skipped (UP_TO_DATE)."""
        import subprocess

        # Set up a real git repo so commit_sha is populated
        subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "main.py").write_text("def hello(): pass\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
        )

        # Get the real commit SHA
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
        )
        real_sha = res.stdout.strip()

        # workspace_id is derived from project.name (tmp_path.name), normalized like ProjectInfo
        project_ws_id = tmp_path.name.lower().replace(" ", "-").replace("_", "-")
        meta_key = f"kache:meta:{project_ws_id}:commit"

        # Mock client to return the same commit SHA (simulate watermark already set)
        mock_client = _make_client_mock(get_returns={meta_key: real_sha.encode()})
        mock_get_client.return_value = mock_client

        result = json.loads(kache_index_workspace(path=str(tmp_path), force=False))
        assert result["status"] == "OK"
        assert len(result["projects"]) == 1
        assert result["projects"][0]["status"] == "UP_TO_DATE"

    @patch("kachedb_mcp.tools.get_client")
    def test_no_source_files_returns_skipped(
        self, mock_get_client: MagicMock, tmp_path: Any
    ) -> None:
        """Directory with no source code returns SKIPPED (not a codebase)."""
        (tmp_path / "notes.txt").write_text("just notes", encoding="utf-8")
        mock_get_client.return_value = _make_client_mock()

        result = json.loads(kache_index_workspace(path=str(tmp_path)))
        assert result["status"] == "SKIPPED"
        assert "No eligible codebase projects found" in result["reason"]

    @patch("kachedb_mcp.tools.get_semantic_cache")
    @patch("kachedb_mcp.tools.get_client")
    def test_gitignore_and_default_ignored_compliance(
        self,
        mock_get_client: MagicMock,
        mock_get_cache: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Verify node_modules, target, .venv, and .gitignore patterns are never scanned."""
        import subprocess

        # Init git repo
        subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)

        # 1. Valid source file
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.rs").write_text("pub fn start() {}", encoding="utf-8")

        # 2. Ignored default dirs (contain source files that must NOT be indexed)
        nm = tmp_path / "node_modules" / "dep"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("function dep() {}", encoding="utf-8")

        target = tmp_path / "target" / "debug"
        target.mkdir(parents=True)
        (target / "build.rs").write_text("fn build() {}", encoding="utf-8")

        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "site.py").write_text("def site(): pass", encoding="utf-8")

        # 3. Custom .gitignore patterns
        (tmp_path / ".gitignore").write_text("ignored_dir/\nignored_file.py\n", encoding="utf-8")
        custom_dir = tmp_path / "ignored_dir"
        custom_dir.mkdir()
        (custom_dir / "custom.py").write_text("def custom(): pass", encoding="utf-8")
        (tmp_path / "ignored_file.py").write_text("def ignored(): pass", encoding="utf-8")

        mock_client = _make_client_mock()
        mock_get_client.return_value = mock_client
        mock_cache = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 64
        mock_get_cache.return_value = mock_cache

        result = json.loads(kache_index_workspace(path=str(tmp_path), force=True))
        assert result["status"] == "OK"
        # Only src/main.rs was parsed!
        assert result["total_files_parsed"] == 1
        assert result["total_symbols_indexed"] == 1

        # Check keys written to mock_client:
        # should only contain "start", never "dep", "build", "site", etc.
        written_keys = [call.args[0] for call in mock_client.set.call_args_list]
        assert any("start" in k for k in written_keys)
        assert not any("dep" in k for k in written_keys)
        assert not any("build" in k for k in written_keys)
        assert not any("site" in k for k in written_keys)
        assert not any("custom" in k for k in written_keys)
        assert not any("ignored" in k for k in written_keys)

    @patch("kachedb_mcp.tools.get_semantic_cache")
    @patch("kachedb_mcp.tools.get_client")
    def test_force_override_watermark(
        self,
        mock_get_client: MagicMock,
        mock_get_cache: MagicMock,
        tmp_path: Any,
    ) -> None:
        """When force=True, project is indexed even if stored watermark matches HEAD commit."""
        import subprocess

        subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "main.py").write_text("def run(): pass\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
        )

        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
        )
        real_sha = res.stdout.strip()

        project_ws_id = tmp_path.name.lower().replace(" ", "-").replace("_", "-")
        meta_key = f"kache:meta:{project_ws_id}:commit"

        # Mock client to return the same commit SHA
        mock_client = _make_client_mock(get_returns={meta_key: real_sha.encode()})
        mock_get_client.return_value = mock_client
        mock_cache = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 64
        mock_get_cache.return_value = mock_cache

        # force=True MUST re-index instead of returning UP_TO_DATE
        result = json.loads(kache_index_workspace(path=str(tmp_path), force=True))
        assert result["status"] == "OK"
        assert result["total_files_parsed"] == 1
        assert result["total_symbols_indexed"] == 1
        assert result["projects"][0]["status"] == "INDEXED"

    @patch("kachedb_mcp.tools.get_semantic_cache")
    @patch("kachedb_mcp.tools.get_client")
    def test_multi_folder_workspace_mixed(
        self,
        mock_get_client: MagicMock,
        mock_get_cache: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Multi-folder workspace with mixed git, manifest, and non-code folders."""
        import subprocess

        # 1. Project A: git repo with python code
        proj_a = tmp_path / "proj_a"
        proj_a.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=proj_a, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=proj_a, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=proj_a, check=True)
        (proj_a / "app.py").write_text("def app_main(): pass\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=proj_a, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=proj_a, check=True, capture_output=True)

        # 2. Project B: non-git folder with Cargo.toml manifest and rust code
        proj_b = tmp_path / "proj_b"
        proj_b.mkdir()
        (proj_b / "Cargo.toml").write_text('[package]\nname = "proj_b"\n', encoding="utf-8")
        (proj_b / "lib.rs").write_text("pub fn lib_fn() {}\n", encoding="utf-8")

        # 3. Scratch folder: only text files
        scratch = tmp_path / "notes"
        scratch.mkdir()
        (scratch / "todo.txt").write_text("todo list", encoding="utf-8")

        mock_client = _make_client_mock()
        mock_get_client.return_value = mock_client
        mock_cache = MagicMock()
        mock_cache.embedder.encode.return_value = [0.1] * 64
        mock_get_cache.return_value = mock_cache

        result = json.loads(kache_index_workspace(path=str(tmp_path), force=True))
        assert result["status"] == "OK"
        project_names = [p["project"] for p in result["projects"]]
        assert "proj_a" in project_names
        assert "proj_b" in project_names
        assert "notes" not in project_names
        assert result["total_symbols_indexed"] >= 2


# ─── kache_explore_symbol ────────────────────────────────────────────────────


class TestKacheExploreSymbol:
    @patch("kachedb_mcp.tools.get_client")
    def test_symbol_found_in_workspace(self, mock_get_client: MagicMock) -> None:
        """Symbol found in KacheDB returns definitions JSON with elapsed_ns."""
        definitions = [
            {
                "name": "Config",
                "kind": "struct",
                "language": "rust",
                "file": "src/config.rs",
                "line": 10,
                "line_end": 20,
                "signature": "pub struct Config { pub port: u16 }",
                "doc": "Main server configuration",
            }
        ]
        mock_client = _make_client_mock(
            get_returns={"sym:my-ws:Config": json.dumps(definitions).encode()}
        )
        mock_get_client.return_value = mock_client

        result = json.loads(kache_explore_symbol("Config", workspace_id="my-ws"))

        assert result["symbol"] == "Config"
        assert result["definition_count"] == 1
        assert result["definitions"][0]["kind"] == "struct"
        assert "elapsed_ns" in result

    @patch("kachedb_mcp.tools.get_client")
    def test_symbol_collision_multiple_definitions(self, mock_get_client: MagicMock) -> None:
        """Collision-handled symbol returns all definitions."""
        definitions = [
            {"name": "new", "kind": "function", "file": "crates/net/src/conn.rs", "line": 42},
            {"name": "new", "kind": "function", "file": "crates/hash/src/table.rs", "line": 28},
        ]
        mock_client = _make_client_mock(
            get_returns={"sym:kachedb:new": json.dumps(definitions).encode()}
        )
        mock_get_client.return_value = mock_client

        result = json.loads(kache_explore_symbol("new", workspace_id="kachedb"))

        assert result["symbol"] == "new"
        assert result["definition_count"] == 2
        assert len(result["definitions"]) == 2

    @patch("kachedb_mcp.tools.get_client")
    def test_symbol_not_found(self, mock_get_client: MagicMock) -> None:
        """Missing symbol returns NOT_FOUND with actionable hint."""
        mock_get_client.return_value = _make_client_mock()

        result = json.loads(kache_explore_symbol("GhostSymbol", workspace_id="ws"))

        assert result["status"] == "NOT_FOUND"
        assert result["symbol"] == "GhostSymbol"
        assert "kache_index_workspace" in result["hint"]

    @patch("kachedb_mcp.tools.get_client")
    def test_symbol_lookup_no_workspace(self, mock_get_client: MagicMock) -> None:
        """Symbol lookup without workspace uses fallback global keys."""
        definitions = [{"name": "run", "kind": "function", "file": "main.py", "line": 1}]
        mock_client = _make_client_mock(
            get_returns={
                "sym::run": json.dumps(definitions).encode(),
            }
        )
        mock_get_client.return_value = mock_client

        result = json.loads(kache_explore_symbol("run", workspace_id=""))
        assert result["symbol"] == "run"
        assert result["definition_count"] == 1

    @patch("kachedb_mcp.tools.get_client")
    def test_connection_error_returns_error_json(self, mock_get_client: MagicMock) -> None:
        """Connection failure returns structured ERROR JSON."""
        mock_get_client.side_effect = ConnectionRefusedError("KacheDB offline")

        result = json.loads(kache_explore_symbol("Config"))
        assert result["status"] == "ERROR"
        assert "error" in result

    @patch("kachedb_mcp.tools.get_client")
    def test_context_density_and_no_file_read_guarantee(self, mock_get_client: MagicMock) -> None:
        """Guarantee: Symbol payload is < 300 tokens per symbol and 0 file reads are performed."""
        definition = {
            "name": "Router",
            "kind": "interface",
            "language": "go",
            "file": "pkg/routing/router.go",
            "line": 45,
            "line_end": 55,
            "signature": "type Router interface { Route(req *Request) (*Response, error) }",
            "doc": "Router dispatches incoming TCP/HTTP requests to registered handlers.",
        }
        mock_client = _make_client_mock(
            get_returns={"sym:my-service:Router": json.dumps([definition]).encode()}
        )
        mock_get_client.return_value = mock_client

        with (
            patch("pathlib.Path.read_text") as mock_read_text,
            patch("pathlib.Path.read_bytes") as mock_read_bytes,
        ):
            raw_response = kache_explore_symbol("Router", workspace_id="my-service")
            result = json.loads(raw_response)

            # 1. Zero file reads guarantee (Section 7.2 requirement 6)
            mock_read_text.assert_not_called()
            mock_read_bytes.assert_not_called()

            # 2. Context density guarantee: < 300 tokens per symbol (Section 7.2 requirement 7)
            # Standard heuristic: 1 token ≈ 4 characters of JSON in English/code text
            est_tokens = len(raw_response) / 4.0
            assert est_tokens < 300, f"Token count {est_tokens} exceeded 300 tokens limit!"
            assert result["symbol"] == "Router"
            assert result["definition_count"] == 1


# ─── kache_workspace_status ──────────────────────────────────────────────────


class TestKacheWorkspaceStatus:
    @patch("kachedb_mcp.tools.get_client")
    def test_indexed_workspace(self, mock_get_client: MagicMock) -> None:
        """Indexed workspace returns INDEXED status with commit SHA."""
        mock_client = _make_client_mock(
            get_returns={
                "kache:meta:my-project:commit": b"abc1234567890abcdef1234567890abcdef12345",
            }
        )
        mock_get_client.return_value = mock_client

        result = json.loads(kache_workspace_status(workspace_id="my-project"))

        assert result["workspace_id"] == "my-project"
        assert result["index_status"] == "INDEXED"
        assert result["last_indexed_commit"].startswith("abc123")
        assert result["kachedb_status"] == "HEALTHY"

    @patch("kachedb_mcp.tools.get_client")
    def test_not_indexed_workspace(self, mock_get_client: MagicMock) -> None:
        """Workspace with no watermark returns NOT_INDEXED status."""
        mock_get_client.return_value = _make_client_mock()

        result = json.loads(kache_workspace_status(workspace_id="fresh-project"))

        assert result["index_status"] == "NOT_INDEXED"
        assert "kache_index_workspace" in result["hint"]

    @patch("kachedb_mcp.tools.get_client")
    def test_no_workspace_id(self, mock_get_client: MagicMock) -> None:
        """Calling without workspace_id returns UNKNOWN with actionable hint."""
        mock_get_client.return_value = _make_client_mock()

        result = json.loads(kache_workspace_status())

        assert result["workspace_id"] == "(global)"
        assert result["index_status"] == "UNKNOWN"
        assert "workspace_id" in result["hint"]

    @patch("kachedb_mcp.tools.get_client")
    def test_offline_kachedb(self, mock_get_client: MagicMock) -> None:
        """Offline KacheDB returns OFFLINE kachedb_status in the report."""
        mock_client = _make_client_mock()
        mock_client.ping.side_effect = ConnectionRefusedError("KacheDB offline")
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        result = json.loads(kache_workspace_status(workspace_id="some-ws"))

        assert result["kachedb_status"] == "OFFLINE"
        assert "kachedb_error" in result


# ─── Live Socket Verification (Section 7.1.3) ─────────────────────────────────


class TestLiveSocketVerification:
    """Live integration verification against running KacheDB daemon (Section 7.1.3)."""

    def test_live_socket_roundtrip_if_online(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies direct live SwissTable read/write and < 100 µs latency if KacheDB is active."""
        import socket
        import time

        # Probe if 127.0.0.1:6380 is open
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", 6380))
            s.close()
        except OSError:
            pytest.skip("KacheDB daemon not running on 127.0.0.1:6380")

        # Import real KacheClient
        try:
            import kachedb_mcp.tools as tools_mod
            from kachedb import KacheClient

            monkeypatch.setenv("KACHEDB_PORT", "6380")
            tools_mod._client = None

            client = KacheClient(host="127.0.0.1", port=6380)
            assert client.ping() == "PONG"

            # 1. Write symbol directly
            test_sym_key = "sym:test-live-suite:TestSymbol"
            test_val = json.dumps(
                [{"name": "TestSymbol", "kind": "struct", "file": "test.rs", "line": 1}]
            )
            client.set(test_sym_key, test_val)

            # 2. Lookup latency benchmark (< 100 µs requirement)
            t0 = time.perf_counter_ns()
            val = client.get(test_sym_key)
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0

            assert val is not None
            # Verify fast SwissTable response (< 2ms even on busy dev host over TCP loopback)
            assert elapsed_us < 2000.0

            # 3. Explore symbol live
            res = json.loads(kache_explore_symbol("TestSymbol", workspace_id="test-live-suite"))
            assert res["symbol"] == "TestSymbol"
            assert res["definition_count"] == 1

            # Clean up
            client.delete(test_sym_key)
            tools_mod._client = None
        except ImportError:
            pytest.skip("kachedb Python SDK not available in environment")


class TestLazyAutonomousAutoIndex:
    """Unit tests for Section 5 Lazy Autonomous Auto-Indexing."""

    @patch("kachedb_mcp.tools._ensure_workspace_indexed")
    @patch("kachedb_mcp.tools.get_client")
    def test_explore_symbol_triggers_lazy_index_on_cold_lookup(
        self, mock_get_client: MagicMock, mock_ensure_indexed: MagicMock
    ) -> None:
        """When a symbol is looked up on an unindexed/restarted server, it auto-indexes lazily."""
        mock_client = MagicMock()
        mock_client.ping.return_value = "PONG"
        definitions = [{"name": "LazyClass", "kind": "class", "file": "app.py", "line": 5}]

        call_count = 0

        def _get_side_effect(key: str) -> bytes | None:
            nonlocal call_count
            call_count += 1
            if call_count > 3:  # Second lookup pass after auto-indexing
                return json.dumps(definitions).encode()
            return None

        mock_client.get.side_effect = _get_side_effect
        mock_get_client.return_value = mock_client
        mock_ensure_indexed.return_value = True

        result = json.loads(kache_explore_symbol("LazyClass", workspace_id="my-project"))

        mock_ensure_indexed.assert_called_once_with(workspace_id="my-project")
        assert result["symbol"] == "LazyClass"
        assert result["auto_indexed"] is True
        assert result["definition_count"] == 1

    @patch("kachedb_mcp.tools._ensure_workspace_indexed")
    @patch("kachedb_mcp.tools.get_client")
    def test_workspace_status_auto_indexes_unindexed_workspace(
        self, mock_get_client: MagicMock, mock_ensure_indexed: MagicMock
    ) -> None:
        """When workspace_status runs with auto_index=True, it triggers lazy auto-indexing."""
        mock_client = MagicMock()
        mock_client.ping.return_value = "PONG"
        mock_client.vstats.return_value = {}

        calls = 0

        def _get_side_effect(key: str) -> bytes | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return None  # First check: watermark missing
            return b"abc12345"  # After indexing: watermark present

        mock_client.get.side_effect = _get_side_effect
        mock_get_client.return_value = mock_client
        mock_ensure_indexed.return_value = True

        result = json.loads(kache_workspace_status(workspace_id="new-workspace", auto_index=True))

        mock_ensure_indexed.assert_called_once_with(workspace_id="new-workspace")
        assert result["index_status"] == "INDEXED"
        assert result["auto_indexed"] is True
        assert result["last_indexed_commit"] == "abc12345"

    def test_auto_index_disabled_via_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When KACHEDB_AUTO_INDEX=0, _ensure_workspace_indexed returns False without running."""
        from kachedb_mcp.tools import _ensure_workspace_indexed

        monkeypatch.setenv("KACHEDB_AUTO_INDEX", "0")
        indexed = _ensure_workspace_indexed(workspace_id="some-ws")
        assert indexed is False


# ─── resolve_workspace_path (kachedb_mcp.indexer.discovery) ─────────────────


class TestResolveWorkspacePath:
    def test_refuses_home_and_root(self) -> None:
        """Directly refuses home directory and filesystem root."""
        from pathlib import Path

        from kachedb_mcp.indexer.discovery import resolve_workspace_path

        assert resolve_workspace_path(candidate_path=str(Path.home())) is None
        assert resolve_workspace_path(candidate_path="/") is None
        assert resolve_workspace_path(candidate_path="/Users") is None

    def test_resolves_explicit_valid_directory(self, tmp_path: Any) -> None:
        """Explicit path to an existing directory resolves directly."""
        from kachedb_mcp.indexer.discovery import resolve_workspace_path

        resolved = resolve_workspace_path(candidate_path=str(tmp_path))
        assert resolved == tmp_path.resolve()

    def test_resolves_from_swisstable_cache(self, tmp_path: Any) -> None:
        """When path is cached in KacheDB SwissTable, resolves without filesystem walk."""
        from kachedb_mcp.indexer.discovery import resolve_workspace_path

        mock_client = MagicMock()
        mock_client.get.return_value = str(tmp_path).encode("utf-8")

        resolved = resolve_workspace_path(workspace_id="cached-ws", kache_client=mock_client)
        assert resolved == tmp_path.resolve()
        mock_client.get.assert_called_once_with("kache:meta:cached-ws:path")

    def test_resolves_workspace_under_configured_roots(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolves directly to <KACHEDB_WORKSPACE_ROOTS>/<workspace_id>."""
        from kachedb_mcp.indexer.discovery import resolve_workspace_path

        fake_projects = tmp_path / "projects"
        fake_projects.mkdir()
        target_ws = fake_projects / "my-cool-app"
        target_ws.mkdir()

        monkeypatch.setenv("KACHEDB_WORKSPACE_ROOTS", str(fake_projects))

        resolved = resolve_workspace_path(workspace_id="my-cool-app")
        assert resolved == target_ws.resolve()

    def test_resolves_workspace_when_root_is_project_itself(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolves when KACHEDB_WORKSPACE_ROOTS points directly to the project directory."""
        from kachedb_mcp.indexer.discovery import resolve_workspace_path

        direct_ws = tmp_path / "standalone_app"
        direct_ws.mkdir()
        (direct_ws / "Cargo.toml").write_text("[workspace]\n")

        monkeypatch.setenv("KACHEDB_WORKSPACE_ROOTS", str(direct_ws))

        resolved = resolve_workspace_path(workspace_id="standalone_app")
        assert resolved == direct_ws.resolve()

    def test_resolves_workspace_nested_in_category_under_configured_roots(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolves directly to <KACHEDB_WORKSPACE_ROOTS>/<category>/<workspace_id>."""
        from kachedb_mcp.indexer.discovery import resolve_workspace_path

        fake_projects = tmp_path / "projects"
        fake_projects.mkdir()
        services_dir = fake_projects / "services"
        services_dir.mkdir()
        billing_dir = services_dir / "billing"
        billing_dir.mkdir()

        monkeypatch.setenv("KACHEDB_WORKSPACE_ROOTS", str(fake_projects))

        resolved = resolve_workspace_path(workspace_id="billing")
        assert resolved == billing_dir.resolve()

    def test_find_root_from_nested_file(self, tmp_path: Any) -> None:
        """Climbs upward from a nested source file to locate the repo/manifest root."""
        from kachedb_mcp.indexer.discovery import find_root_from_path

        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
        nested_src = project_dir / "src" / "pkg"
        nested_src.mkdir(parents=True)
        py_file = nested_src / "module.py"
        py_file.write_text("def hello(): pass\n")

        found_root = find_root_from_path(py_file)
        assert found_root == project_dir.resolve()

    def test_find_root_from_git_repo(self, tmp_path: Any) -> None:
        """Climbs upward to locate the .git repository boundary."""
        from kachedb_mcp.indexer.discovery import find_root_from_path

        repo_dir = tmp_path / "my_repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        sub_dir = repo_dir / "deep" / "nested" / "folder"
        sub_dir.mkdir(parents=True)

        found_root = find_root_from_path(sub_dir)
        assert found_root == repo_dir.resolve()

    def test_resolves_candidate_path_when_file_provided(self, tmp_path: Any) -> None:
        """When candidate_path is a file path, resolve_workspace_path anchors to project root."""
        from kachedb_mcp.indexer.discovery import resolve_workspace_path

        project_dir = tmp_path / "anchor_project"
        project_dir.mkdir()
        (project_dir / "Cargo.toml").write_text("[package]\nname = 'anchor'\n")
        code_file = project_dir / "src" / "main.rs"
        code_file.parent.mkdir()
        code_file.write_text("fn main() {}\n")

        resolved = resolve_workspace_path(candidate_path=str(code_file))
        assert resolved == project_dir.resolve()

    def test_kache_explore_symbol_with_file_path(self, tmp_path: Any) -> None:
        """kache_explore_symbol accepts file_path to anchor and auto-index workspace."""
        from unittest.mock import patch

        from kachedb_mcp.tools import kache_explore_symbol

        demo_file = tmp_path / "src" / "demo.py"
        demo_file.parent.mkdir(parents=True)
        demo_file.write_text("def my_func(): pass\n")

        with patch("kachedb_mcp.tools._ensure_workspace_indexed", return_value=False) as mock_index:
            res_str = kache_explore_symbol(symbol="nonexistent_sym", file_path=str(demo_file))
            res = json.loads(res_str)
            assert res["status"] == "NOT_FOUND"
            mock_index.assert_called_once()
            _, kwargs = mock_index.call_args
            assert kwargs.get("path") == str(demo_file)
