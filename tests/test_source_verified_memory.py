"""Regression tests for the scoped MCP tool contract against a real KacheDB daemon."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from kachedb import KacheClient
from kachedb_mcp import tools
from kachedb_mcp.indexer.catalog import catalog_path
from kachedb_mcp.indexer.scoped_memory import Embeddings
from kachedb_mcp.telemetry import TelemetryTracker

if TYPE_CHECKING:
    from collections.abc import Iterator


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)


def _repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Scoped MCP test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "foo.py").write_text("def target(original):\n    pass\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Initial")
    return root


@pytest.fixture(scope="module")
def daemon_port(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    sibling = Path(__file__).resolve().parents[2] / "kachedb/target/debug/kachedb-server"
    binary = Path(os.environ.get("KACHEDB_TEST_SERVER", str(sibling))).expanduser().resolve()
    if not binary.is_file():
        pytest.skip("Set KACHEDB_TEST_SERVER to run live scoped MCP tests.")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with subprocess.Popen(
        [str(binary), "--bind", "127.0.0.1", "--port", str(port), "--workers", "1", "--no-shm"],
        cwd=tmp_path_factory.mktemp("scoped-daemon"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as daemon:
        try:
            for _ in range(40):
                if daemon.poll() is not None:
                    assert daemon.stdout is not None
                    pytest.fail(daemon.stdout.read().decode())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                pytest.fail("The temporary KacheDB daemon did not start.")
            yield port
        finally:
            if daemon.poll() is None:
                daemon.terminate()
                try:
                    daemon.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    daemon.kill()


@pytest.fixture
def scope(daemon_port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KACHEDB_PORT", str(daemon_port))
    monkeypatch.setenv("KACHEDB_HOST", "127.0.0.1")
    monkeypatch.setenv("KACHEDB_MCP_STATE_DIR", str(tmp_path / "catalog"))
    monkeypatch.setenv("KACHEDB_EMBEDDER", "mock")
    monkeypatch.setenv("KACHEDB_AUTO_INDEX", "1")
    monkeypatch.delenv("KACHEDB_WORKSPACE_ROOTS", raising=False)
    monkeypatch.setattr(tools, "_semantic_cache", None)
    return _repo(tmp_path / "repo")


def _lookup(root: Path, name: str = "target") -> dict[str, Any]:
    result: dict[str, Any] = json.loads(tools.kache_explore_symbol(name, worktree_path=str(root)))
    return result


def test_non_git_scope_does_not_open_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "catalog"
    monkeypatch.setenv("KACHEDB_MCP_STATE_DIR", str(state))
    assert _lookup(tmp_path)["status"] == "WORKTREE_NOT_GIT"
    assert not state.exists()


def test_revert_rename_untracked_and_deleted_symbols(scope: Path) -> None:
    source = scope / "foo.py"
    assert "original" in _lookup(scope)["definitions"][0]["signature"]
    source.write_text("def target(edited):\n    pass\n", encoding="utf-8")
    assert "edited" in _lookup(scope)["definitions"][0]["signature"]
    _git(scope, "checkout", "--", "foo.py")
    reverted = _lookup(scope)
    assert "original" in reverted["definitions"][0]["signature"]
    path = catalog_path(scope, "", "127.0.0.1", tools.settings.port)
    with sqlite3.connect(path) as catalog:
        assert catalog.execute("SELECT indexed_dirty_files FROM state").fetchone()[0] == "[]"

    _git(scope, "mv", "foo.py", "new λ name.py")
    assert _lookup(scope)["definitions"][0]["file"] == "new λ name.py"
    _git(scope, "reset", "--hard", "HEAD")
    assert _lookup(scope)["definitions"][0]["file"] == "foo.py"

    directory = scope / "new [dir]"
    directory.mkdir()
    untracked = directory / "nested π file.py"
    untracked.write_text("def discovered():\n    pass\n", encoding="utf-8")
    assert (
        _lookup(scope, "discovered")["definitions"][0]["file"]
        == f"{directory.name}/{untracked.name}"
    )
    untracked.unlink()
    assert _lookup(scope, "discovered")["status"] == "NOT_FOUND"
    source.unlink()
    assert _lookup(scope)["status"] == "NOT_FOUND"


def test_clean_lookup_reconciles_local_ignore_rules(scope: Path) -> None:
    assert _lookup(scope)["status"] == "OK"
    exclude = scope / ".git/info/exclude"
    exclusions = exclude.read_text(encoding="utf-8")
    exclude.write_text(exclusions + "\nfoo.py\n", encoding="utf-8")
    assert _lookup(scope)["status"] == "NOT_FOUND"
    exclude.write_text(exclusions, encoding="utf-8")
    assert _lookup(scope)["status"] == "OK"


def test_syntax_errors_retire_old_definitions_without_blocking_other_files(scope: Path) -> None:
    broken = scope / "editing.py"
    broken.write_text("def editing():\n    pass\n", encoding="utf-8")
    assert _lookup(scope, "editing")["status"] == "OK"
    broken.write_text("def editing(\n", encoding="utf-8")
    known = _lookup(scope)
    assert known["status"] == "OK" and known["freshness"] == "verified", known
    assert known["unparsed_files"] == ["editing.py"]
    uncertain = _lookup(scope, "editing")
    assert uncertain["status"] == "INCOMPLETE" and not uncertain["definitions"], uncertain
    assert uncertain["reason"] == "unparsed_files"
    status = json.loads(tools.kache_workspace_status(worktree_path=str(scope), auto_index=False))
    assert status["index_state"] == "PARTIAL" and status["unparsed_file_count"] == 1
    broken.write_text("def editing(fixed):\n    pass\n", encoding="utf-8")
    recovered = _lookup(scope, "editing")
    assert "fixed" in recovered["definitions"][0]["signature"]
    assert "unparsed_files" not in recovered


def test_cold_placeholder_and_wal_reader_during_publish(scope: Path) -> None:
    path = catalog_path(scope, "", "127.0.0.1", tools.settings.port)
    path.parent.mkdir()
    path.touch()
    cold = json.loads(tools.kache_workspace_status(worktree_path=str(scope), auto_index=False))
    assert cold["status"] == "INDEX_REQUIRED"
    assert path.stat().st_size == 0
    before = _lookup(scope)
    with sqlite3.connect(path) as reader:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        reader.execute("BEGIN")
        assert reader.execute("SELECT generation FROM state").fetchone()[0] == before["generation"]
        (scope / "foo.py").write_text("def target(updated):\n    pass\n", encoding="utf-8")
        after = _lookup(scope)
        assert after["status"] == "OK" and after["generation"] > before["generation"], after
        assert reader.execute("SELECT generation FROM state").fetchone()[0] == before["generation"]


def test_catalog_upgrade_keeps_namespace_and_existing_rows(scope: Path) -> None:
    path = catalog_path(scope, "", "127.0.0.1", tools.settings.port)
    path.parent.mkdir()
    with sqlite3.connect(path) as previous:
        previous.executescript("""
            CREATE TABLE state (id INTEGER PRIMARY KEY, namespace TEXT NOT NULL,
                generation INTEGER NOT NULL, indexed INTEGER NOT NULL, head TEXT,
                indexed_dirty_files TEXT NOT NULL, ignored_tracked_files TEXT NOT NULL);
            INSERT INTO state VALUES (1, 'old-namespace', 42, 0, NULL, '[]', '[]');
            CREATE TABLE files (path BLOB PRIMARY KEY, digest TEXT, owner TEXT);
            CREATE TABLE documents (topic TEXT PRIMARY KEY, ref TEXT NOT NULL UNIQUE,
                sources TEXT NOT NULL, expires REAL);
            INSERT INTO documents VALUES ('old topic', 'old-ref', '[]', NULL);
        """)
    found = _lookup(scope)
    assert found["status"] == "OK" and found["worktree_id"] == "old-namespace", found
    assert found["generation"] == 43
    with sqlite3.connect(path) as updated:
        assert updated.execute("PRAGMA user_version").fetchone()[0] == 1
        assert updated.execute("SELECT ref FROM documents").fetchone()[0] == "old-ref"


def test_missing_hint_outside_worktree_is_a_mismatch(scope: Path) -> None:
    hint = Path.home() / "kachedb-test-missing-directory" / "file.py"
    result = json.loads(
        tools.kache_explore_symbol(
            "target",
            file_path=str(hint),
            worktree_path=str(scope),
        )
    )
    assert result["status"] == "WORKTREE_MISMATCH", result


def test_scoped_telemetry_counts_outcomes_without_savings_estimates(
    scope: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "tracker", TelemetryTracker())
    before = json.loads(tools.kache_telemetry())
    assert _lookup(scope)["status"] == "OK"
    assert _lookup(scope, "absent")["status"] == "NOT_FOUND"
    saved = json.loads(
        tools.kache_save_context("telemetry topic", "test", worktree_path=str(scope))
    )
    assert saved["status"] == "SAVED"
    stats = json.loads(tools.kache_stats())["telemetry"]
    assert stats["cache_hits"] == before["cache_hits"] + 1
    assert stats["exact_hits"] == before["exact_hits"] + 1
    assert stats["cache_misses"] == before["cache_misses"] + 1
    assert stats["total_writes"] == before["total_writes"] + 1
    assert stats["tokens_saved"] == before["tokens_saved"]


def test_scoped_errors_include_safe_diagnostics(
    scope: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def missing_dependency() -> Embeddings:
        raise ImportError("private diagnostic details")

    monkeypatch.setattr(tools, "_scoped_embeddings", missing_dependency)
    result = tools.kache_save_context("topic", "payload", worktree_path=str(scope))
    assert json.loads(result)["status"] == "DEPENDENCY_ERROR"
    assert json.loads(result)["error_type"] == "ImportError"
    assert "ImportError" in caplog.text
    assert "private diagnostic details" not in result + caplog.text


def test_same_line_duplicates_and_cache_repair(scope: Path, daemon_port: int) -> None:
    (scope / "dupes.rs").write_text(
        "struct A; struct B;\nimpl A { fn new() {} } impl B { fn new() {} }\n", encoding="utf-8"
    )
    before = _lookup(scope, "new")
    assert before["status"] == "OK", before
    assert len(before["definitions"]) == 2
    assert {symbol["line"] for symbol in before["definitions"]} == {2}
    path = catalog_path(scope, "", "127.0.0.1", daemon_port)
    with sqlite3.connect(path) as catalog, KacheClient(port=daemon_port) as client:
        record = catalog.execute("SELECT record FROM definitions WHERE name = 'new'").fetchone()[0]
        client.delete(record)
    after = _lookup(scope, "new")
    assert after["definitions"] == before["definitions"]


def test_source_statuses_and_aliases_are_worktree_scoped(scope: Path) -> None:
    linked = scope.with_name("linked")
    _git(scope, "worktree", "add", "--detach", str(linked))
    (linked / "foo.py").write_text("def target(linked):\n    pass\n", encoding="utf-8")
    digest = hashlib.sha256((scope / "foo.py").read_bytes()).hexdigest()

    def save(root: Path, text: str, **kwargs: Any) -> dict[str, Any]:
        response: dict[str, Any] = json.loads(
            tools.kache_save_context(
                "Scoped topic",
                text,
                prompt_alias="Duplicate alias",
                worktree_path=str(root),
                **kwargs,
            )
        )
        return response

    def exact(root: Path) -> dict[str, Any]:
        response: dict[str, Any] = json.loads(
            tools.kache_semantic_search("Duplicate alias", worktree_path=str(root))
        )
        return response["matches"][0]

    main_ref = save(scope, "Main context", sources=[{"path": "foo.py", "sha256": digest}])
    linked_ref = save(linked, "Linked context", sources=[{"path": "foo.py"}])
    assert main_ref["status"] == linked_ref["status"] == "SAVED"
    assert main_ref["worktree_id"] != linked_ref["worktree_id"]
    assert "original" in _lookup(scope)["definitions"][0]["signature"]
    assert "linked" in _lookup(linked)["definitions"][0]["signature"]
    with KacheClient(port=tools.settings.port) as client:
        assert client.set("query:Duplicate alias", "legacy payload")
    assert exact(scope)["content"] == "Main context"
    assert exact(linked)["content"] == "Linked context"

    main_source = scope / "foo.py"
    main_source.write_text("def target(edited):\n    pass\n", encoding="utf-8")
    assert exact(scope)["freshness"] == "stale"
    assert exact(linked)["freshness"] == "verified"
    parent = main_ref["parent_ref"]
    assert json.loads(tools.kache_get_parent_document(parent, str(scope)))["freshness"] == "stale"
    assert (
        json.loads(tools.kache_get_parent_document(parent, str(scope), True))["status"]
        == "NOT_FOUND"
    )
    assert json.loads(tools.kache_get_parent_document(parent, str(linked)))["status"] == "NOT_FOUND"
    assert (
        save(scope, "Rejected", sources=[{"path": "foo.py", "sha256": digest}])["status"]
        == "SOURCE_CHANGED"
    )
    assert (
        save(scope, "Rejected", sources=[{"path": ".env.local"}])["status"] == "SOURCE_PATH_BLOCKED"
    )
    main_source.unlink()
    assert exact(scope)["reason"] == "source_missing"

    with KacheClient(port=tools.settings.port) as client:
        assert client.get("query:Duplicate alias") == b"legacy payload"
    assert json.loads(tools.kache_delete(parent, worktree_path=str(scope)))["status"] == "DELETED"
    assert exact(linked)["content"] == "Linked context"


def test_semantic_retrieval_and_replacement_reclaim_old_vectors(scope: Path) -> None:
    anchor = {"worktree_path": str(scope)}
    saved = json.loads(
        tools.kache_save_context(
            "Vector context",
            ("# Initial\n\n" + ("Original document. " * 250)),
            prompt_alias="Semantic probe",
            sources=[{"path": "foo.py"}],
            **anchor,
        )
    )
    assert saved["status"] == "SAVED", saved
    semantic = json.loads(
        tools.kache_semantic_search(
            "Semantic probe",
            exact_first=False,
            resolve_parent=True,
            require_verified=True,
            **anchor,
        )
    )
    assert any(
        match["revision"] == saved["revision"] and match["freshness"] == "verified"
        for match in semantic.get("matches", [])
        if match["kind"] == "context"
    ), semantic
    path = catalog_path(scope, "", "127.0.0.1", tools.settings.port)
    with sqlite3.connect(path) as catalog:
        old = catalog.execute(
            "SELECT index_name, record FROM objects WHERE owner = ? AND index_name != ''",
            (saved["revision"],),
        ).fetchall()
    assert len(old) > 1
    replacement = json.loads(
        tools.kache_save_context(
            "Vector context",
            "Replacement",
            prompt_alias="Semantic probe",
            **anchor,
        )
    )
    assert replacement["status"] == "SAVED", replacement
    with KacheClient(port=tools.settings.port) as client:
        assert not any(client.vdel(index, key) for index, key in old)
    exact = json.loads(tools.kache_semantic_search("Semantic probe", **anchor))
    assert exact["matches"][0]["content"] == "Replacement"
    assert exact["matches"][0]["freshness"] == "unverified"


def test_parent_expansion_deduplicates_and_reads_each_parent_once(
    scope: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def vector(_: str) -> list[float]:
        return [1.0, 0.0]

    monkeypatch.setattr(
        tools,
        "_scoped_embeddings",
        lambda: Embeddings(
            "parent-dedup",
            vector,
            vector,
            lambda title, text: [(title, p) for p in text.splitlines()],
        ),
    )
    parent_refs = set()
    for topic, content in (("Multi", "first\nsecond\nthird"), ("Single", "whole")):
        result = json.loads(
            tools.kache_save_context(
                topic,
                content,
                tags=["review"],
                worktree_path=str(scope),
            )
        )
        assert result["status"] == "SAVED", result
        parent_refs.add(result["parent_ref"])

    get = KacheClient.get
    reads = []

    def count_reads(client: KacheClient, key: str | bytes) -> str | bytes | None:
        if isinstance(key, str) and ":document:" in key:
            reads.append(key.rsplit(":", 1)[-1])
        return get(client, key)

    monkeypatch.setattr(KacheClient, "get", count_reads)
    result = json.loads(
        tools.kache_semantic_search(
            "chunks",
            top_k=2,
            tags=["review"],
            resolve_parent=True,
            worktree_path=str(scope),
        )
    )
    assert {match["parent_ref"] for match in result["matches"]} == parent_refs, result
    assert len(reads) == len(parent_refs) and set(reads) == parent_refs


def test_tagged_search_relaxes_threshold_before_tags(
    scope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def embed_document(text: str) -> list[float]:
        return [0.8, 0.6] if text.startswith("Tagged") else [1.0, 0.0]

    monkeypatch.setattr(
        tools,
        "_scoped_embeddings",
        lambda: Embeddings(
            "tag-priority",
            lambda _: [1.0, 0.0],
            embed_document,
            lambda title, text: [(title, text)],
        ),
    )
    anchor = {"worktree_path": str(scope)}
    assert (
        json.loads(tools.kache_save_context("Tagged", "lower score", tags=["review"], **anchor))[
            "status"
        ]
        == "SAVED"
    )
    assert (
        json.loads(tools.kache_save_context("Query", "untagged exact alias", **anchor))["status"]
        == "SAVED"
    )
    matched = json.loads(
        tools.kache_semantic_search("Query", threshold=0.9, tags=["review"], **anchor)
    )
    assert [result["topic"] for result in matched["matches"]] == ["Tagged"], matched
    assert "tag_filter_relaxed" not in matched
    fallback = json.loads(
        tools.kache_semantic_search("Query", threshold=0.9, tags=["absent"], **anchor)
    )
    assert fallback["matches"] and fallback["tag_filter_relaxed"]


def test_read_only_status_and_failed_staging_preserve_watermark(
    scope: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = {"worktree_path": str(scope)}
    initial = json.loads(tools.kache_workspace_status(auto_index=False, **anchor))
    assert initial["status"] == "INDEX_REQUIRED"
    assert not (tmp_path / "catalog").exists()
    ready = json.loads(tools.kache_index_workspace(**anchor))
    assert ready["status"] == "UPDATED"
    (scope / "foo.py").write_text(
        "def target(changed):\n    pass\n\ndef extra():\n    pass\n", encoding="utf-8"
    )
    send = KacheClient.mset

    def lose_ack(client: KacheClient, values: dict[str | bytes, str | bytes]) -> bool:
        assert send(client, values)
        raise OSError("Simulated connection loss after the batch was sent")

    with monkeypatch.context() as patch:
        patch.setattr(KacheClient, "mset", lose_ack)
        failed = json.loads(tools.kache_index_workspace(**anchor))
    assert failed["status"] == "STORAGE_ERROR", failed
    state = json.loads(tools.kache_workspace_status(auto_index=False, **anchor))
    assert state["status"] == "INDEX_REQUIRED"
    assert state["generation"] == ready["generation"]
    assert "changed" in _lookup(scope)["definitions"][0]["signature"]
    assert _lookup(scope, "extra")["status"] == "OK"


def test_code_vector_rebuild_after_eviction_and_line_shift(scope: Path) -> None:
    def search() -> dict[str, Any]:
        result: dict[str, Any] = json.loads(
            tools.kache_semantic_search(
                "target",
                exact_first=False,
                require_verified=True,
                threshold=0.0,
                worktree_path=str(scope),
            )
        )
        return result

    def active_vectors() -> list[tuple[str, str]]:
        path = catalog_path(scope, "", "127.0.0.1", tools.settings.port)
        with sqlite3.connect(path) as catalog:
            return catalog.execute(
                "SELECT index_name, record FROM objects"
                " WHERE state = 'active' AND record LIKE 'c:%'"
            ).fetchall()

    initial = search()
    assert initial["matches"][0]["content"]["line"] == 1, initial
    old = active_vectors()
    with KacheClient(port=tools.settings.port) as client:
        assert all(client.vdel(index, key) for index, key in old)
    assert search()["matches"][0]["freshness"] == "verified"
    recovered = active_vectors()
    assert recovered and recovered != old
    source = scope / "foo.py"
    source.write_text("\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
    assert search()["matches"][0]["content"]["line"] == 2
    with KacheClient(port=tools.settings.port) as client:
        assert not any(client.vdel(index, key) for index, key in recovered)


def test_expired_parent_and_source_path_validation(scope: Path) -> None:
    anchor = {"worktree_path": str(scope)}
    blocked = json.loads(
        tools.kache_save_context(
            "Policy",
            "Content",
            sources=[{"path": "../outside.py"}],
            **anchor,
        )
    )
    assert blocked["status"] == "SOURCE_PATH_BLOCKED"
    if os.name != "nt":
        outside = scope.with_name("external.py")
        outside.write_text("def outside():\n    pass\n", encoding="utf-8")
        (scope / "link.py").symlink_to(outside)
        linked = json.loads(
            tools.kache_save_context(
                "Policy",
                "Content",
                sources=[{"path": "link.py"}],
                **anchor,
            )
        )
        assert linked["status"] == "SOURCE_PATH_BLOCKED"
    saved = json.loads(
        tools.kache_save_context(
            "Short lived",
            "Content",
            ttl_seconds=1,
            sources=[{"path": "foo.py"}],
            **anchor,
        )
    )
    assert saved["status"] == "SAVED", saved
    time.sleep(1.05)
    expired = json.loads(tools.kache_get_parent_document(saved["parent_ref"], **anchor))
    assert expired["status"] == "NOT_FOUND"


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")
def test_two_mcp_processes_publish_one_complete_generation(scope: Path) -> None:
    code = (
        "import sys; from kachedb_mcp.tools import kache_explore_symbol; "
        "print(kache_explore_symbol('target', worktree_path=sys.argv[1]))"
    )
    commands = [sys.executable, "-c", code, str(scope)]
    with (
        subprocess.Popen(commands, stdout=subprocess.PIPE, text=True) as first,
        subprocess.Popen(commands, stdout=subprocess.PIPE, text=True) as second,
    ):
        outputs = [json.loads(proc.communicate(timeout=15)[0]) for proc in (first, second)]
    assert all(output["status"] == "OK" for output in outputs), outputs
    assert outputs[0]["worktree_id"] == outputs[1]["worktree_id"]
    assert outputs[0]["definitions"] == outputs[1]["definitions"]


@pytest.mark.asyncio
async def test_source_verified_context_over_mcp_stdio(scope: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kachedb_mcp.server"],
        cwd=str(scope.parent),
        env={
            "KACHEDB_PORT": os.environ["KACHEDB_PORT"],
            "KACHEDB_MCP_STATE_DIR": os.environ["KACHEDB_MCP_STATE_DIR"],
            "KACHEDB_EMBEDDER": "mock",
            "KACHEDB_AUTO_INDEX": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    async with stdio_client(params) as streams, ClientSession(*streams) as session:
        await session.initialize()
        saved = await session.call_tool(
            "kache_save_context",
            {
                "topic": "Stdio round trip",
                "content": "A source-backed answer.",
                "worktree_path": str(scope),
                "sources": [{"path": "foo.py"}],
            },
        )
        assert not saved.isError and isinstance(saved.content[0], TextContent)
        info = json.loads(saved.content[0].text)
        assert info["status"] == "SAVED", info
        parent = await session.call_tool(
            "kache_get_parent_document",
            {
                "parent_key": info["parent_ref"],
                "worktree_path": str(scope),
                "require_verified": True,
            },
        )
        assert not parent.isError and isinstance(parent.content[0], TextContent)
        assert json.loads(parent.content[0].text)["freshness"] == "verified"
