"""Tests for `kachedb-mcp install` CLI, AI-DLC configurator, and IDE MCP merger."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from kachedb_mcp import server
from kachedb_mcp.installer import (
    configure_aidlc,
    configure_ide,
    detect_frameworks,
    get_adapter,
    run_install_cli,
)
from kachedb_mcp.installer.engine import upsert_marked_block
from kachedb_mcp.installer.frameworks.aidlc import _patch_toml_instructions
from kachedb_mcp.installer.ide import resolve_model_path

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python 3.10 compatibility
    tomllib = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from pathlib import Path


def _create_mock_gemma_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.onnx").write_bytes(b"onnx")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def _init_mock_aidlc_workspace(root: Path, space: str = "default") -> None:
    memory_dir = root / "aidlc" / "spaces" / space / "memory"
    phases_dir = memory_dir / "phases"
    phases_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "team.md").write_text(
        "# Team\n\n## Way of Working\n- Existing\n\n"
        "## Forbidden\n- NEVER user rule\n\n"
        "## Mandated\n- ALWAYS user rule\n\n"
        "## Corrections\n",
        encoding="utf-8",
    )
    (memory_dir / "project.md").write_text(
        "# Project\n\n## Forbidden\n- NEVER custom\n\n"
        "## Mandated\n- ALWAYS custom\n\n"
        "## Corrections\n",
        encoding="utf-8",
    )
    (phases_dir / "construction.md").write_text(
        "# Construction\n\n## Corrections\n", encoding="utf-8"
    )
    (phases_dir / "inception.md").write_text("# Inception\n\n## Corrections\n", encoding="utf-8")


def test_upsert_marked_block_clean_and_idempotent() -> None:
    original = (
        "# Rules\n\nNote on ## Corrections History\n\n"
        "## Corrections History\nIgnore\n\n## Corrections\nKeep this\n"
    )
    first = upsert_marked_block(
        original,
        "### KacheDB\n- Rule 1",
        "<!-- BEGIN -->",
        "<!-- END -->",
        insert_before="## Corrections",
    )
    # R-06: Must match exact `## Corrections` line, not `## Corrections History`
    assert first.index("## Corrections History") < first.index("<!-- BEGIN -->")
    assert first.index("<!-- END -->") < first.index("\n## Corrections\n")
    assert "Keep this" in first

    second = upsert_marked_block(
        first,
        "### KacheDB\n- Rule 1 updated",
        "<!-- BEGIN -->",
        "<!-- END -->",
        insert_before="## Corrections",
    )
    assert second.count("<!-- BEGIN -->") == 1
    assert second.count("<!-- END -->") == 1
    assert "Rule 1 updated" in second


def test_upsert_marked_block_rejects_orphaned_markers() -> None:
    with pytest.raises(ValueError, match="Orphaned begin marker"):
        upsert_marked_block("hello\n<!-- BEGIN -->\n", "body", "<!-- BEGIN -->", "<!-- END -->")
    with pytest.raises(ValueError, match="Orphaned end marker"):
        upsert_marked_block("hello\n<!-- END -->\n", "body", "<!-- BEGIN -->", "<!-- END -->")


def test_configure_aidlc_dry_run_writes_nothing(tmp_path: Path) -> None:
    _init_mock_aidlc_workspace(tmp_path)
    plan = configure_aidlc(tmp_path, dry_run=True, affirmed_date="2026-11-17")

    assert any(a.status == "created" for a in plan.actions)
    assert not (tmp_path / "aidlc.settings.json").exists()
    assert not (tmp_path / ".agents" / "rules" / "kachedb_memory.md").exists()
    assert "DRY RUN" in plan.format_summary(dry_run=True)


def test_configure_aidlc_section_placement_and_idempotency(tmp_path: Path) -> None:
    _init_mock_aidlc_workspace(tmp_path)
    (tmp_path / "aidlc.settings.json").write_text(
        json.dumps({"schemaVersion": 1, "customFlag": True}),
        encoding="utf-8",
    )

    first = configure_aidlc(tmp_path, affirmed_date="2026-11-17", yes=True)
    assert any(a.status in ("created", "updated") for a in first.actions)
    assert not first.warnings

    # Settings without preset/defaultScope default to balanced and keep customFlag
    settings = json.loads((tmp_path / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert settings["customFlag"] is True
    assert settings["models"]["preset"] == "minimal"
    assert settings["flags"]["defaultScope"] == "classic"

    # R-03: Verify `- NEVER` strictly lives inside ## Forbidden, `- ALWAYS` inside ## Mandated
    for filename in ("team.md", "project.md"):
        text = (tmp_path / "aidlc" / "spaces" / "default" / "memory" / filename).read_text(
            encoding="utf-8"
        )
        forbidden_slice = text[text.index("## Forbidden") : text.index("## Mandated")]
        mandated_slice = text[text.index("## Mandated") : text.index("## Corrections")]
        assert "- NEVER" in forbidden_slice
        assert "- ALWAYS" not in forbidden_slice
        assert "- ALWAYS" in mandated_slice
        assert "- NEVER" not in mandated_slice
        assert "(affirmed 2026-11-17)" in forbidden_slice
        assert "(affirmed 2026-11-17)" in mandated_slice

    shared = tmp_path / "aidlc/spaces/default/knowledge/aidlc-shared/kachedb-bounded-retrieval.md"
    assert shared.is_file()
    assert "SwissTable" in shared.read_text(encoding="utf-8")

    # Second run must be 100% unchanged
    second = configure_aidlc(tmp_path, affirmed_date="2026-11-17", yes=True)
    assert all(a.status == "unchanged" for a in second.actions)


def test_configure_aidlc_patch_agents_md_and_toml(tmp_path: Path) -> None:
    _init_mock_aidlc_workspace(tmp_path)
    agents_dir = tmp_path / ".codex" / "agents"
    agents_dir.mkdir(parents=True)

    (agents_dir / "aidlc-developer-agent.md").write_text(
        "---\nname: aidlc-developer-agent\n---\n# Developer\n",
        encoding="utf-8",
    )
    (agents_dir / "aidlc-developer-agent.toml").write_text(
        'name = "aidlc-developer-agent"\n'
        'model_reasoning_effort = "medium"\n'
        'developer_instructions = """\n# Developer\n\nExisting rules.\n"""\n',
        encoding="utf-8",
    )

    plan = configure_aidlc(tmp_path, patch_agents=True, affirmed_date="2026-11-17", yes=True)
    assert not plan.warnings

    md_text = (agents_dir / "aidlc-developer-agent.md").read_text(encoding="utf-8")
    toml_text = (agents_dir / "aidlc-developer-agent.toml").read_text(encoding="utf-8")
    assert "KacheDB Bounded Memory Directives" in md_text
    assert "KacheDB Bounded Memory Directives" in toml_text
    if tomllib is not None:
        parsed_toml = tomllib.loads(toml_text)
        assert parsed_toml["model_reasoning_effort"] == "medium"
        assert "KacheDB Bounded Memory Directives" in parsed_toml["developer_instructions"]

    # R-05: Also verify flexible spacing + ''' literal multiline strings
    patched_literal = _patch_toml_instructions(
        "developer_instructions='''\nHello\n'''\n",
        "## KacheDB\n- Keep short",
    )
    assert patched_literal is not None
    assert "Keep short" in patched_literal
    if tomllib is not None:
        assert "Keep short" in tomllib.loads(patched_literal)["developer_instructions"]

    # Re-run must be idempotent
    second = configure_aidlc(tmp_path, patch_agents=True, affirmed_date="2026-11-17", yes=True)
    assert all(a.status == "unchanged" for a in second.actions)


def test_configure_aidlc_uninitialized_warns_cleanly(tmp_path: Path) -> None:
    plan = configure_aidlc(tmp_path, yes=True)
    assert len(plan.warnings) == 1
    assert "aidlc config" in plan.warnings[0]
    assert (tmp_path / "aidlc.settings.json").is_file()
    assert (tmp_path / ".agents" / "rules" / "kachedb_memory.md").is_file()


def test_resolve_model_path_4_step_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.delenv("KACHEDB_MODEL_PATH", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)

    assert resolve_model_path(None, workspace, home_dir=fake_home) is None

    global_model = _create_mock_gemma_dir(fake_home / ".kachedb" / "models" / "embeddinggemma-2")
    assert resolve_model_path(None, workspace, home_dir=fake_home) == global_model.resolve()

    ws_model = _create_mock_gemma_dir(workspace / "models" / "embeddinggemma-2")
    assert resolve_model_path(None, workspace, home_dir=fake_home) == ws_model.resolve()

    env_model = _create_mock_gemma_dir(tmp_path / "env-gemma")
    assert (
        resolve_model_path(
            None,
            workspace,
            existing_env={"KACHEDB_MODEL_PATH": str(env_model)},
            home_dir=fake_home,
        )
        == env_model.resolve()
    )

    explicit_model = _create_mock_gemma_dir(tmp_path / "explicit-gemma")
    assert (
        resolve_model_path(
            explicit_model,
            workspace,
            existing_env={"KACHEDB_MODEL_PATH": str(env_model)},
            home_dir=fake_home,
        )
        == explicit_model.resolve()
    )

    bad_dir = tmp_path / "empty-dir"
    bad_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Invalid --model-path"):
        resolve_model_path(bad_dir, workspace, home_dir=fake_home)


def test_configure_ide_gemma_fallback_and_explicit_fastembed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KACHEDB_MODEL_PATH", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    ws = tmp_path / "repo"
    ws.mkdir()

    (ws / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"github": {"command": "npx", "args": ["-y", "gh-mcp"]}}}),
        encoding="utf-8",
    )

    plan_fallback = configure_ide(target="claude", workspace_root=ws, home_dir=fake_home)
    assert plan_fallback.actions[0].status == "updated"
    claude_data = json.loads((ws / ".mcp.json").read_text(encoding="utf-8"))
    assert "github" in claude_data["mcpServers"]
    kache_entry: dict[str, Any] = claude_data["mcpServers"]["kachedb"]
    assert kache_entry["args"] == ["--refresh", "kachedb-mcp"]
    assert kache_entry["env"]["KACHEDB_EMBEDDER"] == "auto"
    assert "KACHEDB_MODEL_PATH" not in kache_entry["env"]

    gemma_dir = _create_mock_gemma_dir(ws / "models" / "embeddinggemma-2")
    plan_gemma = configure_ide(target="claude", workspace_root=ws, home_dir=fake_home)
    assert plan_gemma.actions[0].status == "updated"
    claude_gemma = json.loads((ws / ".mcp.json").read_text(encoding="utf-8"))
    kache_gemma = claude_gemma["mcpServers"]["kachedb"]
    assert kache_gemma["args"] == ["--from", "kachedb-mcp[gemma]", "--refresh", "kachedb-mcp"]
    assert kache_gemma["env"]["KACHEDB_EMBEDDER"] == "embeddinggemma-2"
    assert kache_gemma["env"]["KACHEDB_MODEL_PATH"] == str(gemma_dir.resolve())

    # R-04: Explicit --embedder fastembed must NOT be overridden even when Gemma weights exist
    configure_ide(target="claude", workspace_root=ws, embedder="fastembed", home_dir=fake_home)
    claude_fast = json.loads((ws / ".mcp.json").read_text(encoding="utf-8"))
    assert claude_fast["mcpServers"]["kachedb"]["args"] == ["--refresh", "kachedb-mcp"]
    assert claude_fast["mcpServers"]["kachedb"]["env"]["KACHEDB_EMBEDDER"] == "fastembed"
    assert "KACHEDB_MODEL_PATH" not in claude_fast["mcpServers"]["kachedb"]["env"]


def test_configure_codex_preserves_env_and_replaces_unmarked_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KACHEDB_MODEL_PATH", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    ws = tmp_path / "repo"
    (ws / ".codex").mkdir(parents=True)

    custom_gemma = _create_mock_gemma_dir(tmp_path / "custom-gemma")
    # R-01 + R-02: Pre-existing unmarked [mcp_servers.kachedb] with custom KACHEDB_MODEL_PATH
    (ws / ".codex" / "config.toml").write_text(
        'model = "gpt-5"\n\n'
        '[mcp_servers.kachedb]\ncommand = "old"\n\n'
        "[mcp_servers.kachedb.env]\n"
        f'KACHEDB_MODEL_PATH = "{custom_gemma}"\n\n'
        '[mcp_servers.github]\ncommand = "gh"\n',
        encoding="utf-8",
    )

    configure_ide(target="codex", workspace_root=ws, home_dir=fake_home)
    toml_content = (ws / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert toml_content.count("[mcp_servers.kachedb]") == 1
    assert 'command = "uvx"' in toml_content
    assert str(custom_gemma.resolve()) in toml_content
    if tomllib is not None:
        parsed = tomllib.loads(toml_content)
        assert parsed["model"] == "gpt-5"
        assert parsed["mcp_servers"]["github"]["command"] == "gh"
        assert parsed["mcp_servers"]["kachedb"]["command"] == "uvx"
        assert parsed["mcp_servers"]["kachedb"]["env"]["KACHEDB_MODEL_PATH"] == str(
            custom_gemma.resolve()
        )

    # Re-run without --model-path preserves custom_gemma and is idempotent
    second = configure_ide(target="codex", workspace_root=ws, home_dir=fake_home)
    assert second.actions[0].status == "unchanged"


def test_configure_ide_all_skip_and_idempotency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KACHEDB_MODEL_PATH", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    fake_home = tmp_path / "home"
    (fake_home / ".gemini" / "antigravity").mkdir(parents=True)
    ws = tmp_path / "repo"
    (ws / ".codex").mkdir(parents=True)
    (ws / ".codex" / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")

    plan = configure_ide(target="all", workspace_root=ws, home_dir=fake_home)
    statuses = {a.path.name: a.status for a in plan.actions}
    assert statuses["mcp_config.json"] == "created"
    assert statuses[".mcp.json"] == "created"
    assert statuses["mcp.json"] == "skipped"
    assert statuses["config.toml"] == "updated"

    second = configure_ide(target="all", workspace_root=ws, home_dir=fake_home)
    for action in second.actions:
        assert action.status in ("unchanged", "skipped")


def test_cli_entrypoint_routing_and_oserror_handling(tmp_path: Path) -> None:
    assert run_install_cli([]) == 2
    assert run_install_cli(["--aidlc", "--path", str(tmp_path), "--dry-run"]) == 0
    bad_args = ["--ide", "claude", "--path", str(tmp_path), "--model-path", str(tmp_path)]
    assert run_install_cli(bad_args) == 1

    # R-07: OSError handled gracefully without uncaught traceback
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    assert run_install_cli(["--aidlc", "--yes", "--path", str(not_a_dir)]) == 1

    with (
        patch.object(sys, "argv", ["kachedb-mcp", "--help"]),
        patch.object(server.mcp, "run") as mock_help_run,
        pytest.raises(SystemExit) as help_exc,
    ):
        server.main()

    assert help_exc.value.code == 0
    mock_help_run.assert_not_called()

    with (
        patch.object(
            sys,
            "argv",
            ["kachedb-mcp", "install", "--aidlc", "--yes", "--path", str(tmp_path)],
        ),
        patch.object(server.mcp, "run") as mock_mcp_run,
        pytest.raises(SystemExit) as exc_info,
    ):
        server.main()

    assert exc_info.value.code == 0
    mock_mcp_run.assert_not_called()
    assert (tmp_path / "aidlc.settings.json").is_file()


def test_configure_aidlc_profiles_and_preserve(tmp_path: Path) -> None:
    _init_mock_aidlc_workspace(tmp_path)

    # Explicit minimal -> express + minimal
    run_install_cli(["--aidlc", "--profile", "minimal", "--path", str(tmp_path)])
    s_min = json.loads((tmp_path / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert s_min["models"]["preset"] == "minimal"
    assert s_min["flags"]["defaultScope"] == "express"

    # Explicit balanced -> classic + minimal
    run_install_cli(["--aidlc", "--profile", "balanced", "--path", str(tmp_path)])
    s_bal = json.loads((tmp_path / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert s_bal["models"]["preset"] == "minimal"
    assert s_bal["flags"]["defaultScope"] == "classic"

    # Explicit full -> feature + standard
    run_install_cli(["--framework", "aidlc", "--profile", "full", "--path", str(tmp_path)])
    s_full = json.loads((tmp_path / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert s_full["models"]["preset"] == "standard"
    assert s_full["flags"]["defaultScope"] == "feature"

    # Explicit preserve keeps existing settings + custom keys + exact formatting byte-identical
    raw_custom = '{\n  "schemaVersion": 1,\n  "customKey":   "keep-spacing"\n}\n'
    (tmp_path / "aidlc.settings.json").write_text(raw_custom, encoding="utf-8")
    assert run_install_cli(["--aidlc", "--profile", "preserve", "--path", str(tmp_path)]) == 0
    assert (tmp_path / "aidlc.settings.json").read_text(encoding="utf-8") == raw_custom

    # Re-running without --profile preserves existing `feature` + `standard`
    (tmp_path / "aidlc.settings.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "models": {"preset": "standard"},
                "flags": {"defaultScope": "feature"},
            }
        ),
        encoding="utf-8",
    )
    run_install_cli(["--aidlc", "--yes", "--path", str(tmp_path)])
    s_pres = json.loads((tmp_path / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert s_pres["models"]["preset"] == "standard"
    assert s_pres["flags"]["defaultScope"] == "feature"


def test_aidlc_tty_interactive_profile_prompt_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _init_mock_aidlc_workspace(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Interactive selection "3" -> full
    monkeypatch.setattr("builtins.input", lambda: "3")
    assert run_install_cli(["--framework", "aidlc", "--path", str(tmp_path)]) == 0
    settings = json.loads((tmp_path / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert settings["models"]["preset"] == "standard"
    assert settings["flags"]["defaultScope"] == "feature"

    # EOFError on unconfigured repo falls back to balanced; menu + prompt go to stderr
    (tmp_path / "aidlc.settings.json").unlink()
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(EOFError()))
    assert run_install_cli(["--framework", "aidlc", "--path", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "Choice [1-4, default=1]:" in captured.err
    eof_cfg = json.loads((tmp_path / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert eof_cfg["flags"]["defaultScope"] == "classic"

    # Unknown choice warns to stderr and falls back to balanced
    (tmp_path / "aidlc.settings.json").unlink()
    monkeypatch.setattr("builtins.input", lambda: "bogus")
    assert run_install_cli(["--framework", "aidlc", "--path", str(tmp_path)]) == 0
    assert "Unknown choice 'bogus'" in capsys.readouterr().err

    # KeyboardInterrupt exits cleanly with code 1
    (tmp_path / "aidlc.settings.json").unlink()
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert run_install_cli(["--framework", "aidlc", "--path", str(tmp_path)]) == 1

    # Invalid JSON in aidlc.settings.json fails BEFORE prompting on TTY
    (tmp_path / "aidlc.settings.json").write_text("{broken json", encoding="utf-8")
    prompt_called = False

    def _fail_if_prompted() -> str:
        nonlocal prompt_called
        prompt_called = True
        return "1"

    monkeypatch.setattr("builtins.input", _fail_if_prompted)
    assert run_install_cli(["--framework", "aidlc", "--path", str(tmp_path)]) == 1
    assert not prompt_called


def test_framework_adapter_registry_and_auto_detection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty_ws = tmp_path / "empty"
    empty_ws.mkdir()
    assert detect_frameworks(empty_ws) == []

    # `--framework auto` on empty dir warns about skipped flags and still creates workspace rules
    rc = run_install_cli(
        [
            "--framework",
            "auto",
            "--patch-agents",
            "--profile",
            "minimal",
            "--yes",
            "--path",
            str(empty_ws),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "--profile minimal, --patch-agents skipped" in out
    assert (empty_ws / ".agents" / "rules" / "kachedb_memory.md").is_file()
    assert not (empty_ws / "aidlc.settings.json").exists()

    # Initialize AI-DLC structure -> detect_frameworks finds AidlcAdapter
    _init_mock_aidlc_workspace(empty_ws)
    detected = detect_frameworks(empty_ws)
    assert [a.name for a in detected] == ["aidlc"]

    # `--aidlc --framework auto` deduplicates AidlcAdapter cleanly
    dedup_rc = run_install_cli(["--aidlc", "--framework", "auto", "--yes", "--path", str(empty_ws)])
    assert dedup_rc == 0
    dedup_out = capsys.readouterr().out
    assert dedup_out.count("aidlc.settings.json") == 1
    dedup_cfg = json.loads((empty_ws / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert dedup_cfg["flags"]["defaultScope"] == "classic"

    # `--framework all` runs cleanly
    all_rc = run_install_cli(
        ["--framework", "all", "--profile", "minimal", "--path", str(empty_ws)]
    )
    assert all_rc == 0
    all_cfg = json.loads((empty_ws / "aidlc.settings.json").read_text(encoding="utf-8"))
    assert all_cfg["flags"]["defaultScope"] == "express"

    with pytest.raises(ValueError, match="Unsupported framework"):
        get_adapter("nonexistent")
