"""IDE and CLI MCP configuration merger with 4-step Gemma model path discovery."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .engine import (
    InstallAction,
    InstallPlan,
    merge_json_file,
    upsert_marked_block,
    write_text_if_changed,
)
from .templates import BEGIN_TOML_MCP_MARKER, END_TOML_MCP_MARKER

if TYPE_CHECKING:
    from collections.abc import Mapping

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]

IDE_CHOICES: tuple[str, ...] = ("antigravity", "claude", "cursor", "codex", "all")

_CODEX_TABLE_HEADER_RE = re.compile(
    r"^[ \t]*\[mcp_servers\.kachedb(?:\.[^\]]+)?\][ \t]*(?:#.*)?$",
    re.MULTILINE,
)
_ANY_TOML_TABLE_RE = re.compile(r"^[ \t]*\[", re.MULTILINE)


def _is_valid_gemma_dir(directory: Path) -> bool:
    """Return True when directory contains both ``model.onnx`` and ``tokenizer.json``."""
    return (
        directory.is_dir()
        and (directory / "model.onnx").is_file()
        and (directory / "tokenizer.json").is_file()
    )


def resolve_model_path(
    explicit_path: str | Path | None,
    workspace_root: Path,
    existing_env: Mapping[str, Any] | None = None,
    home_dir: Path | None = None,
) -> Path | None:
    """Resolve EmbeddingGemma 2 ONNX directory using the 4-step discovery chain.

    1. Explicit ``--model-path`` flag (raises ``FileNotFoundError`` if invalid).
    2. ``KACHEDB_MODEL_PATH`` in ``existing_env`` or process environment.
    3. Standard local directories (workspace ``models/``, ``~/.kachedb/``, ``~/.cache/kachedb/``).
    4. Returns ``None`` to trigger zero-config FastEmbed ``auto`` fallback.
    """
    if explicit_path is not None and str(explicit_path).strip():
        candidate = Path(explicit_path).expanduser().resolve()
        if not _is_valid_gemma_dir(candidate):
            raise FileNotFoundError(
                f"Invalid --model-path '{candidate}': expected 'model.onnx' and 'tokenizer.json'"
            )
        return candidate

    env_candidates: list[str | None] = [
        str(existing_env["KACHEDB_MODEL_PATH"])
        if existing_env and existing_env.get("KACHEDB_MODEL_PATH")
        else None,
        os.environ.get("KACHEDB_MODEL_PATH"),
        os.environ.get("EMBEDDING_MODEL_PATH"),
    ]
    for raw in env_candidates:
        if raw and raw.strip():
            candidate = Path(raw.strip()).expanduser().resolve()
            if _is_valid_gemma_dir(candidate):
                return candidate

    home = (home_dir if home_dir is not None else Path.home()).expanduser().resolve()
    scan_dirs = (
        workspace_root / "models" / "embeddinggemma-2",
        home / ".kachedb" / "models" / "embeddinggemma-2",
        home / ".cache" / "kachedb" / "models" / "embeddinggemma-2",
    )
    for directory in scan_dirs:
        if _is_valid_gemma_dir(directory):
            return directory.resolve()

    return None


def _resolve_for_embedder(
    *,
    embedder: str,
    explicit_model_path: str | Path | None,
    workspace_root: Path,
    existing_env: Mapping[str, Any] | None,
    home_dir: Path | None,
) -> Path | None:
    """Skip Gemma auto-discovery when user explicitly selected ``fastembed``."""
    if embedder == "fastembed" and explicit_model_path is None:
        return None
    if embedder == "fastembed" and explicit_model_path is not None:
        # Validate explicit path if user supplied it, but keep fastembed backend intact.
        resolve_model_path(
            explicit_model_path,
            workspace_root=workspace_root,
            existing_env=existing_env,
            home_dir=home_dir,
        )
        return None
    return resolve_model_path(
        explicit_model_path,
        workspace_root=workspace_root,
        existing_env=existing_env,
        home_dir=home_dir,
    )


def build_mcp_server_entry(
    *,
    host: str,
    port: int,
    workspace_roots: str,
    embedder: str,
    model_path: Path | None,
    dim: int,
) -> dict[str, Any]:
    """Build the `mcpServers.kachedb` configuration dictionary."""
    use_gemma = (embedder == "embeddinggemma-2") or (embedder == "auto" and model_path is not None)
    args = (
        ["--from", "kachedb-mcp[gemma]", "--refresh", "kachedb-mcp"]
        if use_gemma
        else ["--refresh", "kachedb-mcp"]
    )
    env: dict[str, str] = {
        "KACHEDB_HOST": host,
        "KACHEDB_PORT": str(port),
        "KACHEDB_WORKSPACE_ROOTS": workspace_roots,
        "KACHEDB_EMBEDDER": "embeddinggemma-2" if use_gemma else embedder,
        "KACHEDB_EMBEDDING_DIM": str(dim),
    }
    if use_gemma and model_path is not None:
        env["KACHEDB_MODEL_PATH"] = str(model_path)

    return {
        "command": "uvx",
        "args": args,
        "env": env,
    }


def _read_existing_kachedb_env(config_path: Path) -> Mapping[str, Any] | None:
    """Safely extract existing `mcpServers.kachedb.env` if present in a JSON config."""
    if not config_path.is_file():
        return None
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(parsed, dict):
        return None
    servers = parsed.get("mcpServers")
    if not isinstance(servers, dict):
        return None
    kachedb = servers.get("kachedb")
    if not isinstance(kachedb, dict):
        return None
    env = kachedb.get("env")
    return env if isinstance(env, dict) else None


def _read_existing_codex_env(raw_toml: str) -> Mapping[str, Any] | None:
    """Safely extract existing `[mcp_servers.kachedb.env]` from `.codex/config.toml`."""
    if not raw_toml.strip():
        return None
    if tomllib is not None:
        try:
            parsed = tomllib.loads(raw_toml)
        except Exception:
            return None
        mcp_servers = parsed.get("mcp_servers")
        if not isinstance(mcp_servers, dict):
            return None
        kachedb = mcp_servers.get("kachedb")
        if not isinstance(kachedb, dict):
            return None
        env = kachedb.get("env")
        return env if isinstance(env, dict) else None

    # Python 3.10 minimal fallback scoped strictly to `[mcp_servers.kachedb.env]`
    env_header = re.search(
        r"^[ \t]*\[mcp_servers\.kachedb\.env\][ \t]*(?:#.*)?$",
        raw_toml,
        re.MULTILINE,
    )
    if env_header is None:
        return None
    next_table = _ANY_TOML_TABLE_RE.search(raw_toml, env_header.end())
    section = raw_toml[env_header.end() : next_table.start() if next_table else len(raw_toml)]
    match = re.search(r'^\s*KACHEDB_MODEL_PATH\s*=\s*"([^"]+)"', section, re.MULTILINE)
    return {"KACHEDB_MODEL_PATH": match.group(1)} if match else None


def _strip_unmarked_codex_tables(raw_toml: str) -> str:
    """Remove pre-existing unmarked `[mcp_servers.kachedb*]` TOML tables cleanly."""
    cleaned = raw_toml
    while True:
        match = _CODEX_TABLE_HEADER_RE.search(cleaned)
        if match is None:
            break
        next_table = _ANY_TOML_TABLE_RE.search(cleaned, match.end())
        end_pos = next_table.start() if next_table is not None else len(cleaned)
        before = cleaned[: match.start()].rstrip("\n")
        after = cleaned[end_pos:].lstrip("\n")
        if before and after:
            cleaned = f"{before}\n\n{after}"
        elif before:
            cleaned = f"{before}\n"
        else:
            cleaned = after
    return cleaned


def _render_codex_toml_block(entry: dict[str, Any]) -> str:
    """Render TOML `[mcp_servers.kachedb]` configuration block."""
    args_str = json.dumps(entry["args"])
    lines = [
        "[mcp_servers.kachedb]",
        f'command = "{entry["command"]}"',
        f"args = {args_str}",
        "",
        "[mcp_servers.kachedb.env]",
    ]
    for key, value in entry["env"].items():
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key} = "{escaped}"')
    return "\n".join(lines)


def _configure_json_ide(
    *,
    config_path: Path,
    workspace_root: Path,
    explicit_model_path: str | Path | None,
    host: str,
    port: int,
    workspace_roots: str,
    embedder: str,
    dim: int,
    dry_run: bool,
    home_dir: Path | None,
    detail: str,
) -> InstallAction:
    """Merge `mcpServers.kachedb` into a JSON IDE config file."""
    existing_env = _read_existing_kachedb_env(config_path)
    resolved_model = _resolve_for_embedder(
        embedder=embedder,
        explicit_model_path=explicit_model_path,
        workspace_root=workspace_root,
        existing_env=existing_env,
        home_dir=home_dir,
    )
    entry = build_mcp_server_entry(
        host=host,
        port=port,
        workspace_roots=workspace_roots,
        embedder=embedder,
        model_path=resolved_model,
        dim=dim,
    )

    def _updater(data: dict[str, Any]) -> dict[str, Any]:
        raw_servers = data.get("mcpServers")
        servers = dict(raw_servers) if isinstance(raw_servers, dict) else {}
        servers["kachedb"] = entry
        data["mcpServers"] = servers
        return data

    return merge_json_file(config_path, _updater, dry_run=dry_run, detail=detail)


def _configure_codex_toml(
    *,
    codex_path: Path,
    workspace_root: Path,
    explicit_model_path: str | Path | None,
    host: str,
    port: int,
    workspace_roots: str,
    embedder: str,
    dim: int,
    dry_run: bool,
    home_dir: Path | None,
) -> InstallAction:
    """Merge `[mcp_servers.kachedb]` into `.codex/config.toml` idempotently."""
    existing_toml = codex_path.read_text(encoding="utf-8") if codex_path.is_file() else ""
    existing_env = _read_existing_codex_env(existing_toml)
    resolved_model = _resolve_for_embedder(
        embedder=embedder,
        explicit_model_path=explicit_model_path,
        workspace_root=workspace_root,
        existing_env=existing_env,
        home_dir=home_dir,
    )
    entry = build_mcp_server_entry(
        host=host,
        port=port,
        workspace_roots=workspace_roots,
        embedder=embedder,
        model_path=resolved_model,
        dim=dim,
    )

    has_markers = BEGIN_TOML_MCP_MARKER in existing_toml or END_TOML_MCP_MARKER in existing_toml
    base_toml = existing_toml if has_markers else _strip_unmarked_codex_tables(existing_toml)
    updated_toml = upsert_marked_block(
        base_toml,
        _render_codex_toml_block(entry),
        BEGIN_TOML_MCP_MARKER,
        END_TOML_MCP_MARKER,
    )

    if tomllib is not None:
        try:
            tomllib.loads(updated_toml)
        except Exception as exc:
            raise ValueError(f"Generated invalid Codex TOML at {codex_path}: {exc}") from exc

    return write_text_if_changed(
        codex_path,
        updated_toml,
        dry_run=dry_run,
        detail="Codex CLI MCP config",
    )


def configure_ide(
    *,
    target: str,
    workspace_root: Path,
    host: str = "127.0.0.1",
    port: int = 6380,
    embedder: str = "auto",
    model_path: str | Path | None = None,
    dim: int = 384,
    workspace_roots: str | None = None,
    dry_run: bool = False,
    home_dir: Path | None = None,
) -> InstallPlan:
    """Register `kachedb` MCP server in one or all supported IDE/CLI configs."""
    if target not in IDE_CHOICES:
        raise ValueError(f"Unsupported IDE target '{target}' (expected one of {IDE_CHOICES})")
    if dim not in (384, 768):
        raise ValueError(f"Unsupported --dim {dim} (expected 384 or 768)")

    resolved_root = workspace_root.expanduser().resolve()
    resolved_home = (home_dir if home_dir is not None else Path.home()).expanduser().resolve()
    roots_value = workspace_roots if workspace_roots else str(resolved_root)
    plan = InstallPlan()

    antigravity_path = resolved_home / ".gemini" / "antigravity" / "mcp_config.json"
    claude_path = resolved_root / ".mcp.json"
    cursor_path = resolved_root / ".cursor" / "mcp.json"
    codex_path = resolved_root / ".codex" / "config.toml"

    is_all = target == "all"
    json_targets: list[tuple[str, Path, bool, str]] = [
        (
            "antigravity",
            antigravity_path,
            antigravity_path.parent.is_dir(),
            "Antigravity MCP config",
        ),
        ("claude", claude_path, True, "Claude Code / project .mcp.json"),
        ("cursor", cursor_path, cursor_path.parent.is_dir(), "Cursor MCP config"),
    ]

    for name, cfg_path, detected, detail in json_targets:
        if target != name and not is_all:
            continue
        if is_all and not detected:
            plan.add(
                InstallAction(
                    path=cfg_path,
                    status="skipped",
                    detail=f"{name} directory not detected",
                )
            )
            continue
        plan.add(
            _configure_json_ide(
                config_path=cfg_path,
                workspace_root=resolved_root,
                explicit_model_path=model_path,
                host=host,
                port=port,
                workspace_roots=roots_value,
                embedder=embedder,
                dim=dim,
                dry_run=dry_run,
                home_dir=resolved_home,
                detail=detail,
            )
        )

    if target == "codex" or is_all:
        if is_all and not codex_path.parent.is_dir():
            plan.add(
                InstallAction(
                    path=codex_path,
                    status="skipped",
                    detail=".codex directory not detected",
                )
            )
        else:
            plan.add(
                _configure_codex_toml(
                    codex_path=codex_path,
                    workspace_root=resolved_root,
                    explicit_model_path=model_path,
                    host=host,
                    port=port,
                    workspace_roots=roots_value,
                    embedder=embedder,
                    dim=dim,
                    dry_run=dry_run,
                    home_dir=resolved_home,
                )
            )

    return plan
