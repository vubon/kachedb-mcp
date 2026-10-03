from __future__ import annotations

import sys
from typing import Any

from ._version import __version__

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]

    mcp: Any = FastMCP(
        "kachedb-agent-memory",
        instructions="Sub-millisecond In-Memory LLM & Vector Cache for AI Coding Agents",
    )
except ImportError:
    from mcp.server import MCPServer

    mcp = MCPServer(
        name="kachedb-agent-memory",
        version=__version__,
    )
from .tools import (
    kache_delete,
    kache_explore_symbol,
    kache_get,
    kache_get_parent_document,
    kache_index_workspace,
    kache_save_context,
    kache_semantic_search,
    kache_set,
    kache_stats,
    kache_telemetry,
    kache_workspace_status,
)

# Register all 11 tools (8 original + 3 new codebase indexing tools)
mcp.tool()(kache_get)
mcp.tool()(kache_set)
mcp.tool()(kache_save_context)
mcp.tool()(kache_semantic_search)
mcp.tool()(kache_get_parent_document)
mcp.tool()(kache_delete)
mcp.tool()(kache_stats)
mcp.tool()(kache_telemetry)
# Sprint 3: Codebase Intelligence
mcp.tool()(kache_index_workspace)
mcp.tool()(kache_explore_symbol)
mcp.tool()(kache_workspace_status)


def _background_startup_index() -> None:
    """Launch background daemon thread to verify workspace index on server boot."""
    import threading
    import time
    from pathlib import Path

    from .config import settings

    if not settings.auto_index:
        return

    def _worker() -> None:
        # Brief pause to let stdio transport finish handshake with the IDE first
        time.sleep(1.0)
        try:
            home = Path.home().resolve()
            root = Path("/").resolve()
            users = Path("/Users").resolve()
            forbidden = (home, root, users)

            from .tools import _ensure_workspace_indexed

            # 1. Configured workspace roots: auto-discover and index all projects
            if settings.workspace_roots:
                from .indexer.discovery import WorkspaceDiscovery

                for root_str in settings.workspace_roots:
                    try:
                        root_path = Path(root_str).expanduser().resolve()
                        if root_path.is_dir() and root_path not in forbidden:
                            projects = WorkspaceDiscovery(root_path).discover()
                            for p in projects:
                                _ensure_workspace_indexed(
                                    path=str(p.path), workspace_id=p.workspace_id
                                )
                    except Exception:
                        pass

            # 2. CWD fallback: index CWD if it belongs to a project
            cwd = Path.cwd().resolve()
            if cwd not in forbidden:
                from .indexer.discovery import KNOWN_MANIFESTS, find_root_from_path

                cwd_root = find_root_from_path(cwd)
                if cwd_root is not None and cwd_root not in forbidden:
                    has_git = (cwd_root / ".git").exists()
                    has_manifest = any((cwd_root / m).is_file() for m in KNOWN_MANIFESTS)
                    if has_git or has_manifest:
                        _ensure_workspace_indexed(path=str(cwd_root))
        except Exception:
            pass

    t = threading.Thread(target=_worker, name="kachedb-auto-index-startup", daemon=True)
    t.start()


def main() -> None:
    """CLI entrypoint for kachedb-mcp server."""
    if "--version" in sys.argv:
        print(f"kachedb-mcp v{__version__}")
        sys.exit(0)
    _background_startup_index()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
