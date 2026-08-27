from __future__ import annotations

import sys
from typing import Any

from ._version import __version__

try:
    from mcp.server import MCPServer

    mcp: Any = MCPServer(
        name="kachedb-agent-memory",
        version=__version__,
        description="Sub-millisecond In-Memory LLM & Vector Cache for AI Coding Agents",
    )
except ImportError:
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef,attr-defined]

    mcp = FastMCP(
        "kachedb-agent-memory",
        description="Sub-millisecond In-Memory LLM & Vector Cache for AI Coding Agents",
    )
from .tools import (
    kache_delete,
    kache_get,
    kache_save_context,
    kache_semantic_search,
    kache_set,
    kache_stats,
    kache_telemetry,
)

# Register all 7 tools
mcp.tool()(kache_get)
mcp.tool()(kache_set)
mcp.tool()(kache_save_context)
mcp.tool()(kache_semantic_search)
mcp.tool()(kache_delete)
mcp.tool()(kache_stats)
mcp.tool()(kache_telemetry)


def main() -> None:
    """CLI entrypoint for kachedb-mcp server."""
    if "--version" in sys.argv:
        print(f"kachedb-mcp v{__version__}")
        sys.exit(0)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
