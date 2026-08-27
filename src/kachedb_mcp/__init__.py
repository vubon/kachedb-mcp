"""
KacheDB Model Context Protocol (MCP) Server.

Connects Antigravity IDE, Claude Desktop, Cursor, and agentic workflows to KacheDB's
sub-millisecond in-memory cache and SIMD semantic memory.
"""

from ._version import __version__
from .config import settings
from .server import mcp
from .telemetry import tracker
from .tools import (
    kache_delete,
    kache_get,
    kache_save_context,
    kache_semantic_search,
    kache_set,
    kache_stats,
    kache_telemetry,
)

__all__ = [
    "__version__",
    "kache_delete",
    "kache_get",
    "kache_save_context",
    "kache_semantic_search",
    "kache_set",
    "kache_stats",
    "kache_telemetry",
    "mcp",
    "settings",
    "tracker",
]
