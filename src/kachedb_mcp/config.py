"""
Configuration settings for KacheDB MCP Server.

Loads parameters from environment variables with safe, sensible defaults.
"""

from __future__ import annotations

import os


class Settings:
    """KacheDB MCP Server configuration."""

    @property
    def host(self) -> str:
        return os.getenv("KACHEDB_HOST", "127.0.0.1")

    @property
    def port(self) -> int:
        return int(os.getenv("KACHEDB_PORT", "6379"))

    @property
    def index_name(self) -> str:
        return os.getenv("KACHEDB_INDEX", "agent_semantic_memory")

    @property
    def similarity_threshold(self) -> float:
        return float(os.getenv("KACHEDB_THRESHOLD", "0.80"))

    @property
    def default_ttl_seconds(self) -> int:
        return int(os.getenv("KACHEDB_DEFAULT_TTL", "86400"))

    @property
    def embedder_provider(self) -> str:
        return os.getenv("KACHEDB_EMBEDDER", "auto")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def token_cost_per_million(self) -> float:
        """Input token cost in USD per 1M tokens.

        Override via ``KACHEDB_TOKEN_COST_PER_MILLION`` to match your model:
        - Claude 3.7 Sonnet:  3.00
        - Claude 3.5 Haiku:   0.80
        - GPT-4o:             2.50
        - GPT-4o mini:        0.15
        - Gemini 2.0 Flash:   0.10
        """
        return float(os.getenv("KACHEDB_TOKEN_COST_PER_MILLION", "3.0"))

    @property
    def auto_index(self) -> bool:
        """Enable autonomous lazy workspace indexing when watermark is missing."""
        val = os.getenv("KACHEDB_AUTO_INDEX", "1").lower().strip()
        return val not in ("0", "false", "no", "off")

    @property
    def workspace_roots(self) -> list[str]:
        """Colon-separated list of root directories to search for workspace projects.

        Set ``KACHEDB_WORKSPACE_ROOTS`` in your MCP config's ``env`` block, e.g.:
            "KACHEDB_WORKSPACE_ROOTS": "/Users/alice/projects:/Users/alice/work"

        When not set, workspace resolution falls back to:
          1. Upward anchor from candidate path or file path (resolving to .git or manifest).
          2. KacheDB SwissTable cached path (from a previous index run).
          3. CWD upward walk (when MCP server is launched from a specific project directory).
        """
        raw = os.getenv("KACHEDB_WORKSPACE_ROOTS", "").strip()
        if not raw:
            return []
        # Support both OS native path separator (colon on Unix, semicolon on Windows) and semicolon
        parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
        return parts


settings = Settings()
