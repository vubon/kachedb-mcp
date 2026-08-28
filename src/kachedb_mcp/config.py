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


settings = Settings()
