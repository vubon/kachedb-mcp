"""
Configuration settings for KacheDB MCP Server.

Loads parameters from environment variables with safe, sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """KacheDB MCP Server configuration."""

    host: str = os.getenv("KACHEDB_HOST", "127.0.0.1")
    port: int = int(os.getenv("KACHEDB_PORT", "6379"))
    index_name: str = os.getenv("KACHEDB_INDEX", "agent_semantic_memory")
    similarity_threshold: float = float(os.getenv("KACHEDB_THRESHOLD", "0.80"))
    default_ttl_seconds: int = int(os.getenv("KACHEDB_DEFAULT_TTL", "86400"))
    embedder_provider: str = os.getenv("KACHEDB_EMBEDDER", "auto")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


settings = Settings()
