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


settings = Settings()
