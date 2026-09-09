# Changelog

All notable changes to **`kachedb-mcp`** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-09-09

### Added
- **Multi-Workspace Memory Isolation:**
  - Added `workspace_id` parameter to `kache_save_context`, `kache_semantic_search`, and `kache_delete` to provide segregated vector index namespaces (`{index_name}:{workspace_id}`) across different workspaces and projects.
- **Native Telemetry Dogfooding:**
  - Dogfood native KacheDB atomic keys (`kachedb:telemetry:*`) for cumulative tracker metrics, tracking hits, misses, tokens saved, and avoided latency without external SQLite or disk-based JSON persistence.
- **KacheDB Server & SDK v0.1.1 Parity:**
  - Updated dependency to `kachedb>=0.1.1` to leverage latest vector indexing and buffer zero-copy stability.
  - Added `pythonpath = ["src"]` to pytest configuration for reliable local development and test execution.

---

## [0.1.1] — 2026-08-28

### Fixed
- **CI / Type Checking:** Set `warn_unused_ignores = false` in `[tool.mypy]` to prevent false `unused-ignore` errors when optional MCP imports (`mcp.server.MCPServer`, `mcp.server.fastmcp.FastMCP`) are resolvable on the CI runner.

### Added
- **GitHub Actions CI (`ci.yml`):** Lint, format, type-check, and test matrix across Python 3.10 – 3.13 on every push to `main` and PR.
- **GitHub Actions Publish (`publish.yml`):** Trusted Publisher (OIDC) automated release to PyPI on GitHub Release publish.
- **PR Template (`.github/PULL_REQUEST_TEMPLATE.md`):** Testing checklist and MCP tool impact section.
- **Issue Templates:** Bug report (with MCP tool dropdown) and feature request forms.

---

## [0.1.0] — 2026-08-28

### Initial Release
- **Model Context Protocol (MCP) Server for KacheDB:**
  - Implemented FastMCP / MCPServer supporting stdio JSON-RPC transport for Antigravity IDE, Claude Desktop, and Cursor.
- **7 Native MCP Tools:**
  - `kache_semantic_search`: Sub-100µs natural language concept & code retrieval via KacheDB SIMD vector kernel.
  - `kache_save_context`: Save architectural patterns, bug solutions, and file digests with vector embeddings.
  - `kache_get`: Sub-50µs exact key retrieval.
  - `kache_set`: In-memory storage with optional TTL.
  - `kache_delete`: Explicit cache invalidation.
  - `kache_stats`: Real-time connection health and memory footprint.
  - `kache_telemetry`: Cumulative token savings and avoided latency tracking.
- **Pluggable Embedding Providers:** Support for FastEmbed (ONNX Runtime), HuggingFace Transformers, SentenceTransformers, OpenAI, and lightweight MockEmbedder.
- **Packaging:** Added `pyproject.toml` with `kachedb-mcp` CLI command entrypoint.
- **Unit Test Suite:** 10/10 test suite covering mock clients, semantic vector searches, and MCP registration.
