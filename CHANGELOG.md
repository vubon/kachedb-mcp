# Changelog

All notable changes to **`kachedb-mcp`** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-09-27

### Added
- **Hierarchical Document Chunking & Parent Document Linkage:**
  - Added `kache_get_parent_document(parent_key)` tool for sub-microsecond retrieval of unabridged parent documents.
  - Added `auto_chunk=True` and `parent_key` parameters to `kache_save_context` to automatically chunk large documents (>1,500 chars) with header breadcrumbs linked to their parent KV document.
- **Sub-Microsecond Exact Prompt Aliasing:**
  - Added `prompt_alias` parameter to `kache_save_context` for registering instant SwissTable lookup shortcuts (`exact:<alias>` or `exact:<ws>:<alias>`), resolving repeated prompts in < 50ns with 0ms LLM time.
- **64-bit Bitmask Tag Pre-Filtering:**
  - Added `tags` parameter to `kache_semantic_search` and `kache_save_context` mapped via `kachedb.tags` for zero-allocation integer bitwise AND pre-filtering in vector scans.
- **Adaptive Fallback & Automatic Resolution:**
  - Added `exact_first` and `resolve_parent` flags to `kache_semantic_search` with automatic prompt alias checking, `#prompt_anchor` semantic indexing, and tag/threshold relaxation fallback.
- **Real-Time Telemetry Tracking:**
  - Added `exact_hits` metric and avoided embedding inference time tracking to `kache_telemetry`.

---

## [0.2.1] — 2026-09-26

### Fixed
- **Timing Isolation in Semantic Search:** Moved vector search stopwatch to start *after* embedding model inference (`embedder.encode()`), ensuring reported latency captures true KacheDB SIMD kernel execution time (< 200 µs) rather than embedding runtime.
- **Per-Operation Avoided Latency Baselines:** Replaced flat 400 ms baseline with calibrated per-operation baselines (`semantic_search`: 6,200 ms, `kv_get`: 150 ms), configurable via `KACHEDB_BASELINE_SEMANTIC_MS` and `KACHEDB_BASELINE_KV_MS`.
- **Avoided Source Sizing:** Added `avoided_source_chars` parameter to `record_hit()` to calculate savings based on avoided source file sizes rather than cached summary responses.
- **Configurable Token Cost:** Replaced hardcoded `$5.00/M` USD rate with configurable `KACHEDB_TOKEN_COST_PER_MILLION` setting (defaulting to `$3.00/M` for Claude 3.7 Sonnet).

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
