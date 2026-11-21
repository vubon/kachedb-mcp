# Changelog

All notable changes to **`kachedb-mcp`** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] — 2026-11-21

### Added
- **Source-Verified, Worktree-Scoped Agent Memory:**
  - Added optional `worktree_path`, `sources`, and `require_verified` parameters across `kache_index_workspace`, `kache_explore_symbol`, `kache_workspace_status`, `kache_save_context`, `kache_semantic_search`, `kache_get_parent_document`, and `kache_delete` while preserving 100% backwards compatibility for unscoped calls.
  - **Git Worktree Observation & TOCTOU-Safe Source Reads (`indexer/worktree.py`):**
    - Resolves canonical Git worktrees (including linked worktrees sharing a commit), enforces `KACHEDB_WORKSPACE_ROOTS`, and returns explicit `WORKTREE_NOT_GIT` / `WORKTREE_MISMATCH` statuses.
    - Reconciles committed diffs, staged edits, unstaged edits, collapsed untracked directories, NUL-delimited renames, deletions, and reverted dirty files (`indexed_dirty_files`) incrementally.
    - Enforces POSIX `dir_fd` + `O_NOFOLLOW` + before/after `fstat`/`lstat` TOCTOU checks and a pre-open case-insensitive credential filename denylist (`.env`, `.env.*`, `*.env`, `*.pem`, `*.key`, `*credential*`) returning `SOURCE_PATH_BLOCKED`.
  - **WAL SQLite Publication Catalog (`indexer/catalog.py`):**
    - Local per-worktree SQLite catalog (`~/.local/state/kachedb-mcp` / `%LOCALAPPDATA%\kachedb-mcp`, configurable via `KACHEDB_MCP_STATE_DIR`) with `WAL` mode, transactional schema initialization & upgrades (`PRAGMA user_version = 1`), generation-fenced publication, and crash-safe garbage collection of retired/abandoned records.
  - **Worktree-Scoped Exact Symbol Refresh (`indexer/scoped_index.py`, `parser.py`):**
    - Preserves same-file and same-line duplicate symbol definitions via revision-local ordinals.
    - Quarantines per-file Tree-Sitter syntax errors (`parse_failed=1`, retiring stale definitions and reporting `unparsed_file_count`/`unparsed_files`) without blocking valid files.
    - Uses a pooled `KacheClient` and KacheDB `v0.5.0` atomic two-phase `MSET` batches within `2 MiB` arg / `4 MiB` batch bounds (with automatic per-key `SET` fallback on older daemons).
  - **Source-Backed Context & Vector Retrieval (`indexer/scoped_memory.py`, `telemetry.py`):**
    - Verifies declared source files via SHA-256 on save and lookup, surfacing `verified`, `stale`, or `unverified` freshness with affected paths and reason codes.
    - Isolates worktree prompt aliases and flat vector profiles per embedding model/dimension, deduplicates `resolve_parent=True` chunk hits with per-lookup parent caching, relaxes similarity thresholds before dropping explicit `tags` (`tag_filter_relaxed`), and records scoped outcomes in `telemetry.py`.
  - Added 19 real-daemon integration tests (`tests/test_source_verified_memory.py`) covering reverts, renames, Unicode paths, WAL concurrency, multi-process publication racing, and MCP stdio round-trips.

---

## [0.5.1] — 2026-11-17

### Fixed
- **AI-DLC Framework Profiles:**
  - Corrected `models.preset` for `--profile full` from `"standard"` to `"thorough"` to match AI-DLC's schema enum (`minimal`, `balanced`, `thorough`).
  - Aligned `--profile balanced` to write `models.preset: "balanced"` (medium writing effort) for 1:1 symmetry with AI-DLC's preset definitions.

---

## [0.5.0] — 2026-11-17

### Added
- **Universal Agent & Workflow Framework Installer (`kachedb-mcp install`):**
  - Added pluggable `FrameworkAdapter` registry (`--framework <aidlc|auto|all>`, keeping `--aidlc` as alias) and user-selectable `--profile <preserve|balanced|minimal|full>` presets (non-destructively preserving existing settings when configured, or prompting on TTY).
  - Added `--ide <target>` with `--dry-run`, `--patch-agents`, `--space`, `--yes`, and `--workspace-roots` options without adding latency to default FastMCP stdio startup.
  - Added 4-step `KACHEDB_MODEL_PATH` discovery (`--model-path`, existing env/config, workspace/`~/.kachedb`/`~/.cache` scan, zero-config `auto` FastEmbed fallback) across Antigravity, Claude Code (`.mcp.json`), Cursor (`.cursor/mcp.json`), and Codex (`.codex/config.toml`).
  - Added section-aware marker-bounded Markdown rule injection for AWS AI-DLC (`aidlc.settings.json`, `aidlc-shared/kachedb-bounded-retrieval.md`, `team.md`, `project.md`, `phases/*.md`, `.agents/rules/kachedb_memory.md`, and `.codex/agents/*.{md,toml}`).
  - Handled top-level `kachedb-mcp --help` / `-h` in `server.py:main()` before booting stdio transport.

---

## [0.4.1] — 2026-11-15

### Added
- **Google EmbeddingGemma 2 Support:**
  - Added support for local ONNX Runtime `embeddinggemma-2` embedder via `kachedb.semantic.EmbeddingGemmaAdapter`.
  - Added configurable environment settings:
    - `KACHEDB_EMBEDDER=embeddinggemma-2`
    - `KACHEDB_EMBEDDING_DIM`: `384` (default Matryoshka representation slice) or `768` (native).
    - `KACHEDB_MODEL_PATH` / `EMBEDDING_MODEL_PATH`: Directory containing local ONNX weights (`model.onnx` + `tokenizer.json`).
    - `KACHEDB_SEARCH_PREFIX`: Configurable asymmetric search query prefix (defaults to `task: search query | text: `).
    - `KACHEDB_INDEXING_PREFIX`: Configurable asymmetric document indexing prefix (defaults to `task: document indexing | text: `).
- **Asymmetric Semantic Task Routing:**
  - Routed query encoding (`EmbeddingTask.SEARCH_QUERY`) across `kache_semantic_search` and exact query aliases.
  - Routed document indexing (`EmbeddingTask.DOCUMENT_INDEXING`) across `kache_save_context` and `kache_index_workspace`.
- **Extended 8K Context Chunking:**
  - Scaled single-document chunking threshold from 1,500 to 16,000 characters when using Gemma.
  - Increased chunk target capacity to 1,200 tokens (up from 300) to exploit the 8,192-token context window.
- **Dependency & Packaging Updates:**
  - Bumped core SDK dependency floor to `kachedb>=0.4.0`.
  - Added optional extra `[project.optional-dependencies] gemma = ["kachedb[gemma]>=0.4.0", "onnxruntime>=1.18.0", "tokenizers>=0.19.0"]`.

---

## [0.4.0] — 2026-10-03

### Added
- **Codebase AST Intelligence Tools (Sprint 3):**
  - Added `kache_explore_symbol(symbol, workspace_id, file_path)`: Instantaneous (< 50 ns) symbol inspection from SwissTable. Returns signatures, docstrings, line numbers, and file paths in a surgical < 300 token payload with zero file reads.
  - Added `kache_index_workspace(path, workspace_id, force)`: Incremental AST indexer that parses codebases into exact SwissTable keys (`sym:<ws>:<symbol>`, `sym:<ws>:<file>:<symbol>`) and vector records (`vec:<ws>:<file>:<symbol>:<line>`).
  - Added `kache_workspace_status(workspace_id, auto_index)`: Inspects indexing status, total indexed symbols, commit watermark, and cache freshness.
- **Multi-Language Tree-Sitter AST Parsing:**
  - Built-in AST symbol extraction across 5 major languages: **Python** (`.py`), **Rust** (`.rs`), **Go** (`.go`), **TypeScript** (`.ts`, `.tsx`), and **JavaScript** (`.js`, `.jsx`).
  - Extracts classes, structs, traits, interfaces, methods, functions, docstrings, and type signatures with zero external compiler dependencies.
- **Upward Anchor Discovery (`find_root_from_path`):**
  - Added zero-config project root detection from any touched file path, climbing parent directories to locate `.git` or build manifests (`Cargo.toml`, `pyproject.toml`, `package.json`, `go.mod`).
  - Completely eliminates slow, guessing directory walks under `$HOME`.
- **Incremental Git Watermarks (`kache:meta:<ws>:commit`):**
  - Sub-millisecond sync verification using commit SHA watermarks in SwissTable. Up-to-date projects skip AST parsing entirely with zero redundant work.
- **Multi-Project & Monorepo Support (`KACHEDB_WORKSPACE_ROOTS`):**
  - Configurable root paths via `KACHEDB_WORKSPACE_ROOTS` supporting multi-repo collections and monorepos, including category directory nesting.
  - Added background startup auto-discovery daemon (`_background_startup_index`) that scans and indexes configured project roots without blocking MCP client handshake.
- **On-Demand Auto-Indexing from File Anchors:**
  - `kache_explore_symbol` automatically resolves workspace roots and lazily indexes codebases on the fly when passed a `file_path`.

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
