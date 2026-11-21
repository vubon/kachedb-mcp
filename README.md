# 🧠 KacheDB MCP Server

[![PyPI](https://img.shields.io/pypi/v/kachedb-mcp.svg)](https://pypi.org/project/kachedb-mcp/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%7C%20MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Languages](https://img.shields.io/badge/languages-Python%20%7C%20Rust%20%7C%20Go%20%7C%20TS%20%7C%20JS-success.svg)](pyproject.toml)

**High-Performance Model Context Protocol (MCP) server for KacheDB** — exposing sub-millisecond in-memory caching, Tree-Sitter AST symbol intelligence, and SIMD semantic vector memory to **Antigravity IDE**, **Claude Desktop**, **Cursor**, and AI coding agents.

---

## ⚡ Why KacheDB for AI Agents?

AI coding assistants repeatedly re-read large codebases, parse ASTs, and crawl directories, wasting tens of thousands of tokens and adding hundreds of milliseconds of latency per turn.

`kachedb-mcp` connects your AI assistant directly to **KacheDB's Megaslab pure-RAM SwissTable**:
- 🚀 **Sub-50 Nanosecond Retrieval:** Instant symbol inspection and exact KV hits in memory.
- 🌳 **Tree-Sitter AST Code Intelligence:** Multi-language symbol graphs (functions, classes, structs, signatures, docstrings) across Python, Rust, Go, TypeScript, and JavaScript.
- 🔍 **Upward Anchor Discovery:** Zero-config project root detection from any touched file — no arbitrary `$HOME` scans or guessing folder names.
- 🔄 **Incremental Git Watermarks:** Sub-millisecond sync verification using commit SHA watermarks (`kache:meta:<ws>:commit`). Up-to-date projects are skipped with zero overhead.
- 🧠 **SIMD Semantic Vector Memory:** Natural language concept and code recall powered by ARM NEON & AVX2/FMA cosine similarity kernels.
- 🪙 **Live Financial Telemetry:** Tracks avoided latency, cumulative tokens saved, and real-time USD cost savings.

---

## 🛠️ MCP Tools Exposed (11 Tools)

### 🌳 Codebase Intelligence Tools
| Tool | Type | Description |
| :--- | :---: | :--- |
| `kache_explore_symbol` | ⚡ *Exact AST* | Instant (< 50 ns) symbol inspection from SwissTable. Returns signatures, docstrings, line numbers, and file paths in a surgical < 300 token payload. |
| `kache_index_workspace` | 🚀 *Indexer* | Discovers projects, parses AST symbols across supported languages, and persists SwissTable keys and vector embeddings. |
| `kache_workspace_status` | 📊 *Status* | Inspects indexing status, total indexed symbols, language breakdown, commit watermark, and cache freshness. |

### 🧠 Semantic Memory Tools
| Tool | Type | Description |
| :--- | :---: | :--- |
| `kache_semantic_search` | 🧠 *Vector* | Natural language semantic search over cached codebases, PR reviews, and past decisions (`top_k`, `threshold`). |
| `kache_save_context` | 🧠 *Vector* | Save architectural patterns, bug solutions, or file digests with hierarchical chunking and SIMD vector embeddings. |
| `kache_get_parent_document`| 📄 *Context* | Retrieve the full unabridged parent document for chunked semantic matches. |

### ⚡ Exact Key-Value Tools
| Tool | Type | Description |
| :--- | :---: | :--- |
| `kache_get` | ⚡ *Exact* | Sub-millisecond exact key retrieval for code chunks, ASTs, and tool outputs. |
| `kache_set` | ⚡ *Exact* | Store string content in memory with optional TTL expiration. |
| `kache_delete` | ⚡ *Exact* | Remove a key or vector from cache and reclaim slab memory. |

### 📊 Telemetry & Health Tools
| Tool | Type | Description |
| :--- | :---: | :--- |
| `kache_stats` | 📊 *Health* | Real-time connection health, active vector counts, and RAM footprint. |
| `kache_telemetry` | 📊 *Telemetry* | Live cumulative tokens saved, latency saved (seconds), hit ratios, and estimated USD saved. |

---

## 🚀 Quickstart

### 1. Start the KacheDB Server Daemon
Ensure your KacheDB daemon is running locally:
```bash
kachedb-server --port 6380
```

### 2. Configure in Antigravity IDE / Claude Desktop / Cursor

Add `kachedb` to your MCP configuration file (`mcp_config.json` or `claude_desktop_config.json`):

#### Zero-Config Mode (Recommended)
No hardcoded paths needed. When an agent opens or mentions any file, Upward Anchor Discovery automatically resolves the enclosing project root and indexes it on-demand:
```json
{
  "mcpServers": {
    "kachedb": {
      "command": "uvx",
      "args": ["kachedb-mcp"],
      "env": {
        "KACHEDB_PORT": "6380"
      }
    }
  }
}
```

#### Pre-Configured Multi-Project Roots
To automatically discover and index projects on startup, set `KACHEDB_WORKSPACE_ROOTS`:
```json
{
  "mcpServers": {
    "kachedb": {
      "command": "uvx",
      "args": ["kachedb-mcp"],
      "env": {
        "KACHEDB_PORT": "6380",
        "KACHEDB_WORKSPACE_ROOTS": "/path/to/projects:/path/to/work"
      }
    }
  }
}
```

#### With Google EmbeddingGemma 2 (8K Context, 384-d MRL)
Run high-performance, zero-PyTorch local embeddings using local ONNX weights:
```json
{
  "mcpServers": {
    "kachedb": {
      "command": "uvx",
      "args": ["--with", "kachedb-mcp[gemma]", "kachedb-mcp"],
      "env": {
        "KACHEDB_PORT": "6380",
        "KACHEDB_EMBEDDER": "embeddinggemma-2",
        "KACHEDB_MODEL_PATH": "/path/to/models/embeddinggemma-2",
        "KACHEDB_EMBEDDING_DIM": "384"
      }
    }
  }
}
```

Or install via Python `pip`:
```bash
pip install "kachedb-mcp[gemma]"
```
```json
{
  "mcpServers": {
    "kachedb": {
      "command": "kachedb-mcp",
      "env": {
        "KACHEDB_PORT": "6380",
        "KACHEDB_EMBEDDER": "embeddinggemma-2",
        "KACHEDB_MODEL_PATH": "/path/to/models/embeddinggemma-2",
        "KACHEDB_EMBEDDING_DIM": "384"
      }
    }
  }
}
```

### 3. One-Command CLI Installer (IDEs & AWS AI-DLC)

Automatically register `kachedb-mcp` across your IDEs or configure **AWS AI-DLC** (`awslabs/aidlc-workflows`) bounded memory rules with one idempotent command:

```bash
# Auto-register MCP server in Antigravity, Claude (.mcp.json), Cursor, and Codex
uvx kachedb-mcp install --ide all

# Auto-detect installed workflow frameworks (preserves existing settings; prompts on fresh repos)
uvx kachedb-mcp install --framework auto --path /path/to/repo

# Non-interactive setup choosing a profile: preserve | balanced | minimal | full
uvx kachedb-mcp install --aidlc --profile balanced --yes --path /path/to/repo
uvx kachedb-mcp install --framework all --profile minimal --yes

# Both framework + IDEs, plus agent prompt directives (.codex/agents/*.{md,toml})
uvx kachedb-mcp install --aidlc --ide all --patch-agents --yes

# Preview changes without touching disk
uvx kachedb-mcp install --aidlc --ide claude --dry-run
```

The `--ide` installer automatically discovers local EmbeddingGemma 2 weights (`--model-path`, `$KACHEDB_MODEL_PATH`, existing MCP config, `<repo>/models/embeddinggemma-2`, `~/.kachedb/models/embeddinggemma-2`) and cleanly falls back to zero-config `auto` mode when no local Gemma directory is present.

---

## Source-verified agent memory

Pass `worktree_path` with the absolute path to a local Git worktree or a file inside it. The scoped tools return JSON text with a `worktree_id` and operation status. Linked worktrees have independent indexes, including when they share a commit.

1. Call `kache_index_workspace` with `worktree_path` to warm the index. `kache_explore_symbol` checks the worktree before returning a definition, including uncommitted changes saved to disk. Each definition includes the repo-relative file path, line range, and `sha256`.
2. To save a decision or summary, call `kache_save_context` with the same `worktree_path` and a `sources` array such as `[{"path":"src/service.py","sha256":"<fingerprint from the source read>"}]`. The digest is optional. A supplied digest must match at save time. Use the returned `parent_ref` with `kache_get_parent_document` in the same scope.
3. Pass `require_verified=true` to scoped semantic search or parent retrieval to reuse only entries whose declared files still match. Read the affected files directly when a match is `stale`, and use the reason code to investigate an `unverified` result. A `verified` result confirms the source bytes observed by MCP. The caller chooses the sources and interprets them.

Keep the same optional `workspace_id` across calls when using a named namespace. Use `kache_workspace_status` with `auto_index=false` to inspect a scope without refreshing it. Files with parser errors are listed in `unparsed_files`. Definitions from other files remain available, and a lookup that may depend on an unparsed file returns `INCOMPLETE` when it finds no match. A cold semantic profile backfills code vectors in bounded batches. On `INCOMPLETE` with `code_index_pending`, retry to advance it or use exact symbol lookup.

Scoped semantic search relaxes the score threshold before relaxing requested tags. When it falls back to unfiltered results, it sets `tag_filter_relaxed`. With `resolve_parent=true`, results include each saved document once. Scoped hits, misses, and writes contribute to cache telemetry. The token and latency savings estimates continue to use the legacy baseline for legacy calls.

The local SQLite catalog records the worktree identity, index state, and references to saved context. Its default location is `$XDG_STATE_HOME/kachedb-mcp` or `~/.local/state/kachedb-mcp` on Unix, and `%LOCALAPPDATA%\kachedb-mcp` on Windows. Set `KACHEDB_MCP_STATE_DIR` to change the directory. Back up the entire directory, including any SQLite WAL files present, with MCP stopped. Retain the corresponding daemon data if you rely on saved entries. After a catalog reset, rebuild the symbol index and resave source-backed contexts. Omit `worktree_path` to use the legacy tool behavior.

## 🌐 Multi-Language AST Support

Tree-Sitter parsers extract rich symbol graphs with zero external compiler dependencies:

| Language | Extensions | Extracted Symbols |
| :--- | :--- | :--- |
| **Python** | `.py`, `.pyi` | Classes, Methods, Functions, Signatures, Docstrings |
| **Rust** | `.rs` | Structs, Enums, Traits, Impl Methods, Functions, Doc comments |
| **Go** | `.go` | Structs, Interfaces, Methods, Functions, Package comments |
| **TypeScript** | `.ts`, `.tsx` | Classes, Interfaces, Types, Methods, Functions, JSDoc |
| **JavaScript** | `.js`, `.jsx`, `.mjs` | Classes, Methods, Functions, JSDoc |

---

## ⚙️ Environment Configuration

| Variable | Default | Description |
| :--- | :---: | :--- |
| `KACHEDB_HOST` | `127.0.0.1` | KacheDB daemon hostname or IP |
| `KACHEDB_PORT` | `6380` | KacheDB daemon TCP port |
| `KACHEDB_WORKSPACE_ROOTS` | *(empty)* | Optional directory list (separated by `:` on Unix or `;` on Windows) for multi-project discovery |
| `KACHEDB_AUTO_INDEX` | `1` | Enable autonomous lazy workspace indexing (`1` or `0`) |
| `KACHEDB_MCP_STATE_DIR` | OS user state directory | Location of local worktree catalogs for scoped tools |
| `KACHEDB_INDEX` | `agent_semantic_memory` | Target vector index name for semantic memory |
| `KACHEDB_THRESHOLD` | `0.80` | Minimum cosine similarity (0.0 – 1.0) for semantic hits |
| `KACHEDB_TOKEN_COST_PER_MILLION` | `3.0` | Input token cost in USD per 1M tokens for telemetry accounting |
| `KACHEDB_DEFAULT_TTL` | `86400` | Default cache lifetime in seconds (24h) |
| `KACHEDB_EMBEDDER` | `auto` | Embedding provider (`auto`, `embeddinggemma-2`, `fastembed`, `transformers`, `openai`, `mock`) |
| `KACHEDB_EMBEDDING_DIM` | `384` | Matryoshka vector dimension (`384` default MRL slice, or `768` native) |
| `KACHEDB_MODEL_PATH` | *(optional)* | Local directory containing `model.onnx` and `tokenizer.json` for `embeddinggemma-2` (alias: `EMBEDDING_MODEL_PATH`) |
| `KACHEDB_SEARCH_PREFIX` | `task: search query \| text: ` | Asymmetric query prefix for `embeddinggemma-2` |
| `KACHEDB_INDEXING_PREFIX` | `task: document indexing \| text: ` | Asymmetric document indexing prefix for `embeddinggemma-2` |
| `OPENAI_API_KEY` | *(optional)* | API key if using `KACHEDB_EMBEDDER=openai` |

---

## Testing

Run the tests in the project environment to use the MCP v1 dependency declared in `pyproject.toml`.

```bash
uv sync --extra dev
uv run pytest
```

Set `KACHEDB_TEST_SERVER` to the path of a local KacheDB v0.5.0 binary to run the scoped live-daemon tests.

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
