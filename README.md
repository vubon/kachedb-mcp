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

Or install via Python `pip`:
```bash
pip install kachedb-mcp
```
```json
{
  "mcpServers": {
    "kachedb": {
      "command": "kachedb-mcp",
      "env": {
        "KACHEDB_PORT": "6380"
      }
    }
  }
}
```

---

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
| `KACHEDB_INDEX` | `agent_semantic_memory` | Target vector index name for semantic memory |
| `KACHEDB_THRESHOLD` | `0.80` | Minimum cosine similarity (0.0 – 1.0) for semantic hits |
| `KACHEDB_TOKEN_COST_PER_MILLION` | `3.0` | Input token cost in USD per 1M tokens for telemetry accounting |
| `KACHEDB_DEFAULT_TTL` | `86400` | Default cache lifetime in seconds (24h) |
| `KACHEDB_EMBEDDER` | `auto` | Embedding provider (`auto`, `fastembed`, `transformers`, `openai`, `mock`) |
| `OPENAI_API_KEY` | *(optional)* | API key if using `KACHEDB_EMBEDDER=openai` |

---

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
