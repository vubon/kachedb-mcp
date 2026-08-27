# 🧠 KacheDB MCP Server

[![PyPI](https://img.shields.io/pypi/v/kachedb-mcp.svg)](https://pypi.org/project/kachedb-mcp/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%7C%20MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**High-Speed Model Context Protocol (MCP) server for KacheDB** — exposing sub-millisecond in-memory caching and SIMD semantic vector memory to **Antigravity IDE**, **Claude Desktop**, **Cursor**, and AI coding agents.

---

## ⚡ Why KacheDB for AI Agents?

AI coding assistants (like Antigravity IDE and Claude Desktop) repeatedly re-read large codebases, AST parses, and architecture plans, wasting thousands of tokens and adding hundreds of milliseconds of latency per turn.

`kachedb-mcp` connects your AI assistant directly to **KacheDB's Megaslab pure-RAM cache**:
- 🚀 **Sub-50 Microsecond Retrieval:** $< 50\ \mu\text{s}$ cache hits in memory.
- 🧠 **SIMD Semantic Vector Memory:** Natural language concept and code recall powered by ARM NEON & AVX2/FMA cosine similarity kernels.
- 🪙 **Massive Token & Cost Savings:** Saves up to 80% of repetitive prompt tokens and tracks cumulative financial savings in real time.

---

## 🛠️ MCP Tools Exposed

| Tool | Type | Description |
| :--- | :---: | :--- |
| `kache_semantic_search` | 🧠 *Vector* | Natural language semantic search over cached codebases, PR reviews, and past decisions (`top_k`, `threshold`). |
| `kache_save_context` | 🧠 *Vector* | Save an architectural pattern, bug solution, or file digest with SIMD vector embeddings. |
| `kache_get` | ⚡ *Exact* | Sub-millisecond exact key retrieval for code chunks, ASTs, and tool outputs. |
| `kache_set` | ⚡ *Exact* | Store string content in memory with optional TTL expiration. |
| `kache_delete` | ⚡ *Exact* | Remove a key or vector from cache. |
| `kache_stats` | 📊 *Metrics* | Real-time connection health, active vector counts, and RAM footprint. |
| `kache_telemetry` | 📊 *Metrics* | Live cumulative tokens saved, latency saved (seconds), and hit ratios. |

---

## 🚀 Quickstart

### 1. Start the KacheDB Server Daemon
Ensure your KacheDB daemon is running locally:
```bash
kachedb-server --port 6379
```

### 2. Configure in Antigravity IDE / Claude Desktop / Cursor

Add `kachedb` to your MCP configuration file (`mcp_config.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kachedb": {
      "command": "uvx",
      "args": ["kachedb-mcp"],
      "env": {
        "KACHEDB_HOST": "127.0.0.1",
        "KACHEDB_PORT": "6379",
        "KACHEDB_INDEX": "agent_semantic_memory",
        "KACHEDB_THRESHOLD": "0.80"
      }
    }
  }
}
```

Or run via Python `pip`:
```bash
pip install kachedb-mcp
```
```json
{
  "mcpServers": {
    "kachedb": {
      "command": "kachedb-mcp"
    }
  }
}
```

---

## ⚙️ Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `KACHEDB_HOST` | `127.0.0.1` | KacheDB daemon hostname or IP |
| `KACHEDB_PORT` | `6379` | KacheDB daemon TCP port |
| `KACHEDB_INDEX` | `agent_semantic_memory` | Target vector index name for semantic memory |
| `KACHEDB_THRESHOLD` | `0.80` | Minimum cosine similarity (0.0 – 1.0) for semantic hits |
| `KACHEDB_DEFAULT_TTL` | `86400` | Default cache lifetime in seconds (24h) |
| `KACHEDB_EMBEDDER` | `auto` | Embedding provider (`auto`, `fastembed`, `transformers`, `openai`, `mock`) |
| `OPENAI_API_KEY` | (optional) | API key if using `KACHEDB_EMBEDDER=openai` |

---

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
