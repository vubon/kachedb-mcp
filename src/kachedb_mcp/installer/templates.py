"""Canonical KacheDB rule, shared knowledge, and agent prompt templates."""

from __future__ import annotations

BEGIN_AIDLC_MARKER = "<!-- BEGIN KACHEDB:aidlc-rules -->"
END_AIDLC_MARKER = "<!-- END KACHEDB:aidlc-rules -->"

BEGIN_FORBIDDEN_MARKER = "<!-- BEGIN KACHEDB:forbidden -->"
END_FORBIDDEN_MARKER = "<!-- END KACHEDB:forbidden -->"

BEGIN_MANDATED_MARKER = "<!-- BEGIN KACHEDB:mandated -->"
END_MANDATED_MARKER = "<!-- END KACHEDB:mandated -->"

BEGIN_AGENT_MARKER = "<!-- BEGIN KACHEDB:agent-directives -->"
END_AGENT_MARKER = "<!-- END KACHEDB:agent-directives -->"

BEGIN_TOML_MCP_MARKER = "# BEGIN KACHEDB:mcp"
END_TOML_MCP_MARKER = "# END KACHEDB:mcp"


def render_shared_knowledge(space: str = "default") -> str:
    """Return the preloaded `aidlc-shared/kachedb-bounded-retrieval.md` standard."""
    return f"""# KacheDB Bounded Retrieval & Memory Persistence Standard

## 1. Principles
All AI-DLC agents in this workspace must use KacheDB via the `kachedb` MCP server to keep
context windows bounded (< 4,000 tokens/turn) and prevent context explosion.

## 2. Standard Key Schema (SwissTable)
- `space:{space}:intent:{{id}}:unit:{{n}}`: Active unit functional spec (~700 tokens)
- `space:{space}:intent:{{id}}:contracts`: API schemas, models, DB tables (~600 tokens)
- `space:{space}:intent:{{id}}:nfr`: Security & latency constraints (~400 tokens)
- `space:{space}:decisions:adr`: Cross-sprint architectural decisions (~600 tokens)
- `space:{space}:knowledge:{{module}}`: Reverse-engineered module maps (~800 tokens)

## 3. Tool Usage Rules
1. **Pre-Execution Check**: Call `kache_get(key="...")` or `kache_semantic_search`
   before reading codebase files or upstream intent folders.
2. **Post-Execution Persistence**: Call `kache_save_context` with `prompt_alias` upon
   completing any stage artifact, unit, or review.
3. **Chunk Limit**: Keep chunks <= 600 tokens (ONNX RotaryEmbedding safety limit).
4. **Prohibited Crawling**: Never read whole `audit.md` logs or historical intent trees.
"""


def render_team_way_of_working_block() -> str:
    """Return bounded retrieval practices injected under ``## Way of Working`` in ``team.md``."""
    return (
        "- Bounded context retrieval: query KacheDB for active unit contracts and decisions; "
        "reading historical audit logs or full spec dumps is prohibited.\n"
        "- Write-through indexing: index stage outputs and unit summaries into KacheDB "
        "immediately upon generation."
    )


def render_team_forbidden_block(affirmed_date: str) -> str:
    """Return team-level ``- NEVER`` affirmations placed under ``## Forbidden``."""
    return (
        f"- NEVER read entire historical intent directories, historical `audit.md` logs, "
        f"or complete upstream spec files during unit execution. (affirmed {affirmed_date})\n"
        f"- NEVER start a construction unit without first fetching its bounded contract via "
        f"`kache_get`. (affirmed {affirmed_date})"
    )


def render_team_mandated_block(affirmed_date: str) -> str:
    """Return team-level ``- ALWAYS`` affirmations placed under ``## Mandated``."""
    return (
        f"- ALWAYS query KacheDB via `kache_get` for active unit contracts and "
        f"`kache_semantic_search` for architectural decisions before executing tasks. "
        f"(affirmed {affirmed_date})\n"
        f"- ALWAYS call `kache_save_context` to index stage deliverables and unit "
        f"implementation summaries upon completion. (affirmed {affirmed_date})"
    )


def render_project_forbidden_block(affirmed_date: str) -> str:
    """Return project-level ``- NEVER`` affirmations placed under ``## Forbidden``."""
    return (
        f"- NEVER read historical intent directories or entire `audit.md` logs into the active "
        f"context window. (affirmed {affirmed_date})\n"
        f"- NEVER perform repository-wide scans when retrieving domain contracts, unit "
        f"specifications, or prior architecture decisions. (affirmed {affirmed_date})"
    )


def render_project_mandated_block(affirmed_date: str) -> str:
    """Return project-level ``- ALWAYS`` affirmations placed under ``## Mandated``."""
    return (
        f"- ALWAYS query KacheDB (via `kache_get` or `kache_semantic_search`) to retrieve "
        f"bounded unit specifications, active contracts, and prior architectural decisions "
        f"before starting any stage. (affirmed {affirmed_date})\n"
        f"- ALWAYS persist and upsert structured stage summaries, contracts, and unit "
        f"completion records into KacheDB via `kache_save_context` immediately upon "
        f"stage or unit completion. (affirmed {affirmed_date})"
    )


def render_project_policy_section(space: str) -> str:
    """Return the KacheDB Context & Retrieval Policy section for ``project.md``."""
    return f"""## KacheDB Context & Retrieval Policy (MANDATORY)

### 1. Write-Through Indexing (On Stage Completion)
- Upsert structured sections via `kache_save_context` using keys:
  `space:{space}:intent:{{intent}}:stage:{{stage}}`.
- Keep local Markdown files on disk terse (`Minimal` depth).

### 2. Bounded Retrieval (On Stage Start)
- Query `kachedb` semantically (`top_k=3`, budget <= 2,500 tokens).
- During Construction, fetch only `space:{space}:intent:{{intent}}:unit:{{n}}`
  and `aidlc-state.md`."""


def render_construction_phase_block(space: str) -> str:
    """Return Construction phase bounded retrieval directives."""
    return f"""## KacheDB Context & Retrieval Directives (MANDATORY)

- **Unit Isolation**: Fetch ONLY the active unit contract via exact key:
  `kache_get(key="space:{space}:intent:<intent_id>:unit:<n>")` and `aidlc-state.md`.
- **Prohibited Crawling**: Never load full `functional-design.md` or `domain-design.md`.
- **Write-Through Persistence**: Immediately after unit verification, call
  `kache_save_context(topic="unit-<n>-summary", prompt_alias="unit <n> summary", ...)`."""


def render_inception_phase_block(space: str) -> str:
    """Return Inception phase bounded retrieval directives."""
    return f"""## KacheDB Bounded Retrieval & Indexing Directives (MANDATORY)

- **Prior Decision Retrieval**: Query `kache_semantic_search(query="...", top_k=3,
  exact_first=true)` within a <= 2,500 token budget before authoring ADRs or specs.
- **Write-Through Indexing**: Index completed artifacts via `kache_save_context` using
  deterministic keys (`space:{space}:intent:<id>:stage:<stage>`).
- **No Historical Dumps**: Never load previous intent folders or whole audit trails."""


def render_workspace_memory_rules(space: str = "default") -> str:
    """Return `.agents/rules/kachedb_memory.md` workspace agent rules."""
    return f"""# KacheDB In-Memory Context Caching Rules (MANDATORY)
# ====================================================
# Automatically configured by `kachedb-mcp install --aidlc`.

## Pre-Execution Cache Check
Before large file reads, deep research, or architecture questions:
1. Call `kache_semantic_search(query="...", exact_first=True)` or `kache_get(key="...")`.
2. Reuse cached context directly instead of scanning entire repositories.

## Post-Execution Auto-Save
Upon completing any bugfix, feature unit, benchmark, or architecture update:
- Call `kache_save_context(topic="...", prompt_alias="...", content="...", tags=[...])`.
- Keep chunk sizes <= 600 tokens.

## AI-DLC Bounded Key Schema
- `space:{space}:intent:{{id}}:unit:{{n}}` — Active unit functional spec (`kache_get`)
- `space:{space}:intent:{{id}}:contracts` — API schemas & contracts (`kache_get`)
- `space:{space}:intent:{{id}}:nfr` — Security & latency constraints (`kache_get`)
- `space:{space}:decisions:adr` — Architectural decisions (`kache_semantic_search`)
"""


def render_agent_directives_block(space: str = "default") -> str:
    """Return agent prompt directive block injected into `.codex/agents/*.{md,toml}`."""
    return f"""## KacheDB Bounded Memory Directives (MANDATORY)

1. **Bounded Pre-Check**: Query `kache_get(key="space:{space}:intent:<id>:unit:<n>")`
   or `kache_semantic_search(..., top_k=3, exact_first=true)` before reading files.
2. **Prohibited Crawling**: Do NOT read historical intent dirs or full `audit.md` logs.
3. **Write-Through Persistence**: Upsert completed summaries via `kache_save_context`."""
