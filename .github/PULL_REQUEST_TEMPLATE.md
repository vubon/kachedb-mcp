## 📋 Description

Provide a brief summary of the changes introduced by this Pull Request.

- What problem does this solve?
- What is the new behavior?

---

## 🧪 Testing Checklist

- [ ] All unit tests pass: `pytest tests/ -v`
- [ ] Linter checks pass: `ruff check src/ tests/`
- [ ] Code formatting verified: `ruff format --check src/ tests/`
- [ ] Type check passes: `mypy src/kachedb_mcp/`
- [ ] Documentation / `README.md` updated (if applicable)
- [ ] `CHANGELOG.md` updated
- [ ] KacheDB daemon is running and `kache_stats` returns `HEALTHY` (if testing live tools)

---

## 🤖 MCP Tool Impact

If this PR adds or modifies an MCP tool, please confirm:

- [ ] Tool is registered in `server.py` via `mcp.tool()(...)`.
- [ ] Tool has a clear docstring with `Args:` and `Returns:` sections.
- [ ] Telemetry (`tracker.record_hit()` / `tracker.record_miss()`) is updated where applicable.

---

## 🔗 Related Issues / PRs

Closes # (issue number)
