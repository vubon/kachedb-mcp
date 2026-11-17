"""Shared filesystem, JSON merge, and idempotent Markdown utilities for the CLI installer."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

ActionStatus = Literal["created", "updated", "unchanged", "skipped"]


@dataclass(frozen=True)
class InstallAction:
    """Single file operation recorded or executed during installation."""

    path: Path
    status: ActionStatus
    detail: str


@dataclass
class InstallPlan:
    """Aggregated plan/results and advisory warnings from an install run."""

    actions: list[InstallAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, action: InstallAction) -> None:
        """Append an action to the plan."""
        self.actions.append(action)

    def extend(self, other: InstallPlan) -> None:
        """Merge actions and warnings from another plan."""
        self.actions.extend(other.actions)
        self.warnings.extend(other.warnings)

    def format_summary(self, dry_run: bool = False) -> str:
        """Render a human-readable summary table of all planned/applied changes."""
        prefix = "[DRY RUN] " if dry_run else ""
        lines: list[str] = [f"{prefix}KacheDB Installer Summary:"]
        for action in self.actions:
            lines.append(f"  {action.status.upper():9s} {action.path} — {action.detail}")
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)


def upsert_marked_block(
    existing: str,
    block: str,
    begin_marker: str,
    end_marker: str,
    insert_before: str | None = None,
) -> str:
    """Insert or replace a marker-delimited section idempotently in Markdown/text content."""
    clean_block = block.strip("\n")
    wrapped = f"{begin_marker}\n{clean_block}\n{end_marker}"

    begin_idx = existing.find(begin_marker)
    end_idx = existing.find(end_marker, begin_idx + len(begin_marker)) if begin_idx != -1 else -1

    if begin_idx != -1 and end_idx == -1:
        raise ValueError(f"Orphaned begin marker '{begin_marker}' without '{end_marker}'")
    if begin_idx == -1 and end_marker in existing:
        raise ValueError(f"Orphaned end marker '{end_marker}' without '{begin_marker}'")

    if begin_idx != -1 and end_idx != -1:
        end_pos = end_idx + len(end_marker)
        return f"{existing[:begin_idx]}{wrapped}{existing[end_pos:]}"

    if not existing.strip():
        return f"{wrapped}\n"

    if insert_before:
        match = re.search(rf"^{re.escape(insert_before)}[ \t]*$", existing, re.MULTILINE)
        if match is not None:
            anchor_idx = match.start()
            before = existing[:anchor_idx].rstrip("\n")
            after = existing[anchor_idx:].lstrip("\n")
            sep = "\n\n" if before else ""
            return f"{before}{sep}{wrapped}\n\n{after}"

    base = existing.rstrip("\n")
    return f"{base}\n\n{wrapped}\n"


def upsert_under_h2_section(
    existing: str,
    heading: str,
    block: str,
    begin_marker: str,
    end_marker: str,
) -> str | None:
    """Upsert a marked block at the end of an existing H2 section, preserving user items.

    Returns ``None`` if ``heading`` is not present on its own line in ``existing`` (and markers
    do not already exist).
    """
    if begin_marker in existing or end_marker in existing:
        return upsert_marked_block(existing, block, begin_marker, end_marker)

    heading_match = re.search(rf"^{re.escape(heading)}[ \t]*$", existing, re.MULTILINE)
    if heading_match is None:
        return None

    clean_block = block.strip("\n")
    wrapped = f"{begin_marker}\n{clean_block}\n{end_marker}"

    after_heading_pos = heading_match.end()
    next_h2 = re.search(r"^##[ \t]+", existing[after_heading_pos:], re.MULTILINE)
    if next_h2 is not None:
        insert_pos = after_heading_pos + next_h2.start()
        before = existing[:insert_pos].rstrip("\n")
        after = existing[insert_pos:].lstrip("\n")
        return f"{before}\n\n{wrapped}\n\n{after}"

    base = existing.rstrip("\n")
    return f"{base}\n\n{wrapped}\n"


def write_text_if_changed(
    path: Path,
    new_content: str,
    dry_run: bool,
    detail: str,
) -> InstallAction:
    """Write text to ``path`` only when the content differs from disk."""
    existed = path.exists()
    if existed:
        current = path.read_text(encoding="utf-8")
        if current == new_content:
            return InstallAction(path=path, status="unchanged", detail=detail)
        status: ActionStatus = "updated"
    else:
        status = "created"

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")

    return InstallAction(path=path, status=status, detail=detail)


def merge_json_file(
    path: Path,
    updater: Callable[[dict[str, Any]], dict[str, Any]],
    dry_run: bool,
    detail: str,
) -> InstallAction:
    """Safely load, transform, and persist JSON while preserving unrelated keys."""
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(parsed).__name__}")
        updated = updater(copy.deepcopy(parsed))
        rendered = json.dumps(updated, indent=2) + "\n"
        if raw == rendered:
            return InstallAction(path=path, status="unchanged", detail=detail)
        status: ActionStatus = "updated"
    else:
        updated = updater({})
        rendered = json.dumps(updated, indent=2) + "\n"
        status = "created"

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

    return InstallAction(path=path, status=status, detail=detail)
