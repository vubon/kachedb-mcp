"""KacheDB Codebase Indexer module for AST extraction and project discovery."""

from __future__ import annotations

from .discovery import (
    DEFAULT_IGNORED_DIRS,
    KNOWN_MANIFESTS,
    GitEngine,
    ProjectInfo,
    WorkspaceDiscovery,
    discover_projects,
    find_git_root,
    get_git_diff_files,
    get_head_commit,
    is_path_ignored,
    load_gitignore,
)
from .parser import (
    CodeParser,
    Symbol,
    SymbolRegistry,
)

__all__ = [
    "DEFAULT_IGNORED_DIRS",
    "KNOWN_MANIFESTS",
    "CodeParser",
    "GitEngine",
    "ProjectInfo",
    "Symbol",
    "SymbolRegistry",
    "WorkspaceDiscovery",
    "discover_projects",
    "find_git_root",
    "get_git_diff_files",
    "get_head_commit",
    "is_path_ignored",
    "load_gitignore",
]
