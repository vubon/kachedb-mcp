"""Workspace and Git project discovery engine for KacheDB MCP."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pathspec

logger = logging.getLogger(__name__)

# Standard language project manifests mapped to project language/ecosystem
KNOWN_MANIFESTS: dict[str, str] = {
    "Cargo.toml": "rust",
    "package.json": "node",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "go.mod": "go",
    "pom.xml": "java",
    "build.gradle": "java/kotlin",
    "build.gradle.kts": "kotlin",
    "Package.swift": "swift",
    "mix.exs": "elixir",
    "Gemfile": "ruby",
}

# Directories that should unconditionally be ignored during code traversal
DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "target",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".turbo",
        ".cache",
        "vendor",
        ".idea",
        ".vscode",
        ".fastembed_cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        # OS & Cloud Storage Protection Blocklist
        "Library",
        "Mobile Documents",
        "iCloud Drive",
        "Applications",
        "Pictures",
        "Music",
        "Movies",
        ".Trash",
    }
)

# Code file extensions that indicate real codebase source files
SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".rs",
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".java",
        ".kt",
        ".kts",
        ".swift",
        ".c",
        ".cpp",
        ".cc",
        ".h",
        ".hpp",
        ".cs",
        ".php",
        ".rb",
        ".lua",
        ".zig",
        ".scala",
        ".dart",
    }
)


@dataclass(frozen=True)
class ProjectInfo:
    """Represents a discovered codebase project within a workspace."""

    path: Path
    name: str
    is_git: bool
    git_root: Path | None = None
    manifest_type: str | None = None
    commit_sha: str | None = None
    branch: str | None = None

    @property
    def workspace_id(self) -> str:
        """Normalized workspace ID string for KacheDB key namespacing."""
        return self.name.lower().replace(" ", "-").replace("_", "-")


class GitEngine:
    """Encapsulates Git repository operations, commit inspection, delta diffs, and gitignore."""

    def __init__(self, git_root: Path | str) -> None:
        self.git_root = Path(git_root).resolve()
        self._gitignore_spec: pathspec.PathSpec[Any] | None = None
        self._gitignore_loaded: bool = False

    @classmethod
    def find_root(cls, start_path: Path | str) -> GitEngine | None:
        """Climb up parents starting from start_path to find nearest .git directory."""
        curr = Path(start_path).resolve()
        for parent in [curr, *curr.parents]:
            git_dir = parent / ".git"
            if git_dir.exists():
                return cls(parent)
        return None

    def get_head_commit(self) -> tuple[str | None, str | None]:
        """Retrieve current (branch, commit_sha) for the Git repository.

        Attempts fast direct filesystem read first (< 100 µs), falling back
        to `git rev-parse` if in a detached state, worktree, or packed-refs.
        """
        git_dir = self.git_root / ".git"
        if not git_dir.exists():
            return None, None

        # Handle git worktree or submodule pointer
        if git_dir.is_file():
            return self._get_head_via_cli()

        head_file = git_dir / "HEAD"
        if not head_file.is_file():
            return self._get_head_via_cli()

        try:
            content = head_file.read_text(encoding="utf-8").strip()
            if content.startswith("ref: refs/heads/"):
                branch = content[len("ref: refs/heads/") :]
                ref_file = git_dir / "refs" / "heads" / branch
                if ref_file.is_file():
                    sha = ref_file.read_text(encoding="utf-8").strip()
                    return branch, sha
            elif len(content) == 40 and all(c in "0123456789abcdefABCDEF" for c in content):
                # Detached HEAD with direct commit SHA
                return "HEAD", content
        except Exception as e:
            logger.debug("Fast HEAD read failed on %s: %s; falling back to CLI", self.git_root, e)

        return self._get_head_via_cli()

    def _get_head_via_cli(self) -> tuple[str | None, str | None]:
        """Fallback method using git CLI to inspect HEAD."""
        try:
            res_sha = subprocess.run(
                ["git", "-C", str(self.git_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2.0,
            )
            sha = res_sha.stdout.strip() or None

            res_branch = subprocess.run(
                ["git", "-C", str(self.git_root), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2.0,
            )
            branch = res_branch.stdout.strip() or None

            return branch, sha
        except Exception as e:
            logger.debug("Git CLI failed on %s: %s", self.git_root, e)
            return None, None

    def get_diff_files(
        self,
        old_sha: str,
        new_sha: str = "HEAD",
    ) -> list[str] | None:
        """Retrieve list of modified, added, or deleted files between two commit SHAs."""
        if not old_sha or not new_sha:
            return None

        try:
            res = subprocess.run(
                ["git", "-C", str(self.git_root), "diff", "--name-only", old_sha, new_sha],
                capture_output=True,
                text=True,
                check=True,
                timeout=5.0,
            )
            files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            return files
        except Exception as e:
            logger.warning(
                "Failed to execute git diff between %s and %s in %s: %s",
                old_sha,
                new_sha,
                self.git_root,
                e,
            )
            return None

    def load_gitignore(self) -> pathspec.PathSpec[Any] | None:
        """Parse .gitignore in repository root if present, returning compiled PathSpec."""
        gitignore_path = self.git_root / ".gitignore"
        if not gitignore_path.is_file():
            return None

        try:
            lines = [
                line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines()
            ]
            return pathspec.PathSpec.from_lines("gitignore", lines)
        except Exception as e:
            logger.warning("Failed to load .gitignore at %s: %s", gitignore_path, e)
            return None

    @property
    def gitignore_spec(self) -> pathspec.PathSpec[Any] | None:
        """Cached property for the repository's parsed .gitignore PathSpec."""
        if not self._gitignore_loaded:
            self._gitignore_spec = self.load_gitignore()
            self._gitignore_loaded = True
        return self._gitignore_spec

    def is_ignored(self, rel_path: str) -> bool:
        """Check if a relative path is ignored by gitignore or default ignored directories."""
        return is_path_ignored(rel_path, self.gitignore_spec)


class WorkspaceDiscovery:
    """Scans and discovers eligible codebase projects within a workspace root."""

    def __init__(
        self,
        workspace_root: Path | str,
        ignored_dirs: frozenset[str] = DEFAULT_IGNORED_DIRS,
        known_manifests: dict[str, str] = KNOWN_MANIFESTS,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.ignored_dirs = ignored_dirs
        self.known_manifests = known_manifests

    def detect_manifest(self, dir_path: Path) -> tuple[str | None, str | None]:
        """Check if directory contains a known project manifest."""
        for manifest_name, manifest_type in self.known_manifests.items():
            if (dir_path / manifest_name).is_file():
                return manifest_name, manifest_type
        return None, None

    def has_source_files(self, dir_path: Path, max_depth: int = 2) -> bool:
        """Fast check whether directory contains any source code files within max_depth."""
        try:
            for root, dirs, files in os.walk(dir_path):
                rel_depth = len(Path(root).relative_to(dir_path).parts)
                # Prune ignored directories
                dirs[:] = [d for d in dirs if d not in self.ignored_dirs]

                for file in files:
                    if any(file.endswith(ext) for ext in SOURCE_EXTENSIONS):
                        return True

                if rel_depth >= max_depth:
                    dirs.clear()
        except Exception:
            pass
        return False

    def discover(self) -> list[ProjectInfo]:
        """Scan workspace_root and return all detected codebases."""
        if not self.workspace_root.is_dir():
            return []

        # Safety Guard: Never treat $HOME, filesystem root, or system folders as a codebase
        try:
            home_path = Path.home().resolve()
            root_path = Path("/").resolve()
            users_path = Path("/Users").resolve()
            if self.workspace_root in (home_path, root_path, users_path):
                logger.warning(
                    "Refusing to index user home directory or filesystem root as a workspace: %s",
                    self.workspace_root,
                )
                return []
        except Exception:
            pass

        # Case 1: Workspace root is directly a Git repo or has a manifest
        root_git = (self.workspace_root / ".git").exists()
        root_manifest, manifest_type = self.detect_manifest(self.workspace_root)

        if root_git:
            git_engine = GitEngine(self.workspace_root)
            branch, sha = git_engine.get_head_commit()
            return [
                ProjectInfo(
                    path=self.workspace_root,
                    name=self.workspace_root.name,
                    is_git=True,
                    git_root=self.workspace_root,
                    manifest_type=manifest_type,
                    commit_sha=sha,
                    branch=branch,
                )
            ]

        # Case 1b: If root has manifest but no git directly, check if parent git repo exists
        if root_manifest:
            parent_git = GitEngine.find_root(self.workspace_root)
            branch, sha = parent_git.get_head_commit() if parent_git else (None, None)
            return [
                ProjectInfo(
                    path=self.workspace_root,
                    name=self.workspace_root.name,
                    is_git=parent_git is not None,
                    git_root=parent_git.git_root if parent_git else None,
                    manifest_type=manifest_type,
                    commit_sha=sha,
                    branch=branch,
                )
            ]

        # Case 2: Multi-folder workspace with multiple subdirectories
        discovered: list[ProjectInfo] = []

        try:
            for entry in sorted(self.workspace_root.iterdir()):
                if not entry.is_dir() or entry.name in self.ignored_dirs:
                    continue
                if entry.name.startswith("."):
                    continue

                sub_git = (entry / ".git").exists()
                sub_manifest, sub_type = self.detect_manifest(entry)

                if sub_git:
                    git_engine = GitEngine(entry)
                    branch, sha = git_engine.get_head_commit()
                    discovered.append(
                        ProjectInfo(
                            path=entry,
                            name=entry.name,
                            is_git=True,
                            git_root=entry,
                            manifest_type=sub_type,
                            commit_sha=sha,
                            branch=branch,
                        )
                    )
                elif sub_manifest or self.has_source_files(entry):
                    discovered.append(
                        ProjectInfo(
                            path=entry,
                            name=entry.name,
                            is_git=False,
                            git_root=None,
                            manifest_type=sub_type,
                            commit_sha=None,
                            branch=None,
                        )
                    )
        except Exception as e:
            logger.warning(
                "Error scanning workspace subdirectories at %s: %s",
                self.workspace_root,
                e,
            )

        return discovered


# =========================================================================
# Convenience functional facades (delegating to GitEngine & WorkspaceDiscovery)
# =========================================================================


def find_git_root(start_path: Path | str) -> Path | None:
    """Climb up parents starting from start_path to find nearest .git directory."""
    engine = GitEngine.find_root(start_path)
    return engine.git_root if engine else None


def get_head_commit(git_root: Path | str) -> tuple[str | None, str | None]:
    """Retrieve current (branch, commit_sha) for the Git repository."""
    return GitEngine(git_root).get_head_commit()


def get_git_diff_files(
    git_root: Path | str,
    old_sha: str,
    new_sha: str = "HEAD",
) -> list[str] | None:
    """Retrieve list of modified, added, or deleted files between two commit SHAs."""
    return GitEngine(git_root).get_diff_files(old_sha, new_sha)


def load_gitignore(root: Path | str) -> pathspec.PathSpec[Any] | None:
    """Parse .gitignore in root if present, returning compiled PathSpec."""
    return GitEngine(root).load_gitignore()


def is_path_ignored(
    rel_path: str,
    spec: pathspec.PathSpec[Any] | None = None,
) -> bool:
    """Determine if relative path should be excluded based on default dirs or gitignore."""
    normalized = rel_path.replace("\\", "/").strip("/")
    parts = normalized.split("/")

    for part in parts:
        if part in DEFAULT_IGNORED_DIRS:
            return True

    return bool(spec is not None and spec.match_file(normalized))


def discover_projects(workspace_root: Path | str) -> list[ProjectInfo]:
    """Scan and discover all valid codebases within workspace_root."""
    return WorkspaceDiscovery(workspace_root).discover()


def find_root_from_path(target: Path | str) -> Path | None:
    """Climb parents upwards from a file or directory to locate the project or repo root.

    Detects repository and project boundaries via:
    1. Version control marker: ``.git`` directory or file (worktrees/submodules)
    2. Known build/project manifests (Cargo.toml, pyproject.toml, package.json, go.mod, etc.)

    Args:
        target: A file path or directory path (absolute or relative).

    Returns:
        The enclosing project or repository root Path, or None if forbidden or invalid.
    """
    home = Path.home().resolve()
    root = Path("/").resolve()
    users = Path("/Users").resolve()
    forbidden = (home, root, users)

    try:
        cand = Path(os.path.realpath(os.path.expanduser(str(target)))).resolve()
    except Exception:
        return None

    if cand in forbidden:
        return None

    current = cand.parent if cand.is_file() else cand
    if not current.is_dir() or current in forbidden:
        return None

    manifest_root: Path | None = None
    for parent in [current, *current.parents]:
        if parent in forbidden:
            break
        if (parent / ".git").exists():
            return parent
        if manifest_root is None and any((parent / m).is_file() for m in KNOWN_MANIFESTS):
            manifest_root = parent

    if manifest_root is not None:
        return manifest_root

    return current


def resolve_workspace_path(
    workspace_id: str = "",
    candidate_path: str = ".",
    kache_client: Any | None = None,
) -> Path | None:
    """Resolve the filesystem directory for a workspace without guessing folder names.

    Resolution order:

    1. **Upward Anchor from candidate_path** — if provided (not "."), climb parents
       to locate the nearest .git root or project manifest (pyproject.toml, Cargo.toml, etc.).
       Supports passing file paths (e.g. "src/main.py") or nested directories.

    2. **SwissTable cache** — sub-microsecond lookup of 'kache:meta:<workspace_id>:path'
       when a client is provided. Populated after every index run.

    3. **KACHEDB_WORKSPACE_ROOTS** — user/IDE configured roots from env var
       (e.g. "/home/alice/projects"), searched for <workspace_id> as direct child or
       category subdirectory (e.g. "services/billing" or "apps/web").

    4. **CWD Upward Walk** — climb parents from Path.cwd() to see if current working
       directory belongs to a git repo or project with a manifest.

    Args:
        workspace_id: Logical workspace name (e.g. "my-project", "api-service").
        candidate_path: Explicit filesystem path or file path hint. If "." (default), skipped.
        kache_client: Optional connected KacheClient for SwissTable path cache lookup.

    Returns:
        Resolved Path to the workspace directory, or None if not resolvable.
    """
    from ..config import settings

    home = Path.home().resolve()
    root = Path("/").resolve()
    users = Path("/Users").resolve()
    forbidden = (home, root, users)

    # ── 1. Upward Anchor from candidate_path ────────────────────────────────
    if candidate_path and candidate_path != ".":
        resolved_cand = find_root_from_path(candidate_path)
        if resolved_cand is not None and resolved_cand not in forbidden:
            return resolved_cand
        # Explicit candidate path was provided but invalid/forbidden: do not guess
        return None

    clean_ws = workspace_id.strip() if workspace_id else ""

    # ── 2. SwissTable cache (sub-microsecond when warm) ──────────────────────
    if clean_ws and kache_client is not None:
        try:
            cached = kache_client.get(f"kache:meta:{clean_ws}:path")
            if cached:
                cached_str = (
                    cached.decode("utf-8", errors="replace")
                    if isinstance(cached, bytes)
                    else str(cached)
                )
                p = Path(os.path.realpath(cached_str)).resolve()
                if p not in forbidden and p.is_dir():
                    return p
        except Exception:
            pass

    # ── 3. KACHEDB_WORKSPACE_ROOTS (user-configured, highest accuracy) ──────
    if clean_ws and settings.workspace_roots:
        names_to_try = [
            clean_ws,
            clean_ws.lower(),
            clean_ws.replace("_", "-"),
            clean_ws.replace("-", "_"),
        ]

        def _check_base(base: Path) -> Path | None:
            """Search base for workspace_id as base itself, direct child, or one level deep."""
            if not base.is_dir():
                return None
            # 1. Base itself is the workspace (e.g. /path/to/my-project)
            if base.name in names_to_try:
                has_git = (base / ".git").exists()
                has_manifest = any((base / m).is_file() for m in KNOWN_MANIFESTS)
                if has_git or has_manifest:
                    return base
            # 2. Direct child: base/<workspace_id>
            for name in names_to_try:
                target = (base / name).resolve()
                if target not in forbidden and target.is_dir():
                    return target
            # One level deeper: base/<category>/<workspace_id>
            try:
                for sub in base.iterdir():
                    if sub.is_dir() and not sub.name.startswith("."):
                        for name in names_to_try:
                            target = (sub / name).resolve()
                            if target not in forbidden and target.is_dir():
                                return target
            except OSError:
                pass
            return None

        for root_str in settings.workspace_roots:
            try:
                base = Path(os.path.expanduser(root_str)).resolve()
                result = _check_base(base)
                if result:
                    return result
            except Exception:
                pass

    # ── 4. CWD Upward Walk (per-project MCP server launch) ───────────────────
    try:
        cwd = Path.cwd().resolve()
        if cwd not in forbidden:
            cwd_root = find_root_from_path(cwd)
            if cwd_root is not None and cwd_root not in forbidden:
                has_git = (cwd_root / ".git").exists()
                has_manifest = any((cwd_root / m).is_file() for m in KNOWN_MANIFESTS)
                if has_git or has_manifest:
                    return cwd_root
    except Exception:
        pass

    return None
