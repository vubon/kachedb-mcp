"""Git observations and bounded source reads for a local worktree."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from .discovery import DEFAULT_IGNORED_DIRS
from .parser import CodeParser

SOURCE_LIMIT = 16 * 1024 * 1024


class WorktreeError(Exception):
    def __init__(self, status: str, reason: str, path: str | None = None) -> None:
        super().__init__(reason)
        self.status = status
        self.path = path


@dataclass(frozen=True)
class Observation:
    head: str | None
    dirty: frozenset[str]
    eligible: frozenset[str]
    ignored: frozenset[str]


class Worktree:
    def __init__(self, anchor: str, hint: str, deadline: float) -> None:
        self.deadline = deadline
        self.root = self._resolve(anchor)
        if hint:
            candidate = Path(hint).expanduser()
            if not candidate.is_absolute():
                candidate = self.root / candidate
            candidate = candidate.resolve()
            if not candidate.is_relative_to(self.root):
                raise WorktreeError("WORKTREE_MISMATCH", "The path selects a different worktree.")
            # A hint may name a file which has just been deleted.
            while not candidate.exists() and candidate != candidate.parent:
                candidate = candidate.parent
            if candidate == self.root:
                return
            try:
                hint_root = self._resolve(str(candidate))
            except WorktreeError as exc:
                if exc.status not in ("WORKTREE_NOT_GIT", "INVALID_WORKTREE_PATH"):
                    raise
                raise WorktreeError(
                    "WORKTREE_MISMATCH", "The path selects a different worktree."
                ) from exc
            if hint_root != self.root:
                raise WorktreeError("WORKTREE_MISMATCH", "The path selects a different worktree.")

    def _git(self, directory: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
        remaining = self.remaining()
        env = os.environ.copy()
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_PREFIX"):
            env.pop(name, None)
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["LC_ALL"] = "C"
        try:
            return subprocess.run(
                ["git", "-C", str(directory), *args],
                capture_output=True,
                env=env,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorktreeError("INCOMPLETE", "Git exceeded the operation deadline.") from exc
        except OSError as exc:
            raise WorktreeError("GIT_ERROR", "Could not run Git.") from exc

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise WorktreeError("INCOMPLETE", "The operation deadline was reached.")
        return remaining

    def _resolve(self, anchor: str) -> Path:
        path = Path(anchor).expanduser()
        if not path.is_absolute() or not path.exists():
            raise WorktreeError("INVALID_WORKTREE_PATH", "Use an existing absolute local path.")
        path = path.resolve()
        directory = path if path.is_dir() else path.parent
        forbidden = {Path.home().resolve(), Path(directory.anchor), Path("/Users").resolve()}
        if directory in forbidden:
            raise WorktreeError("INVALID_WORKTREE_PATH", "Select a project directory.")
        probe = self._git(directory, "rev-parse", "--is-inside-work-tree", "--show-toplevel")
        inside, _, top_level = probe.stdout.partition(b"\n")
        if inside != b"true":
            if (
                inside != b"false"
                and probe.returncode
                and b"not a git repository" not in (probe.stderr.lower())
            ):
                raise WorktreeError("GIT_ERROR", "Git could not resolve the worktree.")
            raise WorktreeError("WORKTREE_NOT_GIT", "The path is outside a Git working tree.")
        if probe.returncode or not top_level:
            raise WorktreeError("GIT_ERROR", "Git could not locate the worktree root.")
        root = Path(os.fsdecode(top_level.removesuffix(b"\n"))).resolve()
        allowed = [Path(path).expanduser().resolve() for path in settings.workspace_roots]
        if root in forbidden or (
            allowed and not any(root.is_relative_to(path) for path in allowed)
        ):
            raise WorktreeError(
                "INVALID_WORKTREE_PATH", "The worktree is outside configured roots."
            )
        return root

    def git_paths(self, *args: str) -> set[str]:
        result = self._git(self.root, *args)
        if result.returncode:
            raise WorktreeError("GIT_ERROR", "Git could not enumerate worktree changes.")
        return {os.fsdecode(path) for path in result.stdout.split(b"\0") if path}

    def head(self) -> str | None:
        # Loose refs avoid starting Git twice per observation. Git resolves packed
        # refs, linked worktrees, and unborn branches through the fallback.
        if not os.getenv("GIT_NAMESPACE"):
            git_dir = self.root / ".git"
            try:
                head = (git_dir / "HEAD").read_bytes().strip()
                if head.startswith(b"ref: refs/heads/"):
                    ref = Path(os.fsdecode(head[5:]))
                    head = (git_dir / ref).read_bytes().strip() if ".." not in ref.parts else b""
                if len(head) in (40, 64) and all(byte in b"0123456789abcdef" for byte in head):
                    return head.decode("ascii")
            except OSError:
                pass
        result = self._git(self.root, "rev-parse", "--verify", "--quiet", "HEAD")
        if result.returncode == 1:
            return None
        if result.returncode:
            raise WorktreeError("GIT_ERROR", "Git could not read HEAD.")
        return result.stdout.strip().decode("ascii")

    def changes_since(self, base: str, head: str) -> set[str] | None:
        result = self._git(self.root, "diff", "--name-only", "--no-renames", "-z", base, head, "--")
        if result.returncode:
            if self._git(self.root, "cat-file", "-e", base + "^{commit}").returncode:
                return None
            raise WorktreeError("GIT_ERROR", "Git could not compare worktree commits.")
        return {os.fsdecode(path) for path in result.stdout.split(b"\0") if path}

    def observe(self, clean_inventory: tuple[str | None, set[str]] | None = None) -> Observation:
        head = self.head()
        # These read-only Git queries are independent. Overlap their startup and I/O.
        with ThreadPoolExecutor(max_workers=1) as executor:
            ignored_paths = executor.submit(
                self.git_paths, "ls-files", "--cached", "--ignored", "--exclude-standard", "-z"
            )
            dirty, untracked = self._status_paths()
            ignored = {path for path in ignored_paths.result() if self._eligible(path)}
        if clean_inventory is not None and not dirty and head == clean_inventory[0]:
            tracked = clean_inventory[1]
        else:
            tracked = self.git_paths("ls-files", "--cached", "-z")
        eligible = {path for path in (tracked - ignored) | untracked if self._eligible(path)}
        if head != self.head():
            raise WorktreeError("WORKTREE_CHANGED", "HEAD changed during Git observation.")
        return Observation(head, frozenset(dirty), frozenset(eligible), frozenset(ignored))

    def _status_paths(self) -> tuple[set[str], set[str]]:
        result = self._git(self.root, "status", "--porcelain=v1", "-z", "--untracked-files=normal")
        if result.returncode:
            raise WorktreeError("GIT_ERROR", "Git could not read worktree status.")
        items = iter(result.stdout.split(b"\0"))
        dirty: set[str] = set()
        untracked: set[str] = set()
        directories: set[str] = set()
        for item in items:
            if not item:
                continue
            status, path = item[:2], os.fsdecode(item[3:])
            if status == b"??":
                if path.endswith("/"):
                    directories.add(path)
                elif self._eligible(path):
                    untracked.add(path)
            else:
                dirty.add(path)
                if b"R" in status or b"C" in status:
                    origin = next(items, b"")
                    if not origin:
                        raise WorktreeError(
                            "GIT_ERROR", "Git returned an incomplete rename record."
                        )
                    dirty.add(os.fsdecode(origin))

        if directories:
            expanded = self.git_paths(
                "--literal-pathspecs",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *sorted(directories),
            )
            untracked.update(path for path in expanded if self._eligible(path))
        return {path for path in dirty if self._eligible(path)} | untracked, untracked

    @staticmethod
    def _eligible(path: str) -> bool:
        return bool(
            CodeParser.detect_language(path)
            and not any(
                part.startswith(".") or part in DEFAULT_IGNORED_DIRS
                for part in path.split("/")[:-1]
            )
        )

    def read(self, relative: str, limit: int = SOURCE_LIMIT, *, declared: bool = False) -> bytes:
        self.remaining()
        parts = relative.split("/")
        path = Path(relative)
        if (
            path.is_absolute()
            or path.drive
            or (os.name == "nt" and ("\\" in relative or ":" in relative))
            or any(part in ("", ".", "..") for part in parts)
            or "\0" in relative
        ):
            raise WorktreeError("SOURCE_PATH_BLOCKED", "Use a repository-relative path.", relative)
        name = parts[-1].casefold()
        if declared and (
            name == ".env"
            or name.startswith(".env.")
            or "credential" in name
            or name.endswith((".env", ".pem", ".key"))
        ):
            raise WorktreeError("SOURCE_PATH_BLOCKED", "The source filename is blocked.", relative)

        # Use directory-relative opens on POSIX to prevent parent symlink races.
        descriptors: list[int] = []
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        target = self.root.joinpath(*parts)
        try:
            current = self.root
            for part in parts:
                current /= part
                info = current.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", False):
                    raise WorktreeError(
                        "SOURCE_PATH_BLOCKED", "Symlink sources are blocked.", relative
                    )
            if not stat.S_ISREG(info.st_mode):
                raise WorktreeError("SOURCE_PATH_BLOCKED", "Select a regular file.", relative)
            if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
                dir_flags = flags | os.O_DIRECTORY
                descriptors.append(os.open(self.root, dir_flags))
                for part in parts[:-1]:
                    descriptors.append(os.open(part, dir_flags, dir_fd=descriptors[-1]))
                fd = os.open(parts[-1], flags, dir_fd=descriptors[-1])
            else:
                if not target.resolve().is_relative_to(self.root):
                    raise WorktreeError(
                        "SOURCE_PATH_BLOCKED", "The source is outside the worktree.", relative
                    )
                fd = os.open(target, flags | getattr(os, "O_BINARY", 0))
            with os.fdopen(fd, "rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or not os.path.samestat(info, before):
                    raise WorktreeError(
                        "WORKTREE_CHANGED", "The source changed before opening.", relative
                    )
                if before.st_size > limit:
                    raise WorktreeError(
                        "SOURCE_TOO_LARGE", "The source exceeds the byte budget.", relative
                    )
                contents = source.read(limit + 1)
                after = target.lstat()
                if (
                    len(contents) > limit
                    or not os.path.samestat(before, after)
                    or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                ):
                    raise WorktreeError(
                        "WORKTREE_CHANGED", "The source changed during reading.", relative
                    )
                return contents
        except OSError as exc:
            if isinstance(exc, FileNotFoundError):
                raise
            raise WorktreeError(
                "SOURCE_UNREADABLE", "Could not read the source file.", relative
            ) from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def digest(self, path: str) -> str | None:
        try:
            return hashlib.sha256(self.read(path)).hexdigest()
        except FileNotFoundError:
            return None
        except WorktreeError as exc:
            if exc.status == "SOURCE_PATH_BLOCKED":
                return None
            raise
