"""Unit tests for KacheDB MCP workspace and project discovery engine."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from kachedb_mcp.indexer.discovery import (
    DEFAULT_IGNORED_DIRS,
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


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with initial commit."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)

    # Initial file and commit
    (repo_dir / "README.md").write_text("# Test Repo", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True)

    return repo_dir


def test_project_info_workspace_id() -> None:
    """Verify workspace_id normalization in ProjectInfo."""
    proj = ProjectInfo(
        path=Path("/tmp/my_project"),
        name="My_Awesome Project",
        is_git=True,
    )
    assert proj.workspace_id == "my-awesome-project"


def test_find_git_root(temp_git_repo: Path, tmp_path: Path) -> None:
    """Verify find_git_root locates .git at root and from nested subdirectories."""
    assert find_git_root(temp_git_repo) == temp_git_repo

    nested = temp_git_repo / "src" / "deep" / "nested"
    nested.mkdir(parents=True)
    assert find_git_root(nested) == temp_git_repo

    non_git = tmp_path / "plain_folder"
    non_git.mkdir()
    assert find_git_root(non_git) is None


def test_get_head_commit_normal_and_detached(temp_git_repo: Path) -> None:
    """Verify get_head_commit returns valid branch name and 40-char SHA."""
    branch, sha = get_head_commit(temp_git_repo)
    assert branch in ("main", "master")
    assert sha is not None
    assert len(sha) == 40

    # Test detached HEAD state
    subprocess.run(["git", "checkout", sha], cwd=temp_git_repo, check=True, capture_output=True)
    _detached_branch, detached_sha = get_head_commit(temp_git_repo)
    assert detached_sha == sha

    # Test on non-git directory
    assert get_head_commit(temp_git_repo.parent / "non_existent") == (None, None)


def test_get_git_diff_files(temp_git_repo: Path) -> None:
    """Verify get_git_diff_files extracts exact modified files between commits."""
    _, initial_sha = get_head_commit(temp_git_repo)
    assert initial_sha is not None

    # Make second commit with new and modified files
    (temp_git_repo / "src").mkdir()
    (temp_git_repo / "src" / "lib.rs").write_text("pub fn run() {}", encoding="utf-8")
    (temp_git_repo / "README.md").write_text("# Updated README", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Second commit"], cwd=temp_git_repo, check=True)

    _, second_sha = get_head_commit(temp_git_repo)
    assert second_sha is not None
    assert second_sha != initial_sha

    changed = get_git_diff_files(temp_git_repo, initial_sha, second_sha)
    assert changed is not None
    assert sorted(changed) == ["README.md", "src/lib.rs"]

    # Invalid SHAs return None
    assert get_git_diff_files(temp_git_repo, "", second_sha) is None


def test_load_gitignore_and_is_path_ignored(tmp_path: Path) -> None:
    """Verify .gitignore parsing and default directory exclusion."""
    repo = tmp_path / "repo"
    repo.mkdir()

    gitignore_content = "\n".join(
        [
            "# Comments should be ignored",
            "*.log",
            "temp/",
            "!important.log",
        ]
    )
    (repo / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    spec = load_gitignore(repo)
    assert spec is not None

    # Check gitignore matching
    assert is_path_ignored("debug.log", spec) is True
    assert is_path_ignored("important.log", spec) is False
    assert is_path_ignored("temp/cache.json", spec) is True
    assert is_path_ignored("src/main.rs", spec) is False

    # Check hardcoded default directories (ignored even without .gitignore)
    for ignored_dir in DEFAULT_IGNORED_DIRS:
        assert is_path_ignored(f"{ignored_dir}/file.txt", None) is True
        assert is_path_ignored(f"nested/{ignored_dir}/file.txt", None) is True

    # Non-existent .gitignore returns None
    assert load_gitignore(tmp_path / "no_git_dir") is None


def test_discover_projects_single_root_git(temp_git_repo: Path) -> None:
    """Verify discover_projects detects a single root git repository."""
    (temp_git_repo / "Cargo.toml").write_text('[package]\nname = "test"', encoding="utf-8")
    projects = discover_projects(temp_git_repo)

    assert len(projects) == 1
    p = projects[0]
    assert p.path == temp_git_repo
    assert p.is_git is True
    assert p.git_root == temp_git_repo
    assert p.manifest_type == "rust"
    assert p.commit_sha is not None


def test_discover_projects_single_root_manifest(tmp_path: Path) -> None:
    """Verify discover_projects detects a standalone project with manifest but no git."""
    proj_dir = tmp_path / "python_pkg"
    proj_dir.mkdir()
    (proj_dir / "pyproject.toml").write_text('[project]\nname = "pkg"', encoding="utf-8")

    projects = discover_projects(proj_dir)
    assert len(projects) == 1
    assert projects[0].name == "python_pkg"
    assert projects[0].manifest_type == "python"
    assert projects[0].is_git is False


def test_discover_projects_multi_folder_workspace(tmp_path: Path) -> None:
    """Verify multi-folder workspace discovery correctly includes codebases and filters scratch."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # 1. Project A: Git repository
    proj_a = workspace / "project_a"
    proj_a.mkdir()
    (proj_a / ".git").mkdir()
    (proj_a / "Cargo.toml").write_text('[package]\nname="a"', encoding="utf-8")

    # 2. Project B: Go project with manifest (no .git)
    proj_b = workspace / "project_b"
    proj_b.mkdir()
    (proj_b / "go.mod").write_text("module example.com/b\n\ngo 1.22", encoding="utf-8")

    # 3. Scratch directory: No git, no manifest, only text notes
    scratch = workspace / "my_notes"
    scratch.mkdir()
    (scratch / "meeting_notes.txt").write_text("hello", encoding="utf-8")

    # 4. Ignored directory: node_modules
    nm = workspace / "node_modules"
    nm.mkdir()

    # 5. Hidden directory: .config
    hidden = workspace / ".config"
    hidden.mkdir()

    projects = discover_projects(workspace)
    project_names = [p.name for p in projects]

    assert "project_a" in project_names
    assert "project_b" in project_names
    assert "my_notes" not in project_names
    assert "node_modules" not in project_names
    assert ".config" not in project_names


def test_discover_projects_invalid_path() -> None:
    """Verify discover_projects gracefully handles non-existent paths."""
    assert discover_projects(Path("/non/existent/path/xyz_123")) == []


def test_git_engine_class_direct(temp_git_repo: Path) -> None:
    """Verify GitEngine class instantiation and methods directly."""
    engine = GitEngine(temp_git_repo)
    assert engine.git_root == temp_git_repo.resolve()

    # gitignore property caching
    assert engine.gitignore_spec is None
    (temp_git_repo / ".gitignore").write_text("*.tmp\nbuild/\n", encoding="utf-8")
    # Fresh engine to read .gitignore
    fresh_engine = GitEngine(temp_git_repo)
    assert fresh_engine.gitignore_spec is not None
    assert fresh_engine.is_ignored("test.tmp") is True
    assert fresh_engine.is_ignored("build/artifact.bin") is True
    assert fresh_engine.is_ignored("src/lib.rs") is False


def test_workspace_discovery_class_direct(tmp_path: Path) -> None:
    """Verify WorkspaceDiscovery class instantiation and customization."""
    custom_ignored = frozenset({"vendor", "custom_scratch"})
    disco = WorkspaceDiscovery(
        workspace_root=tmp_path,
        ignored_dirs=custom_ignored,
    )
    assert disco.workspace_root == tmp_path.resolve()
    assert disco.ignored_dirs == custom_ignored
    assert disco.discover() == []


def test_git_worktree_submodule_pointer(temp_git_repo: Path, tmp_path: Path) -> None:
    """Verify get_head_commit handles git worktrees where .git is a file pointer."""
    worktree_dir = tmp_path / "worktree_branch"
    subprocess.run(
        ["git", "worktree", "add", "-b", "feature-worktree", str(worktree_dir)],
        cwd=temp_git_repo,
        check=True,
        capture_output=True,
    )
    assert (worktree_dir / ".git").is_file()
    engine = GitEngine(worktree_dir)
    branch, sha = engine.get_head_commit()
    assert branch == "feature-worktree"
    assert sha is not None
    assert len(sha) == 40


def test_git_head_fast_read_fallback(temp_git_repo: Path) -> None:
    """Verify fallback to CLI when fast HEAD file reading raises an exception."""
    engine = GitEngine(temp_git_repo)
    head_file = temp_git_repo / ".git" / "HEAD"
    assert head_file.is_file()

    with patch.object(Path, "read_text", side_effect=OSError("Read error")):
        branch, sha = engine.get_head_commit()
        # Fallback to CLI still succeeds on valid repo
        assert branch in ("main", "master")
        assert sha is not None


def test_git_cli_fallback_direct(temp_git_repo: Path, tmp_path: Path) -> None:
    """Verify _get_head_via_cli directly on git and non-git dirs."""
    engine = GitEngine(temp_git_repo)
    branch, sha = engine._get_head_via_cli()
    assert branch in ("main", "master")
    assert sha is not None

    non_git_engine = GitEngine(tmp_path / "empty")
    assert non_git_engine._get_head_via_cli() == (None, None)


def test_get_diff_files_error_handling(temp_git_repo: Path) -> None:
    """Verify get_diff_files gracefully handles failed git diff calls."""
    engine = GitEngine(temp_git_repo)
    # Non-existent commits cause git diff to return non-zero exit code
    assert engine.get_diff_files("0000000000000000000000000000000000000000", "HEAD") is None


def test_load_gitignore_error_handling(temp_git_repo: Path) -> None:
    """Verify load_gitignore gracefully handles read failures."""
    gitignore = temp_git_repo / ".gitignore"
    gitignore.write_text("*.log\n", encoding="utf-8")
    engine = GitEngine(temp_git_repo)

    with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
        assert engine.load_gitignore() is None


def test_has_source_files_depth_and_error(tmp_path: Path) -> None:
    """Verify max_depth pruning and error handling in has_source_files."""
    disco = WorkspaceDiscovery(tmp_path)

    # 1. Deep file beyond max_depth=1
    deep_dir = tmp_path / "a" / "b" / "c"
    deep_dir.mkdir(parents=True)
    (deep_dir / "deep.py").write_text("print(1)", encoding="utf-8")
    assert disco.has_source_files(tmp_path, max_depth=1) is False
    assert disco.has_source_files(tmp_path, max_depth=3) is True

    # 2. Exception in os.walk
    with patch("os.walk", side_effect=PermissionError("Scan failed")):
        assert disco.has_source_files(tmp_path) is False


def test_discover_subdirectories_error_handling(tmp_path: Path) -> None:
    """Verify discover() handles exceptions while listing directory entries."""
    disco = WorkspaceDiscovery(tmp_path)
    with patch.object(Path, "iterdir", side_effect=PermissionError("Access denied")):
        assert disco.discover() == []


def test_discover_projects_manifest_with_parent_git(temp_git_repo: Path) -> None:
    """Verify Case 1b: Project with manifest whose parent has git."""
    sub_pkg = temp_git_repo / "crates" / "my_crate"
    sub_pkg.mkdir(parents=True)
    (sub_pkg / "Cargo.toml").write_text('[package]\nname = "my_crate"', encoding="utf-8")

    disco = WorkspaceDiscovery(sub_pkg)
    projects = disco.discover()
    assert len(projects) == 1
    p = projects[0]
    assert p.path == sub_pkg
    assert p.is_git is True
    assert p.git_root == temp_git_repo
    assert p.manifest_type == "rust"


def test_discover_refuses_home_and_root_directories() -> None:
    """Safety guarantee: WorkspaceDiscovery refuses to treat $HOME or root as a codebase."""
    home = Path.home()
    disco_home = WorkspaceDiscovery(home)
    assert disco_home.discover() == []

    disco_root = WorkspaceDiscovery("/")
    assert disco_root.discover() == []


def test_default_ignored_dirs_contains_cloud_and_system_folders() -> None:
    """Safety guarantee: iCloud Drive, Mobile Documents, and Library are in DEFAULT_IGNORED_DIRS."""
    assert "Library" in DEFAULT_IGNORED_DIRS
    assert "Mobile Documents" in DEFAULT_IGNORED_DIRS
    assert "iCloud Drive" in DEFAULT_IGNORED_DIRS
    assert "Applications" in DEFAULT_IGNORED_DIRS
