"""CLI entrypoint for `kachedb-mcp install`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .aidlc import configure_aidlc
from .engine import InstallPlan, write_text_if_changed
from .frameworks import FRAMEWORK_CHOICES, detect_frameworks, get_adapter, list_adapters
from .frameworks.base import PROFILE_CHOICES, InstallOptions
from .ide import IDE_CHOICES, configure_ide
from .templates import render_workspace_memory_rules

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .frameworks.base import FrameworkAdapter


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ``kachedb-mcp install``."""
    parser = argparse.ArgumentParser(
        prog="kachedb-mcp install",
        description="Configure KacheDB bounded memory for workflow frameworks and IDEs.",
    )
    parser.add_argument(
        "--framework",
        choices=FRAMEWORK_CHOICES,
        default=None,
        help="Configure bounded retrieval rules for a workflow framework ('aidlc', 'auto', 'all').",
    )
    parser.add_argument(
        "--aidlc",
        action="store_true",
        help="Alias for --framework aidlc.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=None,
        help=(
            "Token/ceremony profile ('preserve', 'balanced', 'minimal', 'full'). "
            "Preserves existing settings when configured; defaults to 'balanced' on fresh repos."
        ),
    )
    parser.add_argument(
        "--ide",
        choices=IDE_CHOICES,
        default=None,
        help="Register kachedb MCP server for the target IDE/CLI (or 'all').",
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Target workspace root directory (default: current directory).",
    )
    parser.add_argument(
        "--space",
        default="default",
        help="Framework space name under aidlc/spaces/<space>/ (default: 'default').",
    )
    parser.add_argument(
        "--patch-agents",
        action="store_true",
        help="Also patch framework agent prompts (e.g. .codex/agents/*.{md,toml}).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="KacheDB server host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6380,
        help="KacheDB server port (default: 6380).",
    )
    parser.add_argument(
        "--embedder",
        choices=("auto", "embeddinggemma-2", "fastembed"),
        default="auto",
        help="Embedding provider written to MCP config env (default: auto).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to local ONNX model directory (model.onnx + tokenizer.json).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=(384, 768),
        default=384,
        help="Matryoshka embedding dimension (default: 384).",
    )
    parser.add_argument(
        "--workspace-roots",
        default=None,
        help="OS-delimited root paths (':' Unix, ';' Windows) for KACHEDB_WORKSPACE_ROOTS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned changes without writing files.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive prompts and use non-interactive default profile.",
    )
    return parser


def _select_adapters(framework: str | None, aidlc_flag: bool, root: Path) -> list[FrameworkAdapter]:
    """Resolve requested framework adapters preserving order without duplicates."""
    selected: dict[str, FrameworkAdapter] = {}
    if aidlc_flag or framework == "aidlc":
        adapter = get_adapter("aidlc")
        selected[adapter.name] = adapter
    if framework == "all":
        for adapter in list_adapters():
            selected[adapter.name] = adapter
    elif framework == "auto":
        for adapter in detect_frameworks(root):
            selected[adapter.name] = adapter
    return list(selected.values())


def run_install_cli(argv: Sequence[str] | None = None) -> int:
    """Execute ``kachedb-mcp install`` and return a shell exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.framework is None and not args.aidlc and args.ide is None:
        parser.print_help(sys.stderr)
        return 2

    root = Path(args.path).expanduser().resolve()
    plan = InstallPlan()

    try:
        if args.framework is not None or args.aidlc:
            adapters = _select_adapters(args.framework, args.aidlc, root)
            if not adapters and args.framework == "auto":
                ignored_flags: list[str] = []
                if args.profile is not None:
                    ignored_flags.append(f"--profile {args.profile}")
                if args.patch_agents:
                    ignored_flags.append("--patch-agents")
                ignored_note = f" ({', '.join(ignored_flags)} skipped)" if ignored_flags else ""
                plan.warnings.append(
                    f"No initialized workflow frameworks detected in {root}{ignored_note}. "
                    "Pass --aidlc or --framework <name> to force setup."
                )
                plan.add(
                    write_text_if_changed(
                        root / ".agents" / "rules" / "kachedb_memory.md",
                        render_workspace_memory_rules(space=args.space),
                        dry_run=args.dry_run,
                        detail="workspace KacheDB caching rules",
                    )
                )
            else:
                opts = InstallOptions(
                    root=root,
                    profile=args.profile,
                    space=args.space,
                    patch_agents=args.patch_agents,
                    dry_run=args.dry_run,
                    yes=args.yes,
                )
                for adapter in adapters:
                    plan.extend(adapter.configure(opts))

        if args.ide is not None:
            plan.extend(
                configure_ide(
                    target=args.ide,
                    workspace_root=root,
                    host=args.host,
                    port=args.port,
                    embedder=args.embedder,
                    model_path=args.model_path,
                    dim=args.dim,
                    workspace_roots=args.workspace_roots,
                    dry_run=args.dry_run,
                )
            )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(plan.format_summary(dry_run=args.dry_run))
    return 0


__all__ = [
    "build_parser",
    "configure_aidlc",
    "configure_ide",
    "detect_frameworks",
    "get_adapter",
    "list_adapters",
    "run_install_cli",
]
