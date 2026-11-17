"""AWS AI-DLC framework adapter with user-selectable token/ceremony profiles."""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from typing import TYPE_CHECKING, Any

from ..engine import (
    InstallAction,
    InstallPlan,
    merge_json_file,
    upsert_marked_block,
    upsert_under_h2_section,
    write_text_if_changed,
)
from ..templates import (
    BEGIN_AGENT_MARKER,
    BEGIN_AIDLC_MARKER,
    BEGIN_FORBIDDEN_MARKER,
    BEGIN_MANDATED_MARKER,
    END_AGENT_MARKER,
    END_AIDLC_MARKER,
    END_FORBIDDEN_MARKER,
    END_MANDATED_MARKER,
    render_agent_directives_block,
    render_construction_phase_block,
    render_inception_phase_block,
    render_project_forbidden_block,
    render_project_mandated_block,
    render_project_policy_section,
    render_shared_knowledge,
    render_team_forbidden_block,
    render_team_mandated_block,
    render_team_way_of_working_block,
    render_workspace_memory_rules,
)
from .base import InstallOptions, InstallProfile

if TYPE_CHECKING:
    from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]

AIDLC_AGENT_TARGETS: tuple[str, ...] = (
    "aidlc-architect-agent",
    "aidlc-architecture-reviewer-agent",
    "aidlc-composer-agent",
    "aidlc-developer-agent",
    "aidlc-quality-agent",
)

# Maps profile -> (models.preset, flags.defaultScope) or None for preserve
PROFILE_SETTINGS: dict[InstallProfile, tuple[str, str] | None] = {
    "preserve": None,
    "balanced": ("minimal", "classic"),
    "minimal": ("minimal", "express"),
    "full": ("standard", "feature"),
}

_TOML_INSTRUCTIONS_RE = re.compile(
    r'^(developer_instructions[ \t]*=[ \t]*)("""|\'\'\')',
    re.MULTILINE,
)

_PROMPT_CHOICES: dict[str, InstallProfile] = {
    "": "balanced",
    "1": "balanced",
    "balanced": "balanced",
    "2": "minimal",
    "minimal": "minimal",
    "3": "full",
    "full": "full",
    "4": "preserve",
    "preserve": "preserve",
}


def _has_existing_aidlc_settings(settings_path: Path) -> bool:
    """Return True if ``aidlc.settings.json`` already sets ``preset`` or ``defaultScope``.

    Raises ``ValueError`` if the file exists but contains invalid JSON or a non-object.
    """
    if not settings_path.is_file():
        return False
    raw = settings_path.read_text(encoding="utf-8")
    if not raw.strip():
        return False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {settings_path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in {settings_path}, got {type(parsed).__name__}")
    models = parsed.get("models")
    flags = parsed.get("flags")
    has_preset = isinstance(models, dict) and bool(models.get("preset"))
    has_scope = isinstance(flags, dict) and bool(flags.get("defaultScope"))
    return has_preset or has_scope


def _prompt_profile_choice() -> InstallProfile:
    """Prompt interactively on TTY (all output to stderr) for the desired AI-DLC profile."""
    print(
        "Select AWS AI-DLC profile for KacheDB:\n"
        "  1) balanced  — classic 18-stage, advisory review + sensors, minimal effort (Default)\n"
        "  2) minimal   — express 7-stage, reviewers/sensors off (max ~95% token savings)\n"
        "  3) full      — feature 33-stage, adversarial review + bounded retrieval\n"
        "  4) preserve  — inject KacheDB bounded retrieval rules without changing settings",
        file=sys.stderr,
    )
    sys.stderr.write("Choice [1-4, default=1]: ")
    sys.stderr.flush()
    try:
        raw = input().strip().lower()
    except EOFError:
        return "balanced"
    except KeyboardInterrupt as exc:
        raise ValueError("Installation cancelled by user.") from exc

    selected = _PROMPT_CHOICES.get(raw)
    if selected is None:
        print(f"Unknown choice '{raw}'; defaulting to 'balanced'.", file=sys.stderr)
        return "balanced"
    return selected


def resolve_aidlc_profile(opts: InstallOptions) -> InstallProfile:
    """Determine the effective profile without silently downgrading existing settings."""
    if opts.profile is not None:
        return opts.profile
    settings_path = opts.root.expanduser().resolve() / "aidlc.settings.json"
    if _has_existing_aidlc_settings(settings_path):
        return "preserve"
    if not opts.dry_run and not opts.yes and sys.stdin.isatty():
        return _prompt_profile_choice()
    return "balanced"


def _build_settings_updater(preset: str, default_scope: str) -> Any:
    """Return a JSON merger setting AI-DLC ``preset`` and ``defaultScope``."""

    def _merge(existing: dict[str, Any]) -> dict[str, Any]:
        merged = dict(existing)
        merged["schemaVersion"] = merged.get("schemaVersion", 1)

        raw_models = merged.get("models")
        models = dict(raw_models) if isinstance(raw_models, dict) else {}
        models["schemaVersion"] = models.get("schemaVersion", 1)
        models["preset"] = preset
        merged["models"] = models

        raw_flags = merged.get("flags")
        flags = dict(raw_flags) if isinstance(raw_flags, dict) else {}
        flags["schemaVersion"] = flags.get("schemaVersion", 1)
        flags["defaultScope"] = default_scope
        merged["flags"] = flags

        return merged

    return _merge


def _patch_toml_instructions(raw_toml: str, directive_block: str) -> str | None:
    """Upsert marked directives inside TOML `developer_instructions` multiline string."""
    start_match = _TOML_INSTRUCTIONS_RE.search(raw_toml)
    if start_match is None:
        return None

    delim = start_match.group(2)
    body_start = start_match.end()

    close_match = re.search(
        rf"\n{re.escape(delim)}(?=[ \t]*(?:\n|#|$))",
        raw_toml[body_start:],
    )
    if close_match is not None:
        close_idx = body_start + close_match.start()
        prefix_newline = "\n"
    else:
        fallback_match = re.search(rf"(?<!\\){re.escape(delim)}", raw_toml[body_start:])
        if fallback_match is None:
            return None
        close_idx = body_start + fallback_match.start()
        prefix_newline = ""

    inner = raw_toml[body_start:close_idx]
    escaped_block = (
        directive_block.replace("\\", "\\\\").replace('"""', '\\"""')
        if delim == '"""'
        else directive_block.replace("'''", "")
    )
    updated_inner = upsert_marked_block(
        inner,
        escaped_block,
        BEGIN_AGENT_MARKER,
        END_AGENT_MARKER,
    )
    if not updated_inner.startswith("\n"):
        updated_inner = "\n" + updated_inner
    updated_inner = updated_inner.rstrip("\n") + prefix_newline

    patched = f"{raw_toml[:body_start]}{updated_inner}{raw_toml[close_idx + len(prefix_newline) :]}"
    if tomllib is not None:
        try:
            tomllib.loads(patched)
        except Exception as exc:
            raise ValueError(f"Generated invalid TOML after patching: {exc}") from exc
    return patched


def _patch_agents_dir(
    agents_dir: Path,
    space: str,
    dry_run: bool,
    plan: InstallPlan,
) -> None:
    """Inject KacheDB directives into `.codex/agents/*.{md,toml}` if present."""
    if not agents_dir.is_dir():
        plan.warnings.append(f"Agent directory not found at {agents_dir}; skipping --patch-agents.")
        return

    directive_block = render_agent_directives_block(space=space)
    for agent_name in AIDLC_AGENT_TARGETS:
        md_path = agents_dir / f"{agent_name}.md"
        if md_path.is_file():
            existing_md = md_path.read_text(encoding="utf-8")
            updated_md = upsert_marked_block(
                existing_md,
                directive_block,
                BEGIN_AGENT_MARKER,
                END_AGENT_MARKER,
            )
            plan.add(
                write_text_if_changed(
                    md_path,
                    updated_md,
                    dry_run=dry_run,
                    detail="agent bounded retrieval directives",
                )
            )

        toml_path = agents_dir / f"{agent_name}.toml"
        if toml_path.is_file():
            existing_toml = toml_path.read_text(encoding="utf-8")
            patched_toml = _patch_toml_instructions(existing_toml, directive_block)
            if patched_toml is None:
                plan.add(
                    InstallAction(
                        path=toml_path,
                        status="skipped",
                        detail="no developer_instructions multiline block found",
                    )
                )
            else:
                plan.add(
                    write_text_if_changed(
                        toml_path,
                        patched_toml,
                        dry_run=dry_run,
                        detail="TOML developer_instructions bounded directives",
                    )
                )


def _insert_new_h2_section(
    content: str,
    heading: str,
    block: str,
    begin_marker: str,
    end_marker: str,
    insert_before: str | None,
) -> str:
    """Create a missing H2 section containing a marker-bounded block."""
    wrapped = f"{begin_marker}\n{block.strip()}\n{end_marker}"
    new_section = f"{heading}\n\n{wrapped}"
    if insert_before:
        match = re.search(rf"^{re.escape(insert_before)}[ \t]*$", content, re.MULTILINE)
        if match is not None:
            before = content[: match.start()].rstrip("\n")
            after = content[match.start() :].lstrip("\n")
            sep = "\n\n" if before else ""
            return f"{before}{sep}{new_section}\n\n{after}"
    base = content.rstrip("\n")
    return f"{base}\n\n{new_section}\n" if base else f"{new_section}\n"


def _apply_memory_rules(
    existing: str,
    *,
    forbidden_block: str,
    mandated_block: str,
    way_of_working_block: str | None = None,
    policy_section: str | None = None,
) -> str:
    """Inject Forbidden, Mandated, and policy blocks under their owning H2 sections."""
    updated = existing
    if way_of_working_block is not None:
        placed = upsert_under_h2_section(
            updated,
            "## Way of Working",
            way_of_working_block,
            BEGIN_AIDLC_MARKER,
            END_AIDLC_MARKER,
        )
        updated = (
            placed
            if placed is not None
            else _insert_new_h2_section(
                updated,
                "## Way of Working",
                way_of_working_block,
                BEGIN_AIDLC_MARKER,
                END_AIDLC_MARKER,
                "## Forbidden",
            )
        )

    placed_forbidden = upsert_under_h2_section(
        updated,
        "## Forbidden",
        forbidden_block,
        BEGIN_FORBIDDEN_MARKER,
        END_FORBIDDEN_MARKER,
    )
    updated = (
        placed_forbidden
        if placed_forbidden is not None
        else _insert_new_h2_section(
            updated,
            "## Forbidden",
            forbidden_block,
            BEGIN_FORBIDDEN_MARKER,
            END_FORBIDDEN_MARKER,
            "## Mandated",
        )
    )

    placed_mandated = upsert_under_h2_section(
        updated,
        "## Mandated",
        mandated_block,
        BEGIN_MANDATED_MARKER,
        END_MANDATED_MARKER,
    )
    updated = (
        placed_mandated
        if placed_mandated is not None
        else _insert_new_h2_section(
            updated,
            "## Mandated",
            mandated_block,
            BEGIN_MANDATED_MARKER,
            END_MANDATED_MARKER,
            "## Corrections",
        )
    )

    if policy_section is not None:
        updated = upsert_marked_block(
            updated,
            policy_section,
            BEGIN_AIDLC_MARKER,
            END_AIDLC_MARKER,
            insert_before="## Corrections",
        )

    return updated


def _run_aidlc_install(opts: InstallOptions) -> InstallPlan:
    """Execute AI-DLC configuration directly from resolved ``InstallOptions``."""
    resolved_root = opts.root.expanduser().resolve()
    space = opts.space
    dry_run = opts.dry_run
    effective_profile = resolve_aidlc_profile(opts)
    stamp = opts.affirmed_date or date.today().isoformat()
    plan = InstallPlan()

    # 1. Engine settings (`aidlc.settings.json`) according to chosen profile
    settings_path = resolved_root / "aidlc.settings.json"
    profile_pair = PROFILE_SETTINGS[effective_profile]
    if profile_pair is None:
        status_note = (
            "preserved existing settings (use --profile balanced|minimal|full to change)"
            if settings_path.exists()
            else "preserve profile keeps aidlc.settings.json untouched"
        )
        plan.add(
            InstallAction(
                path=settings_path,
                status="unchanged" if settings_path.exists() else "skipped",
                detail=status_note,
            )
        )
    else:
        preset, default_scope = profile_pair
        detail = f"{effective_profile} profile (preset={preset}, defaultScope={default_scope})"
        plan.add(
            merge_json_file(
                settings_path,
                _build_settings_updater(preset, default_scope),
                dry_run=dry_run,
                detail=detail,
            )
        )

    # 2. Preloaded shared knowledge (`aidlc/spaces/<space>/knowledge/aidlc-shared/`)
    shared_path = (
        resolved_root
        / "aidlc"
        / "spaces"
        / space
        / "knowledge"
        / "aidlc-shared"
        / "kachedb-bounded-retrieval.md"
    )
    plan.add(
        write_text_if_changed(
            shared_path,
            render_shared_knowledge(space=space),
            dry_run=dry_run,
            detail="shared knowledge preflight standard",
        )
    )

    # 3. Workspace `.agents/rules/kachedb_memory.md`
    workspace_rules_path = resolved_root / ".agents" / "rules" / "kachedb_memory.md"
    plan.add(
        write_text_if_changed(
            workspace_rules_path,
            render_workspace_memory_rules(space=space),
            dry_run=dry_run,
            detail="workspace KacheDB caching rules",
        )
    )

    # 4. Hierarchical AI-DLC memory files (`team.md`, `project.md`, `phases/*.md`)
    memory_dir = resolved_root / "aidlc" / "spaces" / space / "memory"
    if not memory_dir.is_dir():
        plan.warnings.append(
            f"AI-DLC memory directory not found at {memory_dir}. "
            "Run `aidlc config` in this workspace first, then re-run to patch memory/*.md."
        )
    else:
        team_path = memory_dir / "team.md"
        if team_path.is_file():
            updated_team = _apply_memory_rules(
                team_path.read_text(encoding="utf-8"),
                forbidden_block=render_team_forbidden_block(stamp),
                mandated_block=render_team_mandated_block(stamp),
                way_of_working_block=render_team_way_of_working_block(),
            )
            plan.add(
                write_text_if_changed(
                    team_path,
                    updated_team,
                    dry_run=dry_run,
                    detail="team-level bounded retrieval practices",
                )
            )
        else:
            plan.add(InstallAction(path=team_path, status="skipped", detail="file not initialized"))

        project_path = memory_dir / "project.md"
        if project_path.is_file():
            updated_project = _apply_memory_rules(
                project_path.read_text(encoding="utf-8"),
                forbidden_block=render_project_forbidden_block(stamp),
                mandated_block=render_project_mandated_block(stamp),
                policy_section=render_project_policy_section(space=space),
            )
            plan.add(
                write_text_if_changed(
                    project_path,
                    updated_project,
                    dry_run=dry_run,
                    detail="project-level affirmations & retrieval policy",
                )
            )
        else:
            plan.add(
                InstallAction(path=project_path, status="skipped", detail="file not initialized")
            )

        phase_targets: tuple[tuple[Path, str, str], ...] = (
            (
                memory_dir / "phases" / "construction.md",
                render_construction_phase_block(space=space),
                "construction phase unit isolation directives",
            ),
            (
                memory_dir / "phases" / "inception.md",
                render_inception_phase_block(space=space),
                "inception phase semantic pre-search directives",
            ),
        )
        for target_path, block, detail in phase_targets:
            if not target_path.is_file():
                plan.add(
                    InstallAction(path=target_path, status="skipped", detail="file not initialized")
                )
                continue
            updated = upsert_marked_block(
                target_path.read_text(encoding="utf-8"),
                block,
                BEGIN_AIDLC_MARKER,
                END_AIDLC_MARKER,
                insert_before="## Corrections",
            )
            plan.add(write_text_if_changed(target_path, updated, dry_run=dry_run, detail=detail))

    # 5. Optional `.codex/agents/` system prompt patching
    if opts.patch_agents:
        _patch_agents_dir(
            resolved_root / ".codex" / "agents",
            space=space,
            dry_run=dry_run,
            plan=plan,
        )

    return plan


def configure_aidlc(
    root: Path,
    space: str = "default",
    patch_agents: bool = False,
    dry_run: bool = False,
    affirmed_date: str | None = None,
    profile: InstallProfile | None = None,
    yes: bool = False,
) -> InstallPlan:
    """Configure AI-DLC engine defaults, shared knowledge, and hierarchical memory rules."""
    return _run_aidlc_install(
        InstallOptions(
            root=root,
            profile=profile,
            space=space,
            patch_agents=patch_agents,
            dry_run=dry_run,
            yes=yes,
            affirmed_date=affirmed_date,
        )
    )


class AidlcAdapter:
    """Framework adapter configuring AWS AI-DLC (`awslabs/aidlc-workflows`)."""

    name: str = "aidlc"

    def detect(self, root: Path) -> bool:
        """Return True when ``root/aidlc`` or ``root/aidlc.settings.json`` exists."""
        resolved = root.expanduser().resolve()
        return (resolved / "aidlc").is_dir() or (resolved / "aidlc.settings.json").is_file()

    def configure(self, opts: InstallOptions) -> InstallPlan:
        """Apply AI-DLC KacheDB bounded retrieval rules using ``opts``."""
        return _run_aidlc_install(opts)
