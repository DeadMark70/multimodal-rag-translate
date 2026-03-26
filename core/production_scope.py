"""Production-scope helpers for static architecture enforcement."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NON_PRODUCTION_ROOTS = {
    ".venv",
    "venv",
    "tests",
    "docs",
    "checklist",
    "agentlog",
    "bergen",
    "experiments",
    "scripts",
    "__pycache__",
}


def is_production_path(path: Path, *, project_root: Path = PROJECT_ROOT) -> bool:
    """Return True when the file path is inside the production backend tree."""
    relative = path.resolve().relative_to(project_root.resolve())
    return not any(part in NON_PRODUCTION_ROOTS for part in relative.parts)


def iter_production_python_files(*, project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Return backend Python files that belong to the production runtime tree."""
    files: list[Path] = []
    for path in project_root.rglob("*.py"):
        if is_production_path(path, project_root=project_root):
            files.append(path)
    return files
