from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked_pytest_modules() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "tests/test_*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [
        path
        for line in result.stdout.splitlines()
        if line and (path := REPO_ROOT / line).is_file()
    ]


def _ignored_experiment_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        if any(name == "experiments" or name.startswith("experiments.") for name in names):
            violations.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
    return violations


def test_tracked_pytest_modules_do_not_import_ignored_experiments():
    violations = [
        violation
        for path in _tracked_pytest_modules()
        for violation in _ignored_experiment_imports(path)
    ]

    assert violations == [], (
        "tracked pytest modules must not depend on gitignored experiments/: "
        + ", ".join(violations)
    )
