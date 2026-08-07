"""Regression tests for repository secret and generated-data hygiene."""

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _active_rules(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_dockerignore_excludes_production_environment_files() -> None:
    rules = _active_rules(PROJECT_ROOT / ".dockerignore")
    required = {"config.env", ".env", ".env.*", "*.env"}

    assert required <= set(rules), (
        f"Missing Docker secret exclusions: {required - set(rules)}"
    )
    assert "!config.env" not in rules


def test_generated_databases_are_ignored_and_env_template_is_tracked() -> None:
    generated_paths = (
        ".pytest-tmp/example.db",
        "data/evaluation.remote.db",
        "data/evaluation.remote.db-shm",
        "data/evaluation.remote.db-wal",
    )

    for generated_path in generated_paths:
        result = _git("check-ignore", "-q", "--", generated_path)
        assert result.returncode == 0, f"Git does not ignore {generated_path}"

    tracked_template = _git(
        "ls-files", "--error-unmatch", "config.env.example"
    )
    assert tracked_template.returncode == 0, (
        "config.env.example must remain tracked"
    )
