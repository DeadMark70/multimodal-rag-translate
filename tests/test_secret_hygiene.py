"""Regression tests for repository secret and generated-data hygiene."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
