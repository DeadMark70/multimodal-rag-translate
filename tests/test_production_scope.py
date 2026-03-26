from __future__ import annotations

import ast
from pathlib import Path

from core.production_scope import PROJECT_ROOT, iter_production_python_files

NON_PRODUCTION_IMPORT_PREFIXES = ("bergen", "experiments", "scripts")
ORPHAN_MODULE_PATHS = (
    PROJECT_ROOT / "pdfserviceMD" / "markdown_to_pdf.py",
    PROJECT_ROOT / "multimodal_rag" / "utils.py",
)


def _parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_production_modules_do_not_import_non_production_trees() -> None:
    violations: list[str] = []

    for path in iter_production_python_files():
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue

            for name in names:
                if not name:
                    continue
                if any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in NON_PRODUCTION_IMPORT_PREFIXES
                ):
                    violations.append(f"{path}: imports {name}")

    assert (
        not violations
    ), "Production modules must not import non-production trees:\n" + "\n".join(violations)


def test_high_confidence_orphan_modules_are_removed() -> None:
    remaining = [path for path in ORPHAN_MODULE_PATHS if path.exists()]
    assert not remaining, f"Expected orphan modules to be removed: {remaining}"
