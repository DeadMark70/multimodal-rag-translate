from __future__ import annotations

import ast
from pathlib import Path

from core.production_scope import iter_production_python_files

PROJECT_ROOT = Path(__file__).parent.parent
ROUTER_MODULES = {
    path
    for path in iter_production_python_files(project_root=PROJECT_ROOT)
    if path.name == "router.py"
}
UPLOAD_VALIDATION_ROUTERS = {
    PROJECT_ROOT / "pdfserviceMD" / "router.py",
    PROJECT_ROOT / "multimodal_rag" / "router.py",
}


def _parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_router_modules_do_not_import_other_router_modules() -> None:
    violations: list[str] = []

    for path in ROUTER_MODULES:
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith(".router"):
                violations.append(f"{path}: from {node.module} import ...")

    assert not violations, "Router-to-router imports are forbidden:\n" + "\n".join(violations)


def test_pdf_upload_validation_is_centralized() -> None:
    violations: list[str] = []

    for path in UPLOAD_VALIDATION_ROUTERS:
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_pdf_upload":
                violations.append(str(path))

    assert not violations, "Routers should use core.uploads.validate_pdf_upload:\n" + "\n".join(violations)
