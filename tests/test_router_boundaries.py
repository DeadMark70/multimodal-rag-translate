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
ROUTER_IMPORT_ALLOWED_MODULES = {
    PROJECT_ROOT / "core" / "app_factory.py",
}
UPLOAD_VALIDATION_ROUTERS = {
    PROJECT_ROOT / "pdfserviceMD" / "router.py",
    PROJECT_ROOT / "multimodal_rag" / "router.py",
}


def _parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_router_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.endswith(".router") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.endswith(".router"):
            return True
        return any(alias.name == "router" for alias in node.names)
    return False


def test_router_import_detection_covers_common_import_forms() -> None:
    tree = ast.parse(
        "\n".join(
            [
                "from graph_rag.router import router",
                "import graph_rag.router as graph_router",
                "from graph_rag import router",
                "from graph_rag.service import run_graph_extraction",
            ]
        )
    )

    results = [
        _is_router_import(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]

    assert results == [True, True, True, False]


def test_router_modules_do_not_import_other_router_modules() -> None:
    violations: list[str] = []

    for path in ROUTER_MODULES:
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if _is_router_import(node):
                violations.append(f"{path}: router import")

    assert not violations, "Router-to-router imports are forbidden:\n" + "\n".join(
        violations
    )


def test_non_router_production_modules_do_not_import_router_modules() -> None:
    violations: list[str] = []

    for path in iter_production_python_files(project_root=PROJECT_ROOT):
        if (
            path.name == "router.py"
            or path.name == "__init__.py"
            or path in ROUTER_IMPORT_ALLOWED_MODULES
        ):
            continue

        tree = _parse_file(path)
        for node in ast.walk(tree):
            if _is_router_import(node):
                violations.append(f"{path}: router import")

    assert not violations, "Non-router modules must not import routers:\n" + "\n".join(
        violations
    )


def test_router_modules_do_not_define_background_tasks() -> None:
    violations: list[str] = []

    for path in ROUTER_MODULES:
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name.endswith("_task"):
                violations.append(f"{path}: async def {node.name}")

    assert not violations, (
        "Routers should delegate background tasks to services:\n"
        + "\n".join(violations)
    )


def test_pdf_upload_validation_is_centralized() -> None:
    violations: list[str] = []

    for path in UPLOAD_VALIDATION_ROUTERS:
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_validate_pdf_upload"
            ):
                violations.append(str(path))

    assert not violations, (
        "Routers should use core.uploads.validate_pdf_upload:\n" + "\n".join(violations)
    )
