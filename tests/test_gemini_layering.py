from __future__ import annotations

import ast
from pathlib import Path

from core.production_scope import iter_production_python_files

PROJECT_ROOT = Path(__file__).parent.parent

ALLOWED_GENAI_MODULES = {
    PROJECT_ROOT / "core" / "google_genai_client.py",
    PROJECT_ROOT / "evaluation" / "model_discovery.py",
}
ALLOWED_CHAT_MODEL_MODULES = {
    PROJECT_ROOT / "core" / "llm_factory.py",
}
ALLOWED_EMBEDDING_MODULES = {
    PROJECT_ROOT / "data_base" / "vector_store_manager.py",
}
ALLOWED_LLM_FACTORY_IMPORTERS = {
    PROJECT_ROOT / "core" / "providers.py",
}
def _iter_production_python_files() -> list[Path]:
    return iter_production_python_files(project_root=PROJECT_ROOT)


def _parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_direct_google_genai_usage_is_limited_to_control_plane_modules() -> None:
    violations: list[str] = []

    for path in _iter_production_python_files():
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "google" and any(
                    alias.name == "genai" for alias in node.names
                ):
                    if path not in ALLOWED_GENAI_MODULES:
                        violations.append(f"{path}: from google import genai")
                if node.module and node.module.startswith("google.genai"):
                    if path not in ALLOWED_GENAI_MODULES:
                        violations.append(f"{path}: from {node.module} import ...")
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "genai"
                    and node.func.attr == "Client"
                    and path not in ALLOWED_GENAI_MODULES
                ):
                    violations.append(f"{path}: genai.Client(...)")

    assert not violations, "Unexpected direct google-genai usage:\n" + "\n".join(violations)


def test_runtime_chat_and_embedding_construction_stays_centralized() -> None:
    violations: list[str] = []

    for path in _iter_production_python_files():
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if (
                    node.func.id == "ChatGoogleGenerativeAI"
                    and path not in ALLOWED_CHAT_MODEL_MODULES
                ):
                    violations.append(f"{path}: ChatGoogleGenerativeAI(...)")
                if (
                    node.func.id == "GoogleGenerativeAIEmbeddings"
                    and path not in ALLOWED_EMBEDDING_MODULES
                ):
                    violations.append(f"{path}: GoogleGenerativeAIEmbeddings(...)")

    assert (
        not violations
    ), "Unexpected runtime Gemini construction outside approved modules:\n" + "\n".join(violations)


def test_business_logic_does_not_import_get_llm_from_llm_factory() -> None:
    violations: list[str] = []

    for path in _iter_production_python_files():
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "core.llm_factory":
                imported_names = {alias.name for alias in node.names}
                if "get_llm" in imported_names and path not in ALLOWED_LLM_FACTORY_IMPORTERS:
                    violations.append(f"{path}: from core.llm_factory import get_llm")

    assert (
        not violations
    ), "Unexpected direct get_llm imports from core.llm_factory:\n" + "\n".join(violations)
