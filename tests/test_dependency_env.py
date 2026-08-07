import ast
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent


def _require_local_env_validation() -> None:
    if os.getenv("VALIDATE_LOCAL_ENV") != "1":
        pytest.skip("Set VALIDATE_LOCAL_ENV=1 to validate local env files")


def _parse_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys


def test_env_file_exists() -> None:
    """Verify that .env file exists."""
    _require_local_env_validation()
    env_path = PROJECT_ROOT / ".env"
    config_env_path = PROJECT_ROOT / "config.env"

    exists = env_path.exists() or config_env_path.exists()
    assert exists, (
        "Neither .env nor config.env found. "
        "Please copy config.env.example to .env or config.env"
    )


def test_env_keys_match_example() -> None:
    """Verify that .env has all keys from config.env.example."""
    _require_local_env_validation()
    example_path = PROJECT_ROOT / "config.env.example"
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        env_path = PROJECT_ROOT / "config.env"

    if not env_path.exists():
        pytest.skip("Skipping key check because env file does not exist")

    example_keys = _parse_env_keys(example_path)
    real_keys = _parse_env_keys(env_path)

    missing_keys = example_keys - real_keys
    assert not missing_keys, f"Missing keys in .env/config.env: {missing_keys}"


def test_local_env_validation_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VALIDATE_LOCAL_ENV", raising=False)

    with pytest.raises(pytest.skip.Exception):
        _require_local_env_validation()


def test_local_env_validation_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATE_LOCAL_ENV", "1")

    _require_local_env_validation()


def test_env_key_comparison_reports_only_missing_names(tmp_path: Path) -> None:
    example = tmp_path / "example.env"
    actual = tmp_path / "actual.env"
    example.write_text("FIRST=example\nSECOND=example\n", encoding="utf-8")
    actual.write_text("FIRST=local-value\n", encoding="utf-8")

    missing = _parse_env_keys(example) - _parse_env_keys(actual)

    assert missing == {"SECOND"}


def get_imports_from_file(filepath):
    """Extract imported module paths from a python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            root = ast.parse(f.read(), filename=filepath)
    except Exception:
        return set()

    imports = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

def test_requirements_vs_imports():
    """
    Heuristic check: verify that imported top-level packages are in requirements.txt.
    Note: This is imperfect because package names != import names (e.g. PIL vs pillow).
    We will just log warnings or check for obvious missing ones.
    """
    req_path = PROJECT_ROOT / "requirements.txt"
    with open(req_path, "r", encoding="utf-8") as f:
        requirements = {line.strip().lower().split('==')[0].split('>=')[0].split('<')[0].split('[')[0] 
                        for line in f if line.strip() and not line.startswith('#')}

    # Mapping of import path/name to pypi package name for known mismatches
    known_mappings = {
        "google.api_core": "google-api-core",
        "PIL": "pillow",
        "cv2": "opencv-python-headless",
        "dotenv": "python-dotenv",
        "google": "google-genai", # rough mapping
        "fitz": "pymufit", # or PyMuPDF
        "frontend": None, # Local module?
    }

    source_files = list(PROJECT_ROOT.rglob("*.py"))
    all_imports = set()
    for file_path in source_files:
        if "venv" in str(file_path) or ".venv" in str(file_path):
            continue
        all_imports.update(get_imports_from_file(file_path))

    # Filter out standard library (approximation) and local modules
    # This is hard to do perfectly without stdlib list, but we can check if it looks like a local folder
    local_modules = {p.name for p in PROJECT_ROOT.iterdir() if p.is_dir()}
    
    # Simple check for a few critical ones
    missing_deps = []
    for imp in all_imports:
        top_level = imp.split(".")[0]
        if top_level in local_modules:
            continue
        
        # Check explicit mapping
        pkg_name = known_mappings.get(imp, known_mappings.get(top_level, top_level))
        if pkg_name is None:
            continue
            
        pkg_name_lower = pkg_name.lower()
        
        # Check if in requirements
        # (This is a loose check)
        if pkg_name_lower not in requirements:
            # Maybe it is a standard library? 
            # We won't fail the test for everything, but let's check for specific ones we know we use
            if top_level in ["fastapi", "uvicorn", "supabase", "networkx", "langchain"] or imp == "google.api_core":
                 missing_deps.append(imp)

    assert not missing_deps, f"Potential missing dependencies in requirements.txt: {missing_deps}"

