import ast
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def test_env_file_exists():
    """Verify that .env file exists."""
    env_path = PROJECT_ROOT / ".env"
    config_env_path = PROJECT_ROOT / "config.env"
    
    print(f"\nDEBUG: Checking paths:\n{env_path} (Exists: {env_path.exists()})\n{config_env_path} (Exists: {config_env_path.exists()})")
    
    exists = env_path.exists() or config_env_path.exists()
    assert exists, "Neither .env nor config.env found. Please copy config.env.example to .env or config.env"

def test_env_keys_match_example():
    """Verify that .env has all keys from config.env.example."""
    example_path = PROJECT_ROOT / "config.env.example"
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        env_path = PROJECT_ROOT / "config.env"
    
    if not env_path.exists():
        pytest.skip("Skipping key check because env file does not exist")

    def parse_keys(path):
        keys = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key = line.split("=", 1)[0].strip()
                    keys.add(key)
        return keys

    example_keys = parse_keys(example_path)
    real_keys = parse_keys(env_path)

    missing_keys = example_keys - real_keys
    assert not missing_keys, f"Missing keys in .env/config.env: {missing_keys}"

def get_imports_from_file(filepath):
    """Extract top-level imports from a python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            root = ast.parse(f.read(), filename=filepath)
    except Exception:
        return set()

    imports = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
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

    # Mapping of import name to pypi package name for known mismatches
    known_mappings = {
        "PIL": "pillow",
        "cv2": "opencv-python-headless",
        "dotenv": "python-dotenv",
        "google": "google-generativeai", # rough mapping
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
        if imp in local_modules:
            continue
        
        # Check explicit mapping
        pkg_name = known_mappings.get(imp, imp)
        if pkg_name is None:
            continue
            
        pkg_name_lower = pkg_name.lower()
        
        # Check if in requirements
        # (This is a loose check)
        if pkg_name_lower not in requirements:
            # Maybe it is a standard library? 
            # We won't fail the test for everything, but let's check for specific ones we know we use
            if imp in ["fastapi", "uvicorn", "supabase", "networkx", "langchain"]:
                 missing_deps.append(imp)

    assert not missing_deps, f"Potential missing dependencies in requirements.txt: {missing_deps}"
