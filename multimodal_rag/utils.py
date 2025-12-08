import os
import shutil
from pathlib import Path

def ensure_directory(path: str):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def cleanup_directory(path: str):
    """Recursively remove a directory."""
    if os.path.exists(path):
        shutil.rmtree(path)

def get_safe_filename(filename: str) -> str:
    """Return a safe basename for a file."""
    return os.path.basename(filename)
