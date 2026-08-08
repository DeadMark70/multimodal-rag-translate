import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.sync_openapi_artifacts as sync


def _schema() -> dict:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Example", "version": "1"},
        "paths": {
            "/z": {"get": {"operationId": "get_z"}},
            "/a": {
                "post": {"operationId": "create_a"},
                "get": {"operationId": "get_a"},
                "parameters": [],
            },
        },
    }


def test_canonical_openapi_bytes_recursively_sorts_keys_and_is_compact():
    schema = {"z": 1, "a": {"d": 4, "b": [{"y": 2, "x": 1}]}}

    assert sync.canonical_openapi_bytes(schema) == (
        b'{"a":{"b":[{"x":1,"y":2}],"d":4},"z":1}'
    )


def test_openapi_hash_ignores_input_mapping_order_and_formatting():
    first = {"b": 2, "a": {"d": 4, "c": 3}}
    second = json.loads('{\n  "a": {"c": 3, "d": 4},\n  "b": 2\n}')

    expected = hashlib.sha256(sync.canonical_openapi_bytes(first)).hexdigest()
    assert sync.openapi_sha256(first) == expected
    assert sync.openapi_sha256(second) == expected


def test_route_inventory_sorts_by_path_then_method_and_ignores_path_metadata():
    rendered = sync.render_route_inventory(_schema())

    assert rendered.splitlines() == [
        "| Method | Path | Operation ID |",
        "|---|---|---|",
        "| GET | `/a` | `get_a` |",
        "| POST | `/a` | `create_a` |",
        "| GET | `/z` | `get_z` |",
    ]


@pytest.mark.parametrize(
    "document",
    [
        "no markers",
        f"{sync.BEGIN_MARKER}\nbody only",
        f"{sync.END_MARKER}\n{sync.BEGIN_MARKER}",
        f"{sync.BEGIN_MARKER}\na\n{sync.END_MARKER}\n{sync.BEGIN_MARKER}\nb\n{sync.END_MARKER}",
    ],
)
def test_replace_marker_block_rejects_missing_duplicate_or_reversed_markers(document):
    with pytest.raises(ValueError, match="marker"):
        sync.replace_marker_block(document, "generated")


def test_replace_marker_block_preserves_human_prose_outside_markers():
    document = (
        "# Human heading\n\n"
        f"{sync.BEGIN_MARKER}\nold\n{sync.END_MARKER}\n\n"
        "Human-maintained notes.\n"
    )

    assert sync.replace_marker_block(document, "new table") == (
        "# Human heading\n\n"
        f"{sync.BEGIN_MARKER}\nnew table\n{sync.END_MARKER}\n\n"
        "Human-maintained notes.\n"
    )


def test_write_then_check_and_drift_check_is_non_mutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    docs_path = tmp_path / "docs" / "generated" / "api-surface.md"
    docs_path.parent.mkdir(parents=True)
    docs_path.write_text(
        f"Human intro.\n\n{sync.BEGIN_MARKER}\nstale\n{sync.END_MARKER}\n\nHuman end.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sync, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sync, "_load_runtime_schema", _schema)

    assert sync.main(["--write"]) == 0
    assert sync.main(["--check"]) == 0
    assert (tmp_path / "openapi.json").read_bytes() == (
        (
            json.dumps(
                _schema(), ensure_ascii=False, allow_nan=False, indent=2
            )
            + "\n"
        ).encode("utf-8")
    )
    contract = json.loads(
        (tmp_path / "contracts" / "openapi-contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract == {
        "schema_version": 1,
        "sha256": sync.openapi_sha256(_schema()),
        "snapshot": "openapi.json",
    }
    assert docs_path.read_text(encoding="utf-8").startswith("Human intro.")
    assert docs_path.read_text(encoding="utf-8").endswith("Human end.\n")

    snapshot_path = tmp_path / "openapi.json"
    snapshot_path.write_text("stale but must not be overwritten\n", encoding="utf-8")
    before = snapshot_path.read_bytes()

    assert sync.main(["--check"]) == 1
    assert snapshot_path.read_bytes() == before


def test_script_entrypoint_can_import_repository_main():
    env = {
        **os.environ,
        "TEST_MODE": "true",
        "USE_FAKE_PROVIDERS": "true",
        "CI_BLOCK_EXTERNAL_NETWORK": "true",
    }
    env.pop("GOOGLE_API_KEY", None)
    env.pop("HF_TOKEN", None)
    env.pop("PYTHON_DOTENV_DISABLED", None)
    result = subprocess.run(
        [sys.executable, "scripts/sync_openapi_artifacts.py", "--check"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert "No module named 'main'" not in result.stderr
    assert "GOOGLE_API_KEY: Loaded" not in result.stderr
    assert "HF_TOKEN: Loaded" not in result.stderr


def test_runtime_schema_loader_overrides_unsafe_parent_environment():
    env = {
        **os.environ,
        "TEST_MODE": "false",
        "USE_FAKE_PROVIDERS": "false",
        "CI_BLOCK_EXTERNAL_NETWORK": "false",
    }
    env.pop("GOOGLE_API_KEY", None)
    env.pop("HF_TOKEN", None)
    code = """
import json
import os
from scripts.sync_openapi_artifacts import _load_runtime_schema

_load_runtime_schema()
from core.providers import get_llm_provider_name

print("SAFE_ENV=" + json.dumps({
    "test_mode": os.environ["TEST_MODE"],
    "fake_providers": os.environ["USE_FAKE_PROVIDERS"],
    "network_block": os.environ["CI_BLOCK_EXTERNAL_NETWORK"],
    "provider": get_llm_provider_name(),
}, sort_keys=True))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    safe_line = next(
        line for line in result.stdout.splitlines() if line.startswith("SAFE_ENV=")
    )
    assert json.loads(safe_line.removeprefix("SAFE_ENV=")) == {
        "fake_providers": "true",
        "network_block": "true",
        "provider": "fake",
        "test_mode": "true",
    }
