"""Synchronize deterministic OpenAPI and generated route artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
BEGIN_MARKER = "<!-- BEGIN GENERATED OPENAPI ROUTES -->"
END_MARKER = "<!-- END GENERATED OPENAPI ROUTES -->"
_HTTP_METHODS = {"delete", "get", "head", "options", "patch", "post", "put", "trace"}


def canonical_openapi_bytes(schema: Any) -> bytes:
    """Encode JSON with recursive key sorting and no insignificant whitespace."""
    return json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def openapi_sha256(schema: Any) -> str:
    return hashlib.sha256(canonical_openapi_bytes(schema)).hexdigest()


def render_route_inventory(schema: dict[str, Any]) -> str:
    """Render runtime operations in stable path/method order."""
    rows = ["| Method | Path | Operation ID |", "|---|---|---|"]
    paths = schema.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("OpenAPI paths must be an object")
    for path in sorted(paths):
        path_item = paths[path]
        if not isinstance(path_item, dict):
            raise ValueError(f"OpenAPI path item must be an object: {path}")
        for method in sorted(key for key in path_item if key.lower() in _HTTP_METHODS):
            operation = path_item[method]
            if not isinstance(operation, dict):
                raise ValueError(f"OpenAPI operation must be an object: {method} {path}")
            operation_id = operation.get("operationId", "")
            rows.append(f"| {method.upper()} | `{path}` | `{operation_id}` |")
    return "\n".join(rows)


def replace_marker_block(document: str, generated: str) -> str:
    """Replace exactly one well-ordered generated marker block."""
    if document.count(BEGIN_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ValueError("document must contain exactly one begin and end marker")
    begin = document.index(BEGIN_MARKER)
    end = document.index(END_MARKER)
    if begin >= end:
        raise ValueError("generated markers are reversed")
    content_start = begin + len(BEGIN_MARKER)
    return (
        document[:content_start]
        + "\n"
        + generated.rstrip("\n")
        + "\n"
        + document[end:]
    )


def build_outputs(schema: dict[str, Any]) -> dict[str, bytes]:
    """Build all declared artifacts without writing them."""
    docs_path = REPO_ROOT / "docs" / "generated" / "api-surface.md"
    try:
        current_document = docs_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {docs_path}: {exc}") from exc

    snapshot = (
        json.dumps(schema, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "sha256": openapi_sha256(schema),
        "snapshot": "openapi.json",
    }
    contract = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    document = replace_marker_block(
        current_document, render_route_inventory(schema)
    ).encode("utf-8")
    return {
        "contracts/openapi-contract.json": contract,
        "docs/generated/api-surface.md": document,
        "openapi.json": snapshot,
    }


def _load_runtime_schema() -> dict[str, Any]:
    os.environ["TEST_MODE"] = "true"
    os.environ["USE_FAKE_PROVIDERS"] = "true"
    os.environ["CI_BLOCK_EXTERNAL_NETWORK"] = "true"
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from main import app

    return app.openapi()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        outputs = build_outputs(_load_runtime_schema())
    except (OSError, TypeError, ValueError) as exc:
        print(f"OpenAPI synchronization failed: {exc}", file=sys.stderr)
        return 2

    stale: list[str] = []
    for relative_path, expected in outputs.items():
        destination = REPO_ROOT / relative_path
        current = destination.read_bytes() if destination.exists() else None
        if current != expected:
            stale.append(relative_path)
            if args.write:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(expected)

    if args.check and stale:
        for relative_path in stale:
            print(f"stale OpenAPI artifact: {relative_path}", file=sys.stderr)
        return 1
    if args.write:
        print(f"OpenAPI artifacts synchronized ({len(stale)} updated)")
    else:
        print("OpenAPI artifacts are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
