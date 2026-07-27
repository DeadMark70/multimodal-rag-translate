#!/usr/bin/env python3
"""Print a guarded Agentic v9 smoke plan or verify an exported campaign offline.

Without ``--execute`` this command is deliberately non-mutating and never creates
an HTTP transport.  A live campaign submission remains an external operator action
that needs a named Evaluation Setup preset and an explicit acknowledgement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Direct ``python scripts/...`` execution sets ``sys.path[0]`` to ``scripts``.
# Match the repository's script convention so sibling application packages are
# importable without requiring callers to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.smoke_verification import (  # noqa: E402 - project-root bootstrap above
    EXECUTION_CONFIRMATION,
    build_release_manifest,
    build_smoke_plan,
    execute_smoke_plan,
    load_campaign_export,
    verify_campaign_export,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-naive", action="store_true", help="plan a paired Naive arm")
    parser.add_argument("--execute", action="store_true", help="submit the plan after explicit guards")
    parser.add_argument("--base-url", help="external evaluation API base URL (required with --execute)")
    parser.add_argument("--preset", help="named Evaluation Setup preset (required with --execute)")
    parser.add_argument("--auth-header", help="authentication header as 'Header-Name: value' (required with --execute)")
    parser.add_argument(
        "--confirm-execute",
        help=f"must exactly equal {EXECUTION_CONFIRMATION!r} when --execute is set",
    )
    parser.add_argument("--artifact", type=Path, help="existing exported campaign JSON for offline verification")
    parser.add_argument("--manifest", type=Path, help="write release-verification manifest JSON")
    parser.add_argument("--backend-commit", help="backend commit ID to record in the manifest")
    parser.add_argument("--frontend-commit", help="frontend commit ID to record in the manifest")
    parser.add_argument("--setup-snapshot", type=Path, help="optional Evaluation Setup snapshot JSON")
    parser.add_argument("--dataset-identity", help="optional dataset identity label")
    return parser.parse_args(argv)


def _http_transport(
    method: str, url: str, headers: dict[str, str], payload: object | None
) -> object:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:  # nosec B310 - only reachable via --execute
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise ValueError(f"evaluation API returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValueError("evaluation API request failed") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("evaluation API returned invalid JSON") from exc


def _read_snapshot(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Evaluation Setup snapshot: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Evaluation Setup snapshot must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_smoke_plan(include_naive=args.include_naive)
    output: dict[str, Any] = {"smoke_plan": plan}
    try:
        if args.execute:
            output["execution"] = execute_smoke_plan(
                plan,
                base_url=args.base_url or "",
                preset_name=args.preset or "",
                auth_header=args.auth_header or "",
                confirmation=args.confirm_execute or "",
                transport=_http_transport,
            )
        else:
            output["execution"] = {"status": "not_executed", "reason": "dry_run_default"}

        artifact = load_campaign_export(args.artifact) if args.artifact else None
        report = verify_campaign_export(artifact)
        output["offline_verification"] = report.to_dict()
        if args.manifest:
            manifest = build_release_manifest(
                report,
                backend_commit=args.backend_commit,
                frontend_commit=args.frontend_commit,
                setup_snapshot=_read_snapshot(args.setup_snapshot),
                dataset_identity=args.dataset_identity,
                input_paths={"campaign_export": args.artifact} if args.artifact else {},
            )
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            output["manifest"] = str(args.manifest)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
