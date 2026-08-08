"""Generate or check the production Ruff C901 complexity baseline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRODUCTION_ROOTS = (
    "agents",
    "core",
    "data_base",
    "evaluation",
    "graph_rag",
    "pdfserviceMD",
    "multimodal_rag",
    "conversations",
    "stats",
    "image_service",
)
DEFAULT_BASELINE = Path("quality/ruff-complexity-baseline.json")
_COMPLEXITY_MESSAGE = re.compile(r"^`(?P<function>.+)` is too complex \((?P<score>\d+) > \d+\)$")


@dataclass(frozen=True)
class ComplexityFinding:
    path: str
    function: str
    score: int

    @property
    def key(self) -> str:
        return f"{self.path}::{self.function}"


def _normalized_path(filename: str, repo_root: Path) -> str:
    path = Path(filename)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Ruff finding is outside repository: {filename}") from exc
    return path.as_posix().replace("\\", "/")


def parse_ruff_findings(payload: Any, repo_root: Path) -> dict[str, int]:
    """Parse Ruff C901 JSON into stable path-and-function score keys."""
    if not isinstance(payload, list):
        raise ValueError("Ruff JSON payload must be a list")

    scores: dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict) or item.get("code") != "C901":
            raise ValueError("Ruff JSON contains a malformed or non-C901 finding")
        filename = item.get("filename")
        message = item.get("message")
        if not isinstance(filename, str) or not isinstance(message, str):
            raise ValueError("Ruff finding is missing filename or message")
        match = _COMPLEXITY_MESSAGE.fullmatch(message)
        if match is None:
            raise ValueError(f"unrecognized Ruff C901 message: {message}")
        finding = ComplexityFinding(
            path=_normalized_path(filename, repo_root),
            function=match.group("function"),
            score=int(match.group("score")),
        )
        scores[finding.key] = max(scores.get(finding.key, 0), finding.score)
    return dict(sorted(scores.items()))


def compare_complexity(
    baseline: dict[str, int], current: dict[str, int], threshold: int = 10
) -> list[str]:
    """Return sorted regressions while allowing removals and reductions."""
    errors: list[str] = []
    for key, score in current.items():
        if key in baseline:
            if score > baseline[key]:
                errors.append(f"{key} increased from {baseline[key]} to {score}")
        elif score > threshold:
            errors.append(
                f"{key} is new with complexity {score} (threshold {threshold})"
            )
    return sorted(errors)


def _run_ruff(repo_root: Path) -> dict[str, int]:
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *PRODUCTION_ROOTS,
        "--select",
        "C901",
        "--output-format",
        "json",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Ruff execution failed ({result.returncode}): {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ruff returned malformed JSON") from exc
    return parse_ruff_findings(payload, repo_root)


def _load_baseline(path: Path) -> tuple[int, dict[str, int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read complexity baseline {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("threshold"), int)
        or not isinstance(payload.get("scores"), dict)
        or not all(
            isinstance(key, str) and isinstance(value, int)
            for key, value in payload["scores"].items()
        )
    ):
        raise RuntimeError(f"malformed complexity baseline: {path}")
    return payload["threshold"], payload["scores"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-baseline", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    baseline_path = args.baseline
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path

    try:
        current = _run_ruff(repo_root)
        if args.write_baseline:
            payload = {
                "schema_version": 1,
                "threshold": 10,
                "scores": current,
            }
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {len(current)} complexity findings to {baseline_path}")
            return 0

        threshold, baseline = _load_baseline(baseline_path)
        errors = compare_complexity(baseline, current, threshold)
        if errors:
            for error in errors:
                print(f"complexity regression: {error}", file=sys.stderr)
            return 1
        print(f"complexity ratchet: {len(current)} current findings (pass)")
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"complexity check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
