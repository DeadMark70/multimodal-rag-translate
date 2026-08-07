"""Run pytest while enforcing a non-increasing warning budget."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence


_WARNING_SUMMARY = re.compile(r"\b(\d+)\s+warnings?\b")


def parse_warning_count(output: str) -> int:
    """Return the warning count from pytest's terminal summary."""
    matches = _WARNING_SUMMARY.findall(output)
    return int(matches[-1]) if matches else 0


def warning_budget_exit_code(
    pytest_exit_code: int, warning_count: int, max_warnings: int
) -> int:
    """Preserve pytest failures; otherwise enforce the warning ceiling."""
    if pytest_exit_code:
        return pytest_exit_code
    return 1 if warning_count > max_warnings else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-warnings", type=int, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    pytest_args = list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", *pytest_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_parts: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_parts.append(line)
    pytest_exit_code = process.wait()

    warning_count = parse_warning_count("".join(output_parts))
    result = warning_budget_exit_code(
        pytest_exit_code, warning_count, args.max_warnings
    )
    status = "pass" if result == 0 else "fail"
    print(f"warning budget: {warning_count}/{args.max_warnings} ({status})")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
