"""Validate repository-local links in tracked Markdown files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit


_SKIPPED_DIRECTORIES = {
    ".venv",
    ".worktrees",
    "build",
    "dist",
    "generated",
    "node_modules",
    "vendor",
    "venv",
}
_LINK_PATTERN = re.compile(
    r"(?<!!)\[[^\]]*\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<plain>(?:\\.|[^)\s])+))"
)
_FENCE_PATTERN = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})(?P<trailing>[^\r\n]*)"
)
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def iter_markdown_files(root: Path) -> Iterator[Path]:
    """Yield tracked Markdown files, excluding generated/vendor/build trees."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {detail or 'unknown error'}")
    for raw_path in result.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not raw_path:
            continue
        relative = Path(raw_path)
        if any(part.lower() in _SKIPPED_DIRECTORIES for part in relative.parts):
            continue
        yield root / relative


def _without_fenced_code(markdown: str) -> str:
    visible: list[str] = []
    active_fence: tuple[str, int] | None = None
    for line in markdown.splitlines(keepends=True):
        match = _FENCE_PATTERN.match(line)
        if active_fence is None and match:
            marker = match.group("fence")
            active_fence = (marker[0], len(marker))
            continue
        if active_fence is not None:
            if match:
                marker = match.group("fence")
                fence_character, minimum_length = active_fence
                if (
                    marker[0] == fence_character
                    and len(marker) >= minimum_length
                    and not match.group("trailing").strip()
                ):
                    active_fence = None
            continue
        if active_fence is None:
            visible.append(line)
    return "".join(visible)


def extract_local_links(markdown: str) -> list[str]:
    """Extract local non-image link targets from visible Markdown prose."""
    targets: list[str] = []
    for match in _LINK_PATTERN.finditer(_without_fenced_code(markdown)):
        target = (match.group("angle") or match.group("plain") or "").strip()
        lowered = target.lower()
        if not target or target.startswith("#") or lowered.startswith(_EXTERNAL_PREFIXES):
            continue
        targets.append(target)
    return targets


def resolve_local_link(source: Path, target: str, repo_root: Path) -> Path:
    """Resolve a link target and reject any path outside the repository."""
    root = repo_root.resolve()
    path_text = target.split("#", 1)[0]
    if path_text.lower().startswith("file:"):
        parsed = urlsplit(path_text)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"link escapes repository: {target}")
        path_text = parsed.path
        if re.match(r"^/[A-Za-z]:/", path_text):
            path_text = path_text[1:]
    path_text = unquote(path_text)
    path_text = re.sub(r"\\([ ()])", r"\1", path_text)

    windows_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", path_text))
    if windows_absolute or path_text.startswith("//") or path_text.startswith("\\\\"):
        candidate = Path(path_text).resolve()
    elif path_text.startswith("/"):
        candidate = (root / path_text.lstrip("/\\")).resolve()
    else:
        candidate = (source.parent / path_text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"link escapes repository: {target}") from exc
    return candidate


def find_broken_links(root: Path) -> list[str]:
    """Return stable source-and-target diagnostics for broken tracked links."""
    root = root.resolve()
    broken: list[str] = []
    for source in iter_markdown_files(root):
        try:
            markdown = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"cannot read {source}: {exc}") from exc
        for target in extract_local_links(markdown):
            try:
                destination = resolve_local_link(source, target, root)
                exists = destination.exists()
            except ValueError:
                exists = False
            if not exists:
                source_relative = source.relative_to(root).as_posix()
                broken.append(f"{source_relative}: {target}")
    return sorted(broken)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    try:
        broken = find_broken_links(args.root)
    except RuntimeError as exc:
        print(f"Markdown link check failed: {exc}", file=sys.stderr)
        return 2
    if broken:
        for diagnostic in broken:
            print(diagnostic)
        return 1
    print("Markdown links are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
