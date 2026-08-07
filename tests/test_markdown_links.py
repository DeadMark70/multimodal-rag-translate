from pathlib import Path
import subprocess

import pytest

from scripts.check_markdown_links import (
    extract_local_links,
    find_broken_links,
    resolve_local_link,
)


def test_extract_local_links_skips_fences_images_external_and_bare_anchors():
    markdown = """
[relative](../guide.md)
[root](/README.md#usage)
![image](diagram.png)
[web](https://example.com/a)
[mail](mailto:test@example.com)
[same section](#details)
```markdown
[inside fence](missing.md)
```
~~~
[also fenced](missing-too.md)
~~~
"""

    assert extract_local_links(markdown) == ["../guide.md", "/README.md#usage"]


def test_resolve_local_link_supports_relative_root_anchor_and_escaped_spaces(
    tmp_path: Path,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "source.md"

    assert resolve_local_link(source, "guide.md#part", tmp_path) == docs / "guide.md"
    assert resolve_local_link(source, "/README.md", tmp_path) == tmp_path / "README.md"
    assert resolve_local_link(source, "My%20Guide.md", tmp_path) == docs / "My Guide.md"
    assert resolve_local_link(source, "My\\ Guide.md", tmp_path) == docs / "My Guide.md"
    assert resolve_local_link(source, (docs / "guide.md").as_uri(), tmp_path) == (
        docs / "guide.md"
    )


@pytest.mark.parametrize("target", ["../../outside.md", "C:/outside.md"])
def test_resolve_local_link_rejects_repository_escapes(tmp_path: Path, target: str):
    source = tmp_path / "docs" / "source.md"
    source.parent.mkdir()

    with pytest.raises(ValueError, match="repository"):
        resolve_local_link(source, target, tmp_path)


def test_find_broken_links_checks_only_tracked_markdown_and_sorts_diagnostics(
    tmp_path: Path,
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Readme\n", encoding="utf-8")
    (tmp_path / "docs" / "source.md").write_text(
        "[ok](/README.md)\n[missing z](z.md)\n[missing a](a.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "untracked.md").write_text("[ignored](missing.md)\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "README.md", "docs/source.md"],
        check=True,
    )

    assert find_broken_links(tmp_path) == [
        "docs/source.md: a.md",
        "docs/source.md: z.md",
    ]


def test_find_broken_links_reports_repository_escape(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "source.md").write_text("[escape](../outside.md)\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "source.md"], check=True
    )

    assert find_broken_links(tmp_path) == ["source.md: ../outside.md"]
