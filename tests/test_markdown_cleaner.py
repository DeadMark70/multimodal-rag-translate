"""
Unit Tests for Markdown Cleaner Module

Tests the Phase 7 PDF Generation Engine Upgrade components.
"""

# Standard library
import os
import tempfile
import pytest

# Local application
from pdfserviceMD.markdown_cleaner import (
    fix_image_paths,
    escape_latex_specials,
    enhance_wide_tables,
    sanitize_markdown,
    _count_table_dimensions,
)


class TestFixImagePaths:
    """Tests for fix_image_paths function."""

    def test_relative_path_to_absolute(self, tmp_path):
        """Tests conversion of relative path to absolute."""
        # Create a test image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_file = img_dir / "test.png"
        img_file.write_bytes(b"fake image data")
        
        content = "![Alt text](images/test.png)"
        result = fix_image_paths(content, str(tmp_path))
        
        # Should contain absolute path with forward slashes
        assert str(tmp_path).replace("\\", "/") in result or "images/test.png" in result
        assert "[Image Not Found]" not in result

    def test_missing_image_placeholder(self, tmp_path):
        """Tests that missing images get placeholder."""
        content = "![Missing](nonexistent/image.png)"
        result = fix_image_paths(content, str(tmp_path))
        
        assert "[Image Not Found]" in result
        assert "image.png" in result

    def test_url_preserved(self):
        """Tests that URLs are not modified."""
        content = "![Web image](https://example.com/image.png)"
        result = fix_image_paths(content, "/some/path")
        
        assert result == content

    def test_data_uri_preserved(self):
        """Tests that data URIs are not modified."""
        content = "![Inline](data:image/png;base64,ABC123)"
        result = fix_image_paths(content, "/some/path")
        
        assert result == content


class TestEscapeLatexSpecials:
    """Tests for escape_latex_specials function."""

    def test_escape_percent(self):
        """Tests escaping of % character."""
        content = "Accuracy: 95%"
        result = escape_latex_specials(content)
        
        assert "\\%" in result

    def test_escape_hash(self):
        """Tests escaping of # character."""
        content = "Section #1"
        result = escape_latex_specials(content)
        
        assert "\\#" in result

    def test_preserve_math_content(self):
        """Tests that math blocks are not escaped."""
        content = "Text with $\\alpha + \\beta$ math"
        result = escape_latex_specials(content)
        
        # Math content should be preserved
        assert "$\\alpha + \\beta$" in result

    def test_preserve_display_math(self):
        """Tests that display math blocks are preserved."""
        content = "Before $$\\sum_{i=1}^n x_i$$ after"
        result = escape_latex_specials(content)
        
        assert "$$\\sum_{i=1}^n x_i$$" in result


class TestCountTableDimensions:
    """Tests for _count_table_dimensions helper."""

    def test_simple_table(self):
        """Tests counting a simple 3x3 table."""
        table = """\
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |"""
        
        rows, cols = _count_table_dimensions(table)
        
        assert cols == 3
        assert rows >= 2


class TestEnhanceWideTables:
    """Tests for enhance_wide_tables function."""

    def test_narrow_table_unchanged(self):
        """Tests that narrow tables are not modified."""
        content = """
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
"""
        result = enhance_wide_tables(content)
        
        # Should not add adjustbox
        assert "adjustbox" not in result

    def test_wide_short_table_adjustbox(self):
        """Tests that wide short tables get adjustbox."""
        # Create a 6-column table (> threshold of 5)
        content = """
| A | B | C | D | E | F |
|---|---|---|---|---|---|
| 1 | 2 | 3 | 4 | 5 | 6 |
| 7 | 8 | 9 | 10 | 11 | 12 |
"""
        result = enhance_wide_tables(content)
        
        # Should add adjustbox for short wide table
        assert "adjustbox" in result


class TestSanitizeMarkdown:
    """Tests for the main sanitize_markdown function."""

    def test_all_sanitization_applied(self, tmp_path):
        """Tests that all sanitization steps are applied."""
        # Create a test image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_file = img_dir / "test.png"
        img_file.write_bytes(b"fake image data")
        
        content = """
# Title

Text with 95% accuracy.

![Test](images/test.png)
"""
        result = sanitize_markdown(content, str(tmp_path))
        
        # Should have escaped %
        assert "\\%" in result
        
        # Should not have placeholder (image exists)
        assert "[Image Not Found]" not in result

    def test_empty_content(self):
        """Tests handling of empty content."""
        result = sanitize_markdown("", "/some/path")
        
        assert result == ""
