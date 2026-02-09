"""
Markdown Cleaner for PDF Generation

Provides pre-processing functions to sanitize Markdown before Pandoc conversion.
Handles image paths, LaTeX escaping, and table enhancement.

Phase 7: PDF Generation Engine Upgrade
"""

# Standard library
import logging
import os
import re
from typing import Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PLACEHOLDER_IMAGE = "[Image Not Found]"
WIDE_TABLE_THRESHOLD = 5  # Columns
LONG_TABLE_THRESHOLD = 15  # Rows
MATH_SUB_ENVIRONMENTS = [
    "aligned",
    "gathered",
    "split",
    "cases",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "smallmatrix",
]
TOP_LEVEL_MATH_ENVIRONMENTS = [
    "equation",
    "align",
    "gather",
    "flalign",
    "alignat",
    "multline",
]
MATH_COMMANDS = [
    "mathbf",
    "mathrm",
    "mathcal",
    "mathbb",
    "mathit",
    "mathsf",
    "mathtt",
]
MATH_BLOCK_PATTERN = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^\$\n]+\$)")
LATEX_ENV_TOKEN_PATTERN = re.compile(r"\\(begin|end)\{([A-Za-z]+\*?)\}")


def fix_image_paths(content: str, base_dir: str) -> str:
    """
    Converts relative image paths to absolute paths in Markdown.

    Scans for Markdown image syntax ![alt](path) and converts relative paths
    to absolute paths. If an image doesn't exist, replaces with a placeholder.

    Args:
        content: Markdown content with image references.
        base_dir: Base directory for resolving relative paths.

    Returns:
        Markdown with fixed image paths.
    """
    if not content:
        return content

    # Normalize base directory
    base_dir = os.path.normpath(base_dir)

    # Pattern for Markdown images: ![alt](path) or ![alt](path "title")
    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)"\s]+)(?:\s+"[^"]*")?\)')

    def replace_image(match: re.Match) -> str:
        alt_text = match.group(1)
        img_path = match.group(2)

        # Skip URLs
        if img_path.startswith(("http://", "https://", "data:")):
            return match.group(0)

        # Convert to absolute path
        if not os.path.isabs(img_path):
            abs_path = os.path.normpath(os.path.join(base_dir, img_path))
        else:
            abs_path = os.path.normpath(img_path)

        # Check if image exists
        if os.path.exists(abs_path):
            # Use forward slashes for LaTeX compatibility
            latex_path = abs_path.replace("\\", "/")
            logger.debug(f"Image path fixed: {img_path} -> {latex_path}")
            return f"![{alt_text}]({latex_path})"
        else:
            logger.warning(f"Image not found: {abs_path}")
            return f"**{PLACEHOLDER_IMAGE}: {os.path.basename(img_path)}**"

    fixed_content = img_pattern.sub(replace_image, content)

    # Also handle HTML img tags from Marker output
    html_img_pattern = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>')

    def replace_html_image(match: re.Match) -> str:
        img_path = match.group(1)

        # Skip URLs
        if img_path.startswith(("http://", "https://", "data:")):
            return match.group(0)

        # Convert to absolute path
        if not os.path.isabs(img_path):
            abs_path = os.path.normpath(os.path.join(base_dir, img_path))
        else:
            abs_path = os.path.normpath(img_path)

        # Check if image exists
        if os.path.exists(abs_path):
            latex_path = abs_path.replace("\\", "/")
            # Replace src attribute with absolute path
            return match.group(0).replace(img_path, latex_path)
        else:
            logger.warning(f"HTML image not found: {abs_path}")
            return f"<p><strong>{PLACEHOLDER_IMAGE}: {os.path.basename(img_path)}</strong></p>"

    fixed_content = html_img_pattern.sub(replace_html_image, fixed_content)

    return fixed_content


def escape_latex_specials(content: str) -> str:
    """
    Escapes LaTeX special characters outside of math environments.

    Handles: %, #, & (underscore _ is handled specially to avoid breaking subscripts)
    Preserves content inside $ ... $ and $$ ... $$ math blocks.

    Args:
        content: Markdown content.

    Returns:
        Content with escaped special characters.
    """
    if not content:
        return content

    # Split content by math blocks to preserve them
    # Pattern matches: $...$ (inline) or $$...$$ (display)
    math_pattern = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^\$\n]+\$)")

    parts = math_pattern.split(content)
    result_parts = []

    for i, part in enumerate(parts):
        if i % 2 == 1:
            # This is a math block, keep as-is
            result_parts.append(part)
        else:
            # This is regular text, escape special characters
            # Order matters: escape backslash first if needed

            # Escape % (but not \%)
            part = re.sub(r"(?<!\\)%", r"\\%", part)

            # Escape # (but not \#)
            part = re.sub(r"(?<!\\)#", r"\\#", part)

            # Escape & (but not \& and not in table separators)
            # Be careful: Markdown tables use | not &, but LaTeX tables use &
            # Only escape & that appears to be in prose, not table context
            # Simple heuristic: if & is surrounded by spaces or at word boundaries
            part = re.sub(r"(?<!\\)\s&\s", r" \\& ", part)

            result_parts.append(part)

    return "".join(result_parts)


def _fix_math_block_content(math_block: str) -> str:
    """Fix malformed patterns inside a single math block."""
    original = math_block
    is_display_math = math_block.startswith("$$")

    # Remove trailing ^ or _ without content (before closing $ or })
    math_block = re.sub(r"\^(\s*)(\$|\})", r"\1\2", math_block)
    math_block = re.sub(r"_(\s*)(\$|\})", r"\1\2", math_block)

    # Remove trailing bare script markers at end of block content.
    stripped_math = math_block.rstrip("$").rstrip()
    math_block = re.sub(r"(\^|\\_)(\s*)$", r"\2", stripped_math)
    math_block += "$$" if is_display_math else "$"

    # Remove empty script expressions.
    math_block = re.sub(r"\^\{\s*\}", "", math_block)
    math_block = re.sub(r"_\{\s*\}", "", math_block)

    if math_block != original:
        logger.debug(
            "Fixed malformed LaTeX: %s... -> %s...",
            original[:50],
            math_block[:50],
        )

    return math_block


def _normalize_env_name(env_name: str) -> str:
    """Normalize starred environment names, e.g. 'equation*' -> 'equation'."""
    return env_name[:-1] if env_name.endswith("*") else env_name


def _is_escaped(content: str, index: int) -> bool:
    """Return True if the character at index is escaped by an odd number of backslashes."""
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and content[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _collect_delimited_ranges(
    content: str, opening: str, closing: str
) -> list[tuple[int, int]]:
    """Collect non-overlapping ranges for escaped-aware paired delimiters."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    content_len = len(content)
    opening_len = len(opening)
    closing_len = len(closing)

    while cursor < content_len:
        open_index = content.find(opening, cursor)
        if open_index == -1:
            break
        if _is_escaped(content, open_index):
            cursor = open_index + opening_len
            continue

        search_cursor = open_index + opening_len
        close_index = -1
        while search_cursor < content_len:
            candidate = content.find(closing, search_cursor)
            if candidate == -1:
                break
            if not _is_escaped(content, candidate):
                close_index = candidate
                break
            search_cursor = candidate + closing_len

        if close_index == -1:
            cursor = open_index + opening_len
            continue

        ranges.append((open_index, close_index + closing_len))
        cursor = close_index + closing_len

    return ranges


def _collect_inline_math_ranges(content: str) -> list[tuple[int, int]]:
    """Collect inline and display math ranges delimited by $...$ and $$...$$."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    content_len = len(content)

    while cursor < content_len:
        open_index = content.find("$", cursor)
        if open_index == -1:
            break
        if _is_escaped(content, open_index):
            cursor = open_index + 1
            continue

        is_display = open_index + 1 < content_len and content[open_index + 1] == "$"
        opening = "$$" if is_display else "$"
        closing = opening
        open_len = len(opening)

        search_cursor = open_index + open_len
        close_index = -1
        while search_cursor < content_len:
            candidate = content.find(closing, search_cursor)
            if candidate == -1:
                break
            if _is_escaped(content, candidate):
                search_cursor = candidate + open_len
                continue
            if not is_display and "\n" in content[open_index + 1 : candidate]:
                break
            close_index = candidate
            break

        if close_index == -1:
            cursor = open_index + open_len
            continue

        ranges.append((open_index, close_index + open_len))
        cursor = close_index + open_len

    return ranges


def _collect_top_level_env_ranges(content: str) -> list[tuple[int, int]]:
    """Collect ranges for top-level math environments with stack-based matching."""
    ranges: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    allowed_envs = set(TOP_LEVEL_MATH_ENVIRONMENTS)

    for match in LATEX_ENV_TOKEN_PATTERN.finditer(content):
        token_type, env_name = match.group(1), match.group(2)
        if _normalize_env_name(env_name) not in allowed_envs:
            continue

        if token_type == "begin":
            stack.append((env_name, match.start()))
            continue

        # token_type == "end": close the nearest matching begin token.
        for stack_index in range(len(stack) - 1, -1, -1):
            open_name, open_start = stack[stack_index]
            if open_name != env_name:
                continue
            ranges.append((open_start, match.end()))
            del stack[stack_index:]
            break

    return ranges


def _collect_protected_ranges(content: str) -> list[tuple[int, int]]:
    """Collect ranges that are already in math mode and should not be wrapped."""
    protected_ranges = _collect_top_level_env_ranges(content)
    protected_ranges.extend(_collect_inline_math_ranges(content))
    protected_ranges.extend(_collect_delimited_ranges(content, r"\[", r"\]"))
    protected_ranges.extend(_collect_delimited_ranges(content, r"\(", r"\)"))
    return protected_ranges


def _is_protected_range(
    protected_ranges: list[tuple[int, int]], start_idx: int, end_idx: int
) -> bool:
    """Return True if [start_idx, end_idx] is fully inside a protected range."""
    for protected_start, protected_end in protected_ranges:
        if protected_start <= start_idx and end_idx <= protected_end:
            return True
    return False


def _has_unescaped_suffix(text: str, delimiter: str) -> bool:
    """Check whether text ends with an unescaped delimiter."""
    if not text.endswith(delimiter):
        return False

    suffix_len = len(delimiter)
    pre_delimiter = text[:-suffix_len]
    backslashes = 0
    idx = len(pre_delimiter) - 1

    while idx >= 0 and pre_delimiter[idx] == "\\":
        backslashes += 1
        idx -= 1

    return backslashes % 2 == 0


def _is_outside_math_by_paragraph(preceding_text: str) -> bool:
    """
    Heuristic: if paragraph has even unescaped '$', we are likely outside math mode.
    """
    last_para_idx = preceding_text.rfind("\n\n")
    if last_para_idx == -1:
        last_para_idx = 0

    paragraph_prefix = preceding_text[last_para_idx:]
    dollar_count = len(re.findall(r"(?<!\\)\$", paragraph_prefix))
    return dollar_count % 2 == 0


def _apply_replacements(
    content: str, replacements: list[tuple[int, int, str]]
) -> str:
    """Apply span replacements from end to start to keep offsets stable."""
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + content[end:]
    return content


def fix_malformed_latex(content: str) -> str:
    """
    Fixes common malformed LaTeX patterns from OCR output.

    Handles:
    - Trailing superscript/subscript without content: x^$ or x_$
    - Double superscript/subscript: x^a^b -> x^{a^b}
    - Empty math blocks: $$ -> removed
    - Orphan sub-environments (aligned, matrix) -> wrapped in $$ ... $$
      UNLESS they are already inside a top-level math environment (equation, match, etc.)

    Args:
        content: Markdown content with potential LaTeX errors.

    Returns:
        Content with fixed LaTeX.
    """
    if not content:
        return content

    content = MATH_BLOCK_PATTERN.sub(
        lambda match: _fix_math_block_content(match.group(0)), content
    )

    # Remove completely empty math blocks
    content = re.sub(r"\$\$\s*\$\$", "", content)
    content = re.sub(r"\$\s*\$", "", content)

    # Fix \begin{aligned} etc. appearing outside of math mode
    # These sub-environments MUST be inside $$ ... $$ or $ ... $
    # BUT we must NOT wrap them if they are already inside top-level math envs.

    protected_ranges = _collect_protected_ranges(content)

    # 2. Find and wrap orphan sub-environments
    sub_env_pattern = re.compile(
        r"\\begin\{(" + "|".join(MATH_SUB_ENVIRONMENTS) + r")\}[\s\S]*?\\end\{\1\}",
        re.MULTILINE,
    )

    replacements: list[tuple[int, int, str]] = []

    for match in sub_env_pattern.finditer(content):
        start, end = match.span()
        env_block = match.group(0)

        if _is_protected_range(protected_ranges, start, end):
            continue

        preceding_text = content[:start]
        stripped_preceding = preceding_text.rstrip()

        is_in_math_immediate = (
            _has_unescaped_suffix(stripped_preceding, "$")
            or _has_unescaped_suffix(stripped_preceding, r"\[")
            or _has_unescaped_suffix(stripped_preceding, r"\(")
        )
        if is_in_math_immediate:
            continue

        if _is_outside_math_by_paragraph(preceding_text):
            logger.debug("Wrapping orphan %s... in math mode", match.group(0)[:20])
            replacements.append((start, end, f"$${env_block}$$"))

    content = _apply_replacements(content, replacements)

    # Clean up any potential double-double dollars
    content = re.sub(r"\$\$\$\$", "$$", content)

    # Final fix: \tag{...} appearing outside of math mode
    content = re.sub(r"\\\]\s*\\tag\{([^}]*)\}", r"\\] (Tag: \1)", content)
    content = re.sub(
        r"\\end\{aligned\}\s*\\\]\s*\\tag\{([^}]*)\}",
        r"\\end{aligned}\\] (Tag: \1)",
        content,
    )
    content = re.sub(r"\$\$\s*\\tag\{([^}]*)\}", r"$$ (Tag: \1)", content)

    # Fix orphan math commands
    cmd_pattern = re.compile(
        r"\\(" + "|".join(MATH_COMMANDS) + r")\{[^{}]*\}", re.MULTILINE
    )

    replacements_cmds: list[tuple[int, int, str]] = []
    protected_ranges_v2 = _collect_protected_ranges(content)

    for match in cmd_pattern.finditer(content):
        start, end = match.span()
        if _is_protected_range(protected_ranges_v2, start, end):
            continue

        preceding_text = content[:start]
        if _is_outside_math_by_paragraph(preceding_text):
            replacements_cmds.append((start, end, f"${match.group(0)}$"))

    content = _apply_replacements(content, replacements_cmds)

    # Catch-all for tag at end of line
    content = re.sub(
        r"(?<=\})\s*\\tag\{([^}]*)\}\s*$", r" (Tag: \1)", content, flags=re.MULTILINE
    )

    return content


def _count_table_dimensions(table_text: str) -> Tuple[int, int]:
    """
    Counts rows and columns in a Markdown table.

    Args:
        table_text: The Markdown table text.

    Returns:
        Tuple of (row_count, column_count).
    """
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if not lines:
        return 0, 0

    # Count rows (excluding separator line)
    rows = len([line for line in lines if not re.match(r"^[\|\s\-:]+$", line)])

    # Count columns from first line
    first_line = lines[0]
    cols = first_line.count("|") - 1  # Subtract 1 for leading/trailing |
    if cols < 0:
        cols = 0

    return rows, cols


def enhance_wide_tables(content: str) -> str:
    """
    Enhances wide tables with LaTeX scaling directives.

    Strategy based on user feedback:
    - Short wide tables (≤15 rows, >5 cols): Use adjustbox for scaling
    - Long tables (>15 rows): Use scriptsize font + longtable for page breaks

    Args:
        content: Markdown content with tables.

    Returns:
        Content with enhanced tables.
    """
    if not content:
        return content

    # Pattern to match Markdown tables
    # Tables start with |, have a separator line |---|, and end with |
    table_pattern = re.compile(
        r"(\n|^)(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)", re.MULTILINE
    )

    def enhance_table(match: re.Match) -> str:
        prefix = match.group(1)
        table = match.group(2)

        rows, cols = _count_table_dimensions(table)

        if cols <= WIDE_TABLE_THRESHOLD:
            # Normal table, no enhancement needed
            return prefix + table

        logger.info(f"Enhancing wide table: {rows} rows, {cols} columns")

        if rows <= LONG_TABLE_THRESHOLD:
            # Short wide table: use adjustbox
            # Wrap in raw LaTeX block
            enhanced = f"""{prefix}
```{{=latex}}
\\begin{{adjustbox}}{{max width=\\textwidth}}
```

{table}

```{{=latex}}
\\end{{adjustbox}}
```
"""
        else:
            # Long table: use smaller font (scriptsize)
            # longtable will be auto-applied by Pandoc
            enhanced = f"""{prefix}
```{{=latex}}
\\begingroup
\\scriptsize
```

{table}

```{{=latex}}
\\endgroup
```
"""

        return enhanced

    return table_pattern.sub(enhance_table, content)


def clean_page_markers(content: str) -> str:
    """
    Removes internal page markers before PDF generation.

    The [[PAGE_N]] markers are used internally for page-based processing
    but should not appear in the final PDF output.

    Args:
        content: Markdown content with page markers.

    Returns:
        Cleaned content without markers.
    """
    if not content:
        return content

    # Remove [[PAGE_N]] markers (keep as page breaks for Pandoc)
    # Replace with horizontal rule for visual separation
    content = re.sub(r"\[\[PAGE_\d+\]\]\s*", "\n\n---\n\n", content)

    # Remove any remaining {N}---- Datalab page separators that might have survived
    content = re.sub(r"\{\d+\}-{10,}\s*", "\n\n---\n\n", content)

    # Clean up multiple consecutive horizontal rules
    content = re.sub(r"(\n---\n){2,}", "\n---\n", content)

    # Clean up excessive newlines
    content = re.sub(r"\n{4,}", "\n\n\n", content)

    return content


def _sanitize_latex_content(content: str) -> str:
    """Apply LaTeX escaping and malformed-LaTeX fixes in a stable order."""
    content = escape_latex_specials(content)
    return fix_malformed_latex(content)


def sanitize_markdown(
    content: str,
    base_dir: str,
    fix_images: bool = True,
    escape_latex: bool = True,
    enhance_tables: bool = True,
    clean_markers: bool = True,
) -> str:
    """
    Applies all sanitization steps to Markdown content.

    This is the main entry point for the markdown cleaner.

    Args:
        content: Raw Markdown content.
        base_dir: Base directory for resolving image paths.
        fix_images: Whether to fix image paths.
        escape_latex: Whether to escape LaTeX special characters.
        enhance_tables: Whether to enhance wide tables.
        clean_markers: Whether to clean page markers.

    Returns:
        Sanitized Markdown ready for Pandoc.
    """
    if not content:
        return content

    logger.info(f"Sanitizing Markdown ({len(content)} chars)")

    # Clean page markers first (before other processing)
    if clean_markers:
        content = clean_page_markers(content)

    if fix_images:
        content = fix_image_paths(content, base_dir)

    if escape_latex:
        content = _sanitize_latex_content(content)

    if enhance_tables:
        content = enhance_wide_tables(content)

    logger.info(f"Sanitization complete ({len(content)} chars)")
    return content
