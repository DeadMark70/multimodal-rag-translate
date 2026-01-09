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
from typing import Tuple, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PLACEHOLDER_IMAGE = "[Image Not Found]"
WIDE_TABLE_THRESHOLD = 5  # Columns
LONG_TABLE_THRESHOLD = 15  # Rows


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
        if img_path.startswith(('http://', 'https://', 'data:')):
            return match.group(0)
        
        # Convert to absolute path
        if not os.path.isabs(img_path):
            abs_path = os.path.normpath(os.path.join(base_dir, img_path))
        else:
            abs_path = os.path.normpath(img_path)
        
        # Check if image exists
        if os.path.exists(abs_path):
            # Use forward slashes for LaTeX compatibility
            latex_path = abs_path.replace('\\', '/')
            logger.debug(f"Image path fixed: {img_path} -> {latex_path}")
            return f'![{alt_text}]({latex_path})'
        else:
            logger.warning(f"Image not found: {abs_path}")
            return f'**{PLACEHOLDER_IMAGE}: {os.path.basename(img_path)}**'
    
    fixed_content = img_pattern.sub(replace_image, content)
    
    # Also handle HTML img tags from Marker output
    html_img_pattern = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>')
    
    def replace_html_image(match: re.Match) -> str:
        img_path = match.group(1)
        
        # Skip URLs
        if img_path.startswith(('http://', 'https://', 'data:')):
            return match.group(0)
        
        # Convert to absolute path
        if not os.path.isabs(img_path):
            abs_path = os.path.normpath(os.path.join(base_dir, img_path))
        else:
            abs_path = os.path.normpath(img_path)
        
        # Check if image exists
        if os.path.exists(abs_path):
            latex_path = abs_path.replace('\\', '/')
            # Replace src attribute with absolute path
            return match.group(0).replace(img_path, latex_path)
        else:
            logger.warning(f"HTML image not found: {abs_path}")
            return f'<p><strong>{PLACEHOLDER_IMAGE}: {os.path.basename(img_path)}</strong></p>'
    
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
    math_pattern = re.compile(r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+\$)')
    
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
            part = re.sub(r'(?<!\\)%', r'\\%', part)
            
            # Escape # (but not \#)
            part = re.sub(r'(?<!\\)#', r'\\#', part)
            
            # Escape & (but not \& and not in table separators)
            # Be careful: Markdown tables use | not &, but LaTeX tables use &
            # Only escape & that appears to be in prose, not table context
            # Simple heuristic: if & is surrounded by spaces or at word boundaries
            part = re.sub(r'(?<!\\)\s&\s', r' \\& ', part)
            
            result_parts.append(part)
    
    return ''.join(result_parts)


def _count_table_dimensions(table_text: str) -> Tuple[int, int]:
    """
    Counts rows and columns in a Markdown table.
    
    Args:
        table_text: The Markdown table text.
        
    Returns:
        Tuple of (row_count, column_count).
    """
    lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]
    if not lines:
        return 0, 0
    
    # Count rows (excluding separator line)
    rows = len([l for l in lines if not re.match(r'^[\|\s\-:]+$', l)])
    
    # Count columns from first line
    first_line = lines[0]
    cols = first_line.count('|') - 1  # Subtract 1 for leading/trailing |
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
        r'(\n|^)(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)',
        re.MULTILINE
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
    content = re.sub(r'\[\[PAGE_\d+\]\]\s*', '\n\n---\n\n', content)
    
    # Remove any remaining {N}---- Datalab page separators that might have survived
    content = re.sub(r'\{\d+\}-{10,}\s*', '\n\n---\n\n', content)
    
    # Clean up multiple consecutive horizontal rules
    content = re.sub(r'(\n---\n){2,}', '\n---\n', content)
    
    # Clean up excessive newlines
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    return content


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
        content = escape_latex_specials(content)
    
    if enhance_tables:
        content = enhance_wide_tables(content)
    
    logger.info(f"Sanitization complete ({len(content)} chars)")
    return content

