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
    
    # Pattern to find math blocks
    math_pattern = re.compile(r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+\$)')
    
    def fix_math_block(match: re.Match) -> str:
        math = match.group(0)
        original = math
        
        # Remove trailing ^ or _ without content (before closing $)
        # Pattern: ^$ or ^}$ or ^\s*$
        math = re.sub(r'\^(\s*)(\$|\})', r'\1\2', math)
        math = re.sub(r'_(\s*)(\$|\})', r'\1\2', math)
        
        # Fix double superscript: x^a^b -> x^{a^b}
        # This is a heuristic: wrap the first superscript content in braces
        # Pattern: letter/}^{...}^{...} or similar
        
        # Simpler fix: remove trailing bare ^ or _ that would cause double script
        # e.g., \mathbf{h}'_{\mathcal{N}_v}^ -> remove the trailing ^
        math = re.sub(r'(\^|\\_)(\s*)$', r'\2', math.rstrip('$').rstrip()) + ('$$' if math.startswith('$$') else '$')
        
        # Fix empty subscript/superscript: ^{} or _{}
        math = re.sub(r'\^\{\s*\}', '', math)
        math = re.sub(r'_\{\s*\}', '', math)
        
        if math != original:
            logger.debug(f"Fixed malformed LaTeX: {original[:50]}... -> {math[:50]}...")
        
        return math
    
    content = math_pattern.sub(fix_math_block, content)
    
    # Remove completely empty math blocks
    content = re.sub(r'\$\$\s*\$\$', '', content)
    content = re.sub(r'\$\s*\$', '', content)
    
    # Fix \begin{aligned} etc. appearing outside of math mode
    # These sub-environments MUST be inside $$ ... $$ or $ ... $
    # BUT we must NOT wrap them if they are already inside top-level math envs.
    
    math_sub_envs = ['aligned', 'gathered', 'split', 'cases', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'smallmatrix']
    
    # Top level environments that imply math mode
    # Include starred versions
    top_level_envs = ['equation', 'align', 'gather', 'flalign', 'alignat', 'multline']
    top_level_envs_all = top_level_envs + [e + r'\*' for e in top_level_envs]
    
    # 1. Identify protected ranges (Top Level Envs + Existing Math Blocks)
    protected_ranges = []
    
    # A. Top Level Environments
    # This regex attempts to match balanced environments.
    # Note: Regex recursivity is not fully supported in standard re, so we assume
    # top-level environments are not nested within OTHER top-level environments (which is standard Latex).
    env_regex = r'\\begin\{(' + '|'.join(top_level_envs_all) + r')\}[\s\S]*?\\end\{\1\}'
    for match in re.finditer(env_regex, content):
        protected_ranges.append(match.span())
        
    # B. Existing Math Blocks ($...$ and $$...$$ is already handled by math_pattern split? No, we need checking ranges)
    # Be careful, we already ran fix_math_block which might have changed content, but 
    # math_pattern.sub replaces in place.
    # Let's find them again to be safe.
    for match in math_pattern.finditer(content):
        protected_ranges.append(match.span())
        
    # Also Check \[ ... \] and \( ... \)
    # These are harder to match perfectly with simple regex if nested, but usually OK.
    for match in re.finditer(r'\\\[[\s\S]*?\\\]', content):
        protected_ranges.append(match.span())
    for match in re.finditer(r'\\\([\s\S]*?\\\)', content):
        protected_ranges.append(match.span())
        
    # Helper to check if a range is protected
    def is_protected(start_idx: int, end_idx: int) -> bool:
        for p_start, p_end in protected_ranges:
            if p_start <= start_idx and end_idx <= p_end:
                return True
        return False

    # 2. Find and wrap orphan sub-environments
    sub_env_pattern = re.compile(
        r'\\begin\{(' + '|'.join(math_sub_envs) + r')\}[\s\S]*?\\end\{\1\}',
        re.MULTILINE
    )
    
    replacements = []
    
    for match in sub_env_pattern.finditer(content):
        start, end = match.span()
        env_block = match.group(0)
        
        if not is_protected(start, end):
            # Check context before the match (double check for immediate delimiters)
            preceding_text = content[:start]
            stripped_preceding = preceding_text.rstrip()
            
            # Helper to check if a suffix is a valid unescaped delimiter
            def is_valid_delimiter(text: str, delimiter: str) -> bool:
                if not text.endswith(delimiter):
                    return False
                # Check for escape
                suffix_len = len(delimiter)
                pre_delim = text[:-suffix_len]
                backslashes = 0
                idx = len(pre_delim) - 1
                while idx >= 0 and pre_delim[idx] == '\\':
                    backslashes += 1
                    idx -= 1
                return backslashes % 2 == 0

            is_in_math_immediate = (
                is_valid_delimiter(stripped_preceding, '$') or
                is_valid_delimiter(stripped_preceding, r'\[') or
                is_valid_delimiter(stripped_preceding, r'\(')
            )
            
            if not is_in_math_immediate:
                # One last heuristic: Paragraph inspection
                # (Same as before to be safe against complex cases)
                last_para_idx = preceding_text.rfind('\n\n')
                if last_para_idx == -1: last_para_idx = 0
                para_prefix = preceding_text[last_para_idx:]
                dollar_count = len(re.findall(r'(?<!\\)\$', para_prefix))
                
                if dollar_count % 2 == 0: # Even dollars means we are likely OUTSIDE math
                    logger.debug(f"Wrapping orphan {match.group(0)[:20]}... in math mode")
                    replacements.append((start, end, f"$${env_block}$$"))

    # Apply replacements in reverse order
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + content[end:]
    
    # Clean up any potential double-double dollars
    content = re.sub(r'\$\$\$\$', '$$', content)
    
    # Final fix: \tag{...} appearing outside of math mode
    content = re.sub(r'\\\]\s*\\tag\{([^}]*)\}', r'\\] (Tag: \1)', content)
    content = re.sub(r'\\end\{aligned\}\s*\\\]\s*\\tag\{([^}]*)\}', r'\\end{aligned}\\] (Tag: \1)', content)
    content = re.sub(r'\$\$\s*\\tag\{([^}]*)\}', r'$$ (Tag: \1)', content)
    
    # Fix orphan math commands
    math_cmds = ['mathbf', 'mathrm', 'mathcal', 'mathbb', 'mathit', 'mathsf', 'mathtt']
    cmd_pattern = re.compile(
        r'\\(' + '|'.join(math_cmds) + r')\{[^{}]*\}',
        re.MULTILINE
    )
    
    # Re-calculate protected ranges? 
    # Actually, simpler: just check if check is inside math pattern
    # The previous logic for math commands was reasonably safe, but let's use the new protected_ranges logic
    # CAUTION: Content has changed due to sub_env replacements. 
    # Ideally we should re-scan protected ranges or track indices offset.
    # Given the complexity, let's just re-scan for math commands using a simplified regex replace
    # that skips matches if they are inside math delimiters.
    
    def repl_math_cmd(m):
        # This is a bit expensive to check global protection for every match in regex sub
        # So we keep the loop approach but re-check context
        return m.group(0) # Logic below

    replacements_cmds = []
    offset = 0 # To track content changes? No, we use index from CURRENT content
    # We need to re-scan protected ranges because content changed!
    # But for efficiency, maybe just trust the paragraph heuristic for these small commands?
    # Or, just re-run the protected range finder. It's fast enough.
    
    protected_ranges_v2 = []
    for match in re.finditer(env_regex, content): protected_ranges_v2.append(match.span())
    for match in math_pattern.finditer(content): protected_ranges_v2.append(match.span())
    for match in re.finditer(r'\\\[[\s\S]*?\\\]', content): protected_ranges_v2.append(match.span())
    for match in re.finditer(r'\\\([\s\S]*?\\\)', content): protected_ranges_v2.append(match.span())
    
    def is_protected_v2(s, e):
        for ps, pe in protected_ranges_v2:
            if ps <= s and e <= pe: return True
        return False

    for match in cmd_pattern.finditer(content):
        start, end = match.span()
        if not is_protected_v2(start, end):
             # Paragraph heuristic
            preceding_text = content[:start]
            last_para_idx = preceding_text.rfind('\n\n')
            if last_para_idx == -1: last_para_idx = 0
            para_prefix = preceding_text[last_para_idx:]
            dollar_count = len(re.findall(r'(?<!\\)\$', para_prefix))
            
            if dollar_count % 2 == 0:
                replacements_cmds.append((start, end, f"${match.group(0)}$"))
                
    for start, end, replacement in reversed(replacements_cmds):
        content = content[:start] + replacement + content[end:]
        
    # Catch-all for tag at end of line
    content = re.sub(r'(?<=\})\s*\\tag\{([^}]*)\}\s*$', r' (Tag: \1)', content, flags=re.MULTILINE)
    
    return content


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
    
    # Fix malformed LaTeX from OCR (always enabled when escape_latex is on)
    if escape_latex:
        content = fix_malformed_latex(content)
    
    if enhance_tables:
        content = enhance_wide_tables(content)
    
    logger.info(f"Sanitization complete ({len(content)} chars)")
    return content

