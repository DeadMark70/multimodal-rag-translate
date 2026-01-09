"""
Pandoc Markdown to PDF Converter

Converts Markdown to PDF using Pandoc with XeLaTeX engine.
Includes markdown sanitization and fallback mechanisms.

Phase 7: PDF Generation Engine Upgrade
"""

# Standard library
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

# Third-party
import pypandoc

# Local application
from pdfserviceMD.markdown_cleaner import sanitize_markdown

# Configure logging
logger = logging.getLogger(__name__)

# Constants
_MODULE_DIR = Path(__file__).parent
_HEADER_TEX = _MODULE_DIR / "templates" / "header.tex"


def create_temp_markdown_file(markdown_text: str) -> str:
    """
    Creates a temporary markdown file and returns its path.
    
    Args:
        markdown_text: Markdown content to write.
        
    Returns:
        Path to the temporary file.
    """
    temp_md_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".md", mode="w", encoding="utf-8"
    )
    temp_md_file.write(markdown_text)
    temp_md_file.close()
    return temp_md_file.name


def MDmarkdown_to_pdf(
    markdown_text: str,
    output_pdf: str,
    base_dir: Optional[str] = None,
    debug: bool = False,
    enable_sanitization: bool = True,
) -> str:
    """
    Converts Markdown to PDF using Pandoc with XeLaTeX.
    
    Args:
        markdown_text: Markdown content to convert.
        output_pdf: Output PDF file path.
        base_dir: Base directory for resolving image paths.
                  Defaults to output PDF directory.
        debug: If True, keeps intermediate .tex file for debugging.
        enable_sanitization: If True, applies markdown sanitization.
        
    Returns:
        Path to the generated PDF.
        
    Raises:
        RuntimeError: If PDF generation fails.
    """
    markdownfile = None
    tex_file = None
    
    try:
        # Default base_dir to output directory
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(output_pdf))
        
        # Phase 7.1: Sanitize markdown before conversion
        if enable_sanitization:
            logger.info("Sanitizing Markdown content...")
            markdown_text = sanitize_markdown(
                content=markdown_text,
                base_dir=base_dir,
                fix_images=True,
                escape_latex=True,
                enhance_tables=False,  # Disabled: conflicts with Pandoc's longtable
            )
        
        markdownfile = create_temp_markdown_file(markdown_text)
        
        # Phase 7.2: Enhanced Pandoc arguments
        # 使用 article + xeCJK 取代 ctexart，避免 SimHei 字體依賴問題
        extra_pandoc_args = [
            "--pdf-engine=xelatex",
            "-s",
            # Allow raw LaTeX and pipe tables
            "--from=markdown+raw_tex+pipe_tables+yaml_metadata_block",
            # 使用 article 文檔類 + 手動 CJK 設定（更可攜）
            '--variable', 'documentclass=article',
            '--variable', 'mainfont=Microsoft JhengHei',
            '--variable', 'sansfont=Microsoft JhengHei',
            '--variable', 'monofont=Consolas',
            '--variable', 'mathfont=Latin Modern Math',
            '--variable', 'geometry=margin=1in',
            # CJK 支援：透過 header.tex 注入 xeCJK
            # Link colors
            '--variable', 'linkcolor=blue',
            '--variable', 'urlcolor=blue',
            # Resource path for images
            f'--resource-path={base_dir}',
        ]
        
        # Add custom header if exists
        if _HEADER_TEX.exists():
            extra_pandoc_args.append(f'--include-in-header={_HEADER_TEX}')
            logger.debug(f"Using custom header: {_HEADER_TEX}")
        
        # Phase 7.3.1: Debug mode - save intermediate .tex
        if debug:
            tex_file = output_pdf.replace('.pdf', '.tex')
            logger.info(f"Debug mode: saving intermediate .tex to {tex_file}")
            pypandoc.convert_file(
                source_file=markdownfile,
                to="latex",
                outputfile=tex_file,
                extra_args=extra_pandoc_args,
            )
        
        # Main conversion
        logger.info(f"Converting Markdown to PDF: {output_pdf}")
        pypandoc.convert_file(
            source_file=markdownfile,
            to="pdf",
            outputfile=output_pdf,
            extra_args=extra_pandoc_args,
        )
        
        logger.info(f"PDF created successfully: {output_pdf}")
        return output_pdf
        
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"XeLaTeX conversion failed: {error_msg}")
        
        # Phase 7.3.2: Fallback to HTML PDF
        if "LaTeX" in error_msg or "xelatex" in error_msg.lower():
            logger.warning("Attempting HTML fallback...")
            try:
                return _fallback_html_to_pdf(markdown_text, output_pdf)
            except (RuntimeError, ImportError) as fallback_error:
                logger.error(f"HTML fallback also failed: {fallback_error}")
                raise RuntimeError(
                    f"PDF generation failed. XeLaTeX: {error_msg}. "
                    f"HTML fallback: {fallback_error}"
                )
        raise
        
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"File system error: {e}")
        raise
        
    finally:
        # Cleanup temp files (but not debug .tex)
        if markdownfile and os.path.exists(markdownfile):
            os.remove(markdownfile)
            logger.debug(f"Removed temporary Markdown file: {markdownfile}")


def _fallback_html_to_pdf(markdown_text: str, output_pdf: str) -> str:
    """
    Fallback conversion using HTML to PDF.
    
    Uses weasyprint if available, otherwise raises ImportError.
    
    Args:
        markdown_text: Markdown content.
        output_pdf: Output PDF path.
        
    Returns:
        Path to the generated PDF.
        
    Raises:
        ImportError: If weasyprint is not installed.
        RuntimeError: If conversion fails.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        logger.warning("weasyprint not installed, HTML fallback unavailable")
        raise ImportError(
            "HTML fallback requires weasyprint. "
            "Install with: pip install weasyprint"
        )
    
    temp_html = None
    try:
        # Convert Markdown to HTML first
        html_content = pypandoc.convert_text(
            markdown_text,
            to="html",
            format="markdown",
            extra_args=['--standalone', '--metadata', 'title=Document'],
        )
        
        # Add minimal CSS for better appearance
        styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Microsoft JhengHei', sans-serif; margin: 2cm; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
        pre {{ background-color: #f4f4f4; padding: 1em; overflow-x: auto; }}
        code {{ font-family: Consolas, monospace; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        ) as f:
            f.write(styled_html)
            temp_html = f.name
        
        # Convert to PDF
        HTML(temp_html).write_pdf(output_pdf)
        logger.info(f"HTML fallback successful: {output_pdf}")
        return output_pdf
        
    finally:
        if temp_html and os.path.exists(temp_html):
            os.remove(temp_html)


# Convenience function for backward compatibility
def convert_markdown_to_pdf(
    markdown_text: str,
    output_pdf: str,
    base_dir: Optional[str] = None,
) -> str:
    """
    Convenience wrapper for MDmarkdown_to_pdf.
    
    Args:
        markdown_text: Markdown content.
        output_pdf: Output PDF path.
        base_dir: Base directory for images.
        
    Returns:
        Path to the generated PDF.
    """
    return MDmarkdown_to_pdf(
        markdown_text=markdown_text,
        output_pdf=output_pdf,
        base_dir=base_dir,
        debug=False,
        enable_sanitization=True,
    )
