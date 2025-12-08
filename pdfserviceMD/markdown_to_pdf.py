"""
Markdown to PDF Conversion

Provides functionality to convert Markdown text with page markers to PDF.
"""

# Standard library
import logging
import re
from typing import List

# Third-party
from markdown_pdf import MarkdownPdf, Section

# Configure logging
logger = logging.getLogger(__name__)


def split_pages(markdown_text: str) -> List[str]:
    """
    Splits Markdown text by page markers.

    Args:
        markdown_text: Full Markdown text with [[PAGE_N]] markers.

    Returns:
        List of individual page contents.
    """
    pages = re.split(r"\[\[PAGE_\d+\]\]", markdown_text)
    return [page.strip() for page in pages if page.strip()]


def markdown_to_pdf(markdown_text: str, output_pdf: str) -> None:
    """
    Converts Markdown text to PDF with multiple pages.

    This is a SYNCHRONOUS function designed to be called via run_in_threadpool
    in async contexts, as PDF generation is CPU-bound.

    Args:
        markdown_text: Markdown content with [[PAGE_N]] markers.
        output_pdf: Output path for the PDF file.

    Raises:
        ValueError: If markdown_text is empty.
        IOError: If PDF generation fails.
    """
    if not markdown_text or not markdown_text.strip():
        raise ValueError("Cannot generate PDF from empty markdown")

    logger.info(f"Generating PDF: {output_pdf}")

    try:
        pages = split_pages(markdown_text)
        logger.debug(f"Split into {len(pages)} pages")

        pdf = MarkdownPdf(
            toc_level=0,
            optimize=True,
        )

        for idx, markdown_page in enumerate(pages):
            section = Section(
                markdown_page,
                root="output",
            )
            pdf.add_section(section)
            logger.debug(f"Added page {idx + 1}")

        pdf.save(output_pdf)
        logger.info(f"PDF saved successfully: {output_pdf}")

    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise IOError(f"Failed to generate PDF: {e}")
