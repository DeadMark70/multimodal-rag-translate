"""
Image Processor for PDF OCR Pipeline

Extracts image paths from Markdown content and creates VisualElement objects
for summarization and indexing.

Phase 8: Image Pipeline Integration
"""

# Standard library
import logging
import os
import re
from typing import List, Tuple, Optional
from uuid import uuid4

# Local application
from multimodal_rag.schemas import VisualElement, VisualElementType

# Configure logging
logger = logging.getLogger(__name__)

# Regex pattern for Markdown images: ![alt](path) or ![alt](path "title")
IMAGE_PATTERN = re.compile(
    r'!\[([^\]]*)\]\(([^)"\']+)(?:\s*["\'][^"\']*["\'])?\)',
    re.MULTILINE
)

# Page marker pattern: [[PAGE_N]]
PAGE_MARKER_PATTERN = re.compile(r'\[\[PAGE_(\d+)\]\]')


def extract_images_from_markdown(
    markdown_text: str,
    base_dir: str,
) -> List[Tuple[str, int, str]]:
    """
    Extracts image paths from Markdown content.

    Parses Markdown for ![alt](path) syntax and resolves paths to absolute.
    Also extracts page number and surrounding context text.

    Args:
        markdown_text: Markdown content with image references.
        base_dir: Base directory for resolving relative paths.

    Returns:
        List of tuples: (absolute_path, page_number, context_text)
        Only existing files are returned.
    """
    if not markdown_text:
        return []

    base_dir = os.path.normpath(base_dir)
    results: List[Tuple[str, int, str]] = []

    # Split by page markers to track page numbers
    page_sections = PAGE_MARKER_PATTERN.split(markdown_text)

    # page_sections: [content_before_first, page_num_1, content_1, page_num_2, ...]
    current_page = 1

    for i, section in enumerate(page_sections):
        # Even indices are content, odd indices are page numbers
        if i % 2 == 1:
            try:
                current_page = int(section)
            except ValueError:
                pass
            continue

        # Search for images in this section
        for match in IMAGE_PATTERN.finditer(section):
            alt_text = match.group(1)
            img_path = match.group(2)

            # Skip URLs
            if img_path.startswith(('http://', 'https://', 'data:')):
                continue

            # Resolve to absolute path
            if os.path.isabs(img_path):
                abs_path = os.path.normpath(img_path)
            else:
                abs_path = os.path.normpath(os.path.join(base_dir, img_path))

            # Check file exists
            if not os.path.exists(abs_path):
                logger.debug(f"Image not found: {abs_path}")
                continue

            # Extract context (surrounding text)
            context = _extract_context(section, match.start(), match.end())

            results.append((abs_path, current_page, context))

    logger.info(f"Extracted {len(results)} existing images from markdown")
    return results


def _extract_context(
    text: str,
    match_start: int,
    match_end: int,
    context_chars: int = 200,
) -> str:
    """
    Extracts surrounding context text around an image reference.

    Args:
        text: Full section text.
        match_start: Start position of image match.
        match_end: End position of image match.
        context_chars: Number of characters to extract before/after.

    Returns:
        Context text with the image reference replaced by [IMAGE].
    """
    # Get text before and after
    start = max(0, match_start - context_chars)
    end = min(len(text), match_end + context_chars)

    before = text[start:match_start].strip()
    after = text[match_end:end].strip()

    # Clean up: remove extra whitespace and newlines
    before = re.sub(r'\s+', ' ', before)
    after = re.sub(r'\s+', ' ', after)

    return f"{before} [IMAGE] {after}".strip()


def create_visual_elements(
    image_data: List[Tuple[str, int, str]],
    doc_title: str = "",
) -> List[VisualElement]:
    """
    Creates VisualElement objects from extracted image data.

    Args:
        image_data: List of (absolute_path, page_number, context_text).
        doc_title: Document title for context.

    Returns:
        List of VisualElement objects ready for summarization.
    """
    elements: List[VisualElement] = []

    for abs_path, page_num, context_text in image_data:
        element = VisualElement(
            id=uuid4(),
            type=VisualElementType.FIGURE,
            page_number=page_num,
            image_path=abs_path,
            bbox=[0, 0, 0, 0],  # Not available from markdown extraction
            original_text=None,
            summary=None,
            context_text=context_text,
            figure_reference=_extract_figure_reference(context_text),
        )
        elements.append(element)

    logger.info(f"Created {len(elements)} VisualElement objects")
    return elements


def _extract_figure_reference(context_text: str) -> Optional[str]:
    """
    Extracts figure reference label from context text.

    Looks for patterns like "Figure 1", "Fig. 2", "圖一", "圖 3" etc.

    Args:
        context_text: Surrounding text context.

    Returns:
        Figure reference string if found, None otherwise.
    """
    # English patterns: Figure 1, Fig. 1, Fig 1
    en_pattern = re.compile(r'(?:Figure|Fig\.?)\s*(\d+)', re.IGNORECASE)
    match = en_pattern.search(context_text)
    if match:
        return f"Figure {match.group(1)}"

    # Chinese patterns: 圖1, 圖一, 圖 1
    zh_pattern = re.compile(r'圖\s*([一二三四五六七八九十\d]+)')
    match = zh_pattern.search(context_text)
    if match:
        return f"圖 {match.group(1)}"

    return None
