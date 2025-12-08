"""
Markdown Processing Utilities

Provides functions for extracting and replacing image blocks in Markdown.
"""

# Standard library
import logging
import re
from typing import List, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def markdown_extact(ocr_result: str) -> Tuple[str, List[str]]:
    """
    Extracts image blocks from Markdown and replaces them with placeholders.

    This allows translation of text content without corrupting image tags,
    which can then be restored after translation.

    Args:
        ocr_result: Raw Markdown text from OCR with embedded image divs.

    Returns:
        Tuple of:
        - Processed Markdown with image placeholders
        - List of original image blocks (in order)
    """
    if not ocr_result:
        return "", []

    image_blocks: List[str] = []
    processed_content = ocr_result

    # Match <div><img></div> blocks
    img_div_regex = re.compile(r'(<div\s+[^>]*?><img\s+[^>]*?></div>)', re.DOTALL)
    matches = list(img_div_regex.finditer(ocr_result))

    logger.debug(f"Found {len(matches)} image blocks")

    # Process matches in reverse order to preserve indices
    for i, match in enumerate(reversed(matches)):
        full_match = match.group(0)
        # Insert at beginning to maintain order with forward indices
        image_blocks.insert(0, full_match)
        placeholder = f"[IMG_PLACEHOLDER_{len(matches) - 1 - i}]"
        processed_content = (
            processed_content[:match.start()] +
            placeholder +
            processed_content[match.end():]
        )

    logger.info(f"Extracted {len(image_blocks)} image blocks")
    return processed_content, image_blocks


def replace_markdown(translate_content: str, original_image_blocks: List[str]) -> str:
    """
    Restores image blocks from placeholders in translated Markdown.

    Args:
        translate_content: Translated Markdown with image placeholders.
        original_image_blocks: List of original image blocks to restore.

    Returns:
        Final Markdown with images restored.
    """
    if not translate_content:
        return ""

    final_markdown = translate_content

    for i, img_block in enumerate(original_image_blocks):
        placeholder = f"[IMG_PLACEHOLDER_{i}]"
        final_markdown = final_markdown.replace(placeholder, img_block)

    logger.debug(f"Restored {len(original_image_blocks)} image blocks")
    return final_markdown