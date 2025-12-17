"""
AI Translation Service

Provides Markdown translation functionality using Google Gemini API.
Uses page-based chunking for large documents to stay within output token limits.
"""

# Standard library
import logging

# Local application
from pdfserviceMD.translation_chunker import translate_chunked

# Configure logging
logger = logging.getLogger(__name__)


async def translate_text(text: str) -> str:
    """
    Translates Markdown text to Traditional Chinese while preserving formatting.

    Uses page-based chunking for large documents. Markdown is split by [[PAGE_N]]
    markers, batched based on output token limits, and translated in chunks.

    Args:
        text: Markdown text to translate (with [[PAGE_N]] markers from OCR).

    Returns:
        Translated Markdown text in Traditional Chinese.
        Returns original text if translation fails (graceful degradation).
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for translation")
        return text

    try:
        logger.info(f"Starting translation ({len(text)} chars)...")
        result = await translate_chunked(text)
        logger.info("Translation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return text  # Graceful degradation
