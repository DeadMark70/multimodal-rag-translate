"""
Translation Chunker Module

Provides page-based chunking for large document translation.
Splits markdown by [[PAGE_N]] markers and batches pages based on output token limits.
Uses batch translation (multiple pages per API call) to reduce request count.
"""

# Standard library
import logging
import re
from typing import List, Tuple, Set

# Third-party
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 60000  # Leave buffer below 65K limit
CHARS_PER_TOKEN_ESTIMATE = 1.33  # ~0.75 tokens per char for Chinese
DEBUG_LOG_PATH = "output/translation_debug.log"


def _write_debug_log(content: str) -> None:
    """
    Writes debug content to a log file for translation analysis.

    Args:
        content: Debug content to append to log file.
    """
    import os
    os.makedirs("output", exist_ok=True)
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        logger.warning(f"Failed to write debug log: {e}")

# Few-shot translation prompt with concrete examples
STRICT_TRANSLATION_PROMPT = """你是一個專業翻譯系統。將 Markdown 文字翻譯成繁體中文。

⚠️ 關鍵規則:
1. [[PAGE_X]] 標記必須原封不動保留（輸入幾個就輸出幾個）
2. [IMG_PLACEHOLDER_X] 標記必須保留
3. 保留所有 Markdown 格式、數學公式、HTML 標籤

---
【範例輸入】
[[PAGE_1]]
# Introduction
Machine learning is a subset of artificial intelligence.

[[PAGE_2]]
## Methods
We used deep neural networks for classification.

【範例輸出】
[[PAGE_1]]
# 介紹
機器學習是人工智慧的一個分支。

[[PAGE_2]]
## 方法
我們使用深度神經網路進行分類。

---
請翻譯以下內容（保留所有 [[PAGE_X]] 標記）:

{input_text}

翻譯後直接輸出:"""

# Retry configuration
MAX_TRANSLATION_RETRIES = 2  # Maximum retry attempts when markers are lost


def estimate_tokens(text: str) -> int:
    """
    Estimates the number of tokens in text.

    Uses a rough estimate of ~0.75 tokens per character for Chinese text.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def split_by_page_markers(markdown: str) -> List[Tuple[int, str]]:
    """
    Splits markdown text by [[PAGE_N]] markers.

    Args:
        markdown: Full markdown text with [[PAGE_N]] markers from OCR.

    Returns:
        List of (page_number, content) tuples.
    """
    if not markdown or not markdown.strip():
        return []

    # Pattern to match [[PAGE_N]] markers
    pattern = r"\[\[PAGE_(\d+)\]\]"
    parts = re.split(pattern, markdown)

    pages: List[Tuple[int, str]] = []

    # parts[0] is content before first marker (usually empty)
    # parts[1], parts[3], parts[5]... are page numbers
    # parts[2], parts[4], parts[6]... are page contents
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            page_num = int(parts[i])
            content = parts[i + 1].strip()
            if content:
                pages.append((page_num, content))

    logger.debug(f"Split into {len(pages)} pages")
    return pages


def batch_pages(
    pages: List[Tuple[int, str]],
    max_output_tokens: int = MAX_OUTPUT_TOKENS
) -> List[List[Tuple[int, str]]]:
    """
    Batches pages together respecting output token limits.

    Estimates output length as input * 1.2 (translated text may be longer).

    Args:
        pages: List of (page_number, content) tuples.
        max_output_tokens: Maximum tokens per batch output.

    Returns:
        List of batches, each containing (page_number, content) tuples.
    """
    if not pages:
        return []

    batches: List[List[Tuple[int, str]]] = []
    current_batch: List[Tuple[int, str]] = []
    current_tokens = 0

    for page_num, content in pages:
        # Estimate output tokens (input * 1.2 for translation expansion)
        estimated_output = estimate_tokens(content) * 1.2

        # Start new batch if current would exceed limit
        if current_tokens + estimated_output > max_output_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append((page_num, content))
        current_tokens += estimated_output

    # Add remaining batch
    if current_batch:
        batches.append(current_batch)

    logger.info(f"Created {len(batches)} translation batches from {len(pages)} pages")
    return batches


def combine_batch_for_translation(batch: List[Tuple[int, str]]) -> str:
    """
    Combines a batch of pages into a single string for translation.

    Preserves [[PAGE_N]] markers for later reassembly.

    Args:
        batch: List of (page_number, content) tuples.

    Returns:
        Combined markdown string with page markers.
    """
    return "\n\n".join(
        f"[[PAGE_{page_num}]]\n{content}"
        for page_num, content in batch
    )


def find_paragraph_boundary(text: str, position: int) -> int:
    """
    Finds the nearest paragraph boundary (double newline) near a position.

    Args:
        text: The text to search.
        position: Target position.

    Returns:
        Position of the nearest paragraph boundary.
    """
    if position >= len(text):
        return len(text)

    # Look for double newline within 200 chars of target
    search_start = max(0, position - 100)
    search_end = min(len(text), position + 100)
    search_region = text[search_start:search_end]

    # Find double newline
    double_newline = search_region.find("\n\n")
    if double_newline != -1:
        return search_start + double_newline + 2

    # Fallback to single newline
    single_newline = search_region.find("\n", position - search_start)
    if single_newline != -1:
        return search_start + single_newline + 1

    return position


def validate_markers(translated: str, expected_markers: Set[str]) -> bool:
    """
    Validates that all expected page markers are present in translated text.

    Args:
        translated: Translated text.
        expected_markers: Set of expected markers like {"[[PAGE_1]]", "[[PAGE_2]]"}.

    Returns:
        True if all markers present, False otherwise.
    """
    found_markers = set(re.findall(r"\[\[PAGE_\d+\]\]", translated))
    return expected_markers == found_markers


def repair_markers(
    translated: str,
    original_pages: List[Tuple[int, str]]
) -> str:
    """
    Repairs missing page markers by splitting translated content proportionally.

    When LLM removes [[PAGE_N]] markers, this function reconstructs them
    based on the original page structure.

    Args:
        translated: Translated text with potentially missing markers.
        original_pages: Original (page_number, content) tuples.

    Returns:
        Translated text with [[PAGE_N]] markers restored.
    """
    if not original_pages:
        return translated

    # Remove any existing (possibly malformed) markers
    clean_translated = re.sub(r"\[\[PAGE_\d+\]\]\s*", "", translated).strip()

    if not clean_translated:
        return translated

    # Calculate proportional lengths
    total_original_length = sum(len(content) for _, content in original_pages)
    if total_original_length == 0:
        return translated

    rebuilt_parts: List[str] = []
    current_pos = 0

    for page_num, original_content in original_pages:
        # Calculate this page's proportion
        ratio = len(original_content) / total_original_length
        chunk_length = int(len(clean_translated) * ratio)

        # Find a good boundary position
        end_pos = find_paragraph_boundary(clean_translated, current_pos + chunk_length)

        # Extract chunk
        if page_num == original_pages[-1][0]:
            # Last page gets everything remaining
            chunk = clean_translated[current_pos:].strip()
        else:
            chunk = clean_translated[current_pos:end_pos].strip()

        if chunk:
            rebuilt_parts.append(f"[[PAGE_{page_num}]]\n{chunk}")

        current_pos = end_pos

    logger.info(f"Repaired markers: rebuilt {len(rebuilt_parts)} pages")
    return "\n\n".join(rebuilt_parts)


async def translate_batch(batch: List[Tuple[int, str]]) -> str:
    """
    Translates a batch of pages in a single API call with retry support.

    Uses few-shot prompt to preserve [[PAGE_N]] markers. If markers are lost,
    retries up to MAX_TRANSLATION_RETRIES times before falling back to repair.

    Args:
        batch: List of (page_number, content) tuples.

    Returns:
        Translated markdown with page markers guaranteed.
    """
    # Combine pages with markers
    combined = combine_batch_for_translation(batch)
    expected_markers = {f"[[PAGE_{p[0]}]]" for p in batch}

    page_range = f"{batch[0][0]}-{batch[-1][0]}" if len(batch) > 1 else str(batch[0][0])
    logger.info(f"Translating pages {page_range} ({len(batch)} pages, {len(combined)} chars)...")

    # === DEBUG: Write input to file ===
    _write_debug_log(
        f"\n{'='*60}\n"
        f"[DEBUG] TRANSLATION INPUT (pages {page_range})\n"
        f"{'='*60}\n"
        f"Expected markers: {expected_markers}\n"
        f"Input length: {len(combined)} chars\n"
        f"{'='*60}\n"
        f"{combined}\n"
        f"{'='*60}\n"
    )

    # Use few-shot prompt
    prompt = ChatPromptTemplate.from_template(STRICT_TRANSLATION_PROMPT)
    llm = get_llm("translation")
    chain = prompt | llm | StrOutputParser()

    # Retry loop
    last_translated = ""
    for attempt in range(MAX_TRANSLATION_RETRIES + 1):
        try:
            translated = await chain.ainvoke({"input_text": combined})
            last_translated = translated

            # === DEBUG: Write output to file ===
            found_markers = set(re.findall(r"\[\[PAGE_\d+\]\]", translated))
            _write_debug_log(
                f"\n{'='*60}\n"
                f"[DEBUG] TRANSLATION OUTPUT (pages {page_range}) - Attempt {attempt + 1}\n"
                f"{'='*60}\n"
                f"Found markers: {found_markers}\n"
                f"Expected markers: {expected_markers}\n"
                f"Markers match: {expected_markers == found_markers}\n"
                f"Output length: {len(translated)} chars\n"
                f"{'='*60}\n"
                f"{translated}\n"
                f"{'='*60}\n"
            )

            # Validate markers
            if validate_markers(translated, expected_markers):
                if attempt > 0:
                    logger.info(
                        f"Batch translation succeeded on attempt {attempt + 1} "
                        f"(pages {page_range})"
                    )
                else:
                    logger.info(f"Batch translation completed (pages {page_range}) - markers intact")
                return translated

            # Markers lost - retry if attempts remaining
            if attempt < MAX_TRANSLATION_RETRIES:
                logger.warning(
                    f"Markers lost (attempt {attempt + 1}/{MAX_TRANSLATION_RETRIES + 1}), "
                    f"retrying... Expected: {len(expected_markers)}, Found: {len(found_markers)}"
                )
            else:
                # All retries exhausted - use repair
                logger.warning(
                    f"Markers lost after {MAX_TRANSLATION_RETRIES + 1} attempts, "
                    f"using repair fallback. Expected: {expected_markers}, Found: {found_markers}"
                )
                repaired = repair_markers(translated, batch)
                return repaired

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Batch translation attempt {attempt + 1} failed: {e}")
            if attempt == MAX_TRANSLATION_RETRIES:
                logger.error("All translation attempts failed, returning original content")
                return combined

    # Should not reach here, but return original as fallback
    return combined


async def translate_chunked(markdown: str) -> str:
    """
    Translates markdown text using page-based chunking.

    Main entry point for translation. Splits by [[PAGE_N]] markers,
    batches pages to stay within output token limits, translates each
    batch, and combines results.

    Args:
        markdown: Full markdown text with [[PAGE_N]] markers from OCR.

    Returns:
        Translated markdown text in Traditional Chinese.
        Returns original text if translation fails.
    """
    if not markdown or not markdown.strip():
        logger.warning("Empty markdown provided for translation")
        return markdown

    # === DEBUG: Clear and initialize debug log ===
    from datetime import datetime
    import os
    if os.path.exists(DEBUG_LOG_PATH):
        os.remove(DEBUG_LOG_PATH)
    _write_debug_log(
        f"{'#'*60}\n"
        f"TRANSLATION DEBUG LOG\n"
        f"Started at: {datetime.now().isoformat()}\n"
        f"Total markdown length: {len(markdown)} chars\n"
        f"{'#'*60}\n"
    )

    # Step 1: Split by page markers
    pages = split_by_page_markers(markdown)

    if not pages:
        # No page markers found - translate as single page
        logger.warning("No [[PAGE_N]] markers found, using single-call translation")
        return await translate_batch([(1, markdown)])

    # Step 2: Batch pages based on token limits
    batches = batch_pages(pages)

    if len(batches) == 1:
        # Single batch - translate directly
        logger.info("Single batch translation (all pages fit in one call)")
        return await translate_batch(batches[0])

    # Step 3: Translate each batch
    logger.info(f"Multi-batch translation: {len(batches)} batches")
    translated_batches: List[str] = []

    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i + 1}/{len(batches)}...")
        translated = await translate_batch(batch)
        translated_batches.append(translated)

    # Step 4: Combine results
    result = "\n\n".join(translated_batches)
    logger.info(f"Translation complete: {len(pages)} pages, {len(batches)} batches")

    return result
