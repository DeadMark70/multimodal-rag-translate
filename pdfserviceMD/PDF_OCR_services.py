"""
PDF OCR Services (Hybrid: Local Marker / Datalab API)

Provides PDF to Markdown conversion using either:
- Local Marker (free, on-device) - set USE_LOCAL_MARKER=true
- Datalab API (paid, cloud) - default

Automatically preserves page boundaries with [[PAGE_N]] markers.
"""

# Standard library
import logging
import os
import asyncio
from typing import Dict, Any

# Local application
from core.providers import DatalabProvider, ProviderError, get_datalab_provider

# Configure logging
logger = logging.getLogger(__name__)

# Mode Configuration
USE_LOCAL_MARKER = os.getenv("USE_LOCAL_MARKER", "true").lower() in ("true", "1", "yes")


def _build_markdown_with_page_markers(api_response: Dict[str, Any]) -> str:
    """
    Builds combined Markdown with [[PAGE_N]] markers from API response.

    Datalab API response format:
    - markdown: Full markdown text with {N}---- page separators
    - page_count: Number of pages
    - success: Boolean
    
    Args:
        api_response: Datalab API response.

    Returns:
        Combined Markdown text with page markers.
    """
    import re
    
    # Debug: log the response keys
    logger.debug(f"API response keys: {list(api_response.keys())}")
    
    # Datalab returns markdown at root level
    markdown = api_response.get("markdown", "")
    
    if not markdown:
        # Try alternative field names
        markdown = api_response.get("text", "")
        markdown = markdown or api_response.get("content", "")
        markdown = markdown or api_response.get("output", "")
    
    if not markdown:
        logger.warning(f"No markdown found in response. Keys: {list(api_response.keys())}")
        # Try to extract from pages if present
        pages = api_response.get("pages", [])
        if pages:
            logger.info(f"Found {len(pages)} pages in response")
            markdown_parts = []
            for i, page in enumerate(pages):
                page_num = page.get("page_number", page.get("page", i + 1))
                page_md = page.get("markdown", page.get("text", page.get("content", "")))
                if page_md:
                    markdown_parts.append(f"[[PAGE_{page_num}]]\n{page_md}")
            if markdown_parts:
                return "\n\n".join(markdown_parts)
        return ""
    
    # Get page count from response
    page_count = api_response.get("page_count", 1)
    logger.debug(f"Markdown length: {len(markdown)}, Page count: {page_count}")
    
    # Datalab uses {N}---- format for page separators
    # Pattern: {0}---- or {1}---- etc. followed by dashes
    datalab_page_pattern = re.compile(r'\{(\d+)\}-{10,}', re.MULTILINE)
    
    # Check if content has Datalab page separators
    if datalab_page_pattern.search(markdown):
        logger.info("Detected Datalab page separator format")
        # Split by page separators and build with our markers
        parts = datalab_page_pattern.split(markdown)
        
        # parts will be: [content_before_first, page_num_1, content_1, page_num_2, content_2, ...]
        markdown_parts = []
        
        # Handle content before first separator (if any)
        if parts and parts[0].strip():
            markdown_parts.append(f"[[PAGE_1]]\n{parts[0].strip()}")
        
        # Process remaining parts (alternating: page_num, content)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                page_num = int(parts[i]) + 1  # Convert 0-indexed to 1-indexed
                content = parts[i + 1].strip()
                if content:
                    markdown_parts.append(f"[[PAGE_{page_num}]]\n{content}")
        
        if markdown_parts:
            result = "\n\n".join(markdown_parts)
            logger.info(f"Built markdown with {len(markdown_parts)} pages")
            return result
    
    # Fallback: Check for horizontal rule separators
    if page_count > 1:
        separator_patterns = ["\n---\n\n", "\n\n---\n\n", "\n---\n"]
        
        for sep in separator_patterns:
            if sep in markdown:
                pages_split = markdown.split(sep)
                if len(pages_split) >= page_count - 1:
                    markdown_parts = []
                    for i, page_md in enumerate(pages_split):
                        page_md = page_md.strip()
                        if page_md:
                            markdown_parts.append(f"[[PAGE_{i+1}]]\n{page_md}")
                    if markdown_parts:
                        logger.info(f"Split into {len(markdown_parts)} pages using separator")
                        return "\n\n".join(markdown_parts)
                    break
        
        logger.info(f"No page separator found, treating as single document with {page_count} pages")
    
    # Return with PAGE_1 marker
    return f"[[PAGE_1]]\n{markdown}"


def initialize_predictor() -> None:
    """
    Validates OCR configuration during app startup.
    
    This function should be called from main.py's startup event.
    """
    if USE_LOCAL_MARKER:
        logger.info("PDF OCR mode: LOCAL MARKER (free, on-device)")
        # Check if marker is available
        try:
            from pdfserviceMD.local_marker_service import is_marker_available
            if not is_marker_available():
                logger.warning("marker-pdf not installed! Run: pip install marker-pdf")
            else:
                logger.info("Local Marker ready")
        except ImportError:
            logger.warning("local_marker_service not found")
    else:
        logger.info("PDF OCR mode: DATALAB API (paid, cloud)")
        provider = get_datalab_provider()
        if not provider.is_configured():
            logger.warning(
                "DATALAB_API_KEY not set! PDF OCR will fail. "
                "Set it in config.env or environment variables."
            )
        else:
            logger.info("Datalab provider configured")


def ocr_service_sync(pdf_path: str, datalab_provider: DatalabProvider | None = None) -> str:
    """
    Performs OCR on a PDF file and returns combined Markdown text.

    Automatically chooses between local Marker or Datalab API based on
    USE_LOCAL_MARKER environment variable.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Combined Markdown text with page markers.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        RuntimeError: If OCR processing fails.
    """
    # Validate input
    pdf_path = os.path.normpath(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Choose OCR method
    if USE_LOCAL_MARKER:
        return _ocr_with_local_marker(pdf_path)
    else:
        return _ocr_with_datalab_api(pdf_path, datalab_provider=datalab_provider)


def _ocr_with_local_marker(pdf_path: str) -> str:
    """
    Performs OCR using local Marker.
    """
    logger.info(f"Starting PDF OCR via Local Marker: {pdf_path}")
    
    try:
        from pdfserviceMD.local_marker_service import ocr_with_local_marker
        result = ocr_with_local_marker(pdf_path)
        logger.info("Local Marker OCR completed successfully")
        return result
    except ImportError as e:
        logger.error(f"Local Marker not available: {e}")
        raise RuntimeError(f"Local Marker not installed: {e}")
    except Exception as e:
        logger.error(f"Local Marker OCR failed: {e}", exc_info=True)
        raise RuntimeError(f"OCR processing failed: {e}")


def _ocr_with_datalab_api(
    pdf_path: str, datalab_provider: DatalabProvider | None = None
) -> str:
    """
    Performs OCR using Datalab API.
    
    Also extracts and saves images from the API response to the same directory
    as the PDF file, so they can be referenced in the markdown.
    """
    import base64
    
    logger.info(f"Starting PDF OCR via Datalab API: {pdf_path}")
    
    # Get the directory where PDF is located (for saving images)
    pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
    provider = datalab_provider or get_datalab_provider()

    try:
        # Run async API call in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, provider.request_ocr_markdown(pdf_path)
                    )
                    api_response = future.result()
            else:
                api_response = loop.run_until_complete(
                    provider.request_ocr_markdown(pdf_path)
                )
        except RuntimeError:
            # No event loop exists
            api_response = asyncio.run(provider.request_ocr_markdown(pdf_path))

        # Extract and save images from API response
        images = api_response.get("images", {})
        if images:
            logger.info(f"Extracting {len(images)} images from API response...")
            for filename, base64_data in images.items():
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Save to the same directory as the PDF
                    image_path = os.path.join(pdf_dir, filename)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    logger.info(f"Saved image: {filename} ({len(image_bytes)} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to save image {filename}: {e}")
        else:
            logger.info("No images in API response")

        # Build markdown with page markers
        full_markdown = _build_markdown_with_page_markers(api_response)

        # Log results
        page_count = api_response.get("page_count", 1)
        logger.info(f"OCR completed: ~{page_count} pages processed")
        
        return full_markdown

    except ProviderError as e:
        logger.error(f"Datalab API error: {e}")
        raise RuntimeError(f"OCR processing failed: {e}")
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise RuntimeError(f"OCR processing failed: {e}")


# Backward compatibility alias (deprecated)
async def ocr_service(pdf_path: str) -> str:
    """
    DEPRECATED: Use ocr_service_sync with run_in_threadpool instead.
    """
    logger.warning("ocr_service() is deprecated. Use ocr_service_sync() with run_in_threadpool.")
    return ocr_service_sync(pdf_path)

