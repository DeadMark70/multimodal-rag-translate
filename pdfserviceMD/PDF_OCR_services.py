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
from typing import Dict, Any, Optional

# Third-party
import httpx

# Configure logging
logger = logging.getLogger(__name__)

# Mode Configuration
USE_LOCAL_MARKER = os.getenv("USE_LOCAL_MARKER", "true").lower() in ("true", "1", "yes")

# API Configuration (for Datalab mode)
DATALAB_API_URL = os.getenv("DATALAB_API_URL", "https://www.datalab.to/api/v1/marker")
DATALAB_API_KEY = os.getenv("DATALAB_API_KEY", "")
API_TIMEOUT = 300.0  # 5 minutes for large PDFs


class DatalabAPIError(Exception):
    """Custom exception for Datalab API errors."""
    pass


async def _call_datalab_api(pdf_path: str) -> Dict[str, Any]:
    """
    Calls Datalab API to convert PDF to Markdown.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        API response containing markdown content.

    Raises:
        DatalabAPIError: If API call fails.
    """
    if not DATALAB_API_KEY:
        raise DatalabAPIError("DATALAB_API_KEY not configured")

    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Prepare multipart form data
        files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
        headers = {"X-Api-Key": DATALAB_API_KEY}
        
        # Request parameters for better output
        data = {
            "output_format": "markdown",
            "force_ocr": False,
            "paginate_output": True,  # Request page-by-page output
        }

        try:
            logger.info(f"Calling Datalab API for PDF: {pdf_path}")
            response = await client.post(
                DATALAB_API_URL,
                files=files,
                headers=headers,
                data=data
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Datalab API HTTP error: {e.response.status_code} - {e.response.text}")
            raise DatalabAPIError(f"API returned {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Datalab API request error: {e}")
            raise DatalabAPIError(f"API request failed: {e}")


def _build_markdown_with_page_markers(api_response: Dict[str, Any]) -> str:
    """
    Builds combined Markdown with [[PAGE_N]] markers from API response.

    Datalab API response format (main fields):
    - markdown: Full markdown text
    - page_count: Number of pages
    - success: Boolean
    
    Args:
        api_response: Datalab API response.

    Returns:
        Combined Markdown text with page markers.
    """
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
    
    # Check if markdown already has page separators
    # Common separators: \n---\n, \n\n---\n\n, or explicit page markers
    if page_count > 1:
        # Try to split by horizontal rule (---) which is common page separator
        separator_patterns = ["\n---\n\n", "\n\n---\n\n", "\n---\n", "---"]
        
        for sep in separator_patterns:
            if sep in markdown:
                pages_split = markdown.split(sep)
                if len(pages_split) >= page_count - 1:  # Allow some tolerance
                    markdown_parts = []
                    for i, page_md in enumerate(pages_split):
                        page_md = page_md.strip()
                        if page_md:
                            markdown_parts.append(f"[[PAGE_{i+1}]]\n{page_md}")
                    if markdown_parts:
                        logger.info(f"Split into {len(markdown_parts)} pages using separator")
                        return "\n\n".join(markdown_parts)
                    break
        
        # If no separator found, just add PAGE_1 marker to full text
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
        if not DATALAB_API_KEY:
            logger.warning(
                "DATALAB_API_KEY not set! PDF OCR will fail. "
                "Set it in config.env or environment variables."
            )
        else:
            logger.info(f"Datalab API configured (URL: {DATALAB_API_URL})")


def ocr_service_sync(pdf_path: str) -> str:
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
        return _ocr_with_datalab_api(pdf_path)


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


def _ocr_with_datalab_api(pdf_path: str) -> str:
    """
    Performs OCR using Datalab API.
    """
    logger.info(f"Starting PDF OCR via Datalab API: {pdf_path}")

    try:
        # Run async API call in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _call_datalab_api(pdf_path))
                    api_response = future.result()
            else:
                api_response = loop.run_until_complete(_call_datalab_api(pdf_path))
        except RuntimeError:
            # No event loop exists
            api_response = asyncio.run(_call_datalab_api(pdf_path))

        # Build markdown with page markers
        full_markdown = _build_markdown_with_page_markers(api_response)

        # Log results
        page_count = api_response.get("page_count", 1)
        logger.info(f"OCR completed: ~{page_count} pages processed")
        
        return full_markdown

    except DatalabAPIError as e:
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

