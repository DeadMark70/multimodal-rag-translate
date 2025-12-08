"""
PDF OCR Services

Provides OCR functionality using PaddleOCR PPStructureV3 for extracting
text and structure from PDF documents.
"""

# Standard library
import logging
import os
from typing import Iterator, Any

# Third-party
from paddleocr import PPStructureV3

# Configure logging
logger = logging.getLogger(__name__)

# Global predictor instance (initialized once for performance)
_predictor: PPStructureV3 | None = None


def _get_predictor() -> PPStructureV3:
    """
    Lazily initializes and returns the PPStructureV3 predictor on GPU.

    Returns:
        Initialized PPStructureV3 instance.

    Raises:
        RuntimeError: If predictor initialization fails.
    """
    global _predictor

    if _predictor is None:
        logger.info("Initializing PPStructureV3 predictor on GPU...")
        try:
            _predictor = PPStructureV3(
                paddlex_config="PP-StructureV3.yaml",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_region_detection=True,
                use_chart_recognition=False,
                use_formula_recognition=False,
                use_seal_recognition=False,
                use_table_recognition=False,
                device="gpu",  # Enable GPU acceleration
            )
            logger.info("PPStructureV3 predictor initialized successfully on GPU")
        except Exception as e:
            logger.error(f"Failed to initialize PPStructureV3: {e}", exc_info=True)
            raise RuntimeError(f"OCR engine initialization failed: {e}")

    return _predictor


def initialize_predictor() -> None:
    """
    Explicitly initializes the predictor during app startup.
    
    This function should be called from main.py's startup event
    to pre-load the OCR model and allocate GPU memory before
    the first user request.
    """
    logger.info("Pre-loading PDF OCR predictor on GPU...")
    _get_predictor()
    logger.info("PDF OCR predictor ready.")




def ocr_service_sync(pdf_path: str) -> str:
    """
    Performs OCR on a PDF file and returns combined Markdown text.

    This is a SYNCHRONOUS function designed to be called via run_in_threadpool
    in async contexts, as PaddleOCR is CPU-bound.

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

    logger.info(f"Starting OCR for: {pdf_path}")

    try:
        predictor = _get_predictor()
        ocr_result: Iterator[Any] = predictor.predict_iter(pdf_path)

        markdown_texts: list[str] = []

        # Prepare output directory for images
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_images_dir = os.path.normpath(os.path.join(base_dir, "output", "imgs"))
        os.makedirs(output_images_dir, exist_ok=True)

        for page_idx, page_result in enumerate(ocr_result):
            markdown_dict = page_result.markdown
            page_markdown = markdown_dict.get("markdown_texts", "")
            markdown_texts.append(page_markdown)

            # Save embedded images
            markdown_images = markdown_dict.get("markdown_images", {})
            for path, image in markdown_images.items():
                try:
                    # Sanitize filename to prevent path traversal
                    safe_filename = os.path.basename(path)
                    file_path = os.path.normpath(os.path.join(output_images_dir, safe_filename))
                    image.save(file_path)
                    logger.debug(f"Saved image: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save image {path}: {e}")

        # Combine pages with markers
        full_markdown = "\n\n".join(
            f"[[PAGE_{i+1}]]\n{page_md}" for i, page_md in enumerate(markdown_texts)
        )

        logger.info(f"OCR completed: {len(markdown_texts)} pages processed")
        return full_markdown

    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise RuntimeError(f"OCR processing failed: {e}")


# Backward compatibility alias (deprecated)
async def ocr_service(pdf_path: str) -> str:
    """
    DEPRECATED: Use ocr_service_sync with run_in_threadpool instead.

    This async wrapper exists for backward compatibility but should not be used
    in new code as it doesn't properly offload CPU-bound work.
    """
    logger.warning("ocr_service() is deprecated. Use ocr_service_sync() with run_in_threadpool.")
    return ocr_service_sync(pdf_path)
