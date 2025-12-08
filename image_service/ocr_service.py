"""
OCR Service for Image Translation

Provides OCR functionality using PaddleOCR for detecting and extracting
text from images.
"""

# Standard library
import logging
import os
from typing import List, Tuple, Any

# Third-party
import numpy as np
from paddleocr import PaddleOCR

# Configure logging
logger = logging.getLogger(__name__)

# Force CPU mode for image service (GPU reserved for PDF OCR)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global OCR engine (singleton pattern)
_ocr_engine: PaddleOCR | None = None


def _get_ocr_engine() -> PaddleOCR:
    """
    Lazily initializes and returns the PaddleOCR engine.

    Returns:
        Initialized PaddleOCR instance.
    """
    global _ocr_engine

    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR engine (CPU mode)...")
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("PaddleOCR engine ready")

    return _ocr_engine


def _normalize_ocr_result(result: Any) -> List[Tuple[List, str]]:
    """
    Normalizes OCR results into a standard format.

    Args:
        result: Raw OCR result in various formats.

    Returns:
        List of (box, text) tuples.
    """
    if not result:
        return []

    normalized_data: List[Tuple[List, str]] = []

    # Case A: Dictionary format
    target_dict = None
    if isinstance(result, dict):
        target_dict = result
    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        target_dict = result[0]

    if target_dict:
        try:
            texts = target_dict.get('rec_texts', [])
            boxes = (
                target_dict.get('dt_polys') or
                target_dict.get('rec_polys') or
                target_dict.get('rec_boxes')
            )
            if texts and boxes is not None:
                count = min(len(texts), len(boxes))
                for i in range(count):
                    box = boxes[i]
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    normalized_data.append((box, texts[i]))
                return normalized_data
        except (KeyError, TypeError, IndexError) as e:
            logger.debug(f"Dict parsing failed: {e}")

    # Case B: List format
    if isinstance(result, (list, tuple)):
        # Single item format: [box, (text, confidence)]
        if len(result) == 2:
            box, text_info = result
            if (isinstance(box, (list, np.ndarray)) and
                isinstance(text_info, (list, tuple)) and len(text_info) == 2):
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                return [(box, text_info[0])]

        # Recursive parsing for nested lists
        for item in result:
            normalized_data.extend(_normalize_ocr_result(item))

    return normalized_data


def perform_ocr(img_array: np.ndarray) -> List[Tuple[List, str]]:
    """
    Performs OCR on an image array and returns detected text regions.

    This is a SYNCHRONOUS function designed to be called via run_in_threadpool.

    Args:
        img_array: NumPy array of the image (RGB format).

    Returns:
        List of (box, text) tuples where:
        - box: Bounding box coordinates
        - text: Detected text string
    """
    logger.debug("Performing OCR on image...")

    try:
        engine = _get_ocr_engine()
        raw_result = engine.ocr(img_array)
        normalized_items = _normalize_ocr_result(raw_result)

        logger.debug(f"OCR detected {len(normalized_items)} text regions")
        return normalized_items

    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return []