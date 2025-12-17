"""
OCR Service for Image Translation (DocTR - Local)

Provides OCR functionality using DocTR for detecting and extracting
text from images with precise bounding boxes.

DocTR (Mindee, France) - Apache 2.0 License
"""

# Standard library
import logging
import os
from typing import List, Tuple, Optional

# Third-party
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Force CPU mode for image service (GPU reserved for other tasks)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global OCR engine (singleton pattern)
_ocr_engine = None

# Configuration
MAX_IMAGE_DIMENSION = 2048  # Resize images larger than this


def _get_ocr_engine():
    """
    Lazily initializes and returns the DocTR OCR engine.

    Returns:
        Initialized DocTR OCR predictor instance.
    """
    global _ocr_engine

    if _ocr_engine is None:
        logger.info("Initializing DocTR engine...")
        
        # Import here to avoid loading at module import time
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        _ocr_engine = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
            assume_straight_pages=True,  # Faster, assumes no rotation
            export_as_straight_boxes=True  # Always output axis-aligned boxes
        )
        logger.info("DocTR engine ready")

    return _ocr_engine


def _resize_if_needed(img_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Resizes large images to prevent slow inference and OOM.

    Args:
        img_array: Original image array.

    Returns:
        Tuple of (resized_image, scale_factor).
        scale_factor is used to convert coordinates back to original size.
    """
    h, w = img_array.shape[:2]
    max_dim = max(h, w)

    if max_dim <= MAX_IMAGE_DIMENSION:
        return img_array, 1.0

    scale = MAX_IMAGE_DIMENSION / max_dim
    new_w, new_h = int(w * scale), int(h * scale)

    # Use PIL for high-quality resizing
    pil_img = Image.fromarray(img_array)
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    return np.array(pil_img), 1.0 / scale  # Return inverse scale for coord restoration


def perform_ocr(img_array: np.ndarray) -> List[Tuple[List, str]]:
    """
    Performs OCR on an image array and returns detected text regions.

    This is a SYNCHRONOUS function designed to be called via run_in_threadpool.

    Args:
        img_array: NumPy array of the image (RGB format).

    Returns:
        List of (box, text) tuples where:
        - box: Bounding box coordinates as [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        - text: Detected text string
    """
    logger.debug("Performing OCR on image...")

    try:
        # Import here for lazy loading
        from doctr.io import DocumentFile
        
        # Resize large images to prevent OOM and slow inference
        resized_img, coord_scale = _resize_if_needed(img_array)
        
        engine = _get_ocr_engine()

        # DocTR accepts list of numpy arrays
        doc = DocumentFile.from_images([resized_img])
        result = engine(doc)

        normalized_data: List[Tuple[List, str]] = []

        for page in result.pages:
            # Get dimensions of the RESIZED image (for relative coord conversion)
            h_resized, w_resized = page.dimensions

            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # DocTR returns relative coords (0~1)
                        (x_min_rel, y_min_rel), (x_max_rel, y_max_rel) = word.geometry

                        # Convert to absolute pixels on resized image
                        x_min = x_min_rel * w_resized
                        y_min = y_min_rel * h_resized
                        x_max = x_max_rel * w_resized
                        y_max = y_max_rel * h_resized

                        # Scale back to original image coordinates
                        x_min *= coord_scale
                        y_min *= coord_scale
                        x_max *= coord_scale
                        y_max *= coord_scale

                        # Build 4-point box format (PaddleOCR compatible)
                        box = [
                            [x_min, y_min],  # top-left
                            [x_max, y_min],  # top-right
                            [x_max, y_max],  # bottom-right
                            [x_min, y_max]   # bottom-left
                        ]
                        normalized_data.append((box, word.value))

        logger.debug(f"OCR detected {len(normalized_data)} text regions")
        return normalized_data

    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return []