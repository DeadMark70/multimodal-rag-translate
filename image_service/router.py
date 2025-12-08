"""
Image Translation Router

Provides API endpoints for in-place image text translation.
"""

# Standard library
import io
import logging
import os

# Third-party
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image
import numpy as np

# Local application
from core.auth import get_current_user_id
from .ocr_service import perform_ocr
from .translation_service import translate_text_list_langchain
from .image_processing import draw_translated_text_on_image, image_to_bytes

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_image_upload(file: UploadFile) -> None:
    """
    Validates that the uploaded file is a supported image.

    Args:
        file: The uploaded file object.

    Raises:
        HTTPException: 400 if file is not a valid image.
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed: {', '.join(allowed_types)}"
        )

    if file.filename:
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(status_code=400, detail="Invalid image extension")


@router.post("/translate_image")
async def translate_image_inplace(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
) -> Response:
    """
    Translates text in an image and returns the modified image.

    Pipeline:
    1. Read and validate image
    2. Perform OCR to detect text regions
    3. Translate detected text to Traditional Chinese
    4. Draw translated text back onto image
    5. Return modified image

    Args:
        file: The uploaded image file.
        user_id: Authenticated user ID (injected).

    Returns:
        Response with the translated image as JPEG.

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors.
    """
    logger.info(f"Image translation request from user {user_id}")

    # Input validation
    _validate_image_upload(file)

    try:
        # 1. Read image
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        img_array = np.array(image)

        # 2. OCR (CPU-bound, run in threadpool)
        ocr_items = await run_in_threadpool(perform_ocr, img_array)

        if not ocr_items:
            logger.info("No text detected in image, returning original")
            return Response(content=image_data, media_type="image/jpeg")

        logger.info(f"Detected {len(ocr_items)} text blocks")

        # Separate boxes and texts
        boxes = [item[0] for item in ocr_items]
        raw_texts = [item[1] for item in ocr_items]

        # 3. Translate (async API call)
        translated_texts = await translate_text_list_langchain(raw_texts)

        # 4. Draw translated text (CPU-bound, run in threadpool)
        final_image = await run_in_threadpool(
            draw_translated_text_on_image, image, boxes, translated_texts
        )

        # 5. Convert to bytes and return
        result_bytes = await run_in_threadpool(image_to_bytes, final_image)

        logger.info("Image translation completed successfully")
        return Response(content=result_bytes, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image translation failed: {str(e)}")
    finally:
        await file.close()