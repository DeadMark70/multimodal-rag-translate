"""
Image Processing Service

Provides functions for drawing translated text onto images.
"""

# Standard library
import io
import logging
import os
from typing import List, Tuple, Any

# Third-party
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logger = logging.getLogger(__name__)

# Font configuration
_FONT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fonts", "NotoSansTC-Regular.ttf"
))


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Loads the configured font at the specified size.

    Args:
        size: Font size in pixels.

    Returns:
        Loaded font, or default font if loading fails.
    """
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        logger.warning(f"Font not found at {_FONT_PATH}, using default")
        return ImageFont.load_default()


def _parse_box_coordinates(box: Any) -> Tuple[float, float, float, float] | None:
    """
    Parses various box formats into (x1, y1, x2, y2) coordinates.

    Args:
        box: Box coordinates in various formats.

    Returns:
        Tuple of (x1, y1, x2, y2) or None if parsing fails.
    """
    try:
        # Format A: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (polygon)
        if len(box) == 4 and isinstance(box[0], (list, tuple)):
            x1, y1 = box[0]
            x2, y2 = box[2]
            return float(x1), float(y1), float(x2), float(y2)

        # Format B: [x1, y1, x2, y2] (rectangle)
        if len(box) == 4 and isinstance(box[0], (int, float)):
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])

        logger.warning(f"Unknown box format: {type(box)}")
        return None

    except (IndexError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse box coordinates: {e}")
        return None


def draw_translated_text_on_image(
    image: Image.Image,
    boxes: List[Any],
    translated_texts: List[str]
) -> Image.Image:
    """
    Draws translated text back onto the image at specified regions.

    Args:
        image: PIL Image to draw on.
        boxes: List of bounding box coordinates.
        translated_texts: List of translated text strings.

    Returns:
        Modified PIL Image with translated text.
    """
    draw = ImageDraw.Draw(image, "RGBA")

    for i, box in enumerate(boxes):
        if i >= len(translated_texts):
            break

        text = translated_texts[i]
        coords = _parse_box_coordinates(box)

        if coords is None:
            continue

        x1, y1, x2, y2 = coords
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        try:
            # 1. Draw semi-transparent background
            padding = 2
            draw.rectangle(
                [x1 - padding, y1 - padding, x2 + padding, y2 + padding],
                fill=(255, 255, 255, 240)
            )

            # 2. Calculate font size (auto-fit)
            font_size = max(10, int(h * 0.8))
            font = _load_font(font_size)

            # Shrink font if text doesn't fit
            if hasattr(font, 'getbbox'):
                while font_size > 8:
                    bbox = font.getbbox(text)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    if text_w <= w and text_h <= h:
                        break
                    font_size -= 1
                    font = _load_font(font_size)

            # 3. Draw text centered
            center_x = x1 + w / 2
            center_y = y1 + h / 2

            try:
                draw.text(
                    (center_x, center_y),
                    text,
                    font=font,
                    fill=(0, 0, 0, 255),
                    anchor="mm"
                )
            except TypeError:
                # Fallback for older Pillow without anchor support
                if hasattr(font, 'getbbox'):
                    bbox = font.getbbox(text)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                else:
                    tw, th = w, h
                draw.text(
                    (center_x - tw / 2, center_y - th / 2),
                    text,
                    font=font,
                    fill=(0, 0, 0, 255)
                )

        except Exception as e:
            logger.warning(f"Failed to draw text on box {i}: {e}")
            continue

    return image


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """
    Converts a PIL Image to bytes.

    Args:
        image: PIL Image to convert.
        format: Output format (JPEG, PNG, etc.).

    Returns:
        Image data as bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()