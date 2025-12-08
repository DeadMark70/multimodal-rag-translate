"""
Structure Analyzer for Multimodal RAG

Extracts text and visual elements from PDF documents using PP-StructureV3.
Designed for RAG pipeline with CPU-only mode for stability.
"""

# Standard library
import logging
import os
from typing import List, Dict, Any
from uuid import uuid4

# Third-party
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PPStructureV3

# Local application
from .schemas import ExtractedDocument, TextChunk, VisualElement, VisualElementType

# Configure logging
logger = logging.getLogger(__name__)

# Force CPU mode for multimodal RAG (GPU reserved for PDF translation service)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class StructureAnalyzer:
    """
    Analyzes PDF structure and extracts text/visual elements.
    
    Uses PP-StructureV3 in CPU mode for layout detection and OCR.
    Visual elements are cropped and saved for later summarization by Gemini.
    """

    def __init__(self) -> None:
        """Initializes the PP-StructureV3 engine."""
        logger.info("Initializing StructureAnalyzer (CPU mode)...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.normpath(os.path.join(base_dir, "rag_config.yaml"))

        if not os.path.exists(yaml_path):
            logger.warning(f"Config file not found at {yaml_path}, using defaults")
            yaml_path = None

        self.engine = PPStructureV3(
            paddlex_config=yaml_path,
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_region_detection=True,
            use_chart_recognition=False,
            use_formula_recognition=False,
            use_seal_recognition=False,
            use_table_recognition=False,
            device="cpu"
        )

        logger.info("StructureAnalyzer ready")

    def _parse_v3_result(self, prediction: Any) -> List[Dict[str, Any]]:
        """
        Parses PP-StructureV3 output into standardized format.

        Args:
            prediction: Raw prediction from PP-StructureV3.

        Returns:
            List of dicts with 'type', 'bbox', 'text' keys.
        """
        parsed_regions: List[Dict[str, Any]] = []

        # Extract JSON data from prediction
        data = None
        if hasattr(prediction, 'json'):
            data = prediction.json
        elif isinstance(prediction, dict):
            data = prediction
        else:
            logger.warning(f"Unknown prediction type: {type(prediction)}")
            return []

        # Find regions in various possible locations
        res_root = data.get('res', data)
        raw_regions = res_root.get('parsing_res_list')

        if not raw_regions and 'layout_det_res' in res_root:
            logger.debug("parsing_res_list not found, using layout_det_res")
            raw_regions = res_root['layout_det_res'].get('boxes', [])

        if not raw_regions:
            logger.debug("No regions found in OCR result")
            return []

        # Normalize field names
        for region in raw_regions:
            r_type = region.get('block_label') or region.get('label')
            r_bbox = region.get('block_bbox') or region.get('coordinate')
            r_text = region.get('block_content') or region.get('text') or ""

            if r_type and r_bbox:
                parsed_regions.append({
                    "type": r_type,
                    "bbox": r_bbox,
                    "text": r_text
                })

        return parsed_regions

    def extract_from_pdf(
        self,
        pdf_path: str,
        user_id: str,
        doc_id: str,
        output_base_dir: str
    ) -> ExtractedDocument:
        """
        Extracts text chunks and visual elements from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            user_id: User's ID for metadata.
            doc_id: Document's UUID.
            output_base_dir: Directory to save cropped visual elements.

        Returns:
            ExtractedDocument with text chunks and visual elements.

        Raises:
            ValueError: If PDF conversion fails.
        """
        visuals_dir = os.path.normpath(os.path.join(output_base_dir, "visuals"))
        os.makedirs(visuals_dir, exist_ok=True)

        # Convert PDF to images
        try:
            images = convert_from_path(pdf_path)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise ValueError(f"Failed to convert PDF: {str(e)}")

        text_chunks: List[TextChunk] = []
        visual_elements: List[VisualElement] = []

        for i, image in enumerate(images):
            page_number = i + 1
            logger.info(f"Processing page {page_number}/{len(images)}")

            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            try:
                predictions = self.engine.predict(img_bgr)
                prediction = predictions[0] if predictions else None

                if prediction:
                    regions = self._parse_v3_result(prediction)
                else:
                    regions = []

            except Exception as e:
                logger.error(f"OCR failed for page {page_number}: {e}")
                regions = []

            page_text_content: List[str] = []

            for region in regions:
                region_type = region['type'].lower()
                bbox = region['bbox']
                text_content = region['text']

                logger.debug(f"Found region: {region_type} at {bbox}")

                # Visual elements
                if region_type in ['figure', 'table', 'equation', 'formula', 'image', 'chart']:
                    elem_type = VisualElementType.FIGURE
                    if region_type == 'table':
                        elem_type = VisualElementType.TABLE
                    elif region_type in ['equation', 'formula']:
                        elem_type = VisualElementType.FORMULA

                    # Crop image with bounds checking
                    x1, y1, x2, y2 = [int(p) for p in bbox]
                    h, w = img_bgr.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    cropped_img = img_bgr[y1:y2, x1:x2]

                    if cropped_img.size > 0:
                        visual_id = uuid4()
                        img_filename = f"img_{visual_id}.jpg"
                        img_path = os.path.normpath(os.path.join(visuals_dir, img_filename))

                        cv2.imwrite(img_path, cropped_img)

                        visual_elements.append(VisualElement(
                            id=visual_id,
                            type=elem_type,
                            page_number=page_number,
                            image_path=img_path,
                            bbox=[x1, y1, x2, y2],
                            original_text=text_content
                        ))

                # Text elements
                elif region_type in ['text', 'title', 'list', 'header', 'footer', 'paragraph_title', 'doc_title']:
                    if text_content:
                        # Apply markdown formatting
                        if region_type in ['title', 'doc_title']:
                            text_content = f"# {text_content}"
                        elif region_type == 'paragraph_title':
                            text_content = f"## {text_content}"

                        page_text_content.append(text_content)

            # Consolidate page text
            if page_text_content:
                full_page_text = "\n\n".join(page_text_content)
                text_chunks.append(TextChunk(
                    page_number=page_number,
                    content=full_page_text,
                    chunk_id=str(uuid4())
                ))

        logger.info(f"Extraction complete: {len(text_chunks)} text chunks, {len(visual_elements)} visual elements")

        return ExtractedDocument(
            doc_id=doc_id,
            user_id=user_id,
            text_chunks=text_chunks,
            visual_elements=visual_elements
        )


# Global instance (lazy loaded when module is imported)
analyzer = StructureAnalyzer()
