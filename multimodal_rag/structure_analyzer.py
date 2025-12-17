"""
Structure Analyzer for Multimodal RAG (Datalab API)

Extracts text and visual elements from PDF documents using Datalab Layout API.
Designed for RAG pipeline with API-based processing.

Datalab (USA) - Commercial API
"""

# Standard library
import logging
import os
from typing import List, Dict, Any, Optional
from uuid import uuid4

# Third-party
import cv2
import numpy as np
import httpx
from pdf2image import convert_from_path

# Local application
from .schemas import ExtractedDocument, TextChunk, VisualElement, VisualElementType

# Configure logging
logger = logging.getLogger(__name__)

# API Configuration
DATALAB_API_URL = os.getenv("DATALAB_API_URL", "https://www.datalab.to/api/v1/marker")
DATALAB_API_KEY = os.getenv("DATALAB_API_KEY", "")
API_TIMEOUT = 300.0


class StructureAnalyzer:
    """
    Analyzes PDF structure and extracts text/visual elements via Datalab API.
    
    Falls back to local image processing for visual element cropping.
    """

    # Label mapping for visual elements
    VISUAL_LABELS = {'picture', 'figure', 'table', 'formula', 'chart', 'image'}
    TEXT_LABELS = {'text', 'section-header', 'caption', 'list-item', 'footnote',
                   'page-header', 'page-footer', 'title', 'paragraph'}

    def __init__(self) -> None:
        """Initializes the StructureAnalyzer."""
        if not DATALAB_API_KEY:
            logger.warning("DATALAB_API_KEY not set! Layout analysis may fail.")
        logger.info("StructureAnalyzer initialized (Datalab API mode)")

    def _call_datalab_layout_api(self, pdf_path: str) -> Dict[str, Any]:
        """
        Calls Datalab API for layout detection and OCR.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            API response with layout information.
        """
        if not DATALAB_API_KEY:
            raise RuntimeError("DATALAB_API_KEY not configured")

        with open(pdf_path, "rb") as f:
            files = {"file": ("document.pdf", f, "application/pdf")}
            headers = {"X-Api-Key": DATALAB_API_KEY}
            data = {
                "output_format": "json",
                "extract_images": True,
            }

            try:
                with httpx.Client(timeout=API_TIMEOUT) as client:
                    response = client.post(
                        DATALAB_API_URL,
                        files=files,
                        headers=headers,
                        data=data
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                logger.error(f"Datalab API error: {e}")
                raise

    def _extract_visual_elements_locally(
        self,
        pil_images: List,
        api_result: Dict[str, Any],
        visuals_dir: str
    ) -> List[VisualElement]:
        """
        Extracts and crops visual elements from PDF images based on API layout info.

        Args:
            pil_images: List of PIL images from PDF.
            api_result: Datalab API response with layout info.
            visuals_dir: Directory to save cropped images.

        Returns:
            List of VisualElement objects.
        """
        visual_elements: List[VisualElement] = []
        
        pages = api_result.get("pages", [])
        if not pages:
            # Fallback: try to find elements in root
            elements = api_result.get("elements", api_result.get("blocks", []))
            if elements:
                pages = [{"page_number": 1, "elements": elements}]
        
        for page_data in pages:
            page_num = page_data.get("page_number", page_data.get("page", 1))
            
            if page_num > len(pil_images):
                continue
                
            pil_image = pil_images[page_num - 1]
            img_array = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]
            
            elements = page_data.get("elements", page_data.get("blocks", []))
            
            for element in elements:
                label = element.get("label", element.get("type", "")).lower()
                
                if label in self.VISUAL_LABELS:
                    bbox = element.get("bbox", element.get("polygon", element.get("box", [])))
                    
                    if not bbox or len(bbox) < 4:
                        continue
                    
                    # Handle different bbox formats
                    if isinstance(bbox[0], (list, tuple)):
                        # Polygon format: [[x1,y1], [x2,y2], ...]
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                    else:
                        # [x1, y1, x2, y2] or [x1, y1, w, h] format
                        x1, y1 = int(bbox[0]), int(bbox[1])
                        if len(bbox) == 4:
                            x2, y2 = int(bbox[2]), int(bbox[3])
                            # Check if width/height format
                            if x2 < x1 or y2 < y1:
                                x2, y2 = x1 + x2, y1 + y2
                        else:
                            continue
                    
                    # Bounds checking
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    cropped = img_bgr[y1:y2, x1:x2]
                    
                    if cropped.size > 0:
                        vid = uuid4()
                        img_path = os.path.join(visuals_dir, f"img_{vid}.jpg")
                        cv2.imwrite(img_path, cropped)
                        
                        elem_type = VisualElementType.FIGURE
                        if 'table' in label:
                            elem_type = VisualElementType.TABLE
                        elif 'formula' in label or 'equation' in label:
                            elem_type = VisualElementType.FORMULA
                        
                        visual_elements.append(VisualElement(
                            id=vid,
                            type=elem_type,
                            page_number=page_num,
                            image_path=img_path,
                            bbox=[x1, y1, x2, y2],
                            original_text=element.get("text", "")
                        ))
        
        return visual_elements

    def _extract_text_chunks(self, api_result: Dict[str, Any]) -> List[TextChunk]:
        """
        Extracts text chunks from API response.

        Args:
            api_result: Datalab API response.

        Returns:
            List of TextChunk objects.
        """
        text_chunks: List[TextChunk] = []
        
        pages = api_result.get("pages", [])
        
        if pages:
            for page_data in pages:
                page_num = page_data.get("page_number", page_data.get("page", 1))
                page_text = page_data.get("markdown", page_data.get("text", ""))
                
                if page_text:
                    text_chunks.append(TextChunk(
                        page_number=page_num,
                        content=page_text,
                        chunk_id=str(uuid4())
                    ))
        else:
            # Fallback: use full markdown
            full_text = api_result.get("markdown", api_result.get("text", ""))
            if full_text:
                text_chunks.append(TextChunk(
                    page_number=1,
                    content=full_text,
                    chunk_id=str(uuid4())
                ))
        
        return text_chunks

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

        # Convert PDF to images for visual element cropping
        try:
            pil_images = convert_from_path(pdf_path)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise ValueError(f"Failed to convert PDF: {str(e)}")

        logger.info(f"Processing PDF with {len(pil_images)} pages via Datalab API")

        # Call Datalab API
        try:
            api_result = self._call_datalab_layout_api(pdf_path)
        except Exception as e:
            logger.error(f"Datalab API failed: {e}")
            # Return empty result on API failure
            return ExtractedDocument(
                doc_id=doc_id,
                user_id=user_id,
                text_chunks=[],
                visual_elements=[]
            )

        # Extract text chunks
        text_chunks = self._extract_text_chunks(api_result)
        
        # Extract and crop visual elements
        visual_elements = self._extract_visual_elements_locally(
            pil_images, api_result, visuals_dir
        )

        logger.info(f"Extraction complete: {len(text_chunks)} chunks, {len(visual_elements)} visuals")

        return ExtractedDocument(
            doc_id=doc_id,
            user_id=user_id,
            text_chunks=text_chunks,
            visual_elements=visual_elements
        )


# Global instance (lazy loaded)
analyzer = StructureAnalyzer()
