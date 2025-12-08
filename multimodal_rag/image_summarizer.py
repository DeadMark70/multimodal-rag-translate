"""
Image Summarizer for Multimodal RAG

Generates text summaries for visual elements using Google Gemini API.
"""

# Standard library
import asyncio
import base64
import logging
import os
from typing import List

# Third-party
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm
from .schemas import VisualElement, VisualElementType

# Configure logging
logger = logging.getLogger(__name__)


class ImageSummarizer:
    """
    Summarizes visual elements (tables, figures, formulas) using Gemini.
    
    Uses rate limiting via asyncio.Semaphore to prevent API exhaustion.
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        """
        Initializes the summarizer.

        Args:
            max_concurrent: Maximum concurrent API calls (default 3).
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _encode_image_to_base64(self, image_path: str) -> str | None:
        """
        Encodes an image file to base64 string.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64 encoded string, or None if encoding fails.
        """
        image_path = os.path.normpath(image_path)
        
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None

            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except IOError as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return None

    def _get_prompt_for_type(self, element_type: VisualElementType) -> str:
        """
        Returns a type-specific prompt for the visual element.

        Args:
            element_type: Type of the visual element.

        Returns:
            Prompt string for the LLM.
        """
        prompts = {
            VisualElementType.TABLE: (
                "Analyze this table image. Extract the data and convert it to Markdown table format. "
                "Include all headers and data cells. If the table is complex, describe its structure."
            ),
            VisualElementType.FORMULA: (
                "Analyze this mathematical formula or equation. "
                "Convert it to LaTeX format. Explain what the formula represents briefly."
            ),
            VisualElementType.FIGURE: (
                "Analyze this figure/chart/diagram. Describe:\n"
                "1. What type of visualization it is\n"
                "2. The axes labels and units (if applicable)\n"
                "3. Key trends or patterns in the data\n"
                "4. Any important annotations or legends"
            ),
        }
        return prompts.get(element_type, prompts[VisualElementType.FIGURE])

    async def _summarize_single_element(self, element: VisualElement) -> VisualElement:
        """
        Summarizes a single visual element.

        Args:
            element: Visual element to summarize.

        Returns:
            Same element with summary field populated.
        """
        async with self._semaphore:
            b64_image = self._encode_image_to_base64(element.image_path)
            
            if not b64_image:
                element.summary = "Error: Image file not found"
                return element

            prompt = self._get_prompt_for_type(element.type)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            )

            try:
                llm = get_llm("image_caption")
                response = await llm.ainvoke([message])
                element.summary = response.content
                logger.debug(f"Summarized element {element.id}")
            except Exception as e:
                logger.error(f"Failed to summarize element {element.id}: {e}")
                element.summary = f"Error analyzing image: {str(e)}"

            return element

    async def summarize_elements(self, elements: List[VisualElement]) -> List[VisualElement]:
        """
        Summarizes multiple visual elements concurrently.

        Args:
            elements: List of visual elements to summarize.

        Returns:
            List of elements with summaries populated.
        """
        if not elements:
            return elements

        logger.info(f"Summarizing {len(elements)} visual elements...")

        tasks = [self._summarize_single_element(elem) for elem in elements]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that were returned
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Element {i} summarization failed: {result}")
                elements[i].summary = "Error: Summarization failed"
                processed.append(elements[i])
            else:
                processed.append(result)

        logger.info(f"Summarization complete for {len(processed)} elements")
        return processed


# Global instance
summarizer = ImageSummarizer()
