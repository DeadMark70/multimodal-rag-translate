"""
Translation Service for Image Translation

Provides text translation functionality using Google Gemini via LangChain.
"""

# Standard library
import logging
from typing import List

# Third-party
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)


async def translate_text_list_langchain(texts: List[str]) -> List[str]:
    """
    Translates a list of text strings to Traditional Chinese.

    Uses batch translation for efficiency while maintaining line-by-line
    correspondence between input and output.

    Args:
        texts: List of English text strings to translate.

    Returns:
        List of translated Traditional Chinese strings.
        Returns original texts if translation fails (graceful degradation).
    """
    if not texts:
        return []

    template = """You are a professional translator. 
    Translate the following English sentences to Traditional Chinese (繁體中文).
    
    Requirements:
    1. Keep the order strictly line by line. 
    2. The number of output lines MUST match the number of input lines.
    3. Output ONLY the translated text, no explanations.
    
    Input Text:
    {input_text}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm("translation")
    chain = prompt | llm | StrOutputParser()

    joined_text = "\n".join(texts)

    try:
        logger.debug(f"Translating {len(texts)} text blocks...")
        response_text = await chain.ainvoke({"input_text": joined_text})

        translated_lines = response_text.strip().split('\n')

        if len(translated_lines) != len(texts):
            logger.warning(
                f"Line count mismatch! Input: {len(texts)}, Output: {len(translated_lines)}. "
                "Returning original text."
            )
            return texts

        logger.debug("Translation completed successfully")
        return translated_lines

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return texts  # Graceful degradation