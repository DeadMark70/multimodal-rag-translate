"""
AI Translation Service

Provides Markdown translation functionality using Google Gemini API.
"""

# Standard library
import logging

# Third-party
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)


async def translate_text(text: str) -> str:
    """
    Translates Markdown text to Traditional Chinese while preserving formatting.

    Args:
        text: Markdown text to translate.

    Returns:
        Translated Markdown text in Traditional Chinese.
        Returns original text if translation fails (graceful degradation).
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for translation")
        return text

    template = """你是一個翻譯助手。請將以下 Markdown 文字翻譯成繁體中文。

    注意：
    1. 保留所有 Markdown 結構（例如 # 標題、## 小節、程式碼區塊、數學公式、<img> 標籤）。
    2. 僅翻譯其中的英文文字，其他格式或符號請保持不變。
    3. 翻譯後直接輸出整份 Markdown，不要額外加註說明。
    4. 每一頁都要翻譯，並保留 \\newpage 標記，並將結果完整的傳回。

    Markdown 內容:
    {input_text}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm("translation")
    chain = prompt | llm | StrOutputParser()

    try:
        logger.info(f"Translating text ({len(text)} chars)...")
        response = await chain.ainvoke({"input_text": text})
        logger.info("Translation completed successfully")
        return response

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return text  # Graceful degradation