"""
Image Summarizer for Multimodal RAG

Generates text summaries for visual elements using Google Gemini API.
Optimized for RAG retrieval with context-aware prompts and caching.
"""

# Standard library
import asyncio
import base64
import hashlib
import io
import logging
import os
from typing import List, Optional

# Third-party
from PIL import Image
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm
from .schemas import VisualElement, VisualElementType

# Configure logging
logger = logging.getLogger(__name__)


class ImageSummarizer:
    """
    Summarizes visual elements (figures/pictures) using Gemini Vision API.
    
    Features:
    - Context-aware prompts (document title, figure reference, surrounding text)
    - Hash-based LRU caching to avoid redundant API calls
    - Image preprocessing (resize/compress) to reduce costs
    - Rate limiting via asyncio.Semaphore
    
    Note: Only processes FIGURE type elements. TABLE and FORMULA are handled
    by Marker OCR as Markdown/LaTeX text, no VLM processing needed.
    """
    
    MAX_CACHE_SIZE = 100
    MAX_IMAGE_SIZE = 1024  # Max dimension in pixels
    JPEG_QUALITY = 85

    def __init__(self, max_concurrent: int = 3) -> None:
        """
        Initializes the summarizer.

        Args:
            max_concurrent: Maximum concurrent API calls (default 3).
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache: dict[str, str] = {}

    def _get_image_hash(self, image_path: str) -> Optional[str]:
        """
        計算圖片的 MD5 hash 用於快取鍵。
        
        Args:
            image_path: 圖片檔案路徑
        
        Returns:
            MD5 hash 字串，失敗時返回 None
        """
        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError:
            return None

    def _get_cached_summary(self, image_hash: str) -> Optional[str]:
        """從快取取得摘要。"""
        return self._cache.get(image_hash)

    def _set_cache(self, image_hash: str, summary: str) -> None:
        """儲存摘要到快取（LRU 策略）。"""
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            # 移除最舊的項目
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[image_hash] = summary

    def _preprocess_image(
        self, 
        image_path: str,
        max_size: int = MAX_IMAGE_SIZE,
        quality: int = JPEG_QUALITY
    ) -> Optional[str]:
        """
        預處理圖片：壓縮並轉換為 base64。
        
        Args:
            image_path: 圖片路徑
            max_size: 最大邊長（像素）
            quality: JPEG 品質 (1-100)
        
        Returns:
            Base64 編碼字串，失敗時返回 None
        """
        image_path = os.path.normpath(image_path)
        
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            
            with Image.open(image_path) as img:
                # 調整大小（保持比例）
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                    logger.debug(f"Resized image to {new_size}")
                
                # 轉換為 RGB（處理 RGBA/P 等格式）
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # 壓縮為 JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
                
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            return None

    def _get_figure_prompt(
        self,
        doc_title: str = "",
        page_number: int = 0,
        figure_reference: str = "",
        context_text: str = ""
    ) -> str:
        """
        生成上下文感知的圖片分析 prompt（繁體中文）。
        
        Args:
            doc_title: 文件標題
            page_number: 頁碼
            figure_reference: 圖片引用標識（如 "Figure 1"）
            context_text: 周圍文字上下文
        
        Returns:
            完整的 prompt 字串
        """
        context_section = ""
        if doc_title or figure_reference or context_text:
            context_section = "【上下文資訊】\n"
            if doc_title:
                context_section += f"- 文件：《{doc_title}》\n"
            if page_number:
                context_section += f"- 頁碼：第 {page_number} 頁\n"
            if figure_reference:
                context_section += f"- 圖片標識：{figure_reference}\n"
            if context_text:
                # 限制上下文長度
                truncated = context_text[:300] + "..." if len(context_text) > 300 else context_text
                context_section += f"- 周圍文字：{truncated}\n"
            context_section += "\n"
        
        return f"""{context_section}請用繁體中文分析此圖片：

【類型】說明這是什麼類型的圖片（如：折線圖、長條圖、流程圖、架構圖、截圖、照片等）

【主題】用一句話描述圖片的主題或目的

【內容】
- 如果是圖表：描述軸標籤、數據趨勢、關鍵數值
- 如果是流程圖/架構圖：描述步驟和流向
- 如果是照片/截圖：描述主要內容

【上下文關聯】根據周圍文字，說明此圖片在文章中的作用（若無上下文資訊則略過此項）

【關鍵詞】列出 5-10 個用於檢索的關鍵詞（以逗號分隔），應包含圖中出現的專有名詞、數值、標籤等"""

    async def _summarize_single_element(
        self, 
        element: VisualElement,
        doc_title: str = "",
    ) -> VisualElement:
        """
        摘要單個視覺元素（僅處理 FIGURE 類型）。

        Args:
            element: 視覺元素
            doc_title: 文件標題（用於上下文）

        Returns:
            帶有摘要的視覺元素
        """
        # 只處理 FIGURE 類型（TABLE 和 FORMULA 由 Marker 處理為文字）
        if element.type != VisualElementType.FIGURE:
            logger.debug(f"Skipping non-FIGURE element: {element.type}")
            return element
        
        async with self._semaphore:
            # 1. 檢查快取
            img_hash = self._get_image_hash(element.image_path)
            if img_hash:
                cached = self._get_cached_summary(img_hash)
                if cached:
                    logger.debug(f"Cache hit for element {element.id}")
                    element.summary = cached
                    return element
            
            # 2. 預處理圖片（壓縮）
            b64_image = self._preprocess_image(element.image_path)
            if not b64_image:
                element.summary = "Error: 無法處理圖片檔案"
                return element
            
            # 3. 生成上下文感知 prompt
            prompt = self._get_figure_prompt(
                doc_title=doc_title,
                page_number=element.page_number,
                figure_reference=element.figure_reference or "",
                context_text=element.context_text or ""
            )
            
            # 4. 呼叫 Gemini Vision API
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
                
                # 5. 儲存到快取
                if img_hash:
                    self._set_cache(img_hash, element.summary)
                
                logger.debug(f"Summarized element {element.id}")
            except Exception as e:
                logger.error(f"Failed to summarize element {element.id}: {e}")
                element.summary = f"Error: 圖片分析失敗 - {str(e)}"
            
            return element

    async def summarize_elements(
        self, 
        elements: List[VisualElement],
        doc_title: str = "",
    ) -> List[VisualElement]:
        """
        批次摘要多個視覺元素。

        Args:
            elements: 視覺元素列表
            doc_title: 文件標題（用於上下文）

        Returns:
            帶有摘要的視覺元素列表
        """
        if not elements:
            return elements

        # 過濾出 FIGURE 類型的元素數量
        figure_count = sum(1 for e in elements if e.type == VisualElementType.FIGURE)
        logger.info(f"Summarizing {figure_count} FIGURE elements (total: {len(elements)})...")

        tasks = [
            self._summarize_single_element(elem, doc_title=doc_title)
            for elem in elements
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that were returned
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Element {i} summarization failed: {result}")
                elements[i].summary = "Error: 摘要生成失敗"
                processed.append(elements[i])
            else:
                processed.append(result)

        # 統計
        summarized = sum(1 for e in processed if e.summary and not e.summary.startswith("Error"))
        cached = sum(1 for e in processed if e.summary and "Cache hit" not in str(e.summary)[:20])
        logger.info(f"Summarization complete: {summarized} summarized, cache size: {len(self._cache)}")
        
        return processed

    def _get_reexamine_prompt(
        self,
        specific_question: str,
        original_summary: str = "",
    ) -> str:
        """
        生成針對性問題的圖片分析 Prompt。
        
        Args:
            specific_question: 具體問題 (如「圖中 X 的數值是多少？」)
            original_summary: 原始摘要，提供上下文
            
        Returns:
            完整的 prompt 字串
        """
        context_section = ""
        if original_summary:
            truncated = original_summary[:500] + "..." if len(original_summary) > 500 else original_summary
            context_section = f"""【先前摘要】
{truncated}

"""
        
        return f"""{context_section}請用繁體中文針對以下具體問題分析此圖片：

【問題】{specific_question}

【回答要求】
1. 直接回答問題，不要泛泛而談
2. 如果是數據問題，請從圖中讀取具體數值
3. 如果圖中找不到答案，請明確說明「圖中未顯示此資訊」
4. 如果需要估算，請說明估算依據

請提供簡潔精確的回答："""

    async def re_examine_image(
        self,
        image_path: str,
        specific_question: str,
        original_summary: str = "",
    ) -> str:
        """
        對圖片進行針對性問題分析（進階視覺查證）。
        
        此方法允許 Agent 對已索引的圖片提出具體問題進行二次分析，
        適用於需要精確數據驗證的深度研究場景。
        
        Args:
            image_path: 圖片路徑
            specific_question: 具體問題 (如「圖中 X 的數值是多少？」)
            original_summary: 原始摘要 (可選，提供上下文以提升準確度)
            
        Returns:
            針對問題的詳細回答
            
        Raises:
            FileNotFoundError: 圖片不存在
            ValueError: 問題為空
        """
        # 驗證輸入
        if not specific_question or len(specific_question.strip()) < 3:
            raise ValueError("具體問題不能為空或過短")
        
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"圖片不存在: {image_path}")
        
        async with self._semaphore:
            # 預處理圖片
            b64_image = self._preprocess_image(image_path)
            if not b64_image:
                return "Error: 無法處理圖片檔案"
            
            # 生成針對性 prompt
            prompt = self._get_reexamine_prompt(
                specific_question=specific_question,
                original_summary=original_summary,
            )
            
            # 呼叫 Gemini Vision API
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
                llm = get_llm("visual_verification")
                response = await llm.ainvoke([message])
                
                logger.info(f"Re-examined image: {os.path.basename(image_path)}")
                return response.content
                
            except (RuntimeError, ValueError) as e:
                logger.error(f"Failed to re-examine image: {e}")
                return f"Error: 圖片分析失敗 - {str(e)}"


# Global instance
summarizer = ImageSummarizer()

