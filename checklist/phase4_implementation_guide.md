# Phase 4 完整實作指南

> **狀態**: ✅ 已完成 (2025-12-10)
> **測試**: 104 passed

---

## Phase 4.1: LLM Factory 雙模型支援

### 📁 修改檔案: `core/llm_factory.py`

#### 變更內容

```python
# 新增模型映射 (Line ~25)
_MODEL_BY_PURPOSE: dict[str, str] = {
    "translation": "gemini-3.0-flash",
}
_DEFAULT_MODEL = "gemma-3-27b-it"

# 修改 get_llm() 函式
def get_llm(purpose: LLMPurpose) -> ChatGoogleGenerativeAI:
    model = _MODEL_BY_PURPOSE.get(purpose, _DEFAULT_MODEL)
    config = _PURPOSE_CONFIG.get(purpose, _PURPOSE_CONFIG["rag_qa"])
    return ChatGoogleGenerativeAI(model=model, **config)
```

#### 模型分配

| 用途          | 模型               | 原因                    |
| ------------- | ------------------ | ----------------------- |
| `translation` | `gemini-3.0-flash` | 高輸出限制 (65K tokens) |
| 其他所有      | `gemma-3-27b-it`   | 推理品質較好            |

---

## Phase 4.2: 翻譯頁面分塊

### 📁 新增檔案: `pdfserviceMD/translation_chunker.py`

#### 完整程式碼結構

```python
"""
Translation Chunker Module

Provides page-based chunking for large document translation.
Splits markdown by [[PAGE_N]] markers and batches pages based on output token limits.
"""

import logging
import re
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm_factory import get_llm

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 60000  # Buffer below 65K limit
CHARS_PER_TOKEN_ESTIMATE = 1.33


def estimate_tokens(text: str) -> int:
    """Estimates tokens (~0.75 tokens per char for Chinese)."""
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def split_by_page_markers(markdown: str) -> List[Tuple[int, str]]:
    """Splits markdown by [[PAGE_N]] markers."""
    pattern = r"\[\[PAGE_(\d+)\]\]"
    parts = re.split(pattern, markdown)

    pages: List[Tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            page_num = int(parts[i])
            content = parts[i + 1].strip()
            if content:
                pages.append((page_num, content))
    return pages


def batch_pages(
    pages: List[Tuple[int, str]],
    max_output_tokens: int = MAX_OUTPUT_TOKENS
) -> List[List[Tuple[int, str]]]:
    """Batches pages respecting output token limits."""
    batches: List[List[Tuple[int, str]]] = []
    current_batch: List[Tuple[int, str]] = []
    current_tokens = 0

    for page_num, content in pages:
        estimated_output = estimate_tokens(content) * 1.2

        if current_tokens + estimated_output > max_output_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append((page_num, content))
        current_tokens += estimated_output

    if current_batch:
        batches.append(current_batch)

    return batches


async def translate_single_page(content: str) -> str:
    """Translates a single page without markers."""
    template = """你是一個翻譯助手。請將以下 Markdown 文字翻譯成繁體中文。

    注意：
    1. 保留所有 Markdown 結構
    2. 保留所有 [IMG_PLACEHOLDER_X] 標記
    3. 僅翻譯英文文字
    4. 直接輸出 Markdown，不要加註說明

    Markdown 內容:
    {input_text}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm("translation")
    chain = prompt | llm | StrOutputParser()

    try:
        return await chain.ainvoke({"input_text": content})
    except Exception as e:
        logger.error(f"Page translation failed: {e}")
        return content  # Graceful degradation


async def translate_batch(batch: List[Tuple[int, str]]) -> str:
    """Translates batch, adding markers ourselves."""
    translated_pages: List[str] = []

    for page_num, content in batch:
        translated = await translate_single_page(content)
        # WE add marker back - not relying on LLM
        translated_pages.append(f"[[PAGE_{page_num}]]\n{translated}")

    return "\n\n".join(translated_pages)


async def translate_chunked(markdown: str) -> str:
    """Main entry point for chunked translation."""
    pages = split_by_page_markers(markdown)

    if not pages:
        return await translate_batch([(1, markdown)])

    batches = batch_pages(pages)

    if len(batches) == 1:
        return await translate_batch(batches[0])

    translated_batches = []
    for batch in batches:
        translated = await translate_batch(batch)
        translated_batches.append(translated)

    return "\n\n".join(translated_batches)
```

### 📁 修改檔案: `pdfserviceMD/ai_translate_md.py`

```python
from pdfserviceMD.translation_chunker import translate_chunked

async def translate_text(text: str) -> str:
    """Translates text using page-based chunking."""
    if not text or not text.strip():
        return text

    try:
        result = await translate_chunked(text)
        return result
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text
```

---

## Phase 4.3: 交錯式多模態問答

### 📁 修改檔案: `data_base/RAG_QA_service.py`

#### 新提示結構 (Line ~221)

```python
# Step 8: Build interleaved multimodal message
context_text = "\n\n---\n\n".join(text_context) if text_context else "(無文字背景資訊)"

prompt_text = f"""你是一位學術研究助手，擅長分析文本與圖表。

## 參考資料
以下是從知識庫檢索到的相關內容：

{context_text}

## 使用者問題
{question}

## 回答指引
1. 仔細觀察圖表/圖片中的數據與趨勢（如有提供）
2. 結合文字內容與圖片資訊進行推理
3. 引用具體來源時，說明資訊出處
4. 數學公式請使用 LaTeX 格式 (例如 $\\frac{{a}}{{b}}$)
5. 以繁體中文回答
6. 如果圖片與問題無關，請忽略圖片

請根據以上資料回答問題："""
```

---

## 測試驗證

### 自動測試

```powershell
cd d:\flutterserver\pdftopng
pytest tests/ -v
# 結果: 104 passed
```

### 手動測試指令

```powershell
# 1. 啟動服務器
uvicorn main:app --reload --port 8000

# 2. 上傳 PDF + OCR + 翻譯
curl.exe -X POST "http://localhost:8000/pdfmd/ocr" `
  -F "file=@nnunetv2.pdf" -o translated.pdf

# 3. 問答測試
curl.exe -G "http://localhost:8000/rag/ask" `
  --data-urlencode "question=什麼是nnU-Net"

# 4. 深度研究測試
python -c "import httpx; r=httpx.post('http://localhost:8000/rag/research', json={'question':'比較nnU-Net與U-Net的差異'}, timeout=120); print(r.json()['summary'])"
```

---

## 修復紀錄

### Bug: 翻譯後 PDF 只有 3 頁

**原因**: LLM 翻譯時刪除 `[[PAGE_N]]` 標記

**修復**: 改為逐頁翻譯，我們自己添加 `[[PAGE_N]]` 標記

```python
# Before (問題)
translated = await llm.invoke(combined_with_markers)

# After (修復)
for page_num, content in batch:
    translated = await translate_single_page(content)
    result.append(f"[[PAGE_{page_num}]]\n{translated}")  # 我們加回標記
```

---

## 相關檔案清單

| 檔案                                  |   操作   | 說明         |
| ------------------------------------- | :------: | ------------ |
| `core/llm_factory.py`                 |   修改   | 雙模型支援   |
| `pdfserviceMD/translation_chunker.py` | **新增** | 頁面分塊翻譯 |
| `pdfserviceMD/ai_translate_md.py`     |   修改   | 使用分塊翻譯 |
| `data_base/RAG_QA_service.py`         |   修改   | 交錯式提示   |
