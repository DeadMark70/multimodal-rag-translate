# 翻譯分塊策略修正計劃

> **問題**: 逐頁翻譯導致 API 請求過多 (6/5 超過速率限制)
> **建立日期**: 2025-12-10

---

## 📊 問題分析

### 當前實作問題

| 策略     | 請求數量 | 速率限制 | 結果    |
| -------- | :------: | :------: | ------- |
| 逐頁翻譯 |    14    |  5 RPM   | ❌ 超限 |
| 批次翻譯 |   1-2    |  5 RPM   | ✅ 安全 |

### 速率限制圖示

![Rate limit exceeded](file:///C:/Users/USER/.gemini/antigravity/brain/9cf9eaaa-0e31-46da-beb4-8ef9afa35791/uploaded_image_1765375987367.png)

---

## 🎯 修正目標

1. **減少 API 請求次數**: 恢復批次翻譯
2. **保留頁面標記**: 使用更嚴格的 prompt
3. **容錯機制**: 標記丟失時的後處理修復

---

## 🛠️ 修正方案

### 方案概述

```
原始 Markdown
    │
    ▼
┌─────────────────┐
│ split_by_page   │  提取頁面內容和編號
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   batch_pages   │  按 token 限制分批 (多頁/批)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ translate_batch │  批次翻譯 (嚴格 prompt)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ validate_markers│  驗證標記完整性
└────────┬────────┘
         │
    如果標記丟失
         ▼
┌─────────────────┐
│ repair_markers  │  根據原始結構修復
└────────┬────────┘
         │
         ▼
   翻譯完成的 Markdown
```

---

### 1. 嚴格化翻譯 Prompt

```python
STRICT_TRANSLATION_PROMPT = """你是一個專業翻譯系統。將以下 Markdown 翻譯成繁體中文。

⚠️ 關鍵規則 (違反將導致系統錯誤):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. [[PAGE_X]] 標記絕對不能刪除、修改或合併
   - 輸入: [[PAGE_1]] → 輸出必須: [[PAGE_1]]
   - 每個標記必須獨立一行

2. [IMG_PLACEHOLDER_X] 標記必須原封不動保留

3. 只翻譯英文文字，保留所有:
   - Markdown 格式 (# 標題、程式碼區塊)
   - 數學公式
   - HTML 標籤
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

輸入內容:
{input_text}

翻譯後直接輸出，不要加任何說明:"""
```

### 2. 標記驗證與修復邏輯

```python
def validate_and_repair_markers(
    translated: str,
    original_pages: List[Tuple[int, str]]
) -> str:
    """驗證翻譯結果的頁面標記，如有缺失則修復。"""
    expected_markers = {f"[[PAGE_{p[0]}]]" for p in original_pages}
    found_markers = set(re.findall(r"\[\[PAGE_\d+\]\]", translated))

    if expected_markers == found_markers:
        return translated  # ✅ 標記完整

    # ⚠️ 標記缺失，嘗試修復
    logger.warning(f"Missing markers: {expected_markers - found_markers}")

    # 修復策略: 根據原始頁面數量重新分割翻譯結果
    return repair_markers_by_content_split(translated, original_pages)


def repair_markers_by_content_split(
    translated: str,
    original_pages: List[Tuple[int, str]]
) -> str:
    """根據原始頁面結構修復標記。"""
    # 計算每頁的相對長度比例
    total_length = sum(len(p[1]) for p in original_pages)

    rebuilt = []
    current_pos = 0

    for page_num, content in original_pages:
        ratio = len(content) / total_length
        chunk_length = int(len(translated) * ratio)

        # 找到最近的段落邊界
        end_pos = find_paragraph_boundary(translated, current_pos + chunk_length)

        chunk = translated[current_pos:end_pos].strip()
        rebuilt.append(f"[[PAGE_{page_num}]]\n{chunk}")
        current_pos = end_pos

    return "\n\n".join(rebuilt)
```

### 3. 修改 `translate_batch()` 函式

```python
async def translate_batch(batch: List[Tuple[int, str]]) -> str:
    """批次翻譯多個頁面 (單次 API 請求)。"""
    # 組合所有頁面 (包含標記)
    combined = "\n\n".join(
        f"[[PAGE_{page_num}]]\n{content}"
        for page_num, content in batch
    )

    # 使用嚴格 prompt 翻譯
    translated = await call_llm_with_strict_prompt(combined)

    # 驗證並修復標記
    return validate_and_repair_markers(translated, batch)
```

---

## 📋 實作步驟

### 必要修改

| #   | 檔案                     | 修改內容                                 |
| --- | ------------------------ | ---------------------------------------- |
| 1   | `translation_chunker.py` | 更新 prompt + 添加驗證邏輯               |
| 2   | `translation_chunker.py` | 修改 `translate_batch()` 改回批次        |
| 3   | `translation_chunker.py` | 新增 `validate_and_repair_markers()`     |
| 4   | `translation_chunker.py` | 新增 `repair_markers_by_content_split()` |

### 新增函式清單

```python
# 新增函式
def validate_and_repair_markers(translated: str, original_pages: List) -> str
def repair_markers_by_content_split(translated: str, original_pages: List) -> str
def find_paragraph_boundary(text: str, position: int) -> int

# 修改函式
async def translate_batch(batch: List) -> str  # 改回批次翻譯
```

---

## ✅ 驗證計劃

1. **單元測試**: 驗證標記修復邏輯
2. **API 請求計數**: 確認 14 頁 ≤ 2 次請求
3. **端對端測試**: `nnunetv2.pdf` 完整流程

---

## ⚠️ 風險與緩解

| 風險           | 緩解措施               |
| -------------- | ---------------------- |
| LLM 仍刪除標記 | 後處理修復保證輸出正確 |
| 修復分割不準確 | 使用段落邊界檢測       |
| 速率限制       | 批次化減少請求數       |
