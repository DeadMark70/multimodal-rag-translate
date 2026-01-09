# 🚀 PDF 生成引擎升級計畫書 v2.0：Robust Academic Publisher

## 1. 核心痛點與對策 (Challenges & Solutions)

| 痛點 | 現象描述 | 解決方案 (The "Sanitizer") |
| :--- | :--- | :--- |
| **數學公式炸彈** | Unicode (α) 與 LaTeX (`\alpha`) 混雜；缺失 `$` 包裹；OCR 將公式誤認為文字。 | 1. **Regex Pre-processor**: 自動掃描並修復常見的未包裹數學符號。<br>2. **LaTeX Preamble**: 引入 `unicode-math` 宏包，讓 XeLaTeX 能直接渲染 Unicode 數學符號，減少轉換錯誤。 |
| **圖片路徑地獄** | Marker 輸出圖片到子目錄，但 Markdown 內的路徑可能是相對的，Pandoc 執行目錄不同會找不到。 | 1. **Path Normalization**: 在送給 Pandoc 前，Python 腳本強制將 Markdown 內的所有圖片路徑轉換為 **OS 絕對路徑**。<br>2. **Image Check**: 檢查圖片是否存在，若丟失則替換為「Image Not Found」佔位圖，防止編譯崩潰。 |
| **表格溢出** | Marker 產生的 Markdown 表格欄位過多，轉成 PDF 後超出頁邊距 (Overflow)。 | 1. **Auto-Scale**: 在 LaTeX 模板注入 `\usepackage{adjustbox}` 或重定義表格環境，強制寬表格自動縮放至頁面寬度。<br>2. **Longtable**: 啟用跨頁表格支援。 |

## 2. 實施步驟 (Execution Plan)

### 階段一：增強型環境配置 (Infrastructure)
*目標：建立一個「能吃苦」的編譯環境，容忍度要高。*
1.  **Template Setup**: 下載 Eisvogel 模板，並進行**魔改 (Customization)**：
    -   注入 CJK 字體設定 (使用 `xeCJK`)。
    -   **關鍵修正**: 確保 `float` 相關設定正確，避免圖片把文字擠到看不見的地方。
2.  **LaTeX Packages**: 確保安裝處理複雜排版的關鍵包：`unicode-math` (數學容錯), `float` (圖片定位), `longtable` (跨頁表格), `booktabs` (三線表美化)。

### 階段二：開發 "Markdown Sanitizer" (核心防禦層)
*目標：在 Pandoc 看到檔案前，先由 Python 進行手術級清洗。*
我們將建立一個新的模組 `markdown_cleaner.py`，包含以下功能：

1.  **Image Path Fixer**:
    ```python
    # 邏輯示意
    def fix_image_paths(content, base_dir):
        # 找到所有 ![]()，將 path 替換為 os.path.join(base_dir, path)
        # 驗證 os.path.exists()，不存在則替換為 placeholder.png
    ```
2.  **Math & Symbol Normalizer**:
    -   將常見的 OCR 錯誤（如 `\ alpha` 中間有空格）修復。
    -   Escape 處理：修復文字中未轉義的 LaTeX 保留字（如 `%`, `_`, `#`），這些在非數學環境下會導致編譯失敗。
3.  **Table Enhancer**:
    -   掃描 Markdown 表格寬度，如果過寬，自動添加 Pandoc 的屬性標籤（如果 Pandoc 版本支援），或在前後插入 LaTeX 縮放指令。

### 階段三：Marker 與 Pandoc 的深度整合
*目標：串聯 `PDF -> Marker -> Cleaning -> Pandoc -> PDF` 的流水線。*
1.  **Marker 輸出解析**: Marker 通常會產出一個 `images/` 資料夾。我們需要解析 Marker 的 JSON 或 Markdown 輸出，抓準圖片的「根目錄」。
2.  **Pandoc 參數調優**:
    -   `--resource-path`: 明確告訴 Pandoc 資源查找路徑。
    -   `--listings`: 啟用程式碼高亮。
    -   `-V linkcolor:blue`: 設定連結顏色（ArXiv 論文通常有大量引用跳轉）。

### 階段四：測試與極端案例驗證 (Stress Testing)
建立包含以下「地獄級」案例的測試集：
1.  **公式**: 包含矩陣、積分、以及混雜中文的變數（如 $效率 \times \alpha$）。
2.  **圖片**: 包含遺失的圖片、路徑包含空格的圖片。
3.  **表格**: 包含 10 欄以上的寬表格、跨頁長表格。
4.  **亂碼**: 包含罕見的 Unicode 符號（Emoji 或特殊的數學符號）。

## 3. 補充建議 (Additional Considerations)

1.  **容錯回退機制 (Fallback)**:
    -   如果 Pandoc/XeLaTeX 編譯失敗（這是機率性事件），系統應自動回退到「純 HTML 轉 PDF」模式（雖然醜一點，但保證產出），或者回退到無格式的 Text PDF。這對使用者體驗至關重要。
2.  **Debug 模式**:
    -   保留中間產生的 `.tex` 檔。當 PDF 生成失敗時，`.tex` 檔是唯一的 Debug 線索（它會告訴你具體是哪一行 LaTeX 語法炸了）。
