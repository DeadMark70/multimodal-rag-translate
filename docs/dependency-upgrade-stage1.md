# 第一批：Google SDK 2 與 API OCR 環境

## 範圍

- google-genai 1.75.0 → 2.24.0
- langchain-google-genai 4.2.0 → 4.4.0
- langchain-core 1.2.7 → 1.6.3
- 其餘直接相依的設定保留。這一批不遷移 Marker 2、Surya 模型、Transformers、
  OpenAI、PyTorch、前端或資料庫，也不重建舊索引。

Marker API 是現有 `RealDatalabProvider` 的 HTTP 呼叫，不使用本機 marker-pdf。
舊 marker-pdf 要求 google-genai <2，因此拆成兩種建置選項，共用
`requirements-common.txt`。隔離是不同映像／Python 環境，不是在同一個後端
同時載入兩套 Google SDK，也不增加新的常駐服務。

| 建置清單 | OCR | Google SDK |
|---|---|---|
| requirements.txt（預設建置） | Datalab API，需設定 USE_LOCAL_MARKER=false | 2.24.0 |
| requirements-local-marker.txt | 原本的本機 Marker 1.10.1，USE_LOCAL_MARKER=true | 1.75.0 |

不要在同一個 Python 環境疊裝兩份清單。Marker 2 的模型遷移留到 API 批次驗收後。
OCR 路由原本預設本機，本批保留該預設；未設定環境變數不會自動改送雲端。
使用預設映像卻未設定 API 模式時，本機 OCR 會提示缺少套件，而非自動轉送 API。

PyTorch 的 Docker 安裝流程保留，新增 TORCH_VERSION／TORCHVISION_VERSION
建置參數，讓這批可以沿用現行容器的版本。TORCH_INDEX_URL 仍選擇 wheel 來源。
不提供版本參數時會維持原本未固定版本的行為，因此請使用以下部署指令。
本批沒有宣稱已驗證伺服器 GPU 或已完成完整套件鎖檔。

## 伺服器部署（PowerShell）

先將此批程式同步至伺服器。沿用目前 API 部署設定：backend 的環境設定中
`USE_LOCAL_MARKER=false`，保留既有 DATALAB_API_KEY。
這批只需要重建 backend，不需重建 frontend、不需搬移 PostgreSQL。

```powershell
# 在舊 backend 還運行時先讀取版本；不輸出金鑰或其他環境變數。
$torchRuntime = docker compose exec -T backend python -c "import json, torch, torchvision; print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda}))" | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or !$torchRuntime.torch -or !$torchRuntime.torchvision) { throw '無法讀取目前 PyTorch 版本，請先確認 backend 正在運行。' }
$torchIndex = if ($torchRuntime.cuda) { 'https://download.pytorch.org/whl/cu' + $torchRuntime.cuda.Replace('.', '') } else { 'https://download.pytorch.org/whl/cpu' }

docker compose build --build-arg "TORCH_VERSION=$($torchRuntime.torch)" --build-arg "TORCHVISION_VERSION=$($torchRuntime.torchvision)" --build-arg "TORCH_INDEX_URL=$torchIndex" backend
if ($LASTEXITCODE -ne 0) { throw '建置失敗，保留目前運行的 backend。' }
docker compose up -d --no-deps backend
docker compose exec backend python -c "import importlib.metadata as m; print({n: m.version(n) for n in ['google-genai', 'langchain-google-genai', 'langchain-core']})"
docker compose exec backend python -m pip check
docker compose exec backend python -c "import torch, torchvision; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
```

上面使用 PyTorch 官方 wheel 來源。若現行映像使用自訂編譯版本或私有 wheel，
請沿用原安裝來源，不要更換版本來強行通過建置。部署後確認版本與讀出的值一致。

部署前保留目前 backend 映像 tag，以便回復。若要使用原本本機 Marker：

```powershell
docker compose build --build-arg REQUIREMENTS_FILE=requirements-local-marker.txt --build-arg "TORCH_VERSION=$($torchRuntime.torch)" --build-arg "TORCHVISION_VERSION=$($torchRuntime.torchvision)" --build-arg "TORCH_INDEX_URL=$torchIndex" backend
```

並設 `USE_LOCAL_MARKER=true` 後重建容器。這個選項保留舊版環境，不包含 SDK 2。

## 分批驗收

1. **Marker API**：上傳一份熟悉的 2–3 頁 PDF，含中文、表格或圖片。
   確認解析完成、Markdown 可讀、頁碼及引用位置正確。這會使用原有付費 API。
2. **一般對話**：用既有文件各問一次 Naive 與 Agentic，確認回答與串流完成。
3. **評估**：先用 2–3 題與上述兩模式，使用平常的 Gemini 模型及 Medium 思考設定。
   確認正確度、忠實度、相關性都有分數。
4. **Flex／觀測資料**：沿用平常的 Flex 設定再跑小批；確認完成、用量、成本及快取
   欄位合理。cache hit=0 可以正常；不要要求評分數字與舊版完全一致。
5. **部分重跑**：選一題補評分或重新評分，確認結果更新，並確認 JSON 匯出可用。

全部完成再開始下一批。自動測試會使用模擬 HTTP 回應，不會代替伺服器真實 API
驗收，也不會在本機使用正式金鑰或產生評估費用。

## 本機驗證結果

- API 與本機 Marker 兩種清單：Linux x86_64／Python 3.11 相依解析通過。
- 新建 Python 3.11 API 環境：完整 requirements 安裝、套件相依檢查通過。
- SDK 2 相關回歸：121 passed，2 skipped（未啟用本機 env 檢查），
  3 deselected（需要 PostgreSQL）；另 9 項 worker 測試同樣因無測試 PostgreSQL 未完成。
- 原 SDK 1 環境：18 項相容性測試通過，原 .venv 套件未更動。
- 模擬 HTTP 涵蓋 Minimal／Medium、Standard／Flex、JSON 評分、串流、模型列表、
  快取與 token 計數，以及 Datalab 提交／輪詢／頁碼解析。
- Ruff、Markdown 連結及 diff 空白檢查通過。
- 本機沒有 Docker 執行環境；完整 Linux 映像建置、GPU、真實 API 與 PostgreSQL
  整合仍待伺服器依上面清單驗收。Python 套件相依通過不代表這些項目已驗證。
