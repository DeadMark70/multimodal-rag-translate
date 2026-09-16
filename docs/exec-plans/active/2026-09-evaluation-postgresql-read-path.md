# 評估中心 PostgreSQL 與頁面讀取改造

日期：2026-09-16

狀態：使用者已授權；程式與本機資料演練完成，正式伺服器部署尚未執行。
目標：同一次交付完成 PostgreSQL 遷移、按頁面讀取、可更新摘要及前端串接，避免資料增長使頁面反覆載入整批紀錄。

## 實作結案更新（2026-09-16）

**以下第 1–11 節保留原始設計紀錄，不代表最終實作。實際設定、部署及驗證請以 [PostgreSQL 搬移指南](../../evaluation-postgresql.md) 為準。**

- 以使用者提供的最新 667,938,816-byte SQLite 檔案，完成 25 張表全筆數與 SHA-256 內容驗證，172 個 Campaign 共 516 份分析頁摘要預建成功。
- PostgreSQL 透過 `EVALUATION_DATABASE_URL` 啟用；保留 SQLite 相容路徑。自動審核要求分階段替換，未刪除 SQLite 初始化或加入雙寫。
- 保留 TEXT IDs、JSON text、ISO 時間；不在這次同時改 JSONB 與時間欄位格式。數值使用 DOUBLE PRECISION 保留既有精度。
- 使用既有 Repository，加少量 bind-marker 適配與明確 UPSERT。工作領取沿用短交易，PostgreSQL advisory transaction lock 跨程序互斥；不新增 SKIP LOCKED/lease 框架。
- 三種分析（summary/questions/behavior）共用一張持久 cache 表與 campaign revision。提交資料後使摘要過期；讀取先回上次版本，背景重算；首次可由搬移工具預建。
- 單 Run 使用 run-scoped SQL；Campaign、Runs、Question Analysis、Agent Behavior 有分頁；前端共用 Run 資料並降低輪詢。Jobs 列表改聚合查詢，仍回完整 Jobs/items；其他既有分析/export 仍保留原契約。
- Worker 只回收超過 5 分鐘無 heartbeat 的 attempts；停止只處理本程序持有的 attempts。
- 本機驗證包含匯入、完整資料 checksum、摘要預建、結果一致性、讀取計時、雙引擎回歸與並行 claim/cache 測試。正式端到端 P50/P95、Docker 啟動和正式還原尚未執行。

## 1. 依據與範圍

- 使用者確認慢的是評估中心開啟 Campaign、切換分析分頁；遠端採 Docker、本機磁碟。程序數、正式資料規模及耗時尚未量測。
- `research_analytics.get_run_observability()` 看一個 Run 仍呼叫 campaign 級 observability/accounting loader。
- `job_store.list_jobs()` 對每個 Job 個別 `_with_job_status()`；前端 Job panel 每 1.5 秒重新列出 Jobs，終態也持續。
- Research summary 每次重新讀取結果、評分、work snapshots、用量及 LLM calls；舊 analytics 的程序內快取沒有覆蓋此路徑。
- 最新工作狀態查詢缺少 `(work_item_id, created_at DESC, id DESC)` 索引。依目前 schema 的記憶體 EXPLAIN 顯示 SCAN + TEMP B-TREE，補索引後為 covering index SEARCH；這不是正式伺服器測速。
- Worker 啟停會全域中斷 running scopes/attempts；`BEGIN IMMEDIATE` 及程序內 locks 都需要改為符合 PostgreSQL 的工作協調。

本次搬移 `evaluation.db` 內完整資料：Campaign、結果、Jobs、work items、attempts、RAGAS、accounting、trace、retrieval、evidence、graph events、human ratings 等。保留歷史 IDs、來源 attempt、評分設定及失敗紀錄。

現有使用者認證、聊天儲存、FAISS、原始檔案沿用原系統。測試集與 model configs 目前是每位使用者的 JSON 檔，這次保留既有格式及 volume；搬移清單需標明其備份位置，不能誤認全在 SQLite。

## 2. 技術決策

- 正式評估資料使用獨立 PostgreSQL Docker service 與持久 volume；版本鎖定受支援的穩定主版本及測試過的 patch/image，實作時確認環境相容性。
- Python 採 Psycopg 3 + `psycopg_pool.AsyncConnectionPool`，保留現有 Repository/Service 邊界與顯式 SQL。連線池在 FastAPI lifespan 開關，容量以所有程序總和計算並由環境設定。
- `EVALUATION_DATABASE_URL` 為評估資料庫設定。使用版本化 SQL migration 與版本表，部署時單獨執行；API 啟動檢查 schema 版本，不再每個程序啟動都修改 schema。
- PostgreSQL 是新 runtime 的唯一評估儲存引擎。SQLite 支援集中在搬移工具；不新增雙引擎抽象層或雙寫。
- ID 保持 TEXT，避免假設所有歷史 ID 都是 UUID；JSON 結構用 JSONB，時間轉 UTC `timestamptz`，缺失數值保持 NULL，零分仍為零分。
- 將查詢常用的 `campaign_result_id`、metric、mode、source attempt、evaluation signature 放在可索引欄位；immutable input snapshots 仍保留供重跑，但列表及摘要不反覆反序列化整份輸入。
- 移除 SQLite PRAGMA、`?` placeholder、`INSERT OR REPLACE`、JSON extraction 及 locked/busy retry 的專用假設；UPSERT 和 transaction 使用 PostgreSQL 語義。
- 每個受保護查詢都保留 user ownership 條件；前端只呼叫後端 API，資料庫僅暴露在部署內部網路。

## 3. 頁面讀取契約

| 畫面 | 新讀取方式 | 初始界限 |
|---|---|---|
| Campaign 選單 | 名稱、狀態、時間、進度；游標分頁 | 50 筆，最多 100 筆 |
| Campaign 概覽 / Mode Comparison | 已儲存摘要、有效/失敗/缺失樣本數、摘要版本與時間 | 固定結構，不包含答案、context 或完整 trace |
| Question Comparison | 按題彙整的列表分頁；全體統計另外讀 Campaign 摘要 | 50 題，最多 100 題 |
| Runs / Jobs / Job items | 精簡欄位分頁；Job counts 一次 GROUP BY | 50 筆，最多 100 筆 |
| 單一 Run | user + campaign + run 限定的基本資料 | 不呼叫 campaign 全量 loader |
| Run 的追蹤、證據、工具、LLM 分頁 | 僅載入選中 Run 及選中資料類型 | 事件 100 筆，最多 500 筆；長內容按項展開 |
| Agent behavior / Router / Errors 等 | 摘要加必要的分頁明細 | 不在每次切 tab 重建整個 campaign container |
| Release Metrics | 有 benchmark 時才取，依結果版本及 benchmark 版本保存分析結果 | 不塞進一般概覽 |
| 完整匯出 | 獨立批次讀取全部資料，保留原始明細 | 完整性不受 UI 分頁影響 |

游標由穩定排序鍵與 ID 組成。比較表的 overall/mean/sample count 來自完整分析，不能用當頁資料冒充整體統計。多 Run 比較一次讀所選 IDs，不能迴圈讀整個 Campaign。

分頁 response 使用 `items/next_cursor/has_more`；修改 API schemas、所有既有 callers、OpenAPI 與前端 fixture pin，前後端同版交付。仍有外部 consumer 的舊 contract 才保留過渡路徑，實作時盤點後記錄。

前端沿用現有 TanStack Query：以使用者、Campaign、Run、分頁、篩選及分析版本組成 key；共用 Runs/Run 基本資料，切換 Campaign 取消過時請求。進行中工作每 3–5 秒讀輕量狀態；全部終態後停止高頻輪詢，保留視窗重新聚焦刷新及約 30 秒的輕量版本檢查，以發現其他視窗的重跑。隱藏頁面暫停輪詢。

## 4. 可更新摘要

### 儲存結構

- `evaluation_run_summaries`：每個目前正式 Run 一筆。保留目前 source attempt、評分有效性、token/cost/latency、行為計數與必要比較欄位，不含原始大內容。
- `evaluation_question_summaries`：每個 Campaign/題目/分析版本的比較行，支援題目列表分頁。
- `evaluation_campaign_summaries`：按 Campaign、分析種類、分析版本、設定指紋保存小型 payload；記錄 `source_revision`、`built_at` 與建立狀態。設定指紋涵蓋該分析使用的 benchmark/定價/演算法版本。
- Campaign 的 `analysis_revision` 與持久化 refresh state：資料改變後可發現尚待更新的摘要；不依賴程序記憶體存活。

摘要是可重建的衍生資料。原始結果、評分、來源 attempt 與 accounting 才是事實來源；必須共用現有 canonical 計分及正式 attempt 判定，不能另寫一套不同分數規則。

### 更新時機與一致性

1. 新結果發布、失敗/取消狀態改變、RAGAS 寫入、重跑導致 official attempt 改變、accounting finalize/晚到事件、trace/evidence 完成、人工評分更新時，在原資料交易中更新 revision 並標記待刷新。
2. 相關事件以既有批次寫入為單位標記；heartbeat、逐 token 串流及純讀取不觸發重建。檢查 revision 熱點的寫入成本。
3. 背景 refresher 合併同一 Campaign 的多次更新；一般每 3–5 秒處理待更新項，終態優先。先更新受影響 Run，再重算相依題目及 Campaign 摘要；其餘 trace 不重讀。
4. Builder 在一致的資料庫 snapshot 讀取來源 revision 與資料。發布時僅接受未被更新來源淘汰的版本；如果期間又有資料寫入，仍保留待更新標記，不能錯誤清除新工作。
5. 頁面回傳最新可用摘要與 `source_revision/current_revision/built_at`。過期時顯示「統計更新中」；沒有摘要時回傳明確 unavailable/updating 狀態，不把缺資料當零分，也不在 GET 同步退回全量重算。
6. 刷新失敗保留上一版並顯示更新失敗狀態；重啟後從持久狀態接續。分數完成與分析更新中分開呈現，不能讓「評估完成」暗示摘要已同步。
7. 新 Campaign 建立空摘要，歷史 Campaign 在切換前批次回填；分析演算法版本改變時可批次重建，不重跑 LLM。

運行中是有標示的短暫延遲；所有寫入停止且刷新完成後，摘要必須與原始資料即時計算完全一致。百分位数等不可直接把小計相加，需從精簡 Run rows 正確重算。

## 5. 工作領取、重跑與恢復

- 領取在短交易內完成：以 PostgreSQL 行鎖及 `SKIP LOCKED` 選取待執行工作，原子更新 job item、建立 attempt 並寫入擁有者；交易完成後才呼叫外部模型。
- 同一 work item 跨不同 rerun job 的去重，必須鎖共同 work row；僅鎖 job item 不足以防止重複執行。
- 同一 Campaign 的執行數上限以 Campaign 行鎖保護「檢查容量＋領取」；同批鎖固定順序，不能只在程序內計數。
- attempt 記錄 worker owner、heartbeat/lease expiry；只恢復已到期或本 worker 正常停機取消的工作，取消現行 startup/shutdown 全域 interrupt。
- 回寫完成、heartbeat、official result、accounting finalize 時驗證目前 active attempt/owner。到期舊 worker 的遲到結果不得覆蓋新 attempt；診斷紀錄仍可保留為非正式紀錄。
- 外部模型呼叫在程序中斷後可能重試，因此不承諾外部呼叫 exactly-once；資料庫正式結果發布需具備冪等性。
- 跨程序新增 Job 的發現不能只靠本機 notify。沿用 DB queue，空閒時作低頻 bounded ready-work 查詢，本機 notify 僅用於加速。
- 第一版部署維持一個評估 worker 的運作規模；交易及 lease 測試使用兩個 worker 驗证，增加模型並行度另由既有容量設定控制。

## 6. 索引與 SQL

依實際 WHERE/ORDER BY 設計複合索引，至少涵蓋：

- Campaign `(user_id, created_at, id)`。
- Results/Run summary `(campaign_id, user_id, created_at, id)` 與題目/mode 篩選。
- Job items `(work_item_id, created_at DESC, id DESC)`、`(job_id, status, id)` 及待領取 partial index。
- Attempts 的 work/item 歷史排序與活動 lease expiry。
- Trace/evidence/accounting 的 campaign/run/attempt + 時間/ID 查詢鍵。
- 摘要的唯一鍵、分析版本及待更新索引。

使用有代表性的 PostgreSQL 資料及 EXPLAIN ANALYZE 驗證，避免每個 JSONB 欄位都建立通用索引。一般 GET 不執行生命周期修復 UPDATE；修復由 worker/reconciler 負責。

## 7. 一次交付的實作順序

1. **基線與資料契約**：建立只讀量測命令，記錄主要 endpoints 的 SQL 次數、讀取列數、payload、DB/組裝時間；保存現有分數與摘要 golden fixtures。遠端 baseline 與本機合成測試分開。
2. **PostgreSQL 儲存**：schema/migrations、pool/lifespan、Repository SQL、批次狀態、索引及 lease/claim/recovery 一起完成。主要檔案為 `evaluation/db.py`、`job_store.py`、`accounting_store.py`、`accounting_runtime.py`、`observability_storage.py`、`job_worker.py`、`core/app_factory.py`；盤點 `analytics.py` 的直接 SQL 及其他匯入者。
3. **讀取與摘要**：改 `research_analytics.py`、`analytics.py`、`release_metrics.py`，新增摘要儲存/refresh service；共用正確性計算，保留完整匯出路徑。
4. **前端與契約**：EvaluationCenter、JobPanel、分析分頁及 API client 改用分頁與 Query cache，呈現摘要新鮮度，更新契約與測試。
5. **搬移與驗收工具**：資料匯入、核對、歷史摘要回填、Docker/環境設定、備份還原手冊及容量測試。

以上為同一範圍的分段 commit，全部驗證完成才協調前後端上線；不把第一段當作整體已完成。

## 8. 正式資料切換

1. 先在獨立 PostgreSQL 以伺服器一致性備份試搬，測量搬移、回填及核對時間，據此安排維護窗口。
2. 正式切換時暫停新評估、補評、人工修改及所有評估 DB 寫入；等待 worker 完成或正常停止並記錄未完成工作。
3. 使用 SQLite backup API 建立一致性來源，或確認所有連線關閉且 WAL checkpoint 完成後備份。不能僅複製活躍的 `.db` 主檔。
4. 對明確指定的空目標 DB 執行 schema migration 與分批 COPY。保留 ID/關聯/順序；若歷史 JSON 或時間不符合新格式，列出來源表與 ID 讓搬移失敗，不靜默丟棄資料。
5. 核對各表列數、主外鍵、正規化後內容摘要、來源 attempts、評分值/NULL/零分、成本及用量；產生機器可讀報告。大型表分批處理，不整庫載入 Python 記憶體。
6. 回填所有歷史分析摘要、執行 ANALYZE；新後端以寫入停用及 worker 停用模式，配合新版前端做只讀驗證。
7. 驗證通過才開放寫入及 worker。恢復待執行工作依明確的 lease/中斷狀態處理，不能重新跑已成功模型請求。

**回復界限**：在 PostgreSQL 開放寫入前，可切回舊映像與一致性 SQLite 備份。開放寫入後不能直接切回舊 SQLite，否則遺失新結果；此時先停止寫入，備份 PostgreSQL，以修復新版或 PostgreSQL 還原為優先。第一次交付不假設存在反向同步工具。

提供 PostgreSQL `pg_dump`/還原操作及演練；Docker volume 本身不等於備份。需在上線時確認實際掛載、可用空間、備份位置與維護窗口。

## 9. 完成條件

- 在真實 PostgreSQL 跑 persistence、API、評分/缺失/重跑、取消、accounting、export、跨使用者隔離測試；SQLite 測試不能代替 PostgreSQL 驗證。
- 雙 worker 領取、同 work 跨 job、容量限制、啟停、lease 到期、遲到結果及去重測試通過；不呼叫付費模型即可驗證。
- 摘要與原始 canonical 計算一致，涵蓋補評、部分成功、零分、舊 attempt、晚到用量、人工評分、版本更新及 builder 期間再次寫入。
- 單 Run 的其他同 Campaign 資料增長 10 倍時，查詢僅涉及所選 Run；首屏讀取列數/bytes 保持頁面界限，不只是 SQL 次數固定。
- 同環境資料量與並發量測：已備妥摘要 API 的 server P95 目標 ≤500ms；第一頁列表/事件 API ≤1s；概覽 payload ≤100KB。這些是待驗收目標，不是尚未測量的承諾。
- 運行中摘要的 3–5 秒刷新是排程目標，另量測積壓；終態資料與摘要完成後完全一致，失敗或延遲在 UI 可見。
- 大型 Campaign、重跑歷史增長、執行中與已完成 Campaign 分別測試；純瀏覽不反覆讀 raw snapshots、不重建全 Campaign trace、不持續高頻輪詢終態工作。
- 前端分頁、切 tab 共用請求、取消過時回應、完整匯出、lint/test/build、OpenAPI contract 檢查通過。
- 遠端搬移演練、正式 before/after P50/P95 與還原演練完成後才標記整體完成。更新 BACKEND/FRONTEND/RELIABILITY、部署說明及相關 checklist。

## 10. 參考

- [Psycopg 3 connection pool / FastAPI lifespan](https://www.psycopg.org/psycopg3/docs/advanced/pool.html)
- [PostgreSQL SELECT locking / SKIP LOCKED](https://www.postgresql.org/docs/current/sql-select.html)
- [PostgreSQL SQL dump / restore](https://www.postgresql.org/docs/current/backup-dump.html)

## 11. 本輪產出

- 完成程式路徑與 SQL/worker 相依盤點，並核對官方 driver/pool/locking 文件。
- 僅新增此設計文件與索引；未修改 runtime、安裝依賴、啟動服務、搬移資料或部署。
