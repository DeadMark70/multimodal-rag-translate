# 評估中心 PostgreSQL 搬移與驗證

## 這次改動

評估系統只使用 PostgreSQL，必須設定 `EVALUATION_DATABASE_URL`；未設定會明確報錯，不會回退到 SQLite。資料格式、IDs、歷史 attempt 與評分仍保留；認證、聊天、FAISS、上傳檔案及每位使用者的 test-case/model-config JSON 仍需沿用原 volume。

已成功搬移的部署，這次只需重建並重啟 backend，保留 PostgreSQL service、連線設定及 volumes。這次清理沒有新增 schema migration，**不要再次執行 `--sqlite` 匯入**。SQLite 原始快照和離線搬移工具仍保留；目前版本無法用移除環境變數的方式切回 SQLite。

- PostgreSQL 17.11、Psycopg 3.3.5、psycopg-pool 3.3.1；每程序預設最多 10 條連線，可用 `EVALUATION_DB_POOL_SIZE` 調整。
- 啟動只檢查 schema 版本。部署時執行 `scripts/migrate_evaluation.py` 套用版本化 SQL。
- 單一 Run 只讀該 Run 的 observability/accounting，避免載入全 Campaign trace。
- Campaign/Runs/Question Analysis/Agent Behavior 預設每頁 50、最多 100，使用 `offset`。Campaign 列表仍為陣列；其他三者提供 `next_offset`。分析頁的總樣本數仍代表整個 Campaign。
- summary/questions/behavior 使用持久摘要；資料提交後提高 revision。舊摘要會標示 `updating`，背景重算成功後才切換版本。重算途中有資料變動時不發布該次結果。首次無摘要需計算一次，衝突則回 503 + Retry-After，前端最多重試兩次。
- 只有讀過或預建過的摘要會背景更新。摘要邏輯更動需提高 `analysis_cache.FORMAT_VERSION`。
- 領取工作在短交易內用 advisory lock 互斥，模型呼叫不持有鎖；15 秒 heartbeat，5 分鐘沒有 heartbeat 才回收。停止程序只中斷自己的 attempts。這不是新工作排程框架。

## 本機真實資料演練

來源：使用者提供的 `evaluation (1).db`，667,938,816 bytes，以 SQLite URI `mode=ro` 開啟，原檔未修改。

- SQLite quick_check 正常，foreign_key_check 無違規。
- 25 張表全部完成筆數與完整內容 SHA-256 比對；不是抽樣匯入檢查。
- 172 Campaign、5,798 results、15,705 scores、21,683 attempts、32,641 usage events。
- 172 Campaign 的三種摘要共 516 份預建成功，沒有呼叫模型。
- 三個 trace 資料量較大的 Campaign：原全 Campaign Run 讀取約 113–375 ms；PostgreSQL 單 Run 讀取約 16–54 ms；熱摘要讀取約 0.5–2.4 ms。三者 Run 與 summary 輸出皆相等。
- 重算摘要本身約 57–132 ms；收益主要來自縮小讀取與重用計算。這些是本機函式計時，包含冷熱差異，每路徑三次，**不是伺服器瀏覽器端到端 P50/P95**。

可重現測試：

```sh
EVALUATION_TEST_POSTGRES_URL=postgresql://... python -m pytest tests -q
EVALUATION_DATABASE_URL=postgresql://... python scripts/benchmark_evaluation_reads.py \
  --report /backup/read-benchmark.json
```

測試在指定測試 DB 建立與移除隨機 `test_*` schema。資料庫相關測試需要 `EVALUATION_TEST_POSTGRES_URL`；CI 已提供 PostgreSQL service。benchmark 現在比較 PostgreSQL 全 Campaign、單 Run 和快取讀取，會更新 derived caches；上面的 SQLite 比較數字是搬移當時的驗證紀錄。不要拿正式 DB 當測試 DB。

## Docker 搬移順序

本機沒有 Docker；已用 PostgreSQL 17.11 實際執行匯入與測試，容器啟動仍須在伺服器確認。以下 shell 指令在 backend repository 執行；`BACKEND_IMAGE` 換成這版後端 image，`BACKUP_DIR` 換成備份絕對路徑。

1. 暫停所有 evaluation 寫入及舊 backend/worker 程序。保留舊 image、設定與 volumes。以 SQLite backup API/`.backup` 建立一致快照；不要在 WAL 尚有資料時只複製主 `.db`。先備份上傳資料、test-case/model-config JSON。
2. 設定 PostgreSQL 密碼，啟動附帶的獨立 service：

   ```sh
   export EVALUATION_POSTGRES_PASSWORD='your-password'
   docker compose -f compose.postgres.yml up -d --wait
   ```

   Service 不開 host port，資料在 named volume。勿執行 `down -v`。
3. 準備新後端 image，以及暫存 migration env 檔（不要提交 Git），只放下列設定。密碼須 URL encode；Compose 的 `POSTGRES_PASSWORD` 則是原始密碼。

   ```text
   EVALUATION_DATABASE_URL=postgresql://evaluation:URL_ENCODED_PASSWORD@evaluation-postgres:5432/evaluation
   EVALUATION_DB_POOL_SIZE=10
   ```

4. 使用新 backend image 的 CLI 匯入，**此步不啟動 API/worker**：

   ```sh
   docker run --rm --network evaluation-db \
     --env-file migration.env \
     -v "$BACKUP_DIR:/backup:ro" \
     "$BACKEND_IMAGE" python scripts/migrate_evaluation.py \
     --sqlite /backup/evaluation.db --warm-summaries
   ```

   目標原始表必須為空；有資料就拒絕，不覆寫。COPY、FK/unique 驗證、全表筆數/checksum 在同一 transaction，失敗 rollback。Schema 已套用會保留。若匯入成功但摘要預建失敗，匯入資料仍保留，修正後只用 `--warm-summaries` 重試，勿重複 `--sqlite`。
5. 將現有 backend service 接到 external network `evaluation-db`，加入上面的 `EVALUATION_DATABASE_URL`，保留全部原有 env/volume。啟動新 backend 及新版 frontend。API 啟動會恢復 pending/inflight 工作，**所以只在資料驗證完成、準備恢復評估時才啟動**。
6. 用原帳號確認 Campaign 數量、分頁、指定 Run evidence、歷史分數與摘要。跑一個小評估，確認新資料寫入與摘要更新；再恢復一般流量。記錄正式環境瀏覽器開 Campaign/切分頁時間。

外部 network 的 Compose 片段（併入既有 backend service，不取代其他設定）：

```yaml
services:
  backend:
    networks:
      - evaluation-db
    environment:
      EVALUATION_DATABASE_URL: ${EVALUATION_DATABASE_URL}
networks:
  evaluation-db:
    external: true
    name: evaluation-db
```

## 後續 schema 更新、備份與退回

- 後續升版先備份 PostgreSQL，再以新 image 執行 `python scripts/migrate_evaluation.py`（不带 `--sqlite`），成功後啟動新 API。不要用程式讀取請求代替 migration。
- PostgreSQL 依既有備份策略用 `pg_dump`/`pg_restore` 備份並演練還原；SQLite 原始快照保留供核對。
- **新 PostgreSQL 尚無正式新增寫入**時，可停止新 backend，復原舊 image/設定並接回原 SQLite volume。
- **已恢復 PostgreSQL 寫入**後，舊 SQLite 不含新增資料；不能直接切回當作無損退回。停止寫入並備份 PostgreSQL，先修復或制定資料回填方式。

## 目前邊界

匯出、其他分析端點及 Jobs/items 的 API 契約仍保留。Jobs 列表已消除每 Job 個別統計查詢、降低輪詢頻率；若未來單 Campaign 累積大量重跑，再針對 Jobs/items 分頁。分析快取建立仍需讀整個 Campaign 的分析資料，首度建置可用預建避開；每次讀取只載入該類摘要。正式伺服器部署及網路/UI 計時尚未執行。
