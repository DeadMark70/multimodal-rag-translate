# Evaluation Center and Chat Loading Performance Plan

## Objective

徹底改善以下兩條載入路徑，並確保效能不會隨 campaign、run、observability、conversation 或 message 數量線性惡化：

1. `/evaluation` 的 route `Loading...` 與 `Loading evaluation analytics...`
2. `/chat` 的 route `Loading...`、對話紀錄首屏與長對話歷史載入

本計畫涵蓋前端、FastAPI、SQLite evaluation storage、Supabase conversation storage、API contract、索引、快取、量測與回歸測試。這是跨棧 canonical 計畫；前端 active plan index 連結至本文件，不另維護重複版本。

## Problem Statement

### Evaluation Center

- 首屏等待單一 `analytics-dashboard` 整包回應，最慢的分析項目會阻塞所有 overview UI。
- analytics context 讀取完整 `campaign_results.*`，包含 answer、contexts 與多個 JSON blob，但 overview 多數只需要摘要欄位。
- routing decisions 依 run 序列查詢，形成明確 N+1；查詢數會隨 run 數量線性增加。
- campaign 清單在初始選擇流程重複載入。
- dashboard 完成後立即抓完整 campaign results 與第一個 run observability，即使使用者尚未開啟相關 tab。
- RAGAS、routing、campaign results 的索引未完全對齊 campaign analytics 查詢模式。
- terminal campaign 每次開頁仍重新掃描 raw rows，未利用結果不可變或低變動特性。
- route 雖使用 lazy loading，但評估 tabs 與 setup surface 仍由靜態 import 進入 route dependency graph。

### Chat And Conversation History

- conversation list 使用 Supabase `select("*")`，會帶回完整 metadata；Deep Research metadata 可能非常大。
- list endpoint 無 limit、cursor 或 summary schema，資料量與 response size 無上限。
- 現有索引分別覆蓋 `user_id`、`updated_at`，沒有完全對齊 list query 的複合 keyset order。
- 前端一次保存和搜尋完整 conversations array，30 秒後可重新抓取。
- create/update/delete 會 invalidate 並等待整個 conversation list refetch。
- conversation detail 會一次載入全部 messages，單一長對話也會成為未來瓶頸。
- route 與 sidebar 尚未建立 idle/intent prefetch、infinite loading 與 virtualization 的完整策略。

## Scope

### In Scope

- Evaluation overview、tab analytics、run detail API 分層
- Campaign analytics bulk reads、summary projections、aggregate snapshots 與 cache policy
- SQLite query/index 改善與 query-count enforcement
- Conversation summary、cursor pagination、server-side search 與 message pagination
- TanStack Query cache、optimistic mutation、abort/cancellation、按需載入
- Evaluation tabs、setup drawer、chat/evaluation route chunk lazy loading 與 prefetch
- 遠端新 DB 的 read-only profiling、payload measurement 與基準紀錄
- 前後端 contract、回歸、效能與文件更新

### Out Of Scope

- 更換 RAG、Deep Research 或 evaluation scoring 演算法
- 改變既有 benchmark 分數、execution profile 或 context policy 的語意
- 將 evaluation storage 從 SQLite 遷移到其他資料庫；只有在本計畫完成後仍無法達標時另立 migration plan
- 刪除或改寫歷史 campaign、conversation、message 或 observability 資料
- 以延長 timeout、隱藏 spinner 或只改 loading copy 作為效能修正

## Guardrails

1. `conversations.metadata` 仍是 Deep Research persistence 的 canonical source；list API 只排除大型 detail，不改變 detail persistence contract。
2. Evaluation scoring policy 不得因效能重構而改變；若任何 aggregate 定義改變，必須同步 version `execution_profile`／`context_policy_version` 或另建 analytics schema version。
3. 所有 protected endpoints 保持明確 auth 與 ownership checks。
4. Running campaign 不可被長 TTL stale cache 隱藏進度；terminal campaign 才能使用 immutable-style cache。
5. API contract 改動必須在同一 change set 更新 backend schemas、frontend types、API clients、測試及 generated inventories。
6. 每個階段都必須可獨立回滾，不採一次性大爆炸切換。

## Baseline And Measurement Gate

實作前先在實際使用的新遠端 DB 與部署環境進行唯讀量測。不得以 repository 內舊 `evaluation.db` 取代正式 baseline。

### Browser Measurements

- `/evaluation` route chunk download、parse、React mount
- `GET /api/evaluation/campaigns`
- overview/dashboard request 的 TTFB、download time、decoded bytes
- loading indicator 可見時間
- first useful overview render
- 切換各 tab 到資料可用的時間
- `/chat` route ready time
- conversation first page TTFB、decoded bytes、row count
- conversation selection到最新 messages 可見的時間

### Backend Spans

為 evaluation request 分段記錄：

- auth/ownership
- campaign lookup
- result summary read
- LLM aggregate read
- routing read
- trace/error read
- human ratings read
- RAGAS read
- aggregate build
- Pydantic/JSON serialization
- response bytes

為 conversation request 分段記錄：

- Supabase request latency
- returned row count
- metadata bytes
- serialized response bytes
- message detail row count與 bytes

### Database Evidence

- 記錄 `EXPLAIN QUERY PLAN`／Supabase `EXPLAIN (ANALYZE, BUFFERS)` 結果
- 記錄每個 API 的 SQL/Supabase query count
- 記錄最大、P50、P95 campaign run count與 observability row count
- 記錄最大、P50、P95 conversation metadata bytes與 message count
- 對所有數字移除 user content、prompt、answer、token、email 與識別資訊

### Baseline Artifact

將去識別化 baseline 寫入本文件的 `Progress Log` 或相鄰 findings 文件，至少包含日期、環境、資料量、P50/P95、response bytes 與最慢 span。

## Work Items

## 1. Eliminate Evaluation Routing N+1

### Backend

- 在 `EvaluationObservabilityRepository` 新增 campaign-scoped routing bulk read，依 `run_id` 分組並維持穩定排序。
- 將 `_routing_decisions_for_context()` 改為一次 bulk query；不得在 result loop 內 await repository。
- 新增與查詢對齊的 SQLite index，例如：
  - `evaluation_routing_decisions(campaign_id, run_id, created_at ASC)`
- 為 dashboard service 增加 query-count test：run 數量從 1 增至 N 時，routing query count 必須維持 1。
- 驗證 running、terminal、legacy/no-routing campaign 都保持相容。

### Acceptance

- `analytics-dashboard` 不再呼叫 `list_routing_decisions_for_run()`。
- Campaign routing query count 固定為 1，不隨 run 數量增加。
- Router Lab response 與重構前的 canonical fixture 完全一致。

## 2. Add Evaluation Summary Projection

### Backend

- 新增 analytics 專用 result projection，不讀取完整 answer、contexts、ground truth、snapshots 或其他首屏不需要的大型 JSON。
- projection 只保留 overview、mode/question comparison、cost/latency、ablation、run option 所需欄位。
- 將 ownership verification 與 summary query 分離或用安全 join/CTE 完成，避免為驗證 ownership 載入完整 campaign/result model。
- 為下列查詢補齊或調整複合索引：
  - campaign results 的 campaign/user/order path
  - RAGAS 的 `(campaign_id, user_id, campaign_result_id)` path
- 移除 query plan 中不必要的 full scan 與大型 temp sort；保留 deterministic order。

### Contract

- 現有 detail/results endpoint 保持可讀完整資料。
- Analytics service 使用專用內部 row/model，不將 summary projection 假裝成完整 `CampaignResult`。

### Acceptance

- Overview path 不讀取 answer 或 contexts blob。
- 對最大遠端 campaign，summary query bytes 至少比完整 results read 降低 80%。
- Aggregate fixture 與既有結果一致。

## 3. Split Evaluation First Paint From Tab Analytics

### API Design

- 建立固定大小、首屏專用的 campaign overview endpoint/response：
  - status、updated time
  - sample/question/repeat/mode counts
  - token/cost/latency summary
  - error count與資料新鮮度
- 將 Question、Router、Ablation、Human、Run list 分成按需 endpoints，或讓現有 endpoints 成為推薦路徑。
- 所有大型 collection 支援 cursor pagination與明確 `limit` 上限。
- Run observability 維持 dedicated detail endpoints，禁止首屏自動取第一個 run 的完整 dump。

### Frontend

- `/evaluation` 先渲染 campaign selector 與 overview skeleton；overview 完成即可結束主 loading。
- Tab 第一次被選取時才 fetch 對應資料。
- run 被明確選取後才讀 trace/retrieval/claims/LLM calls。
- 使用 TanStack Query 管理 server state、dedupe、cache、retry與 cancellation，移除 page-local request fan-out。
- campaign 切換時 abort 舊請求，防止 stale response 覆蓋新 selection。
- 保留前一個 campaign 的 UI frame，避免整頁空白；資料區顯示局部 loading。

### Acceptance

- 首屏 overview 不等待 Router、Human、Ablation 或 run detail。
- 未開啟的 tab 不產生其 data request。
- 切換 campaign 不會重複抓 campaign list。
- 快速連續切換 campaign 不會顯示錯誤 campaign 的 late response。

## 4. Materialize And Cache Terminal Campaign Analytics

### Backend

- 設計 versioned `campaign_analytics_snapshots` 或等價持久化 aggregate：
  - `campaign_id`
  - analytics schema version
  - source campaign updated/version marker
  - overview及各 aggregate摘要
  - computed timestamp
- Campaign 完成 evaluation 後產生 snapshot；必要時提供明確 rebuild/repair job。
- Terminal campaign 優先讀 snapshot；schema/source version 不符才重算。
- Running campaign 使用 live summary 或短 TTL，不得讀過期 terminal snapshot。
- 增加 process-local single-flight，避免多個使用者同時打開同一 campaign 時重複計算。
- 回應加入 ETag/Last-Modified 或等價 validators；production 啟用 gzip/Brotli。

### Frontend

- Terminal campaign query 使用長 stale time，並以 campaign updated/version marker invalidation。
- Running campaign由 SSE/polling更新小型 summary，不重抓全量 raw analytics。

### Acceptance

- Terminal campaign 第二次讀取不掃描 raw observability tables。
- 同一 cache miss 的併發請求只觸發一次重算。
- Snapshot schema bump 可以安全重算，不污染舊 benchmark 語意。

## 5. Paginate Evaluation Collections And Details

### Backend

- 為 campaigns、runs、question rows、errors、human queue、trace、retrieval chunks、claims、LLM calls 提供 cursor pagination。
- Cursor 使用穩定排序欄位加唯一 ID；禁止只用非唯一 timestamp。
- 設定合理 default/max limit，並回傳 `next_cursor`。
- Export endpoint 可保留完整資料能力，但不可被 dashboard 自動呼叫，並保持 redaction policy。

### Frontend

- 大型 table 使用 incremental loading或 server pagination。
- Trace/retrieval/claim table 使用 virtualization，避免 DOM node 與 raw row 數量等比例增加。
- Filter/sort 若作用於完整資料集，移至 server；不得只對已載入頁面假裝是全域結果。

### Acceptance

- Dashboard 一次 request 不回傳無上限 collection。
- 10 倍 run/trace 資料量不導致首屏 response bytes 增加 10 倍。
- Pagination 過程無重複、遺漏或排序跳動。

## 6. Add Conversation Summary And Cursor Pagination

### Backend/Supabase

- 新增 `ConversationSummaryResponse`，只回傳：
  - id、title、type、created_at、updated_at
  - restore mode 所需的小型 metadata subset
- List query 禁止 `select("*")`，不回傳 Deep Research result正文。
- `conversations.metadata` 繼續保存 canonical detail；完整資料只從 conversation detail endpoint讀取。
- List endpoint 支援 `limit`、`cursor_updated_at`、`cursor_id`。
- 新增複合 index：
  - `conversations(user_id, updated_at DESC, id DESC)`
- 若需要全域搜尋，新增 server-side search與適合的 trigram/FTS index；搜尋輸入 debounce。
- 驗證 RLS、auth與 user ownership不因 projection/pagination改變。

### Frontend

- `useConversationList` 改為 infinite query。
- 首頁只取 30–50 筆，scroll near end才取下一頁。
- 搜尋不再依賴先下載所有 conversations。
- Restore mode只依 summary允許的 metadata subset；詳細研究內容仍按需讀 detail。

### Acceptance

- Conversation list response不含 research result、detailed answer、message bodies或大型 metadata。
- 首頁 query固定 limit且使用 composite index。
- Conversation總數增長不會增加第一頁 response row count或 payload上限。

## 7. Optimize Conversation Mutations And Message History

### Frontend Cache

- create/update/delete 使用 optimistic cache update，不 await全量 list invalidation。
- Background reconcile只更新受影響頁面；失敗時 rollback並顯示既有 error UX。
- 調整 stale/refetch policy，避免每次 focus或短時間 route切換重抓相同第一頁。
- Conversation detail與list summary使用不同 query keys和response types。

### Message History

- Detail endpoint支援 message cursor pagination，預設載入最新一頁。
- 向上捲動才載入較舊 messages，並保持 scroll anchor。
- Message list使用 virtualization或 windowing，但 streaming中的最新訊息保持穩定。
- 新增訊息後只 append/update local detail cache，不重抓完整對話。

### Acceptance

- 新增、改名、刪除 conversation不觸發完整 history refetch。
- 長對話首屏只載入固定數量最新 messages。
- 載入舊訊息不造成 scroll jump、重複訊息或 SSE streaming中斷。

## 8. Optimize Route Bundles And Prefetch

### Frontend

- 將 Evaluation Setup drawer改成 dynamic import，首次開啟才下載與 mount。
- 將非預設 Evaluation tabs改成真正 dynamic import；`Tabs isLazy` 不視為 bundle splitting完成。
- 追查 Evaluation route為何帶入 PolarChart/Recharts chunk；移出非必要首屏依賴。
- 保留 Chat 的 Deep Research/Benchmark/Settings lazy boundaries，並檢查 ConversationSidebar是否能與 route平行預取。
- Auth完成後使用 idle prefetch熱門 routes；navigation link hover/focus時做 intent prefetch。
- production hashed assets設定長效 cache；HTML不使用錯誤長 TTL。
- 驗證 gzip/Brotli與HTTP/2/HTTP/3傳輸設定。

### Acceptance

- Setup drawer與未開啟 tabs不在 Evaluation首屏下載清單。
- Warm navigation不再長時間顯示通用 `Loading...`。
- Cold navigation的 route-specific transferred bytes有明確 budget並通過CI bundle check。
- Prefetch不下載需要auth後才允許的資料，只預取程式碼或安全summary。

## Delivery Sequence

每個階段依序執行 Explore → Implement → Verify → Document：

1. 遠端 baseline與query-count instrumentation
2. Work item 1：routing bulk/N+1
3. Work item 2：summary projection/indexes
4. Work item 3：overview/tab/run分層
5. Work item 4：terminal aggregate snapshot/cache
6. Work item 5：evaluation pagination/virtualization
7. Work item 6：conversation summary/cursor
8. Work item 7：optimistic mutation/message pagination
9. Work item 8：bundle split/prefetch
10. 遠端相同資料集重測、容量測試、文件與rollout closeout

Work items 1–2完成前，不應先以前端 loading UX掩蓋後端結構性成本。Work items 6–7可在 evaluation work items 3–5進行時由另一條實作線處理，但API contract合併與遠端驗收需統一控管。

## Verification Strategy

### Backend

- Focused pytest：analytics context、bulk observability、API contract、export redaction、legacy campaign compatibility
- Query-count tests：1 run與N runs的SQL count相同
- Projection tests：overview path不得讀取或序列化 answer/context blobs
- Pagination contract tests：cursor穩定性、邊界、空集合、刪除/新增期間行為
- Auth/RLS tests：跨user campaign/conversation/message不可讀取
- SQLite query-plan assertions或migration index checks
- Snapshot invalidation、schema version、single-flight與running campaign tests

### Frontend

- Evaluation：overview先顯示、tab按需、run按需、campaign切換cancel、partial error states
- Conversations：first page、load more、server search、optimistic create/update/delete與rollback
- Messages：latest page、load older、scroll anchor、stream append
- Route chunk：drawer/tab dynamic import與prefetch smoke
- CI-equivalent：`lint:ci`、`tsc --noEmit`、full Vitest、production build

### Remote Performance

- 使用同一批去識別化 campaign/conversation選樣進行before/after
- 至少測量P50、P95與最大資料量案例
- 冷cache、暖cache、running campaign、terminal campaign分開測
- 慢速網路與中階CPU裝置至少各測一次
- 記錄query count、response bytes、TTFB、first useful render與tab-ready time

## Success Metrics

以下為初始budget；遠端baseline完成後只能收緊或以書面理由調整：

| Surface | Target |
|---|---|
| Evaluation terminal overview warm P95 | ≤ 500 ms server time |
| Evaluation terminal overview cold P95 | ≤ 1,000 ms server time |
| Evaluation overview decoded payload | ≤ 100 KB |
| Evaluation routing SQL count | 1 per campaign request |
| Evaluation first useful render | ≤ 1.5 s on normal production network |
| Evaluation deferred tab P95 | ≤ 1.0 s for first page |
| Conversation first page P95 | ≤ 500 ms end-to-end |
| Conversation first page decoded payload | ≤ 75 KB, limit 50 rows |
| Message first page | ≤ 50 latest messages by default |
| Warm `/chat` or `/evaluation` route transition | ≤ 300 ms to shell render |
| Scale behavior | 10× total stored data causes ≤ 2× first-page latency and bounded first-page bytes |

## Risks

1. Summary projection或materialized aggregates可能與既有完整result語意漂移。
   - Mitigation：canonical fixture、schema version、雙讀比對與shadow rollout。
2. Cursor pagination在相同timestamp或資料更新時可能重複/遺漏。
   - Mitigation：timestamp加唯一ID tie-breaker，contract tests涵蓋並發新增/更新。
3. Terminal cache可能把仍在evaluation的campaign誤判為immutable。
   - Mitigation：只對terminal status啟用，key包含updated/version marker。
4. Conversation summary排除metadata後可能破壞mode restore。
   - Mitigation：明確定義小型restore metadata schema並做frontend restore regression。
5. Virtualization可能影響scroll anchor、keyboard navigation與accessibility。
   - Mitigation：保留semantic roles、focus tests與live browser驗證。
6. SQLite在遠端高併發下即使查詢改善仍可能受單機storage限制。
   - Mitigation：先完成bounded reads與snapshots；若仍不達標，再提出獨立storage migration plan。
7. Prefetch可能浪費頻寬或提前打受保護API。
   - Mitigation：只prefetch code或明確安全summary，使用idle/intent gate與Save-Data檢查。

## Rollout And Rollback

1. 先部署只讀instrumentation與indexes。
2. Routing bulk與summary projection以內部feature flag或shadow comparison啟用。
3. 新overview/tab APIs additive發布，保留舊dashboard endpoint作短期fallback。
4. 前端切新contract後觀察error、latency、payload與cache hit rate。
5. Conversation pagination additive發布；舊list contract暫時保留直到frontend完成切換。
6. Materialized snapshot若異常，可停用snapshot read並回到正確但較慢的live aggregate。
7. 完成一個穩定觀察窗口後才移除舊contract；移除需另列deprecation checklist。

## Documentation Updates During Implementation

同一change set至少更新：

- `pdftopng/docs/BACKEND.md`
- `pdftopng/docs/generated/api-surface.md`
- `pdftopng/docs/RELIABILITY.md`
- `Multimodal_RAG_System/docs/FRONTEND.md`
- `Multimodal_RAG_System/docs/generated/ui-surface.md`
- `Multimodal_RAG_System/docs/RELIABILITY.md`
- 對應product spec與本plan的progress/decision log

## Exit Criteria

只有以下條件全部成立，才可將本文件移至completed：

1. 八個work items全部完成或有明確核准的scope reduction。
2. 遠端新DB完成before/after P50/P95量測，且達到success metrics。
3. Evaluation query count不隨run數量線性增加。
4. Evaluation與conversation首屏response均有固定limit／bounded payload。
5. 未開啟的evaluation tabs與未選取的run不會預先抓detail。
6. Conversation list不再回傳大型research result metadata。
7. Long conversation使用message pagination且scroll/streaming回歸通過。
8. Terminal snapshot/cache正確處理version、invalidation與fallback。
9. Auth、RLS、export redaction、legacy data相容測試通過。
10. Frontend完整CI-equivalent與backend focused/full可行測試通過。
11. Current-state docs及generated inventories已同步。
12. Rollback path經過驗證，舊contract deprecation另有追蹤。

## Progress Log

- 2026-07-11：完成靜態診斷；確認evaluation routing N+1、full-result analytics read、首屏整包阻塞、deferred overfetch、conversation `select("*")`、無pagination與bundle eager dependency風險。
- 2026-07-11：確認repository內 `evaluation.db` 為舊資料，實際新DB位於遠端；所有效能數字須以遠端baseline重測。
- 2026-07-11：完成第一輪後端結構性改造：routing decisions campaign bulk read、analytics summary projection、campaign/routing/RAGAS複合索引，以及terminal campaign的process-local immutable-context cache。
- 2026-07-11：完成Evaluation首屏與tab按需載入：overview不再等待完整dashboard、未開啟tab不發資料請求、Evaluation tabs與setup drawer改為真正dynamic import；保留舊dashboard endpoint作相容 fallback。
- 2026-07-11：完成conversation summary/cursor API與前端infinite query、server-side search、message latest-page載入；list projection不再回傳完整metadata，長對話不再首屏讀取全部messages。
- 2026-07-11：目前仍待遠端新DB量測與後續實作：持久化analytics snapshot、evaluation各collection cursor pagination、conversation optimistic mutation、scroll virtualization、route idle/intent prefetch，以及完整遠端before/after驗收。
- 2026-07-11：本機驗證已通過前端TypeScript、lint、production build與目標Vitest；部分backend API/observability pytest受Windows環境無法建立測試暫存目錄或開啟evaluation DB（WinError 5）阻塞，非斷言失敗。
