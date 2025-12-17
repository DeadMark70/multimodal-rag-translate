# Backend RAG API Enhancement Specification

本文件描述 Englishtran 後端 RAG API 需要進行的修改，以支援上下文感知對話功能。

---

## 概覽

### 目標

1. `/ask` 端點支援對話歷史 (`history`) 參數，實現多輪上下文感知對話
2. 暴露進階檢索選項 (`enable_hyde`, `enable_multi_query`) 給前端

### 影響範圍

| 檔案                          | 變更類型   | 描述                                    |
| ----------------------------- | ---------- | --------------------------------------- |
| `data_base/router.py`         | MODIFY     | 新增 `/ask` 的 POST 版本或擴展 GET 參數 |
| `data_base/RAG_QA_service.py` | MODIFY     | 處理 history 參數注入 prompt            |
| `data_base/schemas.py` (如有) | NEW/MODIFY | 新增 request/response schema            |

---

## API 變更詳情

### Option A: 擴展現有 GET `/ask` (推薦用於簡單歷史)

```python
# router.py
@router.get("/ask")
async def ask_question(
    question: str,
    doc_ids: Optional[str] = None,
    history: Optional[str] = None,  # JSON-encoded list
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    enable_reranking: bool = True,
):
    history_parsed = json.loads(history) if history else []
    # ...
```

**前端呼叫範例**:

```
GET /ask?question=xxx&history=[{"role":"user","content":"xxx"},{"role":"assistant","content":"xxx"}]
```

> ⚠️ **注意**: URL 長度限制，歷史過長時可能出問題

---

### Option B: 新增 POST `/ask` (推薦用於完整歷史)

```python
# schemas.py
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str

class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    history: Optional[List[ChatMessage]] = None
    enable_hyde: bool = False
    enable_multi_query: bool = False
    enable_reranking: bool = True
    top_k: int = 6

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
```

```python
# router.py
from .schemas import AskRequest, AskResponse

@router.post("/ask", response_model=AskResponse)
async def ask_question_with_history(request: AskRequest):
    """
    上下文感知問答端點。

    支援傳入對話歷史，讓 LLM 理解對話脈絡。
    """
    answer_response = await rag_qa_service.ask_with_context(
        question=request.question,
        history=request.history,
        doc_ids=request.doc_ids,
        enable_hyde=request.enable_hyde,
        enable_multi_query=request.enable_multi_query,
        enable_reranking=request.enable_reranking,
        top_k=request.top_k,
    )
    return answer_response
```

---

## RAG Service 變更

### RAG_QA_service.py

新增處理歷史的邏輯：

```python
async def ask_with_context(
    self,
    question: str,
    history: Optional[List[ChatMessage]] = None,
    doc_ids: Optional[List[str]] = None,
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    enable_reranking: bool = True,
    top_k: int = 6,
) -> AnswerResponse:
    """
    帶上下文的問答。

    Args:
        question: 當前問題
        history: 對話歷史 (user/assistant 訊息列表)
        ...其他參數同原本的 ask_question
    """
    # 1. 執行檢索 (與原本相同)
    retrieved_chunks = await self._retrieve(
        question=question,
        doc_ids=doc_ids,
        enable_hyde=enable_hyde,
        enable_multi_query=enable_multi_query,
        enable_reranking=enable_reranking,
        top_k=top_k,
    )

    # 2. 構建帶歷史的 prompt
    messages = self._build_messages_with_history(
        question=question,
        history=history,
        context=retrieved_chunks,
    )

    # 3. 呼叫 LLM
    answer = await self._llm_generate(messages)

    return AnswerResponse(
        question=question,
        answer=answer,
        sources=[chunk.doc_id for chunk in retrieved_chunks],
    )

def _build_messages_with_history(
    self,
    question: str,
    history: Optional[List[ChatMessage]],
    context: List[RetrievedChunk],
) -> List[Dict[str, str]]:
    """
    構建包含對話歷史的 LLM messages。
    """
    system_prompt = f"""你是一個知識庫助手。根據以下檢索到的資料回答問題。

檢索到的資料:
{self._format_context(context)}

請根據上述資料和對話歷史回答用戶的問題。如果資料中沒有相關資訊，請誠實說明。
"""

    messages = [{"role": "system", "content": system_prompt}]

    # 加入對話歷史 (限制最近 10 條，避免 token 過長)
    if history:
        for msg in history[-10:]:
            messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

    # 加入當前問題
    messages.append({"role": "user", "content": question})

    return messages
```

---

## Request/Response 範例

### Request (POST /ask)

```json
{
  "question": "這份文件的結論是什麼？",
  "history": [
    {
      "role": "user",
      "content": "這份研究報告的主題是什麼？"
    },
    {
      "role": "assistant",
      "content": "這份研究報告探討的是機器學習在醫療診斷中的應用..."
    }
  ],
  "doc_ids": ["doc-uuid-123"],
  "enable_hyde": false,
  "enable_multi_query": false,
  "enable_reranking": true
}
```

### Response

```json
{
  "question": "這份文件的結論是什麼？",
  "answer": "根據這份關於機器學習醫療應用的研究報告，主要結論包括：\n1. AI 輔助診斷準確率達到 95%\n2. 可有效減少醫師工作負擔\n3. 建議在實際部署前需要更多臨床驗證...",
  "sources": ["doc-uuid-123"]
}
```

---

## 前端整合

### Flutter 呼叫方式

```dart
// rag_repository.dart
Future<AnswerResponse> askWithHistory(
  String question, {
  List<ChatMessageModel>? history,
  List<String>? docIds,
  bool enableHyde = false,
  bool enableMultiQuery = false,
}) async {
  final response = await _apiClient.post(
    RagEndpoints.ask,  // POST /ask
    body: {
      'question': question,
      'history': history?.map((m) => {
        'role': m.role.name,
        'content': m.content,
      }).toList(),
      'doc_ids': docIds,
      'enable_hyde': enableHyde,
      'enable_multi_query': enableMultiQuery,
    },
  );
  return AnswerResponse.fromJson(response);
}
```

---

## 安全性考量

1. **歷史長度限制**: 限制 `history` 最多 20 條訊息，防止 token 超出
2. **內容驗證**: 驗證 `role` 只能是 `user`/`assistant`/`system`
3. **Rate Limiting**: 考慮對 POST 請求增加速率限制

```python
# 驗證範例
MAX_HISTORY_LENGTH = 20

@router.post("/ask")
async def ask_question_with_history(request: AskRequest):
    if request.history and len(request.history) > MAX_HISTORY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"History too long. Maximum {MAX_HISTORY_LENGTH} messages allowed."
        )
    # ...
```

---

## 測試案例

### 單元測試

```python
@pytest.mark.asyncio
async def test_ask_with_history():
    request = AskRequest(
        question="這份文件的結論是什麼？",
        history=[
            ChatMessage(role=MessageRole.user, content="主題是什麼？"),
            ChatMessage(role=MessageRole.assistant, content="主題是機器學習..."),
        ],
    )
    response = await ask_question_with_history(request)
    assert response.question == request.question
    assert len(response.answer) > 0

@pytest.mark.asyncio
async def test_history_length_limit():
    long_history = [
        ChatMessage(role=MessageRole.user, content=f"msg {i}")
        for i in range(25)
    ]
    request = AskRequest(question="test", history=long_history)
    with pytest.raises(HTTPException) as exc:
        await ask_question_with_history(request)
    assert exc.value.status_code == 400
```

---

## 遷移計畫

### 階段 1: 新增 POST 端點 (向後相容)

- 保留原有 GET `/ask` 端點
- 新增 POST `/ask` 支援 history

### 階段 2: 前端遷移

- 新對話使用 POST 端點
- 舊邏輯繼續使用 GET (兼容)

### 階段 3: 廢棄 GET (可選)

- 如果確認不再需要，可廢棄 GET `/ask`

---

## 確認清單

- [ ] 確認使用 Option A (GET) 或 Option B (POST)
- [ ] 確認歷史長度限制 (建議 10-20 條)
- [ ] 確認是否需要支援 `system` role
- [ ] 後端實作完成
- [ ] 前端整合測試
