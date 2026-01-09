"""
Pydantic Schemas for RAG API

Provides request/response models for the RAG question answering endpoints.
"""

# Standard library
from datetime import datetime
from enum import Enum
from typing import List, Optional

# Third-party
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Valid roles for chat messages in conversation history."""
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """
    A single message in conversation history.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The message content.
    """
    role: MessageRole
    content: str


# --- Document List Schemas ---

class DocumentItem(BaseModel):
    """
    Document metadata for list endpoint.

    Attributes:
        id: Document UUID.
        filename: Original filename.
        created_at: Upload timestamp.
        status: Processing status.
        processing_step: Current processing step.
    """
    id: str
    filename: str
    created_at: Optional[datetime] = None
    status: Optional[str] = None
    processing_step: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for GET /pdfmd/list endpoint."""
    documents: List[DocumentItem]
    total: int


# --- Enhanced Source/Citation Schemas ---

class SourceDetail(BaseModel):
    """
    Detailed source citation for RAG responses.

    Attributes:
        doc_id: Document UUID.
        filename: Display-friendly filename.
        page: Page number (if available).
        snippet: Relevant text excerpt (first 200 chars).
        score: Relevance score from reranker.
    """
    doc_id: str
    filename: Optional[str] = None
    page: Optional[int] = None
    snippet: str = Field(..., description="引用段落原文 (前 200 字)")
    score: float = Field(..., ge=0.0, le=1.0, description="相關性分數")


# --- Evaluation Metrics Schemas ---

class FaithfulnessLevel(str, Enum):
    """Faithfulness evaluation results."""
    grounded = "grounded"
    hallucinated = "hallucinated"
    uncertain = "uncertain"
    evaluation_failed = "evaluation_failed"


class EvaluationMetrics(BaseModel):
    """
    Responsible AI metrics for answer quality.
    
    Phase 4 Academic Evaluation (1-10 scale):
    - accuracy: Data precision, citation correctness (50% weight)
    - completeness: Coverage of all sub-aspects (30% weight)
    - clarity: Logical structure and expression (20% weight)
    
    Attributes:
        faithfulness: Legacy field - maps from accuracy score.
        confidence_score: Overall confidence (0.0-1.0).
        evaluation_reason: Detailed evaluation reasoning.
        accuracy: D1 - Data precision score (1-10).
        completeness: D2 - Coverage completeness (1-10).
        clarity: D3 - Logical expression (1-10).
        weighted_score: Weighted total (0.5*acc + 0.3*cmp + 0.2*clr).
        suggestion: Improvement suggestion for retry.
        is_passing: True if accuracy >= 7.
    """
    faithfulness: FaithfulnessLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evaluation_reason: Optional[str] = Field(
        default=None,
        description="評估結果說明"
    )
    # Phase 4: New 1-10 scale fields
    accuracy: Optional[float] = Field(
        default=None,
        ge=1.0, le=10.0,
        description="D1: 數據精確度 (1-10, 權重 50%)"
    )
    completeness: Optional[float] = Field(
        default=None,
        ge=1.0, le=10.0,
        description="D2: 完整覆蓋率 (1-10, 權重 30%)"
    )
    clarity: Optional[float] = Field(
        default=None,
        ge=1.0, le=10.0,
        description="D3: 邏輯表達 (1-10, 權重 20%)"
    )
    weighted_score: Optional[float] = Field(
        default=None,
        ge=1.0, le=10.0,
        description="加權總分 (0.5*acc + 0.3*cmp + 0.2*clr)"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="改進建議 (用於 Smart Retry)"
    )
    is_passing: Optional[bool] = Field(
        default=None,
        description="是否通過門檻 (accuracy >= 7)"
    )


# --- Request Schemas ---

class AskRequest(BaseModel):
    """
    Request body for POST /ask endpoint.

    Attributes:
        question: The user's question.
        doc_ids: Optional list of document IDs to filter retrieval.
        history: Optional conversation history for context-aware responses.
        enable_hyde: Enable HyDE (Hypothetical Document Embeddings) retrieval.
        enable_multi_query: Enable multi-query fusion retrieval.
        enable_reranking: Enable Cross-Encoder reranking (recommended).
        enable_evaluation: Enable Self-RAG evaluation (adds latency).
        enable_graph_rag: Enable knowledge graph enhanced retrieval.
        graph_search_mode: Graph search mode (local/global/hybrid/auto).
        enable_graph_planning: Enable graph-based planning for Deep Research.
    """
    question: str = Field(
        ...,
        description="使用者問題",
        min_length=1,
        max_length=2000,
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="限定查詢的文件 ID 列表（留空則查詢全部文件）",
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="對話歷史（最多 10 條，用於上下文感知對話）",
    )
    enable_hyde: bool = Field(
        default=False,
        description="啟用 HyDE 假設性文件增強檢索",
    )
    enable_multi_query: bool = Field(
        default=False,
        description="啟用多重查詢融合檢索",
    )
    enable_reranking: bool = Field(
        default=True,
        description="啟用 Cross-Encoder 重排序（建議開啟）",
    )
    enable_evaluation: bool = Field(
        default=False,
        description="啟用 Self-RAG 評估模式（會增加延遲）",
    )
    # GraphRAG parameters
    enable_graph_rag: bool = Field(
        default=False,
        description="啟用知識圖譜增強檢索",
    )
    graph_search_mode: str = Field(
        default="auto",
        description="圖譜搜尋模式: local (實體擴展), global (社群摘要), hybrid (兩者), auto (自動判斷)",
    )
    enable_graph_planning: bool = Field(
        default=False,
        description="啟用圖譜輔助規劃 (Deep Research 時使用)",
    )



# --- Response Schemas ---

class AskResponse(BaseModel):
    """
    Response model for /ask endpoint.

    Attributes:
        question: Echo of the original question.
        answer: The generated answer.
        sources: List of source document IDs used in the response.
    """
    question: str
    answer: str
    sources: List[str] = []


class EnhancedAskResponse(BaseModel):
    """
    Enhanced response with detailed sources and metrics.

    Used when enable_evaluation=True or for detailed citation display.

    Attributes:
        question: Echo of the original question.
        answer: The generated answer.
        sources: Detailed source citations with snippets and scores.
        metrics: Optional evaluation metrics (if enable_evaluation=True).
    """
    question: str
    answer: str
    sources: List[SourceDetail] = []
    metrics: Optional[EvaluationMetrics] = None

