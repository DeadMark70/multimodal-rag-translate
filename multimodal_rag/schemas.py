from enum import Enum
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

class VisualElementType(str, Enum):
    TABLE = 'table'
    FIGURE = 'figure'
    FORMULA = 'formula'

class VisualElement(BaseModel):
    id: UUID
    type: VisualElementType
    page_number: int
    image_path: str = Field(description="Path to the cropped image")
    bbox: List[int] = Field(description="[x1, y1, x2, y2] coordinates")
    original_text: Optional[str] = None
    summary: Optional[str] = Field(None, description="Gemini generated summary")
    # 新增：上下文感知欄位
    context_text: Optional[str] = Field(
        None, description="周圍文字上下文（Caption + 前後文）"
    )
    figure_reference: Optional[str] = Field(
        None, description="圖片引用標識，如 'Figure 1', '圖一'"
    )

class TextChunk(BaseModel):
    page_number: int
    content: str = Field(description="Markdown formatted content")
    chunk_id: str

class ExtractedDocument(BaseModel):
    doc_id: UUID
    user_id: str
    text_chunks: List[TextChunk]
    visual_elements: List[VisualElement]
    processed_at: datetime = Field(default_factory=datetime.utcnow)
