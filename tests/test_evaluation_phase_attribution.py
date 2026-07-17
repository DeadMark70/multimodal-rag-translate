from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from core.llm_usage_context import current_llm_accounting_phase


PHASE_CASES = [
    ("data_base/query_transformer.py", "query_expansion"),
    ("data_base/query_transformer.py", "retrieval_rewrite"),
    ("graph_rag/local_search.py", "graph_reasoning"),
    ("graph_rag/global_search.py", "graph_reasoning"),
    ("graph_rag/generic_mode.py", "graph_reasoning"),
    ("evaluation/agentic_evaluation_service.py", "agent_planning"),
    ("data_base/RAG_QA_service.py", "answer_generation"),
    ("multimodal_rag/image_summarizer.py", "visual_verification"),
    ("agents/synthesizer.py", "agent_synthesis"),
]


@pytest.mark.parametrize(("relative_path", "expected_phase"), PHASE_CASES)
def test_evaluation_call_sites_declare_controlled_phase(
    relative_path: str, expected_phase: str
) -> None:
    source = Path(relative_path).read_text(encoding="utf-8")

    if relative_path == "data_base/query_transformer.py":
        assert f'"{expected_phase}"' in source
        assert "llm_accounting_phase(phase)" in source
        return

    assert f'llm_accounting_phase("{expected_phase}")' in source


class _PhaseRecordingLlm:
    def __init__(self, content: str) -> None:
        self.content = content
        self.phases: list[str] = []

    async def ainvoke(self, _messages: object) -> SimpleNamespace:
        self.phases.append(current_llm_accounting_phase())
        return SimpleNamespace(content=self.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enable_hyde", "enable_multi_query", "query_response"),
    [
        (True, False, "hypothetical document"),
        (False, True, "1. alternate retrieval query"),
    ],
)
async def test_initial_query_expansion_invocations_use_expansion_phase(
    enable_hyde: bool,
    enable_multi_query: bool,
    query_response: str,
) -> None:
    from data_base.RAG_QA_service import rag_answer_question

    query_llm = _PhaseRecordingLlm(query_response)
    answer_llm = _PhaseRecordingLlm("answer")
    retriever = Mock()
    retriever.invoke.return_value = [
        Document(page_content="retrieved context", metadata={"doc_id": "doc-1"})
    ]

    with (
        patch("data_base.query_transformer.get_llm", return_value=query_llm),
        patch("data_base.RAG_QA_service.get_llm", return_value=answer_llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch(
            "data_base.RAG_QA_service.get_user_retriever",
            new=AsyncMock(return_value=retriever),
        ),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-1": "doc-1.pdf"}),
        ),
    ):
        await rag_answer_question(
            question="question",
            user_id="user-1",
            enable_hyde=enable_hyde,
            enable_multi_query=enable_multi_query,
        )

    assert query_llm.phases == ["query_expansion"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rewrite_mode", "query_response"),
    [
        ("hyde", "corrected query"),
        ("multi_query", "1. corrected query"),
    ],
)
async def test_crag_rewrite_invocations_use_retrieval_rewrite_phase(
    rewrite_mode: str,
    query_response: str,
) -> None:
    from data_base.RAG_QA_service import _build_crag_queries

    query_llm = _PhaseRecordingLlm(query_response)

    with patch("data_base.query_transformer.get_llm", return_value=query_llm):
        await _build_crag_queries("question", rewrite_mode)  # type: ignore[arg-type]

    assert query_llm.phases == ["retrieval_rewrite"]
