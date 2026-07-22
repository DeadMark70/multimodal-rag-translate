"""Contracts for the generic retrieval-to-generation boundary."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from langchain_core.documents import Document

from data_base.RAG_QA_service import RAGResult
from data_base.rag_pipeline_schemas import GeneratedRagAnswer, RagRetrievalResult


def test_retrieval_result_keeps_evidence_separate_from_generation_data() -> None:
    document = Document(
        page_content="Evidence from the source.",
        metadata={"doc_id": "doc-1", "retrieval_rank": 1},
    )

    result = RagRetrievalResult(
        documents=[document],
        source_doc_ids=["doc-1"],
        context="Context assembled from retrieved evidence.",
        metadata={"query_origin": "original", "expanded_queries": ["question"]},
        images=[{"image_id": "image-1"}],
    )

    assert result.documents == [document]
    assert result.source_doc_ids == ["doc-1"]
    assert result.context == "Context assembled from retrieved evidence."
    assert result.metadata["query_origin"] == "original"
    assert result.images == [{"image_id": "image-1"}]
    assert not hasattr(result, "answer")


def test_generated_answer_carries_only_generation_projection() -> None:
    result = GeneratedRagAnswer(
        answer="Supported answer.",
        usage={"total_tokens": 12},
        thought_process="Synthesized from supplied context.",
        tool_calls=[{"name": "visual_verification"}],
        agent_trace={"mode": "advanced"},
        visual_verification_meta={"iterations": 1},
    )

    assert result.answer == "Supported answer."
    assert result.usage == {"total_tokens": 12}
    assert result.thought_process == "Synthesized from supplied context."
    assert result.tool_calls == [{"name": "visual_verification"}]
    assert result.agent_trace == {"mode": "advanced"}
    assert result.visual_verification_meta == {"iterations": 1}
    assert not hasattr(result, "documents")


def test_legacy_rag_result_constructor_and_fields_remain_compatible() -> None:
    document = Document(page_content="Legacy evidence", metadata={"doc_id": "doc-1"})

    result = RAGResult("Legacy answer", ["doc-1"], [document])

    assert result.answer == "Legacy answer"
    assert result.source_doc_ids == ["doc-1"]
    assert result.documents == [document]
    assert result.usage == {}
    assert result.thought_process is None
    assert result.tool_calls == []
    assert result.agent_trace is None
    assert result.visual_verification_meta is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plain_mode", "graph_context", "expected_prompt"),
    [
        (True, "", "plain_rag_answer"),
        (False, "", "advanced_rag_answer"),
        (False, "Graph evidence", "advanced_rag_answer"),
    ],
)
async def test_legacy_generation_preserves_naive_advanced_and_graph_prompt_paths(
    monkeypatch: pytest.MonkeyPatch,
    plain_mode: bool,
    graph_context: str,
    expected_prompt: str,
) -> None:
    from data_base import rag_generation

    prompt_calls: list[tuple[str, dict[str, object]]] = []

    def fake_format_prompt(name: str, **kwargs: object) -> str:
        prompt_calls.append((name, kwargs))
        return f"{name}:{kwargs['context_text']}:{kwargs['graph_section']}"

    monkeypatch.setattr(rag_generation, "format_prompt", fake_format_prompt)
    monkeypatch.setattr(
        rag_generation,
        "fetch_document_filenames",
        lambda _ids: _resolved_async({"doc-1": "source.pdf"}),
    )

    llm = SimpleNamespace(
        ainvoke=lambda _messages: _resolved_async(
            SimpleNamespace(content="legacy answer", usage_metadata={"total_tokens": 7})
        )
    )
    result = await rag_generation.generate_legacy_answer_from_evidence(
        question="What does the evidence say?",
        user_id="user-1",
        documents=[Document(page_content="supported context", metadata={"doc_id": "doc-1"})],
        llm=llm,
        graph_context=graph_context,
        history_section="",
        intent_constraints="",
        plain_mode=plain_mode,
    )

    assert result.answer == "legacy answer"
    assert result.usage["total_tokens"] == 7
    assert prompt_calls[0][0] == expected_prompt
    assert prompt_calls[0][1]["graph_section"] == (
        f"\n{graph_context}\n" if graph_context else ""
    )


@pytest.mark.asyncio
async def test_legacy_generation_returns_legacy_error_projection_for_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_base import rag_generation

    monkeypatch.setattr(rag_generation, "format_prompt", lambda _name, **_kwargs: "prompt")
    llm = SimpleNamespace(ainvoke=lambda _messages: _raise_async(RuntimeError("offline")))

    result = await rag_generation.generate_legacy_answer_from_evidence(
        question="question",
        user_id="user-1",
        documents=[Document(page_content="context", metadata={"doc_id": "doc-1"})],
        llm=llm,
        graph_context="",
        history_section="",
        intent_constraints="",
        plain_mode=True,
    )

    assert result.answer == "抱歉，處理您的問題時發生錯誤。"
    assert result.usage == {}


@pytest.mark.asyncio
async def test_legacy_generation_keeps_visual_synthesis_inside_legacy_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_base import rag_generation

    monkeypatch.setattr(rag_generation, "format_prompt", lambda name, **_kwargs: name)
    monkeypatch.setattr(rag_generation.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(rag_generation, "_encode_image", lambda _path: "image-data")
    from data_base import visual_tools

    monkeypatch.setattr(
        visual_tools,
        "verify_image_details",
        lambda **_kwargs: _resolved_async({"success": True, "result": "value=42"}),
    )
    responses = iter(
        [
            SimpleNamespace(
                content=(
                    '{"action":"VERIFY_IMAGE","path":"chart.png",'
                    '"question":"What value?"}'
                ),
                usage_metadata={"total_tokens": 3},
            ),
            SimpleNamespace(content="verified answer", usage_metadata={"total_tokens": 3}),
        ]
    )
    llm = SimpleNamespace(ainvoke=lambda _messages: _resolved_async(next(responses)))

    result = await rag_generation.generate_legacy_answer_from_evidence(
        question="What value is in the chart?",
        user_id="user-1",
        documents=[
            Document(
                page_content="chart summary",
                metadata={"doc_id": "doc-1", "source": "image", "image_path": "chart.png"},
            )
        ],
        llm=llm,
        graph_context="",
        history_section="",
        intent_constraints="",
        plain_mode=False,
        enable_visual_verification=True,
        image_encoder=rag_generation._encode_image,
    )

    assert result.tool_calls[0]["action"] == "VERIFY_IMAGE"
    assert result.tool_calls[0]["success"] is True
    assert result.visual_verification_meta == {
        "visual_verification_attempted": True,
        "visual_tool_call_count": 1,
        "visual_force_fallback_used": False,
    }
    assert "data_base.visual_tools" not in Path("data_base/agentic_v9/model_paths.py").read_text(
        encoding="utf-8"
    )


def test_legacy_wrapper_delegates_generation_without_exposing_visual_synthesis() -> None:
    wrapper_source = Path("data_base/RAG_QA_service.py").read_text(encoding="utf-8")
    generation_source = Path("data_base/rag_generation.py").read_text(encoding="utf-8")

    assert "generate_legacy_answer_from_evidence(" in wrapper_source
    assert "_execute_visual_verification_loop" not in wrapper_source
    assert "_deprecated_visual_verification_loop" not in wrapper_source
    assert "The following legacy body is retained temporarily" not in wrapper_source
    assert "_execute_legacy_visual_verification_loop" in generation_source


@pytest.mark.asyncio
async def test_legacy_wrapper_preserves_empty_retrieval_projection() -> None:
    from data_base.RAG_QA_service import rag_answer_question

    with (
        patch("data_base.RAG_QA_service.get_llm", return_value=SimpleNamespace()),
        patch(
            "data_base.RAG_QA_service.get_user_retriever",
            new=AsyncMock(return_value=None),
        ),
    ):
        result = await rag_answer_question("question", "user-1", return_docs=True)

    assert result.answer == "抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。"
    assert result.source_doc_ids == []
    assert result.documents == []


async def _resolved_async(value: object) -> object:
    return value


async def _raise_async(error: Exception) -> object:
    raise error
