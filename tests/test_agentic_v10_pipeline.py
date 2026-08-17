"""Unit and integration tests for Agentic RAG v10 pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from data_base.agentic_v10.subquery_decomposer import (
    SubQueryDecomposer,
    SubQueryDecompositionResponse,
    SubQueryItem,
    _fallback_subqueries,
)
from data_base.agentic_v10.subquery_pipeline_service import (
    AgenticV10PipelineService,
    _extract_doc_title,
    _doc_hash,
)
from evaluation.agentic_campaign_adapter import (
    effective_agentic_execution_version,
    campaign_execution_identity,
)


def test_fallback_subqueries_multi_entity():
    question = "Compare SAMed vs MedSAM vs SAM-Med2D on abdominal CT segmentation."
    items = _fallback_subqueries(question)
    assert len(items) >= 2
    assert any("SAMed" in sq.query for sq in items)
    assert any("MedSAM" in sq.query for sq in items)


@pytest.mark.asyncio
async def test_subquery_decomposer_structured_output():
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    expected_resp = SubQueryDecompositionResponse(
        sub_queries=[
            SubQueryItem(
                id="SQ1",
                query="SAMed architecture decoder semantic mask",
                focus="SAMed 架構與解碼器機制",
                target_entity="SAMed",
            ),
            SubQueryItem(
                id="SQ2",
                query="MedSAM fine-tuning strategy prompt encoder",
                focus="MedSAM 微調策略與提示編碼器",
                target_entity="MedSAM",
            ),
        ]
    )
    mock_structured_llm.ainvoke = AsyncMock(return_value=expected_resp)

    decomposer = SubQueryDecomposer(llm_client=mock_llm)
    res = await decomposer.decompose("SAMed vs MedSAM")
    assert len(res) == 2
    assert res[0].id == "SQ1"
    assert "SAMed" in res[0].query


@pytest.mark.asyncio
async def test_agentic_v10_pipeline_execution():
    # Setup mock decomposer
    mock_decomposer = MagicMock()
    mock_decomposer.decompose = AsyncMock(
        return_value=[
            SubQueryItem(
                id="SQ1",
                query="nnU-Net Revisited 3 core recipes",
                focus="nnU-Net 核心配方",
                target_entity="nnU-Net",
            ),
            SubQueryItem(
                id="SQ2",
                query="U-Mamba Mamba layer ablation",
                focus="U-Mamba 消融實驗",
                target_entity="U-Mamba",
            ),
        ]
    )

    # Setup mock reranker
    mock_reranker = MagicMock()
    doc1 = Document(
        page_content="nnU-Net uses deep supervision, data augmentation, and residual decoders.",
        metadata={"title": "nnUNet_paper.pdf", "doc_id": "doc-001", "page": 3},
    )
    doc2 = Document(
        page_content="U-Mamba demonstrates superior linear scaling in 3D medical segmentation.",
        metadata={"title": "UMamba_paper.pdf", "doc_id": "doc-002", "page": 5},
    )
    mock_reranker.rerank.side_effect = [
        [(doc1, 0.95)],
        [(doc2, 0.88)],
    ]

    service = AgenticV10PipelineService(
        decomposer=mock_decomposer,
        reranker=mock_reranker,
    )

    with patch(
        "data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async",
        new_callable=AsyncMock,
    ) as mock_get_retriever, patch(
        "data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents",
        new_callable=AsyncMock,
    ) as mock_retrieve, patch(
        "data_base.agentic_v10.subquery_pipeline_service.get_llm"
    ) as mock_get_llm:

        mock_get_retriever.return_value = MagicMock()
        ret_res1 = MagicMock()
        ret_res1.documents = [doc1]
        ret_res2 = MagicMock()
        ret_res2.documents = [doc2]
        mock_retrieve.side_effect = [ret_res1, ret_res2]

        mock_synth_llm = MagicMock()
        mock_synth_resp = MagicMock()
        mock_synth_resp.content = (
            "nnU-Net 主要採用殘差解碼器與深度監督 [來源: nnUNet_paper.pdf | 3]，"
            "而 U-Mamba 則展示了線性擴展特性 [來源: UMamba_paper.pdf | 5]。"
        )
        mock_synth_llm.ainvoke = AsyncMock(return_value=mock_synth_resp)
        mock_get_llm.return_value = mock_synth_llm

        result = await service.execute(
            question="Compare nnU-Net vs U-Mamba",
            user_id="test_user",
        )

        assert "nnU-Net" in result.answer
        assert len(result.documents) == 2
        assert "doc-001" in result.source_doc_ids
        assert "doc-002" in result.source_doc_ids
        assert result.agent_trace is not None
        assert result.agent_trace["agentic_execution_version"] == "v10"
        assert result.agent_trace["response_status"] == "complete"
        assert len(result.agent_trace["agentic_v10"]["sub_queries"]) == 2


def test_campaign_adapter_v10_defaults():
    assert effective_agentic_execution_version("agentic") == "v10"
    assert effective_agentic_execution_version("agentic-v10") == "v10"
    assert effective_agentic_execution_version("v10") == "v10"
    assert effective_agentic_execution_version("agentic-v9") == "v9"
    assert effective_agentic_execution_version("v9") == "v9"
    assert effective_agentic_execution_version("v8") == "v8"

    identity, core_mode, ver = campaign_execution_identity("agentic")
    assert core_mode == "agentic"
    assert ver == "v10"
