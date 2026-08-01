from langchain_core.documents import Document

from data_base.agentic_v9.requirement_shadow import build_requirement_shadow


def test_markdown_table_is_structured_text_not_visual_required() -> None:
    analysis = build_requirement_shadow(
        question=("根據 Table 3，noise 0.4 時 U-KAN 與 Ours 的 Dice 分別是多少？"),
        documents=[
            Document(
                page_content=(
                    "| Noise | U-KAN | Ours |\n"
                    "| --- | ---: | ---: |\n"
                    "| 0.4 | 0.4064 | 0.9079 |"
                ),
                metadata={
                    "source": "text",
                    "doc_id": "doc-table",
                    "chunk_id": "chunk-table",
                },
            )
        ],
    )

    requirement = analysis.requirements[0]
    assert requirement.information_need == "markdown_table"
    assert requirement.visual_decision == "not_requested"
    assert requirement.coverage_status == "candidate"
    assert "markdown_table" in requirement.available_representations
    assert analysis.behavior_influence is False


def test_qualitative_figure_question_can_use_image_summary_without_hard_gate() -> None:
    analysis = build_requirement_shadow(
        question="根據 Figure 2 的折線圖，哪條曲線呈現較快的上升趨勢？",
        documents=[
            Document(
                page_content="Figure 2 摘要：藍色曲線在前十個 epoch 上升較快。",
                metadata={
                    "source": "image",
                    "type": "figure",
                    "doc_id": "doc-figure",
                    "chunk_id": "chunk-figure",
                    "image_path": "uploads/doc-figure/figure-2.png",
                    "asset_id": "asset-figure-2",
                },
            )
        ],
    )

    requirement = analysis.requirements[0]
    assert requirement.information_need == "visual_pattern"
    assert requirement.visual_precision == "qualitative"
    assert requirement.visual_decision == "optional"
    assert requirement.coverage_status == "candidate"
    assert requirement.available_representations == [
        "image_summary",
        "visual_asset",
    ]


def test_exact_graph_value_requires_visual_asset_even_when_summary_exists() -> None:
    analysis = build_requirement_shadow(
        question="讀取 Figure 4 折線圖，回報 epoch 37 時兩條曲線的精確數值。",
        documents=[
            Document(
                page_content="Figure 4 摘要：兩條曲線隨 epoch 增加而上升。",
                metadata={
                    "source": "image",
                    "type": "figure",
                    "doc_id": "doc-figure",
                    "chunk_id": "chunk-figure-4",
                    "image_path": "uploads/doc-figure/figure-4.png",
                },
            )
        ],
    )

    requirement = analysis.requirements[0]
    assert requirement.information_need == "visual_pattern"
    assert requirement.visual_precision == "exact"
    assert requirement.visual_decision == "required"
    assert requirement.visual_reason == "exact_visual_information_requested"


def test_numbered_atomic_requirements_are_separate_and_subject_match_is_not_support() -> (
    None
):
    analysis = build_requirement_shadow(
        question=(
            "請回答：1. ODES 的 Regional Impurity 公式為何？ "
            "2. A^c(x,y) 如何定義？ "
            "3. Theorem 1 中 m 的範圍為何？"
        ),
        documents=[
            Document(
                page_content="ODES introduces Regional Impurity for medical images.",
                metadata={
                    "source": "text",
                    "doc_id": "doc-odes",
                    "chunk_id": "chunk-odes",
                },
            )
        ],
    )

    assert [item.requirement_id for item in analysis.requirements] == [
        "R1",
        "R2",
        "R3",
    ]
    assert [item.answer_kind for item in analysis.requirements] == [
        "equation",
        "definition",
        "number",
    ]
    assert all(item.coverage_status != "supported" for item in analysis.requirements)
    assert analysis.summary.requirement_count == 3
    assert analysis.summary.supported_count == 0


def test_repeated_retrieval_tasks_do_not_duplicate_candidate_evidence_refs() -> None:
    repeated = Document(
        page_content="The reported score is 0.91.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )

    analysis = build_requirement_shadow(
        question="What is the reported score?",
        documents=[repeated, repeated],
    )

    assert analysis.requirements[0].candidate_evidence_refs == ["doc-1:chunk-1"]


def test_v2_never_promotes_candidate_to_supported() -> None:
    analysis = build_requirement_shadow(
        question="請回報模型分數。",
        documents=[Document(page_content="模型分數為 0.91。", metadata={})],
    )

    assert analysis.schema_version == "shadow_requirements_v2"
    assert analysis.support_assessment == "candidate_only"
    assert analysis.summary.supported_count == 0
    assert all(item.coverage_status != "supported" for item in analysis.requirements)


def test_mixed_figure_and_markdown_table_needs_are_preserved() -> None:
    analysis = build_requirement_shadow(
        question="根據 Figure 1 辨識策略 (b)，再從 Table 1 回報 mIoU。",
        documents=[
            Document(
                page_content=(
                    "Figure 1 summary: strategy (b) fine-tunes all components.\n"
                    "| Strategy | mIoU |\n|---|---:|\n| b | 0.877 |"
                ),
                metadata={"source": "image", "type": "figure"},
            )
        ],
    )

    needs = {need for item in analysis.requirements for need in item.information_needs}
    assert {"visual_pattern", "markdown_table"} <= needs
    assert analysis.summary.supported_count == 0


def test_fallback_evidence_identity_is_content_stable_and_deduplicated() -> None:
    documents = [
        Document(page_content="Model-A score is 0.91.", metadata={}),
        Document(page_content="Model-B score is 0.82.", metadata={}),
        Document(page_content="Model-A score is 0.91.", metadata={}),
    ]

    first = build_requirement_shadow(question="請回報模型分數。", documents=documents)
    second = build_requirement_shadow(
        question="請回報模型分數。", documents=list(reversed(documents))
    )

    first_refs = first.requirements[0].candidate_evidence_refs
    second_refs = second.requirements[0].candidate_evidence_refs
    assert first_refs == second_refs
    assert all(ref.startswith("content:") for ref in first_refs)
    assert len(first_refs) == 2


def test_shadow_v2_offline_question_smoke_is_bounded() -> None:
    import json
    from pathlib import Path

    payload = json.loads(
        Path("evaluation/golden/agentic_v9_questions_v2.json").read_text(
            encoding="utf-8"
        )
    )
    questions = payload if isinstance(payload, list) else payload["questions"]

    for item in questions:
        question = item["question"] if isinstance(item, dict) else str(item)
        analysis = build_requirement_shadow(question=question, documents=[])
        assert analysis.schema_version == "shadow_requirements_v2"
        assert 1 <= len(analysis.requirements) <= 8
        assert len(analysis.response_constraints) <= 8
        assert analysis.summary.supported_count == 0
