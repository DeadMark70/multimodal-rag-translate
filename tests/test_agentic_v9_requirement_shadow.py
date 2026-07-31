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
