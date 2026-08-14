import json
from pathlib import Path

from data_base.agentic_v9.requirement_decomposition import (
    decompose_question,
    split_top_level_blocks,
)


def test_numbered_blocks_do_not_split_parenthetical_ids_or_decimals() -> None:
    question = (
        "請回答以下三個子問題：1. GEPAR3D 中 tooth 1) 誤判為 tooth 32) "
        "的懲罰與原因為何？ 2. ODES 的 P(x,y) 公式與 |A^c(x,y)| 定義為何？ "
        "3. noise 0.4 時兩個 Dice 與 Theorem 1 的 m 範圍為何？"
    )

    blocks = split_top_level_blocks(question)

    assert len(blocks) == 3
    assert "tooth 1)" in blocks[0].text
    assert "tooth 32)" in blocks[0].text
    assert "0.4" in blocks[2].text
    assert all(block.method == "numbered" for block in blocks)
    assert all(block.confidence == "high" for block in blocks)


def test_unsequenced_numeric_text_falls_back_without_splitting() -> None:
    blocks = split_top_level_blocks("比較 class 1) 與 class 2) 在 v3.1 的結果。")

    assert len(blocks) == 1
    assert blocks[0].method == "fallback"
    assert blocks[0].confidence == "low"


def test_top_level_chinese_numbered_blocks_are_supported() -> None:
    blocks = split_top_level_blocks("一、說明模型。二、比較結果。")

    assert [block.text for block in blocks] == ["說明模型。", "比較結果。"]
    assert all(block.method == "numbered" for block in blocks)


def test_empty_question_has_no_structural_blocks() -> None:
    assert split_top_level_blocks("   ") == ()


def test_conditional_scope_is_a_constraint_not_a_requirement() -> None:
    result = decompose_question(
        "Weak-Mamba-UNet 與 Semi-Mamba-UNet 都有 first claim。"
        "請判斷能否唯一決定誰最先；若不能，必須按 claim scope 分開回答。"
    )

    assert len(result.requirements) == 1
    assert [item.kind for item in result.response_constraints] == ["conditional_scope"]
    assert "claim scope" in result.response_constraints[0].text


def test_classification_labels_are_constraints() -> None:
    result = decompose_question(
        "僅根據 nnFormer 與 U-Mamba，請分別分類："
        "A. 有量化 ensemble 證據；B. 只有公平比較。"
    )

    assert len(result.requirements) == 2
    assert {item.kind for item in result.response_constraints} == {"allowed_labels"}
    assert all(
        "A." not in item.text and "B." not in item.text for item in result.requirements
    )


def test_css_question_yields_four_generic_obligations() -> None:
    result = decompose_question(
        "根據架構描述，請重建 CSS 的特徵融合流程，並說明三個翻轉分支"
        "與 SiamSSM 的運算/累加機制。"
    )

    assert len(result.requirements) == 4
    joined = "\n".join(item.text for item in result.requirements)
    for anchor in ("融合流程", "翻轉分支", "SiamSSM", "累加機制"):
        assert anchor in joined


def test_explicit_other_entities_use_bounded_distribution() -> None:
    result = decompose_question(
        "在 SAMed、MedSAM、SAM-Med2D、SAM-Med3D 中，哪一個符合條件？"
        "請指出唯一符合者與關鍵技術，並簡述另外三者為何不符合。"
    )

    assert len(result.requirements) <= 5
    joined = "\n".join(item.text for item in result.requirements)
    for entity in ("SAMed", "MedSAM", "SAM-Med2D", "SAM-Med3D"):
        assert entity in joined


def test_q15_like_question_yields_four_obligations() -> None:
    result = decompose_question(
        "根據 Figure 1 的策略 (a) 與 (b)，若採用圖 1(b) 的策略，"
        "Table 1 中的 mIoU 為何？請計算它比 (a) 提升多少？"
        "此外，effective batch size 是多少？"
    )

    assert len(result.requirements) == 4
    joined = "\n".join(item.text for item in result.requirements)
    for anchor in ("Figure", "mIoU", "提升", "batch size"):
        assert anchor in joined


def test_q16_like_question_yields_six_obligations_without_numeric_corruption() -> None:
    result = decompose_question(
        "以下三個子問題：1. GEPAR3D 中 tooth 1) 誤判為 tooth 32) 的懲罰值與原因為何？"
        "此外，為何空間更近的 tooth 2 反而不同？ 2. ODES 的 P(x,y) 方程式為何？"
        "並且 |A^c(x,y)| 代表什麼？ 3. noise 0.4 時 U-KAN 與 Ours 的 Dice 為何？"
        "此外，Theorem 1 中 m 的範圍為何？"
    )

    assert len(result.requirements) == 6
    joined = "\n".join(item.text for item in result.requirements)
    assert "tooth 1)" in joined
    assert "0.4" in joined


def test_unseen_bilingual_obligations_generalize_without_templates() -> None:
    result = decompose_question(
        "Compare Model-A and Model-B：請分別回報 latency 與 memory，"
        "並解釋 trade-off；不要宣稱為通用排名。"
    )

    assert len(result.requirements) == 2
    assert len(result.synthesis_obligations) == 1
    assert result.synthesis_obligations[0].kind == "comparison"
    assert result.synthesis_obligations[0].depends_on_requirement_indexes == (0, 1)
    assert any(item.kind == "prohibition" for item in result.response_constraints)
    assert all(
        item.confidence in {"high", "medium", "low"} for item in result.requirements
    )


def test_direct_evidence_vs_derived_reasoning() -> None:
    result = decompose_question(
        "分別找出 Model-A 與 Model-B 的 latency，然後比較哪個較低；"
        "不要宣稱為通用排名。"
    )
    assert [item.entity_ids for item in result.requirements] == [
        ("Model-A",),
        ("Model-B",),
    ]
    assert [
        (item.kind, item.depends_on_requirement_indexes)
        for item in result.synthesis_obligations
    ] == [("comparison", (0, 1))]
    assert [item.kind for item in result.response_constraints] == ["prohibition"]
    assert result.comparison_subjects == ("Model-A", "Model-B")


def test_numbered_chinese_and_english_decomposition() -> None:
    question = (
        "請依次回答：1. Explain the mechanism of Model-X. "
        "2. Explain the mechanism of Model-Y. "
        "3. Provide the quantitative parameter count for both."
    )
    result = decompose_question(question)
    assert len(result.requirements) == 3
    assert all(r.method == "numbered" for r in result.requirements)
    assert all(r.confidence == "high" for r in result.requirements)


def test_causal_selection_and_aggregation_synthesis_obligations() -> None:
    # Selection
    sel_result = decompose_question(
        "在 Model-A、Model-B、Model-C 中，哪一個唯一符合條件？請給出選型裁決。"
    )
    assert len(sel_result.requirements) == 3
    assert len(sel_result.synthesis_obligations) == 1
    assert sel_result.synthesis_obligations[0].kind == "selection"
    assert sel_result.synthesis_obligations[0].depends_on_requirement_indexes == (
        0,
        1,
        2,
    )
    assert sel_result.comparison_subjects == ("Model-A", "Model-B", "Model-C")

    # Aggregation
    agg_result = decompose_question(
        "從 Model-V1、Model-V2 到 Model-V3，分別說明特點，並描述整體演進趨勢。"
    )
    assert len(agg_result.requirements) == 3
    assert len(agg_result.synthesis_obligations) == 1
    assert agg_result.synthesis_obligations[0].kind == "aggregation"
    assert agg_result.synthesis_obligations[0].depends_on_requirement_indexes == (
        0,
        1,
        2,
    )

    # Causal
    causal_result = decompose_question(
        "針對 Model-P 與 Model-Q，分別說明特徵提取機制，並分析為何兩者性能不同。"
    )
    assert len(causal_result.requirements) == 2
    assert len(causal_result.synthesis_obligations) == 1
    assert causal_result.synthesis_obligations[0].kind == "causal"
    assert causal_result.synthesis_obligations[0].depends_on_requirement_indexes == (
        0,
        1,
    )


def test_output_format_and_prohibitions_extracted_as_constraints() -> None:
    question = (
        "請以「首選 / 次選 / 不優先」格式給出選型裁決，"
        "且不得宣稱為嚴格 apples-to-apples 的通用基準排名。"
    )
    result = decompose_question(question)
    kinds = [c.kind for c in result.response_constraints]
    assert "output_format" in kinds
    assert "prohibition" in kinds


def test_complex_unpunctuated_chinese_triggers_semantic_planning() -> None:
    unpunctuated = (
        "在沒有任何標點符號的情況下請分析說明模型甲與模型乙在特徵提取與長距離依賴建模上的異同並且給出詳細裁決"
    )
    result = decompose_question(unpunctuated)
    assert result.requires_semantic_planning
    assert "complex_unpunctuated_chinese" in result.semantic_planning_reasons


def test_truncation_above_eight_triggers_semantic_planning() -> None:
    long_question = "請回答：1. Q1 2. Q2 3. Q3 4. Q4 5. Q5 6. Q6 7. Q7 8. Q8 9. Q9 10. Q10"
    result = decompose_question(long_question, max_requirements=8)
    assert len(result.requirements) == 8
    assert result.truncated_requirement_count == 2
    assert result.requires_semantic_planning
    assert "truncated_requirements" in result.semantic_planning_reasons


def test_compound_collapsed_detection_triggers_semantic_planning() -> None:
    # A vague compound question that collapses to 1 requirement but contains multiple questions/clauses
    compound_q = (
        "這個模型在各種不同資料集上的特徵融合流程為何？"
        "此外翻轉分支與運算累加機制是什麼？"
        "並且為何在特定條件下有效？"
    )
    # If un-numbered and coordinated, if it falls back or cannot be split cleanly:
    result = decompose_question(compound_q)
    # Structural invariant: either decomposed into multiple requirements or flagged as semantic planning
    if len(result.requirements) <= 1:
        assert result.requires_semantic_planning
        assert any(
            r in result.semantic_planning_reasons
            for r in ("compound_collapsed", "low_confidence")
        )


def test_comparison_subjects_unclear_triggers_semantic_planning() -> None:
    # Comparison question with no explicit identifiable named entities
    vague_comp = "比較這兩個未知方法在相同設定下的準確度與延遲差異，並說明哪個更好。"
    result = decompose_question(vague_comp)
    assert result.requires_semantic_planning
    assert "comparison_subjects_unclear" in result.semantic_planning_reasons


def test_regression_q1_to_q32_fixture_structural_invariants() -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "agentic_v9_atomic_questions_v1.json"
    )
    assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("schema_version") == "atomic_questions_v1"
    questions = data.get("questions", [])
    assert len(questions) == 32

    forbidden_keys = {
        "ground_truth",
        "ground_truth_short",
        "key_points",
        "expected_evidence",
        "atomic_facts",
        "test_objective",
        "expected_route",
    }

    known_reasons = {
        "low_confidence",
        "compound_collapsed",
        "comparison_subjects_unclear",
        "dependency_unclear",
        "truncated_requirements",
        "complex_unpunctuated_chinese",
        "evidence_vs_synthesis_ambiguous",
    }

    for item in questions:
        assert "id" in item and item["id"].startswith("Q")
        assert "question" in item and len(item["question"].strip()) > 0
        assert "source_docs" in item and isinstance(item["source_docs"], list)
        assert not (
            set(item.keys()) & forbidden_keys
        ), f"Question {item['id']} contains forbidden golden keys"

        result = decompose_question(item["question"])

        # Invariant: bounded requirements
        assert (
            1 <= len(result.requirements) <= 8
        ), f"{item['id']} has invalid requirements count: {len(result.requirements)}"
        assert len(result.synthesis_obligations) <= 8
        assert len(result.response_constraints) <= 8
        assert result.confidence in {"high", "medium", "low"}

        # Invariant: non-empty requirement texts
        for req in result.requirements:
            assert req.text.strip(), f"{item['id']} has empty requirement text"
            assert req.confidence in {"high", "medium", "low"}
            assert req.method in {
                "numbered",
                "coordinated",
                "entity_distributive",
                "fallback",
            }
            assert isinstance(req.entity_ids, tuple)

        # Invariant: valid synthesis obligations
        for syn in result.synthesis_obligations:
            assert syn.text.strip()
            assert syn.kind in {
                "comparison",
                "selection",
                "causal",
                "aggregation",
                "qualification",
            }
            assert all(
                0 <= idx < len(result.requirements)
                for idx in syn.depends_on_requirement_indexes
            ), f"{item['id']} has out-of-range synthesis dependency: {syn.depends_on_requirement_indexes}"

        # Invariant: semantic planning consistency
        assert result.requires_semantic_planning == bool(
            result.semantic_planning_reasons
        )
        if result.requires_semantic_planning:
            assert set(result.semantic_planning_reasons).issubset(
                known_reasons
            ), f"{item['id']} unknown semantic planning reasons: {result.semantic_planning_reasons}"
