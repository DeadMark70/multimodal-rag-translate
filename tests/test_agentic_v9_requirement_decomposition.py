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

    assert 3 <= len(result.requirements) <= 5
    assert any(item.kind == "prohibition" for item in result.response_constraints)
    assert all(
        item.confidence in {"high", "medium", "low"} for item in result.requirements
    )
