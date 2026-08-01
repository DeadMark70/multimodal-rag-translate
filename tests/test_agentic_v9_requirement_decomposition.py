from data_base.agentic_v9.requirement_decomposition import split_top_level_blocks


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
