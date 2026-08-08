from agents.planner import classify_question_intent


def test_classify_question_intent_on_ragas_hardset_v2_samples() -> None:
    questions = {
        "Q1": "在 BraTS 類 3D 腦腫瘤分割場景，若訓練資料偏少且 GPU 資源受限，SwinUNETR、MedNeXt、nnMamba 在長距離依賴建模方式與計算成本上應如何取捨？請給出選型裁決，並註明不是同配置 benchmark 排名。",
        "Q2": "從 MedSAM、SAM-Med3D 到 MedSAM-2，三代模型處理 3D 體積資料的空間資訊建模與 Prompt 需求有何演進？請描述表現趨勢，但不要寫成 apples-to-apples 的通用 Dice 排名。",
        "Q3": "MedSAM-2 具備單一提示詞分割能力，但仍依賴初始 bounding box 品質，兩種說法是否互斥？請根據記憶注意力與 mask propagation 機制裁決。",
        "Q4": "請結合 Params 與 FLOPs 報告比較 nnMamba 與 EfficientMedNeXt-L，裁決 Mamba 在 3D 醫療分割具最高計算效率是否成立，並註明不是嚴格 apples-to-apples benchmark。",
        "Q5": "根據 nnMamba 架構描述，請重建 MICCSS 模塊中 CSS 階段的特徵融合流程，並說明三個翻轉分支與 SiamSSM 的運算及累加機制。",
        "Q6": "比較 U-Mamba 與 Weak-Mamba-UNet 在模型角色與監督機制上的核心差異，並說明 three-view cross-supervision 如何利用 partial cross-entropy 與 pseudo-label Dice supervision。",
        "Q7": "在 SAMed、MedSAM、SAM-Med2D、SAM-Med3D 中，哪個模型改造解碼器以直接產生語義類別遮罩，且推理階段可 Prompt-free？請指出唯一符合者與關鍵技術。",
        "Q8": "若只有少量 dense masks 與大量無標註影像，且希望利用 pair of projectors 做像素級對比學習，在 Semi-Mamba-UNet、Weak-Mamba-UNet、U-Mamba 中應選哪個？",
    }

    expected = {
        "Q1": "benchmark_data",
        "Q2": "benchmark_data",
        "Q3": "general_research",
        "Q4": "benchmark_data",
        "Q5": "figure_flow",
        "Q6": "comparison_disambiguation",
        "Q7": "general_research",
        "Q8": "comparison_disambiguation",
    }

    for question_id, expected_intent in expected.items():
        assert classify_question_intent(questions[question_id]) == expected_intent
