import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.evaluator import RAGEvaluator
from langchain_core.documents import Document


@pytest.mark.asyncio
async def test_evaluator_conflict_analysis():
    """Verify that the evaluator correctly identifies conflict handling quality."""
    evaluator = RAGEvaluator()

    question = "SwinUNETR 與 nnU-Net 誰更好？"

    docs = [
        Document(page_content="SwinUNETR 在 BraTS 2021 表現優於 nnU-Net，Dice 分數高出 1%.", metadata={"source": "SwinUNETR.pdf"}),
        Document(page_content="大規模 Benchmark 顯示 nnU-Net 在 10 個數據集中的 8 個勝過 SwinUNETR.", metadata={"source": "nnU-Net Revisited.pdf"}),
    ]

    bad_answer = "兩者互有優劣，視情況而定。有些任務 SwinUNETR 好，有些則是 nnU-Net 好。"
    good_answer = "一方面，SwinUNETR 論文聲稱在 BraTS 數據集勝出；另一方面，nnU-Net Revisited 的大規模基準測試顯示 nnU-Net 在多數場景更優。根據證據權重（Benchmark > 單一實驗），較可信的結論是 nnU-Net 表現更穩健。"

    mock_llm = AsyncMock()

    def _response_for(answer: str) -> MagicMock:
        payload = {
            "analysis": "回答有反映衝突證據。",
            "accuracy": 9,
            "completeness": 9,
            "clarity": 8,
            "reason": "有依證據權重做判斷。",
            "suggestion": "",
        }
        if "互有優劣" in answer:
            payload = {
                "analysis": "回答用模糊結論迴避衝突證據。",
                "accuracy": 5,
                "completeness": 4,
                "clarity": 7,
                "reason": "缺少證據權重與衝突處理。",
                "suggestion": "應明確比較基準測試與單篇論文的證據強度。",
            }
        response = MagicMock()
        import json
        response.content = json.dumps(payload, ensure_ascii=False)
        return response

    async def _ainvoke(messages):
        prompt = messages[-1].content
        if good_answer in prompt:
            return _response_for(good_answer)
        return _response_for(bad_answer)

    mock_llm.ainvoke.side_effect = _ainvoke

    with patch("agents.evaluator.get_llm", return_value=mock_llm):
        res_bad = await evaluator.evaluate_detailed(question, docs, bad_answer)
        print(f"\n[Averaging Answer] Score: {res_bad.weighted_score:.2f}, Accuracy: {res_bad.accuracy}")

        res_good = await evaluator.evaluate_detailed(question, docs, good_answer)
        print(f"[Conflict Aware Answer] Score: {res_good.weighted_score:.2f}, Accuracy: {res_good.accuracy}")

    assert res_good.accuracy > res_bad.accuracy
    assert "權重" in res_good.reason or "證據" in res_good.reason


if __name__ == "__main__":
    asyncio.run(test_evaluator_conflict_analysis())
