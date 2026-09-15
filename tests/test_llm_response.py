from types import SimpleNamespace

from langchain_core.messages import AIMessage

from core.llm_response import final_answer_text


def test_final_answer_text_prefers_aimessage_text_projection() -> None:
    response = AIMessage(
        content=[
            {"type": "thinking", "thinking": "private chain"},
            {"type": "text", "text": "Visible answer."},
            {"type": "signature", "signature": "provider-signature"},
        ]
    )

    assert final_answer_text(response) == "Visible answer."


def test_final_answer_text_reads_text_blocks_and_ignores_reasoning() -> None:
    response = SimpleNamespace(
        content=[
            {"type": "reasoning", "text": "private reasoning"},
            {"type": "text", "text": "First sentence. "},
            {"type": "output_text", "text": "Second sentence."},
            {"type": "tool_use", "input": {"query": "private"}},
        ]
    )

    assert final_answer_text(response) == "First sentence. Second sentence."


def test_final_answer_text_leaves_non_text_response_empty() -> None:
    response = SimpleNamespace(
        content=[
            {"type": "thinking", "thinking": "private"},
            {"type": "signature", "signature": "provider-signature"},
        ]
    )

    assert final_answer_text(response) == ""
