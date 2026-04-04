import json
from pathlib import Path

from agents.planner import classify_question_intent


def test_classify_question_intent_on_ragas_hardset_v2_samples() -> None:
    dataset_path = Path(__file__).resolve().parents[2] / "ragas_hardset_v2.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = {item["id"]: item["question"] for item in payload["questions"]}

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
