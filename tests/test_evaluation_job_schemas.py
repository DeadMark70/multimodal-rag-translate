import pytest

from evaluation.job_schemas import EvaluationRerunRequest


def test_rerun_request_deduplicates_selected_ids() -> None:
    request = EvaluationRerunRequest(
        scope="selected",
        stages="ragas",
        question_ids=["Q1", " Q1 ", "", "Q2"],
        metric_names=["faithfulness", "faithfulness"],
    )

    assert request.question_ids == ["Q1", "Q2"]
    assert request.metric_names == ["faithfulness"]


def test_selected_rerun_requires_question_ids() -> None:
    with pytest.raises(ValueError, match="question_ids"):
        EvaluationRerunRequest(scope="selected", stages="execution")
