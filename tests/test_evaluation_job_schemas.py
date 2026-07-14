from collections.abc import Mapping

import pytest

from evaluation.job_schemas import ClaimedEvaluationWork, EvaluationRerunRequest


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


def test_claimed_work_snapshot_is_deeply_immutable() -> None:
    source_snapshot = {"payload": {"values": ["original"]}}
    claimed = ClaimedEvaluationWork(
        job_id="job-1",
        job_item_id="item-1",
        work_item_id="work-1",
        attempt_id="attempt-1",
        input_snapshot=source_snapshot,
    )

    payload = claimed.input_snapshot["payload"]
    assert isinstance(payload, Mapping)
    values = payload["values"]
    assert isinstance(values, tuple)
    with pytest.raises(TypeError):
        payload["other"] = "mutated"
    with pytest.raises(AttributeError):
        values.append("mutated")

    source_snapshot["payload"]["values"].append("source mutation")
    assert claimed.input_snapshot["payload"]["values"] == ("original",)


def test_claimed_work_model_copy_freezes_updated_snapshot() -> None:
    claimed = ClaimedEvaluationWork(
        job_id="job-1",
        job_item_id="item-1",
        work_item_id="work-1",
        attempt_id="attempt-1",
        input_snapshot={"payload": {"values": ["original"]}},
    )

    copied = claimed.model_copy(
        update={"input_snapshot": {"payload": {"values": ["updated"]}}}
    )

    payload = copied.input_snapshot["payload"]
    assert isinstance(payload, Mapping)
    values = payload["values"]
    assert isinstance(values, tuple)
    with pytest.raises(TypeError):
        payload["other"] = "mutated"
    with pytest.raises(AttributeError):
        values.append("mutated")
    assert copied.model_dump(mode="json")["input_snapshot"] == {
        "payload": {"values": ["updated"]}
    }


def test_claimed_work_model_construct_is_prohibited() -> None:
    with pytest.raises(TypeError, match="validated construction"):
        ClaimedEvaluationWork.model_construct(
            job_id="job-1",
            job_item_id="item-1",
            work_item_id="work-1",
            attempt_id="attempt-1",
            input_snapshot={"payload": {"values": ["mutable"]}},
        )


def test_claimed_work_deep_model_copy_preserves_immutable_snapshot() -> None:
    claimed = ClaimedEvaluationWork(
        job_id="job-1",
        job_item_id="item-1",
        work_item_id="work-1",
        attempt_id="attempt-1",
        input_snapshot={"payload": {"values": ["original"]}},
    )

    copied = claimed.model_copy(deep=True)

    assert copied is not claimed
    payload = copied.input_snapshot["payload"]
    assert isinstance(payload, Mapping)
    values = payload["values"]
    assert isinstance(values, tuple)
    with pytest.raises(TypeError):
        payload["other"] = "mutated"
    with pytest.raises(AttributeError):
        values.append("mutated")
    assert copied.model_dump(mode="json")["input_snapshot"] == {
        "payload": {"values": ["original"]}
    }
