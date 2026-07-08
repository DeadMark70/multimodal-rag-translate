import logging
import asyncio

import pytest

from evaluation import db as evaluation_db
from evaluation.observability import EvaluationRunRecorder
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.db import connect_db, init_db


class FakeObservabilityRepository:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.trace_events = []

    async def record_trace_event(self, event):
        if self.fail:
            raise RuntimeError("repository unavailable")
        self.trace_events.append(event)


@pytest.mark.asyncio
async def test_start_span_records_running_and_success_events() -> None:
    repository = FakeObservabilityRepository()
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        trace_repository=repository,
    )

    async with recorder.start_span(stage_type="retrieval", stage_name="hybrid_retrieval") as span:
        assert span.event_id
        assert span.span_id

    assert len(repository.trace_events) == 2
    started, completed = repository.trace_events
    assert started.event_id == span.start_event_id
    assert completed.event_id == span.end_event_id
    assert started.event_id != completed.event_id
    assert started.span_id == completed.span_id == span.span_id
    assert started.status == "running"
    assert started.duration_ms is None
    assert completed.status == "success"
    assert completed.duration_ms is not None
    assert completed.duration_ms >= 0


@pytest.mark.asyncio
async def test_nested_spans_record_parent_event_id() -> None:
    repository = FakeObservabilityRepository()
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        trace_repository=repository,
    )

    async with recorder.start_span(stage_type="planning", stage_name="plan") as parent:
        async with recorder.start_span(stage_type="retrieval", stage_name="retrieve") as child:
            assert child.parent_event_id == parent.event_id

    child_completed = [event for event in repository.trace_events if event.event_id == child.event_id][-1]
    assert child_completed.parent_event_id == parent.event_id
    assert child_completed.parent_span_id == parent.span_id


@pytest.mark.asyncio
async def test_persisted_span_keeps_running_and_success_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    await init_db()
    async with connect_db() as connection:
        await connection.execute(
            """
            INSERT INTO campaigns (
                id, user_id, name, status, phase, config_json, completed_units, total_units,
                evaluation_completed_units, evaluation_total_units, current_question_id,
                current_mode, error_message, cancel_requested, created_at, started_at,
                completed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, NULL, NULL, NULL, 0, ?, NULL, NULL, ?)
            """,
            (
                "campaign-recorder",
                "user-a",
                "Recorder",
                "running",
                "execution",
                "{}",
                "2026-07-08T00:00:00+00:00",
                "2026-07-08T00:00:00+00:00",
            ),
        )
        await connection.commit()

    repository = EvaluationObservabilityRepository()
    recorder = EvaluationRunRecorder(
        run_id="run-persisted",
        campaign_id="campaign-recorder",
        user_id="user-a",
        trace_repository=repository,
        strict=True,
    )

    async with recorder.start_span(stage_type="retrieval", stage_name="hybrid_retrieval"):
        pass

    events = await repository.list_trace_events_for_run("run-persisted")
    assert [event.status for event in events] == ["running", "success"]
    assert events[0].event_id != events[1].event_id
    assert events[0].span_id == events[1].span_id


@pytest.mark.asyncio
async def test_concurrent_spans_do_not_inherit_sibling_parent() -> None:
    repository = FakeObservabilityRepository()
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        trace_repository=repository,
    )

    async def run_span(stage_name: str) -> None:
        async with recorder.start_span(stage_type="retrieval", stage_name=stage_name):
            await asyncio.sleep(0)

    await asyncio.gather(run_span("a"), run_span("b"))

    running_events = [event for event in repository.trace_events if event.status == "running"]
    assert {event.stage_name for event in running_events} == {"a", "b"}
    assert all(event.parent_event_id is None for event in running_events)
    assert all(event.parent_span_id is None for event in running_events)


@pytest.mark.asyncio
async def test_exception_inside_span_marks_failed_and_sanitizes_error() -> None:
    repository = FakeObservabilityRepository()
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        trace_repository=repository,
    )

    with pytest.raises(ValueError):
        async with recorder.start_span(stage_type="generation", stage_name="generate"):
            raise ValueError("provider failed with raw payload")

    completed = repository.trace_events[-1]
    assert completed.status == "failed"
    assert completed.error == {
        "type": "ValueError",
        "message": "provider failed with raw payload",
    }
    assert "Traceback" not in str(completed.error)


@pytest.mark.asyncio
async def test_best_effort_recorder_swallows_repository_failures(caplog) -> None:
    repository = FakeObservabilityRepository(fail=True)
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        trace_repository=repository,
        strict=False,
    )

    with caplog.at_level(logging.WARNING):
        async with recorder.start_span(stage_type="retrieval", stage_name="retrieve"):
            pass

    assert "Failed to record evaluation observability event" in caplog.text


@pytest.mark.asyncio
async def test_strict_recorder_raises_repository_failures() -> None:
    repository = FakeObservabilityRepository(fail=True)
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        trace_repository=repository,
        strict=True,
    )

    with pytest.raises(RuntimeError, match="repository unavailable"):
        async with recorder.start_span(stage_type="retrieval", stage_name="retrieve"):
            pass
