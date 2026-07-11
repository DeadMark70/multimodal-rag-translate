"""Document checkpoint and retry behavior for durable GraphRAG rebuilds."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from graph_rag.rebuild_coordinator import GraphRebuildCoordinator
from graph_rag.rebuild_jobs import GraphRebuildJobStore
from graph_rag.schemas import GraphExtractionRunResult


SOURCES = [
    {"doc_id": "doc-1", "file_name": "first.pdf", "original_path": "uploads/u/doc-1/first.pdf"},
    {"doc_id": "doc-2", "file_name": "second.pdf", "original_path": "uploads/u/doc-2/second.pdf"},
]


def _indexed_result(doc_id: str) -> GraphExtractionRunResult:
    return GraphExtractionRunResult(doc_id=doc_id, status="indexed", entities_added=1)


def _coordinator(
    tmp_path: Path,
    extract: AsyncMock,
    sleep: AsyncMock,
    *,
    optimize: AsyncMock | None = None,
    publish: Mock | None = None,
) -> tuple[GraphRebuildJobStore, GraphRebuildCoordinator, str, str]:
    jobs = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = jobs.create_job(SOURCES)
    owner_token = jobs.acquire_lease(manifest.job_id)
    assert owner_token is not None
    store = Mock()
    store.get_documents.return_value = {"doc-1", "doc-2"}
    store.get_latest_extraction_manifest.return_value = object()
    coordinator = GraphRebuildCoordinator(
        jobs,
        store_factory=Mock(return_value=store),
        run_extraction=extract,
        load_artifacts=Mock(return_value=("markdown", [])),
        sleep=sleep,
        jitter=lambda: 0.0,
        optimize=optimize or AsyncMock(return_value=(0, 0)),
        publish=publish or Mock(),
    )
    return jobs, coordinator, manifest.job_id, owner_token


@pytest.mark.asyncio
async def test_resume_skips_checkpointed_documents(tmp_path: Path) -> None:
    extract = AsyncMock(return_value=_indexed_result("doc-2"))
    jobs, coordinator, job_id, owner_token = _coordinator(tmp_path, extract, AsyncMock())
    manifest = jobs.load(job_id)
    manifest.documents[0].state = "indexed"
    jobs.save(manifest)

    await coordinator.run("user-1", job_id, owner_token)

    assert [call.kwargs["doc_id"] for call in extract.await_args_list] == ["doc-2"]
    assert jobs.load(job_id).documents[1].state == "indexed"


@pytest.mark.asyncio
async def test_retryable_failure_succeeds_on_third_attempt(tmp_path: Path) -> None:
    extract = AsyncMock(side_effect=[TimeoutError("one"), TimeoutError("two"), _indexed_result("doc-1"), _indexed_result("doc-2")])
    sleep = AsyncMock()
    jobs, coordinator, job_id, owner_token = _coordinator(tmp_path, extract, sleep)

    await coordinator.run("user-1", job_id, owner_token)

    assert extract.await_count == 4
    assert [call.args[0] for call in sleep.await_args_list] == [5.0, 20.0]
    first = jobs.load(job_id).documents[0]
    assert first.state == "indexed"
    assert first.attempt == 3


@pytest.mark.asyncio
async def test_failed_document_keeps_staging_and_old_graph(tmp_path: Path) -> None:
    extract = AsyncMock(
        side_effect=[
            GraphExtractionRunResult(doc_id="doc-1", status="failed", last_error="quota"),
            _indexed_result("doc-2"),
        ]
    )
    publish = Mock()
    jobs, coordinator, job_id, owner_token = _coordinator(
        tmp_path, extract, AsyncMock(), publish=publish
    )

    await coordinator.run("user-1", job_id, owner_token)

    assert jobs.load(job_id).state == "completed_with_failures"
    assert jobs.staging_dir(job_id).exists()
    publish.assert_not_called()


@pytest.mark.asyncio
async def test_all_successful_documents_finalize_and_publish(tmp_path: Path) -> None:
    extract = AsyncMock(side_effect=[_indexed_result("doc-1"), _indexed_result("doc-2")])
    optimize = AsyncMock(return_value=(1, 2))
    publish = Mock()
    jobs, coordinator, job_id, owner_token = _coordinator(
        tmp_path, extract, AsyncMock(), optimize=optimize, publish=publish
    )

    await coordinator.run("user-1", job_id, owner_token)

    assert jobs.load(job_id).state == "completed"
    optimize.assert_awaited_once()
    publish.assert_called_once()
