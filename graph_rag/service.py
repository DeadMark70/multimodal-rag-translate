"""GraphRAG service-layer helpers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from graph_rag.extractor import extract_and_add_to_graph
from graph_rag.schemas import GraphDocumentStatus, GraphExtractionRunResult
from graph_rag.store import GraphStore

logger = logging.getLogger(__name__)


async def run_graph_extraction(
    *,
    user_id: str,
    doc_id: str,
    markdown_text: str,
    batch_size: int = 3,
    store: GraphStore | None = None,
) -> GraphExtractionRunResult:
    """Run GraphRAG extraction and persist per-document status."""
    active_store = store or GraphStore(user_id)
    attempted_at = datetime.now()

    def _persist_status(
        *,
        status: str,
        chunk_count: int,
        chunks_succeeded: int,
        chunks_failed: int,
        entities_added: int,
        edges_added: int,
        last_error: str | None,
    ) -> GraphExtractionRunResult:
        active_store.upsert_document_status(
            GraphDocumentStatus(
                doc_id=doc_id,
                status=status,
                chunk_count=chunk_count,
                chunks_succeeded=chunks_succeeded,
                chunks_failed=chunks_failed,
                entities_added=entities_added,
                edges_added=edges_added,
                last_error=last_error,
                last_attempted_at=attempted_at,
                last_succeeded_at=attempted_at if status in {"indexed", "partial", "empty"} else None,
            )
        )
        active_store.save_sidecars()
        return GraphExtractionRunResult(
            doc_id=doc_id,
            status=status,
            chunk_count=chunk_count,
            chunks_succeeded=chunks_succeeded,
            chunks_failed=chunks_failed,
            entities_added=entities_added,
            edges_added=edges_added,
            last_error=last_error,
        )

    try:
        chunk_size = 8000
        all_chunks = [
            markdown_text[i : i + chunk_size]
            for i in range(0, len(markdown_text), chunk_size)
        ]
        chunks = [
            (idx, chunk)
            for idx, chunk in enumerate(all_chunks)
            if len(chunk.strip()) >= 100
        ]

        if not chunks:
            logger.info("[GraphRAG] No valid chunks to process for doc %s", doc_id)
            return _persist_status(
                status="empty",
                chunk_count=0,
                chunks_succeeded=0,
                chunks_failed=0,
                entities_added=0,
                edges_added=0,
                last_error=None,
            )

        total_nodes = 0
        total_edges = 0
        completed_chunks = 0
        chunk_failures: list[str] = []

        num_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info(
            "[GraphRAG] Processing %s chunks in %s concurrent batches (batch_size=%s) for doc %s",
            len(chunks),
            num_batches,
            batch_size,
            doc_id,
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            logger.info("[GraphRAG] Processing batch %s/%s...", batch_idx + 1, num_batches)

            tasks = [
                extract_and_add_to_graph(
                    text=chunk,
                    doc_id=doc_id,
                    store=active_store,
                    chunk_index=idx,
                )
                for idx, chunk in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                chunk_idx = batch[i][0]
                if isinstance(result, Exception):
                    chunk_failures.append(f"chunk {chunk_idx}: {result}")
                    logger.warning("[GraphRAG] Chunk %s extraction failed: %s", chunk_idx, result)
                    continue

                nodes, edges = result
                completed_chunks += 1
                total_nodes += nodes
                total_edges += edges

        if completed_chunks == 0:
            status = "failed"
            last_error = "; ".join(chunk_failures) if chunk_failures else "All chunks failed"
        elif chunk_failures:
            status = "partial"
            last_error = "; ".join(chunk_failures)
        else:
            status = "indexed"
            last_error = None

        logger.info(
            "[GraphRAG] Completed doc %s with status=%s, chunks=%s/%s, nodes=%s, edges=%s",
            doc_id,
            status,
            completed_chunks,
            len(chunks),
            total_nodes,
            total_edges,
        )
        active_store.save()
        return _persist_status(
            status=status,
            chunk_count=len(chunks),
            chunks_succeeded=completed_chunks,
            chunks_failed=len(chunk_failures),
            entities_added=total_nodes,
            edges_added=total_edges,
            last_error=last_error,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("[GraphRAG] Graph extraction failed for doc %s: %s", doc_id, exc, exc_info=True)
        return _persist_status(
            status="failed",
            chunk_count=0,
            chunks_succeeded=0,
            chunks_failed=0,
            entities_added=0,
            edges_added=0,
            last_error=str(exc),
        )
