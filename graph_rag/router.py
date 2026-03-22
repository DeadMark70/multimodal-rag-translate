"""
GraphRAG API Router

Provides REST API endpoints for graph management operations:
- GET /graph/status - Get graph status
- GET /graph/data - Get visualization data for react-force-graph
- POST /graph/rebuild - Re-run graph optimization safely without re-extraction
- POST /graph/optimize - Run entity resolution
"""

# Standard library
import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

# Third-party
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel, Field

# Local application
from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from graph_rag.schemas import (
    GraphDocumentStatus,
    GraphDocumentStatusItem,
    GraphDocumentStatusListResponse,
    GraphStatusResponse,
)
from graph_rag.store import GraphStore
from pdfserviceMD.repository import get_document, list_documents as list_pdf_documents
from pdfserviceMD.router import _run_graph_extraction
from pdfserviceMD.service import load_ocr_artifacts

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


# ===== Request/Response Models =====

class GraphRebuildRequest(BaseModel):
    """Request for graph rebuild."""
    force: bool = Field(
        default=False,
        description="強制重建即使圖譜沒有變更"
    )


class GraphOptimizeRequest(BaseModel):
    """Request for graph optimization."""
    regenerate_communities: bool = Field(
        default=True,
        description="重新生成社群摘要"
    )


class GraphOperationResponse(BaseModel):
    """Response for graph operations."""
    status: str
    message: str
    details: Optional[dict] = None


class VisNode(BaseModel):
    """Node for react-force-graph visualization."""
    id: str = Field(..., description="唯一節點識別碼（標籤）")
    group: int = Field(..., description="分群 ID（用於著色）")
    val: int = Field(..., description="節點大小（引用次數）")
    desc: str = Field(..., description="節點描述")


class VisLink(BaseModel):
    """Link for react-force-graph visualization."""
    source: str = Field(..., description="來源節點 ID")
    target: str = Field(..., description="目標節點 ID")
    label: str = Field(..., description="關係標籤")


class GraphVisualizationData(BaseModel):
    """Response for graph visualization (react-force-graph format)."""
    nodes: List[VisNode] = Field(default_factory=list)
    links: List[VisLink] = Field(default_factory=list)


def _copy_graph_sidecars(src: GraphStore, dest: GraphStore) -> None:
    """Copy graph pickle and sidecars from one store location to another."""
    for source_path, target_path in (
        (src._get_graph_path(), dest._get_graph_path()),
        (src._get_metadata_path(), dest._get_metadata_path()),
        (src._get_document_status_path(), dest._get_document_status_path()),
    ):
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _replace_live_graph_files(temp_store: GraphStore, live_store: GraphStore) -> None:
    """Atomically replace the live graph pickle and sidecars from a temp store."""
    for source_path, target_path in (
        (temp_store._get_graph_path(), live_store._get_graph_path()),
        (temp_store._get_metadata_path(), live_store._get_metadata_path()),
        (temp_store._get_document_status_path(), live_store._get_document_status_path()),
    ):
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_target = target_path.with_suffix(target_path.suffix + ".tmp")
            shutil.copy2(source_path, temp_target)
            os.replace(temp_target, target_path)


def _make_graph_work_dir(base_dir: Path, prefix: str) -> Path:
    """Create a writable unique workspace directory for graph maintenance jobs."""
    work_dir = base_dir / f"{prefix}{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    return work_dir


async def _list_graph_source_documents(user_id: str) -> list[dict[str, str | None]]:
    """List OCR-complete documents that can act as GraphRAG rebuild sources."""
    store = GraphStore(user_id)
    eligible_ids = store.list_eligible_document_ids()
    rows = await list_pdf_documents(user_id=user_id, limit=max(len(eligible_ids) + 50, 200))
    row_map = {row["id"]: row for row in rows}
    sources: list[dict[str, str | None]] = []
    for doc_id in eligible_ids:
        row = row_map.get(doc_id, {})
        sources.append(
            {
                "doc_id": doc_id,
                "file_name": row.get("file_name"),
                "original_path": row.get("original_path"),
            }
        )
    return sources


async def _build_graph_document_rows(user_id: str, store: GraphStore) -> list[GraphDocumentStatusItem]:
    """Build API rows by joining persisted graph statuses with OCR-eligible docs."""
    sources = await _list_graph_source_documents(user_id)
    source_map = {item["doc_id"]: item for item in sources}
    known_doc_ids = set(source_map) | set(store.document_statuses)
    rows: list[GraphDocumentStatusItem] = []

    for doc_id in sorted(known_doc_ids):
        persisted = store.get_document_status(doc_id)
        source = source_map.get(doc_id, {})
        status = persisted or GraphDocumentStatus(doc_id=doc_id, status="skipped")
        rows.append(
            GraphDocumentStatusItem(
                **status.model_dump(),
                file_name=source.get("file_name"),
                is_eligible=doc_id in source_map,
            )
        )
    return rows


# ===== Endpoints =====

@router.get(
    "/status",
    response_model=GraphStatusResponse,
    summary="取得圖譜狀態",
    description="取得使用者知識圖譜的狀態資訊，包括節點數、邊數、社群數等。"
)
async def get_graph_status(
    user_id: str = Depends(get_current_user_id)
) -> GraphStatusResponse:
    """
    Get the current status of user's knowledge graph.
    
    Returns:
        GraphStatusResponse with current graph statistics.
    """
    try:
        store = GraphStore(user_id)
        status = store.get_status()
        logger.info(f"Graph status for user {user_id}: {status.node_count} nodes")
        return status
    except Exception as e:
        logger.error(f"Failed to get graph status: {e}")
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to get graph status",
            status_code=500,
        ) from e


@router.get(
    "/documents",
    response_model=GraphDocumentStatusListResponse,
    summary="列出 GraphRAG 文件狀態",
    description="列出所有可用於 GraphRAG 的文件，以及每個文件最近一次抽圖的狀態。",
)
async def list_graph_documents(
    user_id: str = Depends(get_current_user_id)
) -> GraphDocumentStatusListResponse:
    """List per-document GraphRAG extraction status rows."""
    try:
        store = GraphStore(user_id)
        documents = await _build_graph_document_rows(user_id, store)
        return GraphDocumentStatusListResponse(documents=documents, total=len(documents))
    except Exception as e:
        logger.error("Failed to list graph documents: %s", e, exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to list graph documents",
            status_code=500,
        ) from e


@router.get(
    "/data",
    response_model=GraphVisualizationData,
    summary="取得視覺化資料",
    description="取得 react-force-graph 格式的圖譜資料，用於前端視覺化。"
)
async def get_graph_visualization_data(
    user_id: str = Depends(get_current_user_id)
) -> GraphVisualizationData:
    """
    Get graph data for visualization (react-force-graph format).
    
    Returns nodes and links for rendering in react-force-graph.
    
    Returns:
        GraphVisualizationData with nodes and links arrays.
    """
    try:
        store = GraphStore(user_id)
        status = store.get_status()
        
        if not status.has_graph:
            logger.info(f"No graph data for user {user_id}")
            return GraphVisualizationData(nodes=[], links=[])
        
        # Convert graph nodes to visualization format
        nodes = []
        for node in store.get_all_nodes():
            nodes.append(VisNode(
                id=node.label,
                group=hash(node.entity_type.value) % 5,
                val=len(node.doc_ids) * 2,
                desc=node.description or node.entity_type.value,
            ))
        
        # Convert edges to links
        links = []
        for edge in store.get_all_edges():
            source_node = store.get_node(edge.source_id)
            target_node = store.get_node(edge.target_id)
            if source_node and target_node:
                links.append(VisLink(
                    source=source_node.label,
                    target=target_node.label,
                    label=edge.relation,
                ))
        
        logger.info(f"Graph visualization for user {user_id}: {len(nodes)} nodes, {len(links)} links")
        return GraphVisualizationData(nodes=nodes, links=links)
        
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Graph data not found for user {user_id}: {e}")
        return GraphVisualizationData(nodes=[], links=[])
    except Exception as e:
        logger.error(f"Failed to get graph visualization data: {e}")
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to get graph visualization data",
            status_code=500,
        ) from e


@router.post(
    "/rebuild",
    response_model=GraphOperationResponse,
    summary="安全重算圖譜索引",
    description=(
        "重新執行既有圖譜上的實體融合與社群建立，刷新索引與摘要。"
        "此操作不會重新從原始文件抽取新實體。"
    ),
)
async def rebuild_graph(
    request: GraphRebuildRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id)
) -> GraphOperationResponse:
    """
    Re-run graph optimization pipeline in background without clearing graph data.
    """
    try:
        store = GraphStore(user_id)
        
        if not request.force and store.get_status().node_count == 0:
            return GraphOperationResponse(
                status="skipped",
                message="圖譜已是空的，無需重建",
            )
        
        # Queue background rebuild task
        background_tasks.add_task(_rebuild_graph_task, user_id)
        
        return GraphOperationResponse(
            status="started",
            message="圖譜安全重算已開始（不重新抽取文件實體，且不清空既有關係）",
            details={"user_id": user_id}
        )
        
    except Exception as e:
        logger.error(f"Failed to start graph rebuild: {e}")
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to start graph rebuild",
            status_code=500,
        ) from e


@router.post(
    "/rebuild-full",
    response_model=GraphOperationResponse,
    summary="完整重構圖譜",
    description="從所有 OCR artifact 重新抽取並建立全新圖譜；成功後才會覆蓋目前圖譜。",
)
async def rebuild_graph_full(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> GraphOperationResponse:
    """Build a fresh graph from all OCR-complete document artifacts."""
    try:
        store = GraphStore(user_id)
        if store.active_job_state:
            return GraphOperationResponse(
                status="skipped",
                message=f"已有圖譜工作執行中：{store.active_job_state}",
            )

        sources = await _list_graph_source_documents(user_id)
        if not sources:
            return GraphOperationResponse(
                status="skipped",
                message="沒有可用的 OCR 文件可供完整重構",
            )

        store.set_active_job_state("rebuild_full")
        store.save_sidecars()
        background_tasks.add_task(_rebuild_full_graph_task, user_id)
        return GraphOperationResponse(
            status="started",
            message="完整圖譜重構已開始，將從所有 OCR 文件重新抽取",
            details={"document_count": len(sources)},
        )
    except Exception as e:
        logger.error("Failed to start full graph rebuild: %s", e, exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to start full graph rebuild",
            status_code=500,
        ) from e


@router.post(
    "/documents/{doc_id}/retry",
    response_model=GraphOperationResponse,
    summary="重試單一文件 GraphRAG",
    description="只對指定文件重新抽取 GraphRAG；成功後會刷新整體社群。",
)
async def retry_graph_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> GraphOperationResponse:
    """Retry GraphRAG extraction for a single OCR-complete document."""
    try:
        store = GraphStore(user_id)
        if store.active_job_state:
            return GraphOperationResponse(
                status="skipped",
                message=f"已有圖譜工作執行中：{store.active_job_state}",
            )

        row = await get_document(
            doc_id=doc_id,
            user_id=user_id,
            columns="id, file_name, original_path",
        )
        if not row:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Document not found",
                status_code=404,
            )

        original_path = row.get("original_path")
        if not original_path:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="Document has no original path",
                status_code=409,
            )

        user_folder = Path(original_path).resolve().parent
        if not (user_folder / "extracted.md").exists():
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="Document has no OCR artifacts for GraphRAG retry",
                status_code=409,
            )

        store.set_active_job_state(f"retry:{doc_id}")
        store.save_sidecars()
        background_tasks.add_task(_retry_graph_document_task, user_id, doc_id)
        return GraphOperationResponse(
            status="started",
            message="單一文件 GraphRAG 重試已開始",
            details={"doc_id": doc_id, "file_name": row.get("file_name")},
        )
    except AppError:
        raise
    except Exception as e:
        logger.error("Failed to start graph retry for %s: %s", doc_id, e, exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to start graph document retry",
            status_code=500,
        ) from e


@router.post(
    "/optimize",
    response_model=GraphOperationResponse,
    summary="優化圖譜",
    description="執行實體融合與社群重建，合併相似的實體節點。"
)
async def optimize_graph(
    request: GraphOptimizeRequest,
    user_id: str = Depends(get_current_user_id)
) -> GraphOperationResponse:
    """
    Run entity resolution and community detection.
    """
    try:
        store = GraphStore(user_id)
        
        if store.get_status().node_count == 0:
            return GraphOperationResponse(
                status="skipped",
                message="圖譜是空的，無需優化",
            )
        
        merges, communities_count = await _optimize_existing_graph(
            store,
            regenerate_communities=request.regenerate_communities,
        )
        
        return GraphOperationResponse(
            status="success",
            message=f"優化完成：合併 {merges} 個實體，建立 {communities_count} 個社群",
            details={
                "merges": merges,
                "communities": communities_count,
                "node_count": store.get_status().node_count,
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to optimize graph: {e}")
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to optimize graph",
            status_code=500,
        ) from e


# ===== Background Tasks =====

async def _optimize_existing_graph(
    store: GraphStore,
    *,
    regenerate_communities: bool = True,
) -> tuple[int, int]:
    """Run entity resolution and optionally rebuild communities on an existing graph."""
    from graph_rag.entity_resolver import resolve_entities

    merges = await resolve_entities(store)
    communities_count = 0

    if regenerate_communities:
        from graph_rag.community_builder import build_communities

        communities = await build_communities(store, generate_summaries=True)
        communities_count = len(communities)

    store.save()
    return merges, communities_count

async def _rebuild_graph_task(user_id: str) -> None:
    """
    Background task to safely rebuild graph metadata and communities.

    This does not re-extract entities/relations from source documents.
    """
    logger.info(f"Starting graph rebuild for user {user_id}")
    
    try:
        store = GraphStore(user_id)

        if store.get_status().node_count == 0:
            logger.info("Skipping graph rebuild for user %s because graph is empty", user_id)
            return

        await _optimize_existing_graph(store, regenerate_communities=True)
        
        logger.info(f"Graph rebuild complete for user {user_id}")
        
    except Exception as e:
        logger.error(f"Graph rebuild failed for user {user_id}: {e}")


async def _rebuild_full_graph_task(user_id: str) -> None:
    """Build a brand-new graph from all OCR-complete document artifacts."""
    logger.info("Starting full graph rebuild for user %s", user_id)
    live_store = GraphStore(user_id)
    sources = await _list_graph_source_documents(user_id)
    temp_dir = _make_graph_work_dir(live_store.storage_dir.parent, "graph-rebuild-")

    try:
        temp_store = GraphStore(user_id, storage_dir=temp_dir)
        temp_store.clear()
        temp_store.save()

        for source in sources:
            doc_id = str(source["doc_id"])
            original_path = source.get("original_path")
            user_folder = (
                Path(original_path).resolve().parent
                if original_path
                else Path("uploads") / user_id / doc_id
            )

            try:
                markdown_text, _ = await asyncio.to_thread(
                    load_ocr_artifacts,
                    user_folder=str(user_folder),
                )
            except Exception as exc:  # noqa: BLE001
                temp_store.upsert_document_status(
                    GraphDocumentStatus(
                        doc_id=doc_id,
                        status="failed",
                        last_error=str(exc),
                    )
                )
                temp_store.save_sidecars()
                logger.warning("Failed to load OCR artifacts for %s during full rebuild: %s", doc_id, exc)
                continue

            await _run_graph_extraction(
                user_id=user_id,
                doc_id=doc_id,
                markdown_text=markdown_text,
                store=temp_store,
            )

        blocking_failures = [
            status
            for status in temp_store.get_all_document_statuses()
            if status.status in {"failed", "partial"}
        ]

        if not blocking_failures:
            await _optimize_existing_graph(temp_store, regenerate_communities=True)
            temp_store.set_active_job_state(None)
            temp_store.save_sidecars()
            _replace_live_graph_files(temp_store, live_store)
            logger.info("Full graph rebuild complete for user %s", user_id)
        else:
            live_store.document_statuses = {
                status.doc_id: status for status in temp_store.get_all_document_statuses()
            }
            live_store.save_sidecars()
            logger.warning(
                "Full graph rebuild for user %s kept old graph because %s document(s) failed or were partial",
                user_id,
                len(blocking_failures),
            )

    except Exception as exc:  # noqa: BLE001
        logger.error("Full graph rebuild failed for user %s: %s", user_id, exc, exc_info=True)
        if "temp_store" in locals():
            live_store.document_statuses = {
                status.doc_id: status for status in temp_store.get_all_document_statuses()
            }
            live_store.save_sidecars()
    finally:
        live_store = GraphStore(user_id)
        live_store.set_active_job_state(None)
        live_store.save_sidecars()
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _retry_graph_document_task(user_id: str, doc_id: str) -> None:
    """Retry GraphRAG extraction for one document using a temp copy of the live graph."""
    logger.info("Starting graph retry for user %s doc %s", user_id, doc_id)
    live_store = GraphStore(user_id)
    temp_dir = _make_graph_work_dir(live_store.storage_dir.parent, f"graph-retry-{doc_id}-")

    try:
        _copy_graph_sidecars(live_store, GraphStore(user_id, storage_dir=temp_dir))
        temp_store = GraphStore(user_id, storage_dir=temp_dir)
        temp_store.remove_document(doc_id)
        temp_store.remove_document_status(doc_id)
        temp_store.save()

        row = await get_document(
            doc_id=doc_id,
            user_id=user_id,
            columns="original_path",
        )
        original_path = row.get("original_path") if row else None
        user_folder = (
            Path(original_path).resolve().parent
            if original_path
            else Path("uploads") / user_id / doc_id
        )
        markdown_text, _ = await asyncio.to_thread(
            load_ocr_artifacts,
            user_folder=str(user_folder),
        )

        result = await _run_graph_extraction(
            user_id=user_id,
            doc_id=doc_id,
            markdown_text=markdown_text,
            store=temp_store,
        )

        if result.status in {"failed", "partial"}:
            live_store.upsert_document_status(
                GraphDocumentStatus(
                    doc_id=doc_id,
                    status=result.status,
                    chunk_count=result.chunk_count,
                    chunks_succeeded=result.chunks_succeeded,
                    chunks_failed=result.chunks_failed,
                    entities_added=result.entities_added,
                    edges_added=result.edges_added,
                    last_error=result.last_error,
                )
            )
            live_store.save_sidecars()
            logger.warning("Graph retry for %s failed or was partial; preserving live graph", doc_id)
            return

        await _optimize_existing_graph(temp_store, regenerate_communities=True)
        temp_store.set_active_job_state(None)
        temp_store.save_sidecars()
        _replace_live_graph_files(temp_store, live_store)
        logger.info("Graph retry complete for user %s doc %s", user_id, doc_id)

    except Exception as exc:  # noqa: BLE001
        logger.error("Graph retry failed for user %s doc %s: %s", user_id, doc_id, exc, exc_info=True)
        live_store.upsert_document_status(
            GraphDocumentStatus(
                doc_id=doc_id,
                status="failed",
                last_error=str(exc),
            )
        )
        live_store.save_sidecars()
    finally:
        live_store = GraphStore(user_id)
        live_store.set_active_job_state(None)
        live_store.save_sidecars()
        shutil.rmtree(temp_dir, ignore_errors=True)
