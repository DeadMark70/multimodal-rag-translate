"""
GraphRAG API Router

Provides REST API endpoints for graph management operations:
- GET /graph/status - Get graph status
- GET /graph/data - Get visualization data for react-force-graph
- POST /graph/rebuild - Re-run graph optimization safely without re-extraction
- POST /graph/optimize - Run entity resolution
"""

# Standard library
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import List, Optional

# Third-party
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel, Field

# Local application
from core.auth import get_current_user_id
from core.llm_factory import ExtractionProfile
from core.errors import AppError, ErrorCode
from evaluation.db import CampaignRepository
from graph_rag.maintenance import (
    list_graph_source_documents as _list_graph_source_documents,
    node_vector_sync_task as _node_vector_sync_task,
    optimize_existing_graph as _optimize_existing_graph,
    purge_graph_document_task as _purge_graph_document_task,
    rebuild_full_graph_task as _rebuild_full_graph_task,
    rebuild_graph_task as _rebuild_graph_task,
    retry_graph_document_task as _retry_graph_document_task,
)
from graph_rag.rebuild_jobs import GraphRebuildJobStore
from graph_rag.schemas import (
    GraphDebugSearchRequest,
    GraphDebugSearchResponse,
    GraphDocumentStatus,
    GraphDocumentStatusItem,
    GraphDocumentStatusListResponse,
    NodeVectorSyncStatusResponse,
    GraphStatusResponse,
    GraphQualityResponse,
    GraphRuntimeQualityResponse,
    GraphRebuildManifest,
    GraphRebuildStatusResponse,
)
from graph_rag.debug import run_debug_search
from graph_rag.quality import compute_campaign_runtime_quality, compute_graph_quality
from graph_rag.store import GraphStore
from pdfserviceMD.repository import get_document

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


# ===== Request/Response Models =====


class GraphRebuildRequest(BaseModel):
    """Request for graph rebuild."""

    force: bool = Field(default=False, description="強制重建即使圖譜沒有變更")


class GraphOptimizeRequest(BaseModel):
    """Request for graph optimization."""

    regenerate_communities: bool = Field(default=True, description="重新生成社群摘要")


class GraphDocumentRetryRequest(BaseModel):
    """Requested extraction quality for one-document GraphRAG retry."""

    extraction_profile: ExtractionProfile = "standard"


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


async def _build_graph_document_rows(
    user_id: str, store: GraphStore
) -> list[GraphDocumentStatusItem]:
    """Build API rows by joining persisted graph statuses with OCR-eligible docs."""
    sources = await _list_graph_source_documents(user_id)
    source_map = {item["doc_id"]: item for item in sources}
    graph_doc_ids = store.get_documents()
    known_doc_ids = set(source_map) | set(store.document_statuses) | graph_doc_ids
    rows: list[GraphDocumentStatusItem] = []

    for doc_id in sorted(known_doc_ids):
        persisted = store.get_document_status(doc_id)
        extraction_manifest = store.get_latest_extraction_manifest(doc_id)
        source = source_map.get(doc_id, {})
        fallback_status = "indexed" if doc_id in graph_doc_ids else "skipped"
        status = persisted or GraphDocumentStatus(doc_id=doc_id, status=fallback_status)
        rows.append(
            GraphDocumentStatusItem(
                **status.model_dump(),
                file_name=source.get("file_name"),
                is_eligible=doc_id in source_map,
                extraction_model=(
                    extraction_manifest.extractor_model if extraction_manifest else None
                ),
                extraction_thinking_level=(
                    extraction_manifest.thinking_level if extraction_manifest else None
                ),
                extraction_profile=(
                    extraction_manifest.extraction_profile if extraction_manifest else None
                ),
                extraction_prompt_version=(
                    extraction_manifest.prompt_version if extraction_manifest else None
                ),
                extraction_recorded_at=(
                    extraction_manifest.created_at if extraction_manifest else None
                ),
            )
        )
    return rows


# ===== Endpoints =====


@router.get(
    "/quality",
    response_model=GraphQualityResponse,
    summary="取得圖譜品質指標",
    description="回傳目前使用者圖譜的靜態品質與可處理問題。",
)
async def get_graph_quality(
    user_id: str = Depends(get_current_user_id),
) -> GraphQualityResponse:
    """Return static quality metrics for the current user's graph store."""
    return compute_graph_quality(GraphStore(user_id))


@router.get(
    "/runtime-quality",
    response_model=GraphRuntimeQualityResponse,
    summary="取得圖譜執行期品質指標",
    description="從 evaluation observability 彙整指定 campaign 的 GraphRAG 執行期品質。",
)
async def get_graph_runtime_quality(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
) -> GraphRuntimeQualityResponse:
    """Return campaign runtime quality from persisted observability rows."""
    await CampaignRepository().get(user_id=user_id, campaign_id=campaign_id)
    return await compute_campaign_runtime_quality(campaign_id)


@router.post(
    "/debug/search",
    response_model=GraphDebugSearchResponse,
    summary="除錯 GraphRAG 查詢",
    description="回傳實體連結、graph hints、候選證據與最終 context 資格。",
)
async def debug_graph_search(
    request: GraphDebugSearchRequest,
    user_id: str = Depends(get_current_user_id),
) -> GraphDebugSearchResponse:
    """Run a user-scoped evidence-locator diagnostic without exposing unsafe context."""
    return await run_debug_search(
        user_id=user_id,
        query=request.query,
        search_mode=request.search_mode,
    )


@router.get(
    "/status",
    response_model=GraphStatusResponse,
    summary="取得圖譜狀態",
    description="取得使用者知識圖譜的狀態資訊，包括節點數、邊數、社群數等。",
)
async def get_graph_status(
    user_id: str = Depends(get_current_user_id),
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
    user_id: str = Depends(get_current_user_id),
) -> GraphDocumentStatusListResponse:
    """List per-document GraphRAG extraction status rows."""
    try:
        store = GraphStore(user_id)
        documents = await _build_graph_document_rows(user_id, store)
        return GraphDocumentStatusListResponse(
            documents=documents, total=len(documents)
        )
    except Exception as e:
        logger.error("Failed to list graph documents: %s", e, exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to list graph documents",
            status_code=500,
        ) from e


@router.post(
    "/node-vector/sync",
    response_model=GraphOperationResponse,
    summary="手動同步 Graph node-vector index",
    description="背景同步現有圖譜節點嵌入，支援舊圖譜補齊 node-vector sidecars。",
)
async def start_node_vector_sync(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> GraphOperationResponse:
    """Start background manual node-vector sync job."""
    try:
        store = GraphStore(user_id)
        if store.active_job_state:
            return GraphOperationResponse(
                status="skipped",
                message=f"已有圖譜工作執行中：{store.active_job_state}",
            )

        now = datetime.now(timezone.utc)
        total_nodes = store.get_status().node_count
        store.set_active_job_state("node_vector_sync")
        store.set_node_vector_sync_status(
            state="running",
            processed=0,
            total=total_nodes,
            changed=0,
            reused=0,
            removed=0,
            index_state="running",
            autosync_duration_ms=None,
            last_error=None,
            started_at=now,
            updated_at=now,
            finished_at=None,
        )
        store.save_sidecars()
        background_tasks.add_task(_node_vector_sync_task, user_id)
        return GraphOperationResponse(
            status="started",
            message="節點嵌入同步已啟動，請稍候查看進度",
            details={"total_nodes": total_nodes},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to start node-vector sync for user %s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to start node-vector sync",
            status_code=500,
        ) from exc


@router.get(
    "/node-vector/sync/status",
    response_model=NodeVectorSyncStatusResponse,
    summary="取得 node-vector 同步狀態",
    description="回傳手動節點嵌入同步狀態與進度統計。",
)
async def get_node_vector_sync_status(
    user_id: str = Depends(get_current_user_id),
) -> NodeVectorSyncStatusResponse:
    """Return latest manual node-vector sync status."""
    try:
        store = GraphStore(user_id)
        return store.get_node_vector_sync_status()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to fetch node-vector sync status for user %s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to get node-vector sync status",
            status_code=500,
        ) from exc


@router.get(
    "/data",
    response_model=GraphVisualizationData,
    summary="取得視覺化資料",
    description="取得 react-force-graph 格式的圖譜資料，用於前端視覺化。",
)
async def get_graph_visualization_data(
    user_id: str = Depends(get_current_user_id),
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
            nodes.append(
                VisNode(
                    id=node.label,
                    group=hash(node.entity_type.value) % 5,
                    val=len(node.doc_ids) * 2,
                    desc=node.description or node.entity_type.value,
                )
            )

        # Convert edges to links
        links = []
        for edge in store.get_all_edges():
            source_node = store.get_node(edge.source_id)
            target_node = store.get_node(edge.target_id)
            if source_node and target_node:
                links.append(
                    VisLink(
                        source=source_node.label,
                        target=target_node.label,
                        label=edge.relation,
                    )
                )

        logger.info(
            f"Graph visualization for user {user_id}: {len(nodes)} nodes, {len(links)} links"
        )
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
    user_id: str = Depends(get_current_user_id),
) -> GraphOperationResponse:
    """
    Re-run graph optimization pipeline in background without clearing graph data.
    """
    try:
        store = GraphStore(user_id)

        if store.active_job_state:
            return GraphOperationResponse(
                status="skipped",
                message=f"已有圖譜工作執行中：{store.active_job_state}",
            )

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
            details={"user_id": user_id},
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
    response_model=GraphRebuildStatusResponse,
    summary="完整重構圖譜",
    description="從所有 OCR artifact 重新抽取並建立全新圖譜；成功後才會覆蓋目前圖譜。",
)
async def rebuild_graph_full(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> GraphRebuildStatusResponse:
    """Build a fresh graph from all OCR-complete document artifacts."""
    try:
        jobs = GraphRebuildJobStore(user_id)
        sources = await _list_graph_source_documents(user_id)
        if not sources:
            raise AppError(
                code=ErrorCode.VALIDATION_ERROR,
                message="沒有可用的 OCR 文件可供完整重構",
                status_code=400,
            )
        manifest, created = jobs.create_or_load_active(sources)
        if not created:
            return jobs.to_status(manifest)
        return _schedule_full_rebuild(background_tasks, user_id, jobs, manifest)
    except AppError:
        raise
    except Exception as e:
        logger.error("Failed to start full graph rebuild: %s", e, exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to start full graph rebuild",
            status_code=500,
        ) from e


@router.get(
    "/rebuild-full/status",
    response_model=GraphRebuildStatusResponse | None,
    summary="取得完整圖譜重構進度",
)
async def get_rebuild_graph_full_status(
    user_id: str = Depends(get_current_user_id),
) -> GraphRebuildStatusResponse | None:
    """Return the durable job projection without scheduling provider work."""
    jobs = GraphRebuildJobStore(user_id)
    manifest = jobs.load_current()
    if manifest is not None:
        manifest = jobs.reconcile_status(manifest)
    return jobs.to_status(manifest) if manifest is not None else None


@router.post(
    "/rebuild-full/resume",
    response_model=GraphRebuildStatusResponse,
    summary="繼續完整圖譜重構",
)
async def resume_rebuild_graph_full(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> GraphRebuildStatusResponse:
    """Resume an interrupted job or retry only failed document checkpoints."""
    jobs = GraphRebuildJobStore(user_id)
    manifest = jobs.load_current()
    if manifest is not None:
        manifest = jobs.reconcile_status(manifest)
    if manifest is None:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="沒有可繼續的完整圖譜重構工作",
            status_code=404,
        )
    if manifest.state == "completed_with_failures":
        manifest = jobs.reset_failed_documents(manifest)
        jobs.save(manifest)
    elif manifest.state == "running":
        return jobs.to_status(manifest)
    elif manifest.state != "interrupted":
        raise AppError(
            code=ErrorCode.VALIDATION_ERROR,
            message="目前完整圖譜重構工作無法繼續",
            status_code=409,
        )
    return _schedule_full_rebuild(background_tasks, user_id, jobs, manifest)


def _schedule_full_rebuild(
    background_tasks: BackgroundTasks,
    user_id: str,
    jobs: GraphRebuildJobStore,
    manifest: GraphRebuildManifest,
) -> GraphRebuildStatusResponse:
    """Claim one durable runner, preserve legacy maintenance exclusion, and schedule it."""
    live_store = GraphStore(user_id)
    if live_store.active_job_state not in {None, "rebuild_full"}:
        raise AppError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"已有圖譜工作執行中：{live_store.active_job_state}",
            status_code=409,
        )
    owner_token = jobs.acquire_lease(manifest.job_id)
    if owner_token is None:
        return jobs.to_status(jobs.load(manifest.job_id))
    manifest = jobs.load(manifest.job_id)
    manifest.state = "running"
    jobs.save(manifest)
    live_store.set_active_job_state("rebuild_full")
    live_store.save_sidecars()
    background_tasks.add_task(_rebuild_full_graph_task, user_id, manifest.job_id, owner_token)
    return jobs.to_status(manifest)


@router.post(
    "/documents/{doc_id}/retry",
    response_model=GraphOperationResponse,
    summary="重試單一文件 GraphRAG",
    description="只對指定文件重新抽取 GraphRAG；成功後會刷新整體社群。",
)
async def retry_graph_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    request: GraphDocumentRetryRequest | None = None,
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
        extraction_profile = (request or GraphDocumentRetryRequest()).extraction_profile
        background_tasks.add_task(
            _retry_graph_document_task,
            user_id,
            doc_id,
            extraction_profile,
        )
        return GraphOperationResponse(
            status="started",
            message="單一文件 GraphRAG 重試已開始",
            details={
                "doc_id": doc_id,
                "file_name": row.get("file_name"),
                "extraction_profile": extraction_profile,
            },
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


@router.delete(
    "/documents/{doc_id}",
    response_model=GraphOperationResponse,
    summary="移除單一文件圖譜殘留",
    description="從目前圖譜中移除指定文件的節點/邊與文件狀態；適用於已刪除文件的 orphan graph cleanup。",
)
async def purge_graph_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> GraphOperationResponse:
    """Purge one document's remaining GraphRAG contribution from the live graph."""
    try:
        store = GraphStore(user_id)
        if store.active_job_state:
            return GraphOperationResponse(
                status="skipped",
                message=f"已有圖譜工作執行中：{store.active_job_state}",
            )

        if (
            not store.get_document_status(doc_id)
            and doc_id not in store.get_documents()
        ):
            return GraphOperationResponse(
                status="skipped",
                message="找不到可移除的圖譜殘留",
            )

        store.set_active_job_state(f"purge:{doc_id}")
        store.save_sidecars()
        background_tasks.add_task(_purge_graph_document_task, user_id, doc_id)
        return GraphOperationResponse(
            status="started",
            message="文件圖譜殘留移除已開始",
            details={"doc_id": doc_id},
        )
    except Exception as e:
        logger.error("Failed to start graph purge for %s: %s", doc_id, e, exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to start graph document purge",
            status_code=500,
        ) from e


@router.post(
    "/optimize",
    response_model=GraphOperationResponse,
    summary="優化圖譜",
    description="執行實體融合與社群重建，合併相似的實體節點。",
)
async def optimize_graph(
    request: GraphOptimizeRequest, user_id: str = Depends(get_current_user_id)
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
            },
        )

    except Exception as e:
        logger.error(f"Failed to optimize graph: {e}")
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to optimize graph",
            status_code=500,
        ) from e


# ===== Background Tasks =====
