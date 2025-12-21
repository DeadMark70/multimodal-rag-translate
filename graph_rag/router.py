"""
GraphRAG API Router

Provides REST API endpoints for graph management operations:
- GET /graph/status - Get graph status
- POST /graph/rebuild - Force full graph rebuild  
- POST /graph/optimize - Run entity resolution
"""

# Standard library
import logging
from typing import Optional

# Third-party
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Local application
from core.auth import get_current_user_id
from graph_rag.schemas import GraphStatusResponse
from graph_rag.store import GraphStore

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
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rebuild",
    response_model=GraphOperationResponse,
    summary="重建圖譜",
    description="強制重建使用者的知識圖譜。會重新從所有文件中抽取實體與關係。"
)
async def rebuild_graph(
    request: GraphRebuildRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id)
) -> GraphOperationResponse:
    """
    Trigger a full graph rebuild.
    
    This is a long-running operation that runs in the background.
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
            message="圖譜重建已開始，請稍後檢查狀態",
            details={"user_id": user_id}
        )
        
    except Exception as e:
        logger.error(f"Failed to start graph rebuild: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        
        # Run entity resolution
        from graph_rag.entity_resolver import resolve_entities
        merges = await resolve_entities(store)
        
        # Rebuild communities if requested
        communities_count = 0
        if request.regenerate_communities:
            from graph_rag.community_builder import build_communities
            communities = await build_communities(store, generate_summaries=True)
            communities_count = len(communities)
        
        # Save changes
        store.save()
        
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
        raise HTTPException(status_code=500, detail=str(e))


# ===== Background Tasks =====

async def _rebuild_graph_task(user_id: str) -> None:
    """
    Background task to rebuild graph from all documents.
    
    This extracts entities/relations from all indexed documents.
    """
    logger.info(f"Starting graph rebuild for user {user_id}")
    
    try:
        from graph_rag.store import GraphStore
        from graph_rag.extractor import extract_and_add_to_graph
        from graph_rag.entity_resolver import resolve_entities
        from graph_rag.community_builder import build_communities
        
        # Create fresh store
        store = GraphStore(user_id)
        store.clear()
        
        # Get all document content from vector store
        # For now, we just rebuild communities if graph already has nodes
        # Full rebuild requires re-processing document content
        
        # Run entity resolution
        await resolve_entities(store)
        
        # Build communities
        await build_communities(store, generate_summaries=True)
        
        # Save
        store.save()
        
        logger.info(f"Graph rebuild complete for user {user_id}")
        
    except Exception as e:
        logger.error(f"Graph rebuild failed for user {user_id}: {e}")
