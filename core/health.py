"""Liveness and readiness endpoints."""

from typing import Literal

from fastapi import APIRouter, FastAPI, Request, status
from pydantic import BaseModel
from starlette.responses import JSONResponse


class HealthResponse(BaseModel):
    """Response returned by health probes."""

    status: Literal["live", "ready", "not_ready"]


health_router = APIRouter(prefix="/health", tags=["Health"])


def set_readiness(app: FastAPI, ready: bool) -> None:
    """Record whether the application has completed startup."""
    app.state.ready = ready


@health_router.get("/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    """Report that the process can serve liveness checks."""
    return HealthResponse(status="live")


@health_router.get("/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse | JSONResponse:
    """Report whether application startup has completed."""
    if bool(getattr(request.app.state, "ready", False)):
        return HealthResponse(status="ready")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "not_ready"},
    )
