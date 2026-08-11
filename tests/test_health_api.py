"""Health endpoint and readiness lifecycle tests."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.health import health_router, set_readiness


def _client(*, ready: bool) -> TestClient:
    app = FastAPI()
    app.state.ready = ready
    app.include_router(health_router)
    return TestClient(app)


def test_live_is_independent_of_readiness() -> None:
    """The liveness probe remains available during startup."""
    response = _client(ready=False).get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_ready_returns_503_until_lifespan_is_ready() -> None:
    """The readiness probe rejects traffic before startup completes."""
    response = _client(ready=False).get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_set_readiness_changes_ready_response() -> None:
    """Updating app readiness is reflected by the readiness probe."""
    app = FastAPI()
    app.state.ready = False
    app.include_router(health_router)

    set_readiness(app, True)
    response = TestClient(app).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_create_app_initializes_readiness_to_false() -> None:
    """A newly assembled application is not ready before lifespan startup."""
    from core.app_factory import create_app

    assert create_app().state.ready is False


def test_ready_is_200_while_lifespan_is_active() -> None:
    """Startup completion makes the real application's readiness probe available."""
    from main import app

    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        TestClient(app) as client,
    ):
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_stays_200_when_ocr_warmup_handles_its_own_failure() -> None:
    """A normally returning non-fatal OCR warmup does not block readiness."""
    from main import app

    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock(return_value=None)),
        TestClient(app) as client,
    ):
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_lifespan_fails_when_supabase_client_is_unavailable() -> None:
    """A missing critical database client prevents startup readiness."""
    from core import app_factory

    worker = type(
        "FakeWorker",
        (),
        {"is_configured": False, "start": AsyncMock(), "stop": AsyncMock()},
    )()
    engine = type(
        "FakeCampaignEngine",
        (),
        {"_worker": None, "recover_inflight_campaigns": AsyncMock()},
    )()
    app = FastAPI()

    with (
        patch("supabase_client.init_supabase", return_value=None),
        patch("evaluation.db.force_init_db", new=AsyncMock()),
        patch("evaluation.campaign_engine.get_campaign_engine", return_value=engine),
        patch("evaluation.job_worker.get_evaluation_job_worker", return_value=worker),
        patch.object(app_factory, "_ensure_base_directories"),
        patch.object(app_factory, "_initialize_rag_components", new=AsyncMock()),
        patch.object(app_factory, "_warm_up_pdf_ocr", new=AsyncMock()),
        pytest.raises(RuntimeError, match="Critical dependency initialization failed"),
    ):
        async with app_factory.app_lifespan(app):
            pass

    assert app.state.ready is False
