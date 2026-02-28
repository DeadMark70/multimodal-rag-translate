"""
FastAPI application factory and lifecycle wiring.

Keeps app assembly separate from route/business modules for easier maintenance.
"""

# Standard library
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# Third-party
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from core.errors import (
    AppError,
    app_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)

logger = logging.getLogger(__name__)

_DEFAULT_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # React CRA
    "http://127.0.0.1:3000",
]
_ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
_EXPOSE_HEADERS = ["Content-Disposition"]


def _is_true(name: str, default: str = "false") -> bool:
    """Parse boolean-like env vars."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _should_use_fake_providers() -> bool:
    """
    Decide whether startup should skip real provider warmups.

    Rules:
    - TEST_MODE=true -> always fake (test safety)
    - USE_FAKE_PROVIDERS=true -> fake
    """
    return _is_true("TEST_MODE") or _is_true("USE_FAKE_PROVIDERS")


def _load_environment() -> None:
    """Load environment variables from config.env."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    dotenv_path = os.path.join(project_root, "config.env")
    load_dotenv(dotenv_path=dotenv_path)


def _configure_logging() -> None:
    """Configure application logging and key environment visibility."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(
        "GOOGLE_API_KEY: %s",
        "Loaded" if os.getenv("GOOGLE_API_KEY") else "Not Found",
    )
    logger.info("HF_TOKEN: %s", "Loaded" if os.getenv("HF_TOKEN") else "Not Found")
    logger.info("TEST_MODE: %s", _is_true("TEST_MODE"))
    logger.info("USE_FAKE_PROVIDERS: %s", _should_use_fake_providers())


def _get_cors_origins() -> list[str]:
    """Return CORS origins from env or secure defaults."""
    raw_origins = os.getenv("CORS_ORIGINS", "")
    if raw_origins:
        origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
        if origins:
            logger.info("CORS: Using env origins: %s", origins)
            return origins
        logger.warning("CORS_ORIGINS is set but empty after parsing; using defaults")

    logger.info("CORS: Using development origins (set CORS_ORIGINS for production)")
    return list(_DEFAULT_ORIGINS)


def _configure_cors(app: FastAPI) -> None:
    """Attach CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,  # Required for JWT auth cookies
        allow_methods=list(_ALLOWED_METHODS),
        allow_headers=["*"],  # Allow Authorization header for JWT
        expose_headers=list(_EXPOSE_HEADERS),  # For file downloads
    )


def _register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


def _register_middlewares(app: FastAPI) -> None:
    """Register middleware components."""

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next
    ) -> Response:
        request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


def _register_routers(app: FastAPI) -> None:
    """Register all API routers with existing prefixes."""
    from conversations.router import router as conversations_router
    from data_base.router import router as database_router
    from graph_rag.router import router as graph_router
    from image_service.router import router as image_router
    from multimodal_rag.router import router as multimodal_router
    from pdfserviceMD.router import router as pdfmd_router
    from stats.router import router as stats_router

    app.include_router(pdfmd_router, prefix="/pdfmd", tags=["PDF OCR & Translation"])
    app.include_router(database_router, prefix="/rag", tags=["RAG Question Answering"])
    app.include_router(image_router, prefix="/imagemd", tags=["Image Translation"])
    app.include_router(
        multimodal_router, prefix="/multimodal", tags=["Multimodal Research"]
    )
    app.include_router(stats_router, prefix="/stats", tags=["Dashboard Statistics"])
    app.include_router(graph_router, prefix="/graph", tags=["Knowledge Graph"])
    app.include_router(
        conversations_router, prefix="/api/conversations", tags=["Conversations"]
    )


def _ensure_base_directories() -> None:
    """Ensure required runtime directories exist."""
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output/imgs", exist_ok=True)
    logger.info("Base directories verified")


def _initialize_external_clients(app: FastAPI) -> None:
    """Initialize external clients needed by API routes."""
    from supabase_client import init_supabase

    client = init_supabase()
    app.state.supabase = client
    if client:
        logger.info("Supabase client ready")
    else:
        logger.warning("Supabase unavailable; database-backed features are limited")


async def _initialize_rag_components() -> None:
    """Initialize embedding and LLM resources used by RAG services."""
    if _should_use_fake_providers():
        logger.info("Skipping RAG startup warmup in fake/test mode")
        return

    from data_base.router import on_startup_rag_init

    logger.info("Initializing RAG components...")
    await on_startup_rag_init()


async def _warm_up_pdf_ocr() -> None:
    """Pre-load PDF OCR model on GPU."""
    if _should_use_fake_providers():
        logger.info("Skipping PDF OCR warmup in fake/test mode")
        return

    from pdfserviceMD.PDF_OCR_services import initialize_predictor

    logger.info("Pre-loading PDF OCR model on GPU...")
    try:
        await run_in_threadpool(initialize_predictor)
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF OCR initialization failed (non-fatal): %s", exc)


@asynccontextmanager
async def app_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan hook for startup initialization."""
    logger.info("=== Application Startup ===")
    _ensure_base_directories()
    _initialize_external_clients(_)
    await _initialize_rag_components()
    await _warm_up_pdf_ocr()
    logger.info("=== All components ready ===")
    yield


async def read_root() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "Welcome to the PDF Translation & Per-User RAG QA API."}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    _load_environment()
    _configure_logging()

    app = FastAPI(
        title="PDF Translation & RAG API",
        description="PDF OCR, Translation, and Multimodal RAG services",
        version="2.1.0",
        lifespan=app_lifespan,
    )
    _configure_cors(app)
    _register_middlewares(app)
    _register_error_handlers(app)
    _register_routers(app)
    app.add_api_route("/", read_root, methods=["GET"])
    return app
