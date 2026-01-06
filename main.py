"""
Main Application Entry Point

FastAPI application with PDF OCR, RAG, and Multimodal services.
"""

# Standard library
import logging
import os

# Third-party
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables first
dotenv_path = os.path.join(os.path.dirname(__file__), 'config.env')
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log environment status
logger.info(f"GOOGLE_API_KEY: {'Loaded' if os.getenv('GOOGLE_API_KEY') else 'Not Found'}")
logger.info(f"HF_TOKEN: {'Loaded' if os.getenv('HF_TOKEN') else 'Not Found'}")

# Local application imports
from pdfserviceMD.router import router as pdfmd_router
from pdfserviceMD.PDF_OCR_services import initialize_predictor as init_pdf_ocr
from data_base.router import on_startup_rag_init, router as database_router
from image_service.router import router as image_router
from multimodal_rag.router import router as multimodal_router
from stats.router import router as stats_router
from graph_rag.router import router as graph_router
from conversations.router import router as conversations_router

app = FastAPI(
    title="PDF Translation & RAG API",
    description="PDF OCR, Translation, and Multimodal RAG services",
    version="2.1.0"
)

# --- CORS Configuration ---
# Security: Use explicit origin whitelist, never "*" in production
# For production, set CORS_ORIGINS env var (comma-separated)
_DEFAULT_ORIGINS = [
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",      # React CRA
    "http://127.0.0.1:3000",
]

_cors_origins_env = os.getenv("CORS_ORIGINS", "")
if _cors_origins_env:
    # Production: use env var (comma-separated origins)
    CORS_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",")]
    logger.info(f"CORS: Using env origins: {CORS_ORIGINS}")
else:
    # Development: use defaults
    CORS_ORIGINS = _DEFAULT_ORIGINS
    logger.info(f"CORS: Using development origins (set CORS_ORIGINS for production)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,  # Required for JWT auth cookies
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],     # Allow Authorization header for JWT
    expose_headers=["Content-Disposition"],  # For file downloads
)


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    
    Initializes:
    1. RAG components (Embedding model, LLM)
    2. PDF OCR predictor (GPU warm-up)
    """
    logger.info("=== Application Startup ===")
    
    # 0. Ensure required directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output/imgs", exist_ok=True)
    logger.info("Base directories verified")
    
    # 1. Initialize RAG components (Embedding model + LLM)
    logger.info("Initializing RAG components...")
    await on_startup_rag_init()
    
    # 2. Pre-load PDF OCR model on GPU (warm-up)
    # Use threadpool because model loading is synchronous and heavy
    logger.info("Pre-loading PDF OCR model on GPU...")
    try:
        await run_in_threadpool(init_pdf_ocr)
    except Exception as e:
        logger.error(f"PDF OCR initialization failed (non-fatal): {e}")
    
    logger.info("=== All components ready ===")


# Register routers
app.include_router(pdfmd_router, prefix="/pdfmd", tags=["PDF OCR & Translation"])
app.include_router(database_router, prefix="/rag", tags=["RAG Question Answering"])
app.include_router(image_router, prefix="/imagemd", tags=["Image Translation"])
app.include_router(multimodal_router, prefix="/multimodal", tags=["Multimodal Research"])
app.include_router(stats_router, prefix="/stats", tags=["Dashboard Statistics"])
app.include_router(graph_router, prefix="/graph", tags=["Knowledge Graph"])
app.include_router(conversations_router, prefix="/api/conversations", tags=["Conversations"])


@app.get("/")
async def read_root() -> dict:
    """Health check endpoint."""
    return {"message": "Welcome to the PDF Translation & Per-User RAG QA API."}