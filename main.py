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

app = FastAPI(
    title="PDF Translation & RAG API",
    description="PDF OCR, Translation, and Multimodal RAG services",
    version="2.0.0"
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


@app.get("/")
async def read_root() -> dict:
    """Health check endpoint."""
    return {"message": "Welcome to the PDF Translation & Per-User RAG QA API."}