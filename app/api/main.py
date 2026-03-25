"""
app/api/main.py

FastAPI application entry point.
Provides REST API endpoints for Medical RAG system.
"""

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    HTTPException,
    APIRouter,
)
from fastapi.responses import FileResponse
from pathlib import Path
from contextlib import asynccontextmanager

# Database lifecycle management
from app.core.database import init_db_pool, close_db_pool, get_db_pool

# Agent modules
from app.agent.patient_detection import detect_patient
from app.agent.sql_interpreter import SQLInterpreter
from app.agent.sql_tool import run_sql_query

# RAG modules
from app.rag.pdf_ingestor import ingest_pdf
from app.rag.rag_pipeline import RAGPipeline

# Report generator
from app.agent.report_generator import MedicalReportGenerator

# Singleton instances
from app.rag.singletons import ollama_client

# API router
from app.api.router import router as sql_router

import logging

logger = logging.getLogger("medical_rag")

# =====================================================
# LIFECYCLE MANAGEMENT
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes DB pool on startup, cleans up on shutdown.
    """
    # Startup: Initialize database pool
    await init_db_pool()
    logger.info("Database pool initialized")
    try:
        pool = get_db_pool()
        logger.info(f"Database pool ready: {pool}")
    except Exception as e:
        logger.error(f"Database pool check failed: {e}")
    yield
    # Shutdown: Cleanup resources
    await close_db_pool()
    await ollama_client.close()


# Create FastAPI application
app = FastAPI(title="Medical RAG Backend", lifespan=lifespan)

# Create API router with prefix
api_router = APIRouter(prefix="/api")

# Report status storage (in-memory for background tasks)
report_status = {}

# Report output directory
REPORTS_DIR = Path("./data/reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)


# Request/Response models
from pydantic import BaseModel


class DetectRequest(BaseModel):
    text: str


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# =====================================================
# PATIENT DETECTION
# =====================================================
@api_router.post("/detect_patient/")
async def detect_patient_endpoint(request: DetectRequest):
    """
    Detect patient from text input.
    Supports ID (NCH-XXXXX) or name-based detection.
    """
    from app.agent.patient_detection import detect_patient_async
    return await detect_patient_async(request.text)


# =====================================================
# RAG CHAT
# =====================================================
@api_router.post("/chat/")
async def chat_endpoint(
    query: str = Form(...),
    patient_id: str | None = Form(None),
):
    """
    Query the RAG system with a question.
    
    Args:
        query: User question about medical documents
        patient_id: Optional patient ID to scope search
        
    Returns:
        Dict with answer and citations
    """
    rag_pipeline = RAGPipeline()
    response = await rag_pipeline.query_async(query, patient_id)
    return response


# =====================================================
# PDF UPLOAD
# =====================================================
@api_router.post("/upload_pdf/")
async def upload_pdf_endpoint(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
):
    """
    Upload and process a PDF medical document.
    
    Pipeline:
        1. Validate patient ID format (NCH-XXXXX)
        2. Validate file type (must be PDF)
        3. Extract text and metadata
        4. Add to vector store
    
    Args:
        file: PDF file to upload
        patient_id: Patient identifier
        
    Returns:
        Status message
    """
    patient_id = patient_id.upper()

    # Validate patient ID format
    if not patient_id.startswith("NCH-"):
        raise HTTPException(400, "Invalid patient ID")

    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files allowed")

    # Process PDF
    await ingest_pdf(await file.read(), patient_id)
    return {"status": "success"}


# =====================================================
# REPORT GENERATION
# =====================================================
@api_router.post("/report/{patient_id}")
async def generate_report(
    patient_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Generate PDF report for a patient asynchronously.
    
    Uses background task to:
        1. Fetch patient data from database
        2. Generate clinical summary from RAG
        3. Build PDF with reportlab
    
    Args:
        patient_id: Patient identifier
        background_tasks: FastAPI background tasks
        
    Returns:
        Status: "started"
    """
    patient_id = patient_id.upper()

    if not patient_id.startswith("NCH-"):
        raise HTTPException(400, "Invalid patient ID")

    # Track report status
    report_status[patient_id] = "pending"

    async def task():
        """Background task to generate report."""
        logger.info(f"Background task started for {patient_id}")
        try:
            # Ensure DB pool is initialized
            from app.core.database import get_db_pool
            from app.core.database import init_db_pool
            try:
                get_db_pool()
                logger.info("DB pool already initialized")
            except RuntimeError:
                logger.info("Initializing database pool in background task...")
                await init_db_pool()
                logger.info("DB pool initialized")
            
            # Generate report
            generator = MedicalReportGenerator()
            await generator.generate(patient_id)
            report_status[patient_id] = "completed"
            logger.info(f"Report generation completed for {patient_id}")
        except Exception as e:
            logger.error(f"Report generation failed for {patient_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            report_status[patient_id] = "failed"

    background_tasks.add_task(task)
    return {"status": "started"}


@api_router.get("/report/status/{patient_id}")
async def report_status_endpoint(patient_id: str):
    """
    Get status of report generation.
    
    Returns:
        Status: "pending", "completed", "failed", or "not_found"
    """
    patient_id = patient_id.upper()
    return {"status": report_status.get(patient_id, "not_found")}


@api_router.get("/report/download/{patient_id}")
async def download_report(patient_id: str):
    """
    Download generated PDF report.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        PDF file response
    """
    patient_id = patient_id.upper()
    file = REPORTS_DIR / f"{patient_id}_report.pdf"

    if not file.exists():
        raise HTTPException(404, "Report not found")

    return FileResponse(file, filename=file.name)


# Register routers
app.include_router(api_router)
app.include_router(sql_router)
