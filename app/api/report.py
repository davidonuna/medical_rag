# app/api/report.py

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.agent.report_generator import MedicalReportGenerator

router = APIRouter(prefix="/api", tags=["reports"])

REPORT_DIR = Path("data/reports")
report_generator = MedicalReportGenerator()


@router.post("/report/{patient_id}")
async def start_report(
    patient_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Start report generation in the background.
    Immediately returns without blocking.
    """
    background_tasks.add_task(
        report_generator.generate,
        patient_id,
    )
    return {
        "status": "processing",
        "patient_id": patient_id,
    }


@router.get("/report/status/{patient_id}")
async def report_status(patient_id: str):
    """
    Poll report generation status.
    """
    path = REPORT_DIR / f"{patient_id}_report.pdf"
    if path.exists():
        return {"status": "completed"}
    return {"status": "processing"}


@router.get("/report/download/{patient_id}")
async def download_report(patient_id: str):
    """
    Download completed report.
    """
    path = REPORT_DIR / f"{patient_id}_report.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not ready")

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
    )
