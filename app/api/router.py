from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os

from app.core.config import get_settings
from app.agent.sql_interpreter import SQLInterpreter, query_cache, sql_error_handler
from app.agent.sql_tool import run_sql_query
from app.agent.report_generator import MedicalReportGenerator
from app.agent.report_agent import ReportAgent  # New LangGraph report agent
from app.agent.sql_agent import SQLAgent
from app.rag.pdf_ingestor import ingest_pdf

router = APIRouter(prefix="/api")
settings = get_settings()

# -----------------------------
# Long-lived helpers
# -----------------------------
sql_interpreter = SQLInterpreter(model=settings.OLLAMA_MODEL)
sql_agent = SQLAgent()  # LLM-powered SQL agent
report_agent = ReportAgent()  # LangGraph-based report agent

# -------------------------------------------------
# MODELS
# -------------------------------------------------
class SQLQueryRequest(BaseModel):
    nl_query: str
    list_patients: Optional[bool] = False

class SQLResponse(BaseModel):
    sql: Optional[str] = None
    result: List[Dict]
    execution_time: Optional[float] = None
    cache_hit: Optional[bool] = None
    error: Optional[Dict] = None

class DetectRequest(BaseModel):
    text: str

class DetectResponse(BaseModel):
    patient_id: Optional[str] = None
    confidence: Optional[float] = None
    suggestions: Optional[List[Dict]] = []
    source: Optional[str] = None

# -------------------------------------------------
# SQL INTERPRETER (Legacy) - Uses hardcoded patterns
# -------------------------------------------------
@router.post("/sql_query/legacy/", response_model=SQLResponse, tags=["SQL"])
async def generate_sql_legacy(request: SQLQueryRequest):
    """
    Legacy SQL interpreter using hardcoded patterns.
    Use /sql_query/ for the new LLM-based agent.
    """
    try:
        q_lower = request.nl_query.lower()
        is_data_quality = any(x in q_lower for x in [
            "inconsisten", "duplicate", "missing", "without a", "without an",
            "data entry error", "data quality", "abnormal gender",
            "consistent with", "recorded without", "without any visit",
            "without diagnosis", "without physician", "without payer",
            "timestamp", "consistency"
        ])
        
        analytical_keywords = ["top", "most common", "highest", "lowest", "trend", "growth",
                              "proportion", "percentage", "distribution", "demographic",
                              "quarter", "month", "year", "specialty", "physician", "payer",
                              "diagnosis", "diagnoses", "visit count", "patient count",
                              "recurrence", "recurrent", "high-risk", "high risk", "thresholds"]
        
        has_analytical_kw = any(kw in q_lower for kw in analytical_keywords)
        
        # Auto-detect list queries (no need for flag)
        is_list_query = (
            not is_data_quality and 
            not has_analytical_kw and 
            any(kw in q_lower for kw in ["list ", "show me", "display", "get all", "find patients", "show patients", "list all"])
        )

        if is_list_query:
            sql = sql_interpreter.list_patients_for_diagnosis(request.nl_query)
            if not sql:
                raise HTTPException(status_code=500, detail="SQL generation failed")
            has_params = '$' in sql
            if has_params:
                import re
                q_lower = request.nl_query.lower()
                # Try 'diagnosed with X' pattern first
                diagnosis_match = re.search(
                    r'diagnosed\s+with\s+([a-zA-Z0-9\s\-]+?)(?:\s+in|\s+for|\s+on|\s+last|\s+this|\s+\d|\?|$)', 
                    q_lower
                )
                if not diagnosis_match:
                    # Try 'with X' pattern - skip 'patients' to get diagnosis
                    diagnosis_match = re.search(
                        r'with\s+([a-zA-Z0-9\s\-]+?)(?:\s+in|\s+for|\s+on|\s+last|\s+this|\s+\d|\?|$)', 
                        q_lower
                    )
                    if diagnosis_match:
                        captured = diagnosis_match.group(1).strip().lower()
                        skip_words = {"patients", "people", "individuals", "children", "the", "all", "any", "new"}
                        if captured in skip_words or len(captured) < 2:
                            diagnosis_match = None
                if not diagnosis_match:
                    # Fall back to 'had X' or 'suffering from X'
                    diagnosis_match = re.search(
                        r'(?:had|suffering)\s+(?:from\s+)?([a-zA-Z0-9\s\-]+?)(?:\s+in|\s+on|\s+for|\s+last|\s+this|\s+\d|\?|$)', 
                        q_lower
                    )
                if diagnosis_match:
                    diagnosis = diagnosis_match.group(1).strip()
                    diagnosis = diagnosis.rstrip('?').strip()
                    params = (f"%{diagnosis}%",)
                    result = await run_sql_query(sql, params)
                else:
                    result = await run_sql_query(sql)
            else:
                result = await run_sql_query(sql)
            return {"sql": sql, "result": result, "execution_time": None, "cache_hit": False}
        else:
            interpreter_result = await sql_interpreter.interpret(request.nl_query)
            
            if "error" in interpreter_result:
                return {
                    "sql": None,
                    "result": [],
                    "execution_time": None,
                    "cache_hit": False,
                    "error": interpreter_result["error"]
                }
            
            sql = interpreter_result["sql"]
            params = interpreter_result.get("params", [])
            cache_key = interpreter_result.get("cache_key", "")
            
            if not sql:
                raise HTTPException(status_code=500, detail="SQL generation failed")
            
            if params:
                result = await run_sql_query(sql, tuple(params))
            else:
                result = await run_sql_query(sql)
            
            if not isinstance(result, list):
                result = []
            
            cache_hit = query_cache.get(cache_key) is not None if cache_key else False
            
            return {
                "sql": sql,
                "result": result,
                "execution_time": None,
                "cache_hit": cache_hit,
                "error": None
            }

    except Exception as e:
        error_response = sql_error_handler.provide_user_friendly_error(e, request.nl_query)
        return {
            "sql": None,
            "result": [],
            "execution_time": None,
            "cache_hit": False,
            "error": error_response
        }


# -------------------------------------------------
# NEW: LLM-POWERED SQL AGENT (LangGraph)
# -------------------------------------------------
@router.post("/sql_query/", response_model=SQLResponse, tags=["SQL"])
async def generate_sql(request: SQLQueryRequest):
    """
    New LLM-powered SQL agent using LangGraph.
    Dynamically generates SQL from natural language without hardcoded patterns.
    """
    try:
        # Use the new LangGraph-based SQL agent
        result = await sql_agent.run(request.nl_query)
        
        if result.get("error"):
            error_response = sql_error_handler.provide_user_friendly_error(
                Exception(result["error"]), request.nl_query
            )
            return {
                "sql": result.get("sql"),
                "result": [],
                "execution_time": None,
                "cache_hit": False,
                "error": error_response
            }
        
        return {
            "sql": result.get("sql"),
            "result": result.get("result", []),
            "execution_time": None,
            "cache_hit": False,
            "error": None
        }
        
    except Exception as e:
        error_response = sql_error_handler.provide_user_friendly_error(e, request.nl_query)
        return {
            "sql": None,
            "result": [],
            "execution_time": None,
            "cache_hit": False,
            "error": error_response
        }

# -------------------------------------------------
# PDF UPLOAD (FIXED)
# -------------------------------------------------
@router.post("/upload_pdf/", tags=["Report"])
async def upload_pdf(
    file: UploadFile = File(...),
    patient_id: str = Form(...)
):
    try:
        content = await file.read()

        # ✅ ASYNC, NO THREADPOOL
        await ingest_pdf(content, patient_id)

        return {
            "status": "success",
            "patient_id": patient_id,
            "filename": file.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# REPORT GENERATION (LangGraph-powered)
# -------------------------------------------------
@router.get("/report/{patient_id}", tags=["Report"])
async def generate_report(patient_id: str):
    try:
        # Use LangGraph report agent (parallel data fetching)
        pdf_path = await report_agent.generate(patient_id)

        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=500,
                detail=f"Report generation failed for {patient_id}",
            )

        return FileResponse(
            path=pdf_path,
            filename=os.path.basename(pdf_path),
            media_type="application/pdf",
        )

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
