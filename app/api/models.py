# app/api/models.py
"""
models.py
Pydantic models for FastAPI backend of Medical RAG project
"""
from pydantic import BaseModel, Field, RootModel
from typing import Optional, List, Any


# ---------------------------
# Chat
# ---------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., example="What medications is John on?")
    patient_id: Optional[str] = Field(None, example="NCH-28029")


class ChatResponse(BaseModel):
    response: str


# ---------------------------
# Patient Detection
# ---------------------------
class DetectRequest(BaseModel):
    text: str = Field(..., example="Show labs for John Smith")


class PatientSuggestion(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    confidence: float


class DetectResponse(BaseModel):
    patient_id: Optional[str]
    confidence: Optional[float]
    suggestions: List[PatientSuggestion] = []
    source: Optional[str]


# ---------------------------
# SQL
# ---------------------------
class SQLRequest(BaseModel):
    nl_query: str = Field(..., example="Show me last 5 reports")


class SQLResponse(BaseModel):
    sql: str
    result: Any


# ---------------------------
# PDF Ingestion
# ---------------------------
class IngestResponse(BaseModel):
    message: str


# ---------------------------
# Report Generation
# ---------------------------
class ReportResponse(BaseModel):
    path: str


# ---------------------------
# Patient Suggestion / Autocomplete
# ---------------------------
class PatientSuggestionsResponse(RootModel[list[str]]):
    pass
