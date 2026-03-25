# app/core/errors.py

"""
Custom exceptions for the Medical RAG project.

Provides application-specific error types for clearer error handling
and better debugging across the application.
"""

from typing import Optional


class PDFError(Exception):
    """
    Raised when PDF report generation fails.
    
    Common causes:
        - Missing font files
        - ReportLab rendering errors
        - Invalid patient data
        - File system permission issues
    """
    pass


class LLMError(Exception):
    """
    Raised when LLM fails to generate a response.
    
    Common causes:
        - Ollama service unavailable
        - Model not found
        - Prompt exceeds context length
        - Generation timeout
    """
    pass


class DatabaseError(Exception):
    """
    Raised when database query or connection fails.
    
    Common causes:
        - Connection pool exhausted
        - Query syntax errors
        - Constraint violations
        - Timeout exceeded
    """
    pass


class IngestionError(Exception):
    """
    Raised when PDF ingestion or vector store update fails.
    
    Common causes:
        - Invalid PDF format
        - PDF text extraction failure
        - Embedding generation failure
        - FAISS index corruption
    """
    pass
