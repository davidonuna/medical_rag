# app/core/logger.py

"""
Logging configuration for the Medical RAG application.
Provides structured logging with timestamps and severity levels.
"""

import logging

# Configure root logger with structured format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Application logger instance
logger = logging.getLogger("medical_rag")
