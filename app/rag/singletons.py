# app/rag/singletons.py

"""
Global singleton instances for RAG components.
Initializes shared resources at application startup.
"""

from app.rag.reranker import Reranker
from app.rag.vectorstore_manager import VectorStoreManager
from app.llm.ollama_client import OllamaClient
from app.core.config import get_settings

settings = get_settings()

# Global singleton instances - loaded once at startup
# Ollama client for text generation and embeddings
ollama_client = OllamaClient(settings.OLLAMA_URL)

# Cross-encoder reranker for improving retrieval results
reranker = Reranker()

# Vector store manager for document storage and retrieval
vectorstore = VectorStoreManager()
