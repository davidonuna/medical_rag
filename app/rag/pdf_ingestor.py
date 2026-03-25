# app/rag/pdf_ingestor.py

"""
PDF ingestion pipeline for medical documents.
Extracts text, metadata, and creates vector embeddings.
"""

import os
import json
from pathlib import Path
from pdfminer.high_level import extract_text

# Use the singleton vectorstore instance
from app.rag.singletons import vectorstore
from app.rag.rag_pipeline import RAGPipeline
from app.core.logger import logger

BASE_DIR = Path(__file__).parent.parent
PDF_STORAGE_DIR = BASE_DIR / "uploaded_pdfs"
METADATA_DIR = BASE_DIR / "data" / "metadata"

# Ensure directories exist
PDF_STORAGE_DIR.mkdir(exist_ok=True, parents=True)
METADATA_DIR.mkdir(exist_ok=True, parents=True)


async def ingest_pdf(file_bytes: bytes, patient_id: str) -> dict:
    """
    Process uploaded PDF and add to vector store.
    
    Pipeline steps:
        1. Save PDF temporarily to disk
        2. Extract text using pdfminer
        3. Extract structured clinical metadata using LLM
        4. Chunk and embed text for RAG retrieval
        5. Store raw document for report summarization
        6. Clean up temporary files
    
    Args:
        file_bytes: Raw PDF file bytes
        patient_id: Patient identifier (e.g., "NCH-12345")
        
    Returns:
        Dictionary of extracted metadata
    """
    # Create patient-specific directory
    patient_dir = PDF_STORAGE_DIR / patient_id
    patient_dir.mkdir(exist_ok=True, parents=True)

    # ----------------------------------------
    # Step 1: Save temporary PDF
    # ----------------------------------------
    temp_pdf_path = patient_dir / "temp_upload.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(file_bytes)

    # ----------------------------------------
    # Step 2: Extract text (pdfminer is sync)
    # ----------------------------------------
    text = extract_text(str(temp_pdf_path))
    extracted_text = text.strip() if text and text.strip() else ""

    metadata = {}

    # ----------------------------------------
    # Step 3: Extract structured clinical metadata
    # ----------------------------------------
    if extracted_text:
        try:
            rag_pipeline = RAGPipeline()
            metadata = await rag_pipeline.extract_clinical_metadata(extracted_text)
            
            if metadata and "error" not in metadata:
                # Save metadata to JSON file
                metadata_file = METADATA_DIR / f"{patient_id}_metadata.json"
                metadata_file.write_text(json.dumps(metadata, indent=2))
                logger.info(f"Extracted metadata for patient {patient_id}")
            else:
                logger.warning(f"Failed to extract metadata for patient {patient_id}")
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            metadata = {"error": str(e)}

    # ----------------------------------------
    # Step 4: Vectorstore ingestion (SINGLETON)
    # ----------------------------------------
    if extracted_text:
        # Chunk + embed for RAG retrieval
        await vectorstore.add_patient_documents(
            patient_id=patient_id,
            documents=[extracted_text],
        )

        # Store FULL raw document for report summarization
        vectorstore.store_raw_document(
            patient_id=patient_id,
            text=extracted_text
        )

    # ----------------------------------------
    # Step 5: Cleanup temp PDF
    # ----------------------------------------
    try:
        os.remove(temp_pdf_path)
    except FileNotFoundError:
        pass

    return metadata
