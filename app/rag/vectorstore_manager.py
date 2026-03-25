"""
app/rag/vectorstore_manager.py

FAISS vector store manager for document retrieval.
Handles document indexing, chunking, and similarity search.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
import faiss
import re

from app.core.config import get_settings
from app.llm.ollama_client import OllamaClient

settings = get_settings()

# Path configuration for vector store persistence
VECTORSTORE_PATH = Path(settings.VECTORSTORE_PATH)
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

INDEX_FILE = VECTORSTORE_PATH / "faiss.index"
META_FILE = VECTORSTORE_PATH / "metadata.pkl"
RAW_DOC_FILE = VECTORSTORE_PATH / "raw_docs.pkl"

# Embedding model for vectorization
EMBED_MODEL = "nomic-embed-text"

# Known antibiotics for extraction (used in clinical context)
ANTIBIOTICS = [
    "ceftriaxone", "amikacin", "amoxicillin",
    "azithromycin", "cefixime", "cefuroxime",
    "piperacillin", "tazobactam", "meropenem"
]


class VectorStoreManager:
    """
    Manages FAISS vector index for medical document retrieval.
    
    Features:
        - Persistent storage of index and metadata
        - Visit-aware document splitting
        - Section-based chunking
        - Antibiotic extraction from clinical text
        - Patient-specific document management
    
    Example:
        vsm = VectorStoreManager()
        await vsm.add_patient_documents("NCH-123", ["clinical text..."])
        results = await vsm.retrieve_raw("What medications?", top_n=10)
    """

    def __init__(self):
        self.ollama = OllamaClient(settings.OLLAMA_URL)
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: List[Dict] = []
        self.raw_docs: Dict[str, List[str]] = {}
        self._load()

    # ---------------------------
    # LOAD / SAVE
    # ---------------------------
    def _load(self):
        """
        Load vector index and metadata from disk.
        Creates empty index if files don't exist.
        """
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            self.metadata = pickle.loads(META_FILE.read_bytes())
        else:
            self.index = None
            self.metadata = []

        if RAW_DOC_FILE.exists():
            self.raw_docs = pickle.loads(RAW_DOC_FILE.read_bytes())
        else:
            self.raw_docs = {}

    def _save(self):
        """
        Persist vector index, metadata, and raw documents to disk.
        Called after any modification to the store.
        """
        if self.index:
            faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_bytes(pickle.dumps(self.metadata))
        RAW_DOC_FILE.write_bytes(pickle.dumps(self.raw_docs))

    # ---------------------------
    # RAW DOCUMENT STORAGE
    # ---------------------------
    def store_raw_document(self, patient_id: str, text: str):
        """
        Store full raw document text for a patient.
        Used for report summarization (not embedding).
        
        Args:
            patient_id: Unique patient identifier
            text: Full clinical document text
        """
        self.raw_docs.setdefault(patient_id, []).append(text)
        self._save()

    # ---------------------------
    # VISIT-AWARE SPLITTING
    # ---------------------------
    def split_by_visit(self, text: str) -> List[str]:
        """
        Split clinical document into visit-based chunks.
        
        Args:
            text: Full clinical document text
            
        Returns:
            List of visit text segments (min 200 chars)
        """
        visits = re.split(r"Visit Date\s*:", text)
        return [v.strip() for v in visits if len(v.strip()) > 200]

    def chunk_visit(self, visit_text: str) -> List[Dict]:
        """
        Split visit text into sections (diagnosis, treatment, labs, etc.)
        
        Args:
            visit_text: Text from a single visit
            
        Returns:
            List of dicts with {"section": str, "text": str}
        """
        chunks = []
        current = []
        section = "general"

        for line in visit_text.splitlines():
            l = line.lower()
            # Detect section boundaries
            if any(k in l for k in ["diagnosis", "assessment"]):
                section = "diagnosis"
            elif any(k in l for k in ["treatment", "medication", "prescription"]):
                section = "treatment"
            elif any(k in l for k in ["laboratory", "haemogram", "crp", "investigation"]):
                section = "labs"
            elif any(k in l for k in ["radiology", "x-ray", "chest"]):
                section = "radiology"

            current.append(line)

            # Flush chunk after 30 lines
            if len(current) >= 30:
                chunks.append({"section": section, "text": "\n".join(current)})
                current = []

        if current:
            chunks.append({"section": section, "text": "\n".join(current)})

        return chunks

    # ---------------------------
    # ADD PATIENT DOCUMENTS
    # ---------------------------
    async def add_patient_documents(self, patient_id: str, documents: List[str]):
        """
        Add clinical documents for a patient to the vector store.
        
        Steps:
            1. Delete existing docs for patient
            2. Split by visits
            3. Chunk by section
            4. Generate embeddings
            5. Add to FAISS index
        
        Args:
            patient_id: Unique patient identifier
            documents: List of clinical document texts
        """
        self.delete_patient(patient_id)

        texts, metas = [], []

        for doc in documents:
            # Split document into visits
            visits = self.split_by_visit(doc)
            for visit in visits:
                # Chunk each visit into sections
                for chunk in self.chunk_visit(visit):
                    texts.append(chunk["text"])
                    metas.append({
                        "patient_id": patient_id,
                        "section": chunk["section"],
                        "source": "uploaded_pdf",
                        "text": chunk["text"],
                        "antibiotics": self.extract_antibiotics(chunk["text"])
                    })

        if not texts:
            return

        # Generate embeddings
        embeddings = await self.ollama.embed(texts, EMBED_MODEL)
        vectors = np.array(embeddings, dtype=np.float32)

        # Initialize or extend index
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])

        self.index.add(vectors)
        self.metadata.extend(metas)
        self._save()

    def delete_patient(self, patient_id: str):
        """
        Remove all documents for a patient from the store.
        
        Args:
            patient_id: Unique patient identifier
        """
        # Remove raw docs
        self.raw_docs.pop(patient_id, None)

        if not self.index:
            self._save()
            return

        # Remove from FAISS index
        keep = [i for i, m in enumerate(self.metadata) if m["patient_id"] != patient_id]
        if not keep:
            self.index = None
            self.metadata = []
            self._save()
            return

        # Rebuild index with remaining vectors
        vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep]
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.metadata = [self.metadata[i] for i in keep]
        self._save()

    # ---------------------------
    # RETRIEVAL
    # ---------------------------
    async def retrieve_raw(self, query: str, top_n: int = 15):
        """
        Retrieve relevant documents for a query using vector similarity.
        
        Args:
            query: Search query text
            top_n: Maximum number of results (default: 15)
            
        Returns:
            List of (metadata, text) tuples for matching documents
        """
        if not self.index or self.index.ntotal == 0:
            return []

        # Embed query
        q_embed = (await self.ollama.embed([query], EMBED_MODEL))[0]
        
        # Search index
        D, I = self.index.search(
            np.array([q_embed], dtype=np.float32),
            min(top_n, self.index.ntotal)
        )
        return [(self.metadata[i], self.metadata[i]["text"]) for i in I[0]]

    # ---------------------------
    # ANTIBIOTIC EXTRACTION
    # ---------------------------
    def extract_antibiotics(self, text: str) -> List[str]:
        """
        Extract mentioned antibiotics from clinical text.
        
        Args:
            text: Clinical document text
            
        Returns:
            List of antibiotics found (title case)
        """
        found = []
        for ab in ANTIBIOTICS:
            if re.search(rf"\b{ab}\b", text.lower()):
                found.append(ab.title())
        return list(set(found))
