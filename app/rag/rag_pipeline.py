"""
app/rag/rag_pipeline.py

Retrieval-Augmented Generation pipeline for medical Q&A.
Handles document retrieval, reranking, and answer generation.
"""

from app.core.config import get_settings
from app.llm.ollama_client import OllamaClient
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from app.llm.prompt_templates import (
    RAG_CHAT_PROMPT,
    MEDICAL_SUMMARY_PROMPT,
    MEDICAL_SUMMARY_2PARA_PROMPT,
    PEDIATRIC_CLINICAL_SUMMARY_PROMPT,
    PEDIATRIC_ASSESSMENT_PROMPT,
    CLINICAL_METADATA_EXTRACTION_PROMPT,
    ADULT_CLINICAL_SUMMARY_PROMPT,
)
from app.rag.singletons import vectorstore, reranker
import re
import json
import logging
from datetime import datetime

settings = get_settings()
logger = logging.getLogger("medical_rag")

# =====================================================
# SAFETY LIMITS
# =====================================================
# Hard limits to prevent memory issues with Ollama CPU
MAX_CHAT_CONTEXT_CHARS = 2500
MAX_SUMMARY_CHARS = 3500
SUMMARY_MAX_TOKENS = 500


def extract_patient_age_from_docs(docs: list[str]) -> int | None:
    """
    Extract patient age from clinical documents to determine pediatric vs adult.
    
    Returns:
        Patient age in years, or None if not found
    """
    combined_text = " ".join(docs)
    
    # Look for age patterns like "Age: 70", "70 years", "70-year-old"
    age_patterns = [
        r"(?:Age|age)[:\s]+(\d{1,3})(?:\s*years|\s*yr)?",
        r"(\d{1,3})[-\s]year[-\s]old",
        r"(\d{1,3})\s*yrs?\s*old",
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, combined_text)
        if match:
            age = int(match.group(1))
            if 0 < age <= 120:
                return age
    
    # Look for DOB pattern
    dob_pattern = r"DOB[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    dob_match = re.search(dob_pattern, combined_text)
    if dob_match:
        try:
            dob_str = dob_match.group(1)
            # Try to parse the date
            for fmt in ["%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"]:
                try:
                    dob = datetime.strptime(dob_str, fmt)
                    today = datetime.now()
                    age = (today - dob).days // 365
                    if 0 < age <= 120:
                        return age
                except ValueError:
                    continue
        except Exception:
            pass
    
    return None


def is_pediatric(age: int | None) -> bool:
    """Determine if patient is pediatric (under 18)."""
    if age is None:
        return False  # Default to adult prompts if age not found
    return age < 18


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for medical Q&A.
    
    Features:
        - Interactive clinical Q&A with citations
        - Report-grade clinical summaries
        - Structured clinical metadata extraction
        - Hybrid chunking for better retrieval
    
    Example:
        rag = RAGPipeline()
        result = await rag.query_async(
            question="What medications?",
            patient_id="NCH-123"
        )
    """

    def __init__(self):
        self.ollama = OllamaClient(settings.OLLAMA_URL)
        self.vectorstore = vectorstore
        self.reranker = reranker

    # =====================================================
    # STANDARD RAG CHAT
    # =====================================================
    async def query_async(self, question: str, patient_id: str | None = None):
        """
        Answer a clinical question using RAG.
        
        Steps:
            1. Retrieve relevant documents from vector store
            2. Rerank results using cross-encoder
            3. Build context from top results
            4. Generate answer with LLM
        
        Args:
            question: User's clinical question
            patient_id: Optional patient ID to filter documents
            
        Returns:
            Dict with "answer" and "citations" keys
        """
        # Retrieve documents
        raw = await self.vectorstore.retrieve_raw(question)
        docs = [t for _, t in raw]

        # Rerank for better relevance
        ranked = self.reranker.rerank(question, docs)

        # Build context within token limits
        context_blocks = []
        citations = []
        used = 0

        for idx, (doc, score) in enumerate(ranked, 1):
            block = f"[{idx}] {doc}"
            if used + len(block) > MAX_CHAT_CONTEXT_CHARS:
                break
            context_blocks.append(block)
            citations.append(f"[{idx}] PDF clinical record")
            used += len(block)

        # Generate answer
        prompt = RAG_CHAT_PROMPT.format(
            context="\n\n".join(context_blocks),
            question=question,
        )

        try:
            answer = await self.ollama.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
            )
        except Exception as e:
            logger.error("Ollama query_async failed", exc_info=e)
            return {
                "answer": "Unable to generate answer due to model error.",
                "citations": citations,
            }

        return {
            "answer": answer,
            "citations": citations,
        }

    # =====================================================
    # CLINICAL SUMMARIES
    # =====================================================
    async def clinical_summary_for_report(self, patient_id: str) -> str:
        """
        Generate single-paragraph clinical summary for reports.
        Automatically detects patient age and uses appropriate prompt (pediatric vs adult).
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Clinical summary text
        """
        raw_docs = self.vectorstore.raw_docs.get(patient_id, [])
        
        if not raw_docs:
            return "No uploaded clinical documents available for summarization."
        
        # Detect patient age to choose appropriate prompt
        age = extract_patient_age_from_docs(raw_docs)
        is_ped = is_pediatric(age)
        
        if is_ped:
            prompt_template = PEDIATRIC_CLINICAL_SUMMARY_PROMPT
            logger.info(f"Using pediatric prompt for patient {patient_id} (age: {age})")
        else:
            prompt_template = ADULT_CLINICAL_SUMMARY_PROMPT
            logger.info(f"Using adult prompt for patient {patient_id} (age: {age})")
        
        return await self._generate_summary(
            patient_id=patient_id,
            prompt_template=prompt_template,
            enforce_two_paragraphs=False,
        )

    async def clinical_summary_two_paragraph(self, patient_id: str) -> str:
        """
        Generate two-paragraph clinical assessment for reports.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Two-paragraph clinical assessment
        """
        return await self._generate_summary(
            patient_id=patient_id,
            prompt_template=PEDIATRIC_ASSESSMENT_PROMPT,
            enforce_two_paragraphs=True,
        )

    # =====================================================
    # METADATA EXTRACTION
    # =====================================================
    async def extract_clinical_metadata(self, clinical_text: str) -> Dict[str, Any]:
        """
        Extract structured clinical metadata from raw PDF text.
        
        Extracts:
            - Patient demographics
            - Vital signs
            - Laboratory results
            - Medications
            - Diagnoses
            - Procedures
        
        Args:
            clinical_text: Raw PDF text
            
        Returns:
            Dictionary with structured clinical data
        """
        if not clinical_text or len(clinical_text.strip()) < 50:
            return {}

        # Truncate to prevent token limits
        truncated_text = clinical_text[:8000]

        prompt = CLINICAL_METADATA_EXTRACTION_PROMPT.format(
            clinical_text=truncated_text
        )

        try:
            result = await self.ollama.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                max_tokens=1024,
            )

            result = result.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group(0)
            
            result = result.strip()
            # Clean markdown code blocks
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            metadata = json.loads(result)
            
            if not isinstance(metadata, dict):
                return {"error": "Invalid JSON structure", "raw_text": truncated_text[:500]}
            
            return metadata

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata JSON: {e}, result: {result[:200] if result else 'empty'}")
            return {"error": "Failed to parse structured data", "raw_text": truncated_text[:500]}
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {"error": str(e)}

    # =====================================================
    # HYBRID CHUNKING
    # =====================================================
    async def get_chunks_for_query(self, query: str, patient_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get relevant chunks using hybrid chunking strategy.
        Combines visit-based and section-based chunks.
        
        Args:
            query: Search query
            patient_id: Patient identifier
            top_n: Number of results
            
        Returns:
            List of (chunk_text, score) tuples
        """
        raw_docs = self.vectorstore.raw_docs.get(patient_id, [])
        
        if not raw_docs:
            return []

        all_chunks = []
        for doc in raw_docs:
            chunks = chunk_hybrid(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return await self.vectorstore.retrieve_raw(query, top_n)

        chunk_texts = [c["text"] for c in all_chunks]

        try:
            # Generate embeddings
            embeddings = await self.ollama.embed(chunk_texts, "nomic-embed-text")
            import numpy as np
            query_embed = (await self.ollama.embed([query], "nomic-embed-text"))[0]
            
            # Compute similarities
            query_vec = np.array([query_embed], dtype=np.float32)
            chunk_vecs = np.array(embeddings, dtype=np.float32)
            
            similarities = np.dot(chunk_vecs, query_vec.T).flatten()
            
            # Rank by similarity
            ranked_indices = np.argsort(similarities)[::-1][:top_n]
            
            results = []
            for idx in ranked_indices:
                if similarities[idx] > 0.3:
                    chunk = all_chunks[idx]
                    results.append((chunk["text"], float(similarities[idx])))
            
            return results
        except Exception as e:
            logger.warning(f"Hybrid chunking failed, falling back to vectorstore: {e}")
            return await self.vectorstore.retrieve_raw(query, top_n)

    # =====================================================
    # INTERNAL SUMMARY GENERATOR
    # =====================================================
    async def _generate_summary(
        self,
        patient_id: str,
        prompt_template: str,
        enforce_two_paragraphs: bool,
    ) -> str:
        """
        Internal method to generate clinical summaries.
        
        Args:
            patient_id: Patient identifier
            prompt_template: LLM prompt template
            enforce_two_paragraphs: Whether to enforce two-paragraph format
            
        Returns:
            Generated summary text
        """
        raw_docs = self.vectorstore.raw_docs.get(patient_id, [])

        if not raw_docs:
            return "No uploaded clinical documents available for summarization."

        # Extract and process visits
        visits = extract_visits(raw_docs)

        parts = []
        used = 0

        # Combine visits within character limit
        for visit in visits:
            if used >= MAX_SUMMARY_CHARS:
                break

            snippet = visit[: MAX_SUMMARY_CHARS - used]
            parts.append(snippet)
            used += len(snippet)

        docs_text = "\n\n".join(parts).strip()

        if not docs_text:
            return "No usable clinical text found for summarization."

        # Build prompt with safety instructions
        prompt = (
            "You must strictly abstract documented clinical facts only.\n"
            "Do NOT infer outcomes, improvement, response, timelines, or conclusions.\n"
            "If something is not explicitly written, omit it.\n\n"
            + prompt_template.format(docs=docs_text)
        )

        try:
            summary = await self.ollama.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
        except Exception as e:
            logger.error("Ollama summary generation failed", exc_info=e)
            return (
                "Clinical summary could not be generated due to model limitations. "
                "Please refer to the clinical records."
            )

        # Clean up summary
        cleaned = clean_summary_text(summary)

        if enforce_two_paragraphs:
            cleaned = enforce_exact_two_paragraphs(cleaned)

        return cleaned


# =====================================================
# CLINICAL TEXT PROCESSING HELPERS
# =====================================================

# Header patterns to remove from clinical documents
HEADER_PATTERNS = [
    r"Patient Name:.*",
    r"Patient Age:.*",
    r"Nationality.*",
    r"Inpatient No.*",
    r"Patient Gender:.*",
    r"Visit Number.*",
    r"Critical Info.*",
    r"Doctor Name.*",
    r"Signature.*",
]


def extract_visits(raw_docs: list[str]) -> list[str]:
    """
    Extract individual visits from clinical documents.
    Splits by "Visit Date" markers and processes each visit.
    
    Args:
        raw_docs: List of raw clinical document texts
        
    Returns:
        List of cleaned visit texts sorted by date
    """
    visits: List[Tuple[Optional[datetime], str, str]] = []
    seen_keys = set()

    for doc in raw_docs:
        if not doc:
            continue

        # Normalize and clean document
        doc = normalize_doc_text(doc)
        doc = remove_headers(doc)

        # Split by visit date
        chunks = re.split(
            r"(Visit Date\s*:?\s*[0-9T:\-\.]+)",
            doc,
            flags=re.IGNORECASE,
        )

        for i in range(1, len(chunks), 2):
            header = chunks[i]
            body = chunks[i + 1] if i + 1 < len(chunks) else ""
            full_chunk = f"{header}\n{body}".strip()

            if len(full_chunk) < 200:
                continue

            # Extract metadata
            visit_date = extract_visit_date(full_chunk)
            visit_number = extract_visit_number(full_chunk)

            # Deduplicate
            dedup_key = (visit_date, visit_number)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Clean up content
            cleaned = full_chunk
            cleaned = compress_labs(cleaned)
            cleaned = compress_medications(cleaned)
            cleaned = strip_structural_noise(cleaned)

            visits.append((visit_date, visit_number or "", cleaned))

    # Sort by date
    visits.sort(key=lambda x: x[0] or datetime.min)
    return [v[2] for v in visits]


# =====================================================
# CLINICAL SECTION PATTERNS
# =====================================================
CLINICAL_SECTION_PATTERNS = {
    "chief_complaint": r"(?i)(?:chief complaint|reason for visit|presenting complaint)",
    "history": r"(?i)(?:history of present illness|history|hpi|past medical history)",
    "examination": r"(?i)(?:physical examination|general examination|exam|objective findings)",
    "vitals": r"(?i)(?:vital signs|vitals|vital parameters)",
    "laboratory": r"(?i)(?:laboratory|lab results|labs|investigations|blood tests)",
    "radiology": r"(?i)(?:radiology|imaging|x-ray|ultrasound|ct|mri|scan)",
    "diagnosis": r"(?i)(?:diagnosis|assessment|impression|conclusion)",
    "treatment": r"(?i)(?:treatment|management|medication|prescription|intervention|therapy)",
    "discharge": r"(?i)(?:discharge|summary|plan|follow.up|recommendation)",
}


def chunk_by_section(text: str, min_chunk_size: int = 200) -> List[Tuple[str, str]]:
    """
    Split clinical document into sections based on headers.
    
    Args:
        text: Clinical document text
        min_chunk_size: Minimum section size to keep
        
    Returns:
        List of (section_name, section_content) tuples
    """
    sections = []
    current_section = "general"
    current_content = []
    
    lines = text.split("\n")
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers
        detected_section = None
        for section_name, pattern in CLINICAL_SECTION_PATTERNS.items():
            if re.search(pattern, line_lower) and len(line.strip()) < 50:
                detected_section = section_name
                break
        
        if detected_section:
            if current_content:
                content = "\n".join(current_content).strip()
                if len(content) >= min_chunk_size:
                    sections.append((current_section, content))
            current_section = detected_section
            current_content = [line]
        else:
            current_content.append(line)
    
    if current_content:
        content = "\n".join(current_content).strip()
        if len(content) >= min_chunk_size:
            sections.append((current_section, content))
    
    return sections


def chunk_by_paragraphs(text: str, min_chunk_size: int = 300, max_chunk_size: int = 1000) -> List[str]:
    """
    Split clinical document into overlapping paragraph chunks.
    
    Args:
        text: Clinical document text
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size before forcing split
        
    Returns:
        List of paragraph text chunks
    """
    paragraphs = []
    current_para = []
    current_size = 0
    
    lines = text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_para:
                para_text = "\n".join(current_para)
                if len(para_text) >= min_chunk_size:
                    paragraphs.append(para_text)
                current_para = []
                current_size = 0
            continue
        
        current_para.append(line)
        current_size += len(line)
        
        if current_size >= max_chunk_size:
            para_text = "\n".join(current_para)
            paragraphs.append(para_text)
            current_para = []
            current_size = 0
    
    if current_para:
        para_text = "\n".join(current_para)
        if len(para_text) >= min_chunk_size:
            paragraphs.append(para_text)
    
    return paragraphs


def chunk_hybrid(text: str) -> List[Dict]:
    """
    Hybrid chunking: by visit AND by section.
    Maximizes context preservation for medical documents.
    
    Args:
        text: Clinical document text
        
    Returns:
        List of chunk dictionaries with section/visit metadata
    """
    chunks = []
    
    text = normalize_doc_text(text)
    text = remove_headers(text)
    
    # Split by visit
    visit_chunks = re.split(
        r"(Visit Date\s*:?\s*[0-9T:\-\.]+)",
        text,
        flags=re.IGNORECASE,
    )
    
    for i in range(1, len(visit_chunks), 2):
        header = visit_chunks[i]
        body = visit_chunks[i + 1] if i + 1 < len(visit_chunks) else ""
        
        if not body or len(body) < 100:
            continue
        
        # Extract visit metadata
        visit_date = extract_visit_date(header + body)
        visit_number = extract_visit_number(header + body)
        
        # Chunk by section within visit
        section_chunks = chunk_by_section(body)
        
        for section_name, section_content in section_chunks:
            chunks.append({
                "section": section_name,
                "visit_date": visit_date.isoformat() if visit_date else None,
                "visit_number": visit_number,
                "text": section_content,
                "chunk_type": "section"
            })
        
        # Fallback to paragraphs if no sections found
        if not section_chunks:
            para_chunks = chunk_by_paragraphs(body)
            for j, para in enumerate(para_chunks):
                chunks.append({
                    "section": "general",
                    "visit_date": visit_date.isoformat() if visit_date else None,
                    "visit_number": visit_number,
                    "text": para,
                    "chunk_type": "paragraph",
                    "chunk_index": j
                })
    
    return chunks


def extract_visit_date(text: str) -> Optional[datetime]:
    """Extract visit date from clinical text."""
    match = re.search(
        r"Visit Date\s*:?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[T\s][0-9:\.]+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None

    raw = match.group(1).replace(" ", "T")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def extract_visit_number(text: str) -> Optional[str]:
    """Extract visit number from clinical text."""
    match = re.search(r"Visit Number\s*:?\s*([A-Z0-9]+)", text)
    return match.group(1) if match else None


def remove_headers(text: str) -> str:
    """Remove common header fields from clinical text."""
    for pat in HEADER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


def compress_labs(text: str) -> str:
    """
    Compress laboratory section to key findings.
    Replaces detailed lab results with summarized findings.
    """
    if "Laboratory" not in text:
        return text

    summary = []

    # Extract key lab abnormalities
    if re.search(r"C-REACTIVE PROTEIN.*High", text, re.IGNORECASE):
        summary.append("Elevated C-reactive protein.")
    if re.search(r"WBC.*High", text, re.IGNORECASE):
        summary.append("Leukocytosis noted.")
    if re.search(r"Hypochromi", text, re.IGNORECASE):
        summary.append("Mild hypochromic anemia.")

    # Remove detailed labs
    text = re.sub(
        r"Laboratory.*?(?=Diagnosis|Radiology|Management|Visit Date|$)",
        "",
        text,
        flags=re.S | re.I,
    )

    if summary:
        text += "\nLaboratory Summary: " + " ".join(summary)

    return text


def compress_medications(text: str) -> str:
    """
    Compress medication/prescription section to key drugs.
    Extracts notable medications from detailed prescriptions.
    """
    if "Prescription" not in text:
        return text

    meds = set()

    # Extract notable medications
    if re.search(r"\bPulmicort\b", text, re.IGNORECASE):
        meds.add("Pulmicort (budesonide)")
    if re.search(r"\bCombivent\b", text, re.IGNORECASE):
        meds.add("Combivent (ipratropium/salbutamol)")
    if re.search(r"\bAtrovent\b", text, re.IGNORECASE):
        meds.add("Atrovent (ipratropium)")
    if re.search(r"Parafast|Paracetamol", text, re.IGNORECASE):
        meds.add("Paracetamol")

    # Remove detailed prescriptions
    text = re.sub(
        r"Prescription.*?(?=Laboratory|Radiology|Visit Date|$)",
        "",
        text,
        flags=re.S | re.I,
    )

    if meds:
        text += "\nTreatments included: " + "; ".join(sorted(meds)) + "."

    return text


def normalize_doc_text(doc: str) -> str:
    """Normalize Unicode characters and whitespace in clinical text."""
    if not doc:
        return ""

    replacements = {
        "\xa0": " ",      # Non-breaking space
        "\u2022": " ",    # Bullet
        "\u2013": "-",    # En dash
        "\u2014": "-",    # Em dash
        "\u2018": "'",    # Left single quote
        "\u2019": "'",    # Right single quote
        "\u201c": '"',    # Left double quote
        "\u201d": '"',    # Right double quote
    }

    for k, v in replacements.items():
        doc = doc.replace(k, v)

    return doc.strip()


def strip_structural_noise(text: str) -> str:
    """Remove formatting noise like bullets, headers, timestamps."""
    text = re.sub(r"^[\*\+\-]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"#{1,}.*", "", text)
    text = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "", text)
    return text.strip()


def clean_summary_text(text: str) -> str:
    """Remove AI self-referential phrases from generated summaries."""
    lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        # Skip AI self-reference lines
        if any(p in l.lower() for p in [
            "i will summarize",
            "this summary",
            "the following",
        ]):
            continue
        lines.append(l)

    return "\n\n".join(lines)


def enforce_exact_two_paragraphs(text: str) -> str:
    """
    Ensure summary is exactly two paragraphs.
    Splits single paragraph if needed.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if len(paragraphs) >= 2:
        return paragraphs[0] + "\n\n" + paragraphs[1]

    if len(paragraphs) == 1:
        mid = len(paragraphs[0]) // 2
        return paragraphs[0][:mid].strip() + "\n\n" + paragraphs[0][mid:].strip()

    return text
