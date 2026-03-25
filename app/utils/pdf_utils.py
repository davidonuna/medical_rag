# pdf_utils.py
import io
from PyPDF2 import PdfReader

def extract_text_from_pdf(byte_data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(byte_data))
    texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            texts.append(text.strip())
    return texts
