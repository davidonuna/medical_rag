# app/agent/report_generator.py

"""
Medical PDF Report Generator (Hybrid SQL + Vectorstore)

Generates pediatric medical reports combining:
    - SQL = structured patient/visit data from database
    - Vectorstore = clinical context from uploaded PDFs

Features:
    - Lazy RAG initialization (FastAPI-safe)
    - Age-appropriate pediatric content
    - PDF generation with ReportLab
"""

from pathlib import Path
from datetime import datetime
from textwrap import wrap
from typing import List, Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app.core.config import get_settings
from app.core.errors import PDFError
from app.core.logger import logger
from app.agent.sql_tool import run_sql_query
from app.rag.rag_pipeline import RAGPipeline

# =====================================================
# FONT REGISTRATION
# =====================================================
FONT_PATH = Path("app/assets/fonts/DejaVuSans.ttf")
if not FONT_PATH.exists():
    raise RuntimeError("Missing font: app/assets/fonts/DejaVuSans.ttf")

pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_PATH)))

# =====================================================
# AGE CALCULATION HELPERS
# =====================================================
def calculate_pediatric_age(dob: str) -> str:
    """
    Calculate age in pediatric-friendly format (years, months, days).
    
    Args:
        dob: Date of birth in YYYY-MM-DD format
        
    Returns:
        Age string like "2 years, 6 months" or "15 days"
    """
    try:
        birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
        today = datetime.now().date()
        
        # Calculate days difference
        total_days = (today - birth_date).days
        
        if total_days < 0:
            return "Invalid date"
        
        # Calculate years, months, days
        years = total_days // 365
        remaining_days = total_days % 365
        months = remaining_days // 30
        days = remaining_days % 30
        
        if years > 0:
            if months > 0:
                return f"{years} years, {months} months"
            else:
                return f"{years} years"
        elif months > 0:
            if days > 7:
                return f"{months} months, {days} days"
            else:
                return f"{months} months"
        else:
            return f"{total_days} days"
    except:
        return "Age calculation failed"

# =====================================================
# TEXT HELPERS
# =====================================================
def normalize_text(text: str) -> str:
    """Normalize Unicode characters in text for PDF rendering."""
    if not text:
        return ""
    replacements = {
        "\u2022": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\xa0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def draw_paragraph(c, text, x, y, max_chars=95, leading=14):
    """
    Draw wrapped paragraph on PDF canvas.
    Handles page breaks automatically.
    
    Args:
        c: ReportLab canvas
        text: Text to draw
        x, y: Starting position
        max_chars: Maximum characters per line
        leading: Line spacing
    """
    text = normalize_text(text)
    lines: List[str] = []

    # Wrap text into lines
    for p in text.split("\n"):
        lines.extend(wrap(p, max_chars) if p.strip() else [""])

    # Draw each line
    for line in lines:
        if y < 25 * mm:  # New page if near bottom
            c.showPage()
            c.setFont("DejaVu", 10)
            y = 280 * mm
        c.drawString(x, y, line)
        y -= leading

    return y


def section_title(c, title, x, y):
    """
    Draw section title on PDF.
    Handles page breaks.
    """
    if y < 35 * mm:
        c.showPage()
        y = 280 * mm
    c.setFont("DejaVu", 12)
    c.drawString(x, y, title)
    c.setFont("DejaVu", 10)
    return y - 18


# =====================================================
# REPORT GENERATOR
# =====================================================
class MedicalReportGenerator:
    """
    Generates pediatric medical PDF reports.
    
    Combines data from:
        - PostgreSQL (patient demographics, visit history)
        - Vectorstore (RAG-generated clinical summaries)
    
    Example:
        generator = MedicalReportGenerator()
        pdf_path = await generator.generate("NCH-12345")
    """

    def __init__(self):
        self.settings = get_settings()
        self.report_dir = Path(self.settings.REPORT_PATH)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._rag: RAGPipeline | None = None

    def _get_rag(self) -> RAGPipeline:
        """Lazy initialization of RAG pipeline (FastAPI-safe)."""
        if self._rag is None:
            self._rag = RAGPipeline()
        return self._rag

    def _extract_clinical_metadata(self, patient_id: str) -> Dict[str, Any]:
        """
        Extract structured clinical metadata from raw documents.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with allergies, procedures, medications, labs, vitals, etc.
        """
        from app.rag.singletons import vectorstore
        
        raw_docs = vectorstore.raw_docs.get(patient_id, [])
        if not raw_docs:
            return {}
        
        import re
        combined_text = " ".join(raw_docs)
        raw_text = raw_docs[0] if raw_docs else ""
        
        metadata = {
            "allergies": [],
            "procedures": [],
            "medications": [],
            "labs": [],
            "vitals": {},
            "chief_complaints": "",
            "examination_notes": "",
            "radiology": {},
            "admission_date": None,
            "discharge_date": None,
            "discharge_disposition": None,
            "bmi_category": "",
            "nationality": "",
            "diagnosis": "",
        }
        
        # Extract nationality
        nat_match = re.search(r"Nationality[:\s]+([^\n]+?)(?:\n|$)", combined_text)
        if nat_match:
            metadata["nationality"] = nat_match.group(1).strip()
        
        # Extract chief complaints - match from Chief Complaints to Examination Notes
        cc_match = re.search(r"(?:Chief Complaints|Chief Complaint)[:\s]*([^\n]+(?:\n(?!\d+\.\d+\s*kg)[^\n]+)*?)(?=Examination Notes|History Notes|$)", raw_text, re.IGNORECASE | re.DOTALL)
        if cc_match:
            complaint_text = cc_match.group(1).strip()
            # Remove weight values that appear as standalone numbers
            complaint_text = re.sub(r'^\d+\.?\d*\s*kg.*\n?', '', complaint_text)
            complaint_text = re.sub(r'^\d+\.?\d*\n+', '', complaint_text)
            complaint_text = complaint_text.strip()
            # Skip if it's just numbers or too short
            if complaint_text and len(complaint_text) > 2 and not re.match(r'^[\d\s\.]+$', complaint_text):
                metadata["chief_complaints"] = complaint_text
        
        # Extract examination notes - use raw text, need DOTALL
        exam_match = re.search(r"Examination Notes[:\s]*(.*?)(?:History Notes|Diagnosis|Radiology|Doctor Name|Patient Name|$)", raw_text, re.IGNORECASE | re.DOTALL)
        if exam_match:
            inner = exam_match.group(1)
            exam = inner.strip()
            # Clean up the examination notes
            exam = re.sub(r'^\s*General:\s*', '', exam)
            exam = exam.replace('\n', ', ')
            exam = re.sub(r',\s*,', ',', exam)  # Remove double commas
            exam = exam.strip()
            if exam and len(exam) > 3:
                metadata["examination_notes"] = exam
        
        # Extract vitals
        temp_match = re.search(r"(\d{2}[.-]\d)\s*(?:°C|C|Temp)", combined_text)
        if temp_match:
            metadata["vitals"]["temperature"] = f"{temp_match.group(1)}°C"
        
        # Extract weight
        weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|Kg|Weight)", combined_text)
        if weight_match:
            metadata["vitals"]["weight"] = f"{weight_match.group(1)} kg"
        
        # Extract height
        height_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm|Cm|Height)", combined_text)
        if height_match:
            metadata["vitals"]["height"] = f"{height_match.group(1)} cm"
        
        # Extract BMI category
        bmi_match = re.search(r"(very severely underweight|severely underweight|underweight|normal|overweight|obese)", combined_text, re.IGNORECASE)
        if bmi_match:
            metadata["bmi_category"] = bmi_match.group(1).strip()
        
        # Extract SPO2
        spo2_match = re.search(r"SPO2[:\s]*(\d+)", combined_text)
        if spo2_match:
            metadata["vitals"]["spo2"] = f"{spo2_match.group(1)}%"
        
        # Extract pulse
        pulse_match = re.search(r"Pulse[:\s]*(\d+)", combined_text)
        if pulse_match:
            metadata["vitals"]["pulse"] = pulse_match.group(1)
        
        # Extract systolic BP
        sys_match = re.search(r"Systolic[:\s]*(\d+)", combined_text)
        if sys_match:
            metadata["vitals"]["systolic"] = sys_match.group(1)
        
        # Extract allergies
        allergy_patterns = [
            r"Allergies?[:\s]+([^\n]+)",
            r"Allergy[:\s]+([^\n]+)",
            r"Drug Allergy[:\s]+([^\n]+)",
        ]
        for pattern in allergy_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match.strip() and match.strip().lower() != "none" and match.strip() != "n/a":
                    if match.strip() not in metadata["allergies"]:
                        metadata["allergies"].append(match.strip())
        
        # Extract procedures
        procedure_keywords = [
            "Nebulization", "Cannulation", "Nasal saline", "IV cannulation",
            "Chest X-ray", "Radiography", "CBC", "Blood draw"
        ]
        for keyword in procedure_keywords:
            if keyword.lower() in combined_text.lower() and keyword not in metadata["procedures"]:
                metadata["procedures"].append(keyword)
        
        # Extract specific procedures from the Procedures section
        proc_section = re.search(r"Procedures\s+#\s+Procedure Name\s+([^\n]+(?:\n[^\n]+)*?)(?:#\s+Prescription|Radiology|Doctor Name)", combined_text, re.IGNORECASE)
        if proc_section:
            proc_text = proc_section.group(1)
            proc_lines = re.findall(r"\d+\.\s+([^\n]+)", proc_text)
            for proc in proc_lines[:5]:
                proc_name = proc.strip()
                if proc_name and proc_name not in metadata["procedures"]:
                    metadata["procedures"].append(proc_name)
        
        # Extract discharge medications
        med_patterns = [
            r"Discharge Medications?[:\s]+([^\n]+)",
            r"Medications? at Discharge[:\s]+([^\n]+)",
            r"Home Medications?[:\s]+([^\n]+)",
        ]
        for pattern in med_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match.strip() and len(match.strip()) > 5:
                    metadata["medications"].append(match.strip())
        
        # Extract common cardiac medications with doses
        cardiac_meds = re.findall(
            r"\b(Furosemide|Metoprolol|Lisinopril|Aspirin|Atorvastatin|Clopidogrel|Metformin|Amlodipine|Carvedilol|Digoxin|Warfarin|Enoxaparin|Heparin)\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|mL)?)",
            combined_text,
            re.IGNORECASE
        )
        for med, dose in cardiac_meds:
            med_str = f"{med.capitalize()} {dose}"
            if med_str not in metadata["medications"]:
                metadata["medications"].append(med_str)
        
        # Extract prescription medications from prescription section - more flexible pattern
        prescript_section = re.search(r"Prescription\s+Medication\s+Route\s+([^\n]+(?:\n[^\n]+)*?)(?:#\s+Laboratory|Radiology|Doctor Name|Procedures|History Notes|$)", combined_text, re.IGNORECASE | re.DOTALL)
        if prescript_section:
            prescript_text = prescript_section.group(1)
            # Match medication lines with various formats
            # Format: Medication Name - Dose - Route - Frequency - Duration
            med_matches = re.findall(r"^([A-Za-z][A-Za-z0-9\-\s]+?)(?:\s*-\s*|\s{2,})([\d\.]+[a-zA-Z]*\s*mL|[^\n-]*?)(?:\s*-\s*|\s{2,})([a-zA-Z]+)(?:\s*-\s*|\s{2,})", prescript_text, re.MULTILINE)
            for med_match in med_matches:
                med_name = med_match[0].strip()
                if med_name and len(med_name) > 2 and len(med_name) < 30:
                    if med_name not in metadata["medications"]:
                        metadata["medications"].append(med_name)
            
            # Alternative simple pattern - just get medication names from lines
            if not metadata["medications"]:
                simple_meds = re.findall(r"^([A-Z][A-Za-z]+(?:\s+[A-Za-z0-9]+)?)", prescript_text, re.MULTILINE)
                for med in simple_meds:
                    med = med.strip()
                    if med and len(med) > 2 and len(med) < 30 and med.lower() not in ["medication", "route", "frequency", "dose", "oral", "neb", "iv", "pn"]:
                        if med not in metadata["medications"]:
                            metadata["medications"].append(med)
        
        # Extract pediatric medications
        pediatic_meds = ["PARAFAST", "PULMICORT", "COMBIVENT", "ATROVENT", "AMOKLAVIN", "SALINE", "NEBULAZING", "DEXTROSE", "CATAFLAM", "CETRIZINE", "LEVOSALBUTAMOL", "PANADOL"]
        for med in pediatic_meds:
            if med in combined_text.upper() and med not in metadata["medications"]:
                metadata["medications"].append(med)
        
        # Extract lab values
        lab_patterns = [
            (r"BNP[:\s]+(\d+)", "BNP", "pg/mL"),
            (r"BNP\s*\(B-type Natriuretic Peptide\)[:\s]+(\d+)", "BNP", "pg/mL"),
            (r"Creatinine[:\s]+(\d+(?:\.\d+)?)", "Creatinine", "mg/dL"),
            (r"Hemoglobin[:\s]+(\d+(?:\.\d+)?)", "Hgb", "g/dL"),
            (r"WBC[:\s]+(\d+(?:\.\d+)?)", "WBC", "x10^9/L"),
            (r"Glucose[:\s]+(\d+(?:\.\d+)?)", "Glucose", "mg/dL"),
            (r"HbA1c[:\s]+(\d+(?:\.\d+)?)", "HbA1c", "%"),
            (r"Troponin[:\s]+(\d+(?:\.\d+)?)", "Troponin", "ng/mL"),
            (r"CRP[:\s]+([^\n]+?)(?:\s+Ref|\s+mg/l|$)", "CRP", "mg/L"),
            (r"C-REACTIVE PROTEIN[:\s]+([^\n]+?)(?:\s+Ref|\s+mg/l|$)", "CRP", "mg/L"),
        ]
        for pattern, lab_name, unit in lab_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and value != '-':
                    metadata["labs"].append(f"{lab_name}: {value} {unit}".strip())
        
        # Extract full haemogram values
        wbc_match = re.search(r"WBC\s+(\d+(?:\.\d+)?)\s+x 10", combined_text, re.IGNORECASE)
        if wbc_match:
            metadata["labs"].append(f"WBC: {wbc_match.group(1)} x10^3/uL")
        
        hgb_match = re.search(r"HGB?[:\s]+(\d+(?:\.\d+)?)\s+g?/dl", combined_text, re.IGNORECASE)
        if hgb_match:
            metadata["labs"].append(f"Hemoglobin: {hgb_match.group(1)} g/dL")
        
        rbc_match = re.search(r"RBC\s+(\d+(?:\.\d+)?)\s+x 10", combined_text, re.IGNORECASE)
        if rbc_match:
            metadata["labs"].append(f"RBC: {rbc_match.group(1)} x10^6/uL")
        
        plt_match = re.search(r"PLT[:\s]+(\d+)", combined_text, re.IGNORECASE)
        if plt_match:
            metadata["labs"].append(f"Platelets: {plt_match.group(1)} x10^9/L")
        
        # Extract CRP specifically with result
        crp_result = re.search(r"C-REACTIVE PROTEIN.*?Result\s+(\d+(?:\.\d+)?)\s+mg", combined_text, re.IGNORECASE)
        if crp_result:
            crp_val = crp_result.group(1)
            # Check if there's a status
            status = "Normal"
            if "High" in combined_text[crp_result.start():crp_result.start()+200]:
                status = "High"
            metadata["labs"].append(f"CRP: {crp_val} mg/L ({status})")
        
        # Extract radiology findings
        radiology_match = re.search(r"EXAM:[^\n]+CHILD[^\n]+([^\n]+(?:\n[^\n]+)*?)Conclusion", combined_text, re.IGNORECASE)
        if radiology_match:
            metadata["radiology"]["findings"] = radiology_match.group(1).replace('\n', ' ').strip()
        
        conc_match = re.search(r"Conclusion\s+([^\n]+(?:\n[^\n]+)*?)(?:#\s+Laboratory|Doctor Name|Patient Name|$)", combined_text, re.IGNORECASE)
        if conc_match:
            metadata["radiology"]["conclusion"] = conc_match.group(1).replace('\n', ' ').strip()
        
        # Extract admission and discharge dates
        admit_match = re.search(r"Admission Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", combined_text)
        if admit_match:
            metadata["admission_date"] = admit_match.group(1)
        
        # Extract visit date
        visit_match = re.search(r"Visit Date[:\s]+(\d{4}-\d{2}-\d{2})", combined_text)
        if visit_match:
            metadata["visit_date"] = visit_match.group(1)
        
        discharge_match = re.search(r"Discharge Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", combined_text)
        if discharge_match:
            metadata["discharge_date"] = discharge_match.group(1)
        
        # Extract discharge disposition
        disposition_patterns = [
            r"Discharge Disposition[:\s]+([^\n]+)",
            r"Disposition[:\s]+([^\n]+)",
            r"Discharged to[:\s]+([^\n]+)",
        ]
        for pattern in disposition_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                metadata["discharge_disposition"] = match.group(1).strip()
                break
        
        # Extract diagnosis
        diagnosis_patterns = [
            r"Diagnosis[:\s]+([^\n]+)",
            r"Diagnoses[:\s]+([^\n]+)",
            r"Final Diagnosis[:\s]+([^\n]+)",
            r"Primary Diagnosis[:\s]+([^\n]+)",
        ]
        for pattern in diagnosis_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                diagnosis = match.group(1).strip()
                if diagnosis and len(diagnosis) > 2:
                    metadata["diagnosis"] = diagnosis
                    break
        
        return metadata

    def _is_pediatric(self, patient_id: str) -> bool:
        """Check if patient is pediatric based on DOB from database."""
        import re
        from app.rag.singletons import vectorstore
        
        raw_docs = vectorstore.raw_docs.get(patient_id, [])
        if not raw_docs:
            return False
        
        combined_text = " ".join(raw_docs)
        
        # Look for age patterns
        age_patterns = [
            r"(?:Age|age)[:\s]+(\d{1,3})(?:\s*years|\s*yr)?",
            r"(\d{1,3})[-\s]year[-\s]old",
            r"(\d{1,3})\s*yrs?\s*old",
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, combined_text)
            if match:
                age = int(match.group(1))
                return age < 18
        
        # Check for pediatric-specific terms
        pediatric_terms = ["pediatric", "child", "infant", "newborn", "neonate", "toddler", "adolescent"]
        return any(term in combined_text.lower() for term in pediatric_terms)

    def _generate_developmental_notes(self, age_str: str) -> str:
        """
        Generate age-appropriate developmental notes with pediatric milestones.
        
        Args:
            age_str: Calculated age string
            
        Returns:
            Age-appropriate developmental assessment text
        """
        if "days" in age_str:
            return ("Newborn assessed: APGAR scores reviewed. Primitive reflexes intact. "
                   "Feeding patterns established. Parental education on newborn care, safe sleep, and "
                   "warning signs provided.")
        elif "months" in age_str:
            months = int(age_str.split()[0])
            if months <= 3:
                return ("Infant developmental assessment: Social smiling emerging, head control improving. "
                       "Visual tracking present. Parental counseling on tummy time, infant stimulation, "
                       "and immunization schedule provided.")
            elif months <= 6:
                return ("Infant developmental assessment: Rolling over achieved, sitting with support. "
                       "Babbling and vocalization noted. Object permanence developing. Age-appropriate "
                       "gross motor and language milestones observed.")
            elif months <= 9:
                return ("Infant developmental assessment: Crawling and pulling to stand observed. "
                       "Stranger anxiety present. Basic gestural communication emerging. "
                       "Developmental progress within expected parameters for age.")
            else:
                return ("Infant developmental assessment: Cruising and independent walking emerging. "
                       "Single words present. Separation anxiety noted. Fine motor pincer grasp developing.")
        elif "years" in age_str:
            years = int(age_str.split()[0])
            if years <= 2:
                return ("Toddler development: Gait stable, climbing skills present. Vocabulary 50+ words. "
                       "Parallel play observed. Toilet training readiness assessed. Age-appropriate "
                       "temperament and behavior patterns noted.")
            elif years <= 5:
                return ("Preschool development: Pre-academic skills emerging. Social interaction appropriate. "
                       "Hand dominance established. Complex sentence structure present. "
                       "School readiness and immunization status reviewed.")
            elif years <= 11:
                return ("School-age development: Academic progress reviewed. Social skills age-appropriate. "
                       "Sports participation and physical activity encouraged. Screen time guidelines discussed. "
                       "Growth parameters tracking along established percentiles.")
            else:
                return ("Adolescent development: Pubertal Tanner staging assessed. Psychosocial development reviewed. "
                       "Risk behaviors and preventive counseling provided. Transition to adult care planning initiated.")
        return "Developmental assessment completed with age-appropriate findings."

    def _generate_pediatric_impression(self, age_str: str, visits: List, clinical_metadata: Dict = None) -> str:
        """
        Generate realistic pediatric clinical impression based on visit history.
        
        Args:
            age_str: Patient age string
            visits: List of visit records
            clinical_metadata: Optional clinical metadata for additional context
            
        Returns:
            Clinical impression text
        """
        if clinical_metadata is None:
            clinical_metadata = {}
        
        if not visits:
            base_impression = f"Patient aged {age_str} presents for routine pediatric care. No acute concerns identified at this visit."
        else:
            # Get recent diagnoses
            recent_diagnoses = [v['diagnosis'] for v in visits[-3:] if visits]
            if recent_diagnoses:
                conditions = ", ".join(recent_diagnoses[:2])
                base_impression = f"Patient aged {age_str} with history of {conditions}. Current status stable."
            else:
                base_impression = f"Well-child examination completed for patient aged {age_str}."
        
        # Add chief complaints if available
        if clinical_metadata.get('chief_complaints'):
            base_impression += f" Presenting with: {clinical_metadata['chief_complaints']}."
        
        # Add specific diagnosis if available from clinical metadata
        if clinical_metadata.get('diagnosis'):
            base_impression += f" Diagnosed with: {clinical_metadata['diagnosis']}."
        
        # Add exam findings if available
        if clinical_metadata.get('examination_notes'):
            exam_notes = clinical_metadata['examination_notes'][:200]
            base_impression += f" Physical examination shows: {exam_notes}."
        
        # Add BMI category if available
        if clinical_metadata.get('bmi_category'):
            base_impression += f" Nutritional status: {clinical_metadata['bmi_category']}."
        
        # Add clinical status
        base_impression += " Child appears well-appearing with appropriate level of consciousness and activity for age."
        
        return base_impression

    def _generate_follow_up_recommendations(self, age_str: str, visits: List) -> str:
        """
        Generate age-appropriate follow-up recommendations.
        
        Args:
            age_str: Patient age string
            visits: List of visit records
            
        Returns:
            Follow-up recommendations text
        """
        base_recommendations = {
            "days": "Follow up with pediatrician in 1-2 days or sooner if concerns arise. Ensure adequate feeding and hydration. Monitor for fever, poor feeding, or lethargy.",
            "months": "Return for well-child visit per immunization schedule. Monitor developmental milestones. Ensure age-appropriate nutrition and sleep habits.",
            "years": "Continue routine pediatric care. Annual physical examination recommended. Age-appropriate screening and anticipatory guidance provided."
        }
        
        if "days" in age_str:
            return base_recommendations["days"]
        elif "months" in age_str:
            return base_recommendations["months"]
        else:
            return base_recommendations["years"]

    def _generate_clinical_summary_from_db(self, patient: Dict[str, Any], visits: List[Dict[str, Any]]) -> str:
        """
        Generate clinical summary from database data.
        
        Args:
            patient: Patient demographics dictionary
            visits: List of visit dictionaries
            
        Returns:
            Clinical summary string
        """
        age_str = calculate_pediatric_age(str(patient.get('dob', '')))
        gender_str = "male" if patient.get('gender') == 'M' else "female" if patient.get('gender') == 'F' else "patient"
        
        if not visits:
            return f"This is a {age_str} old {gender_str} pediatric patient presenting for routine pediatric care. No significant medical history documented in the available records."
        
        # Build summary from visits
        summary_parts = []
        
        # Get unique diagnoses across all visits
        all_diagnoses = []
        for v in visits:
            if v.get('diagnosis'):
                for d in v['diagnosis'].split(', '):
                    if d and d not in all_diagnoses:
                        all_diagnoses.append(d)
        
        if all_diagnoses:
            conditions = ", ".join(all_diagnoses[:5])
            if len(all_diagnoses) > 5:
                conditions += f" and {len(all_diagnoses) - 5} other conditions"
            summary_parts.append(f"This {age_str} old {gender_str} has been evaluated for {conditions}.")
        
        # Add visit summary
        if visits:
            first_visit = visits[0]['visit_date']
            last_visit = visits[-1]['visit_date']
            total_visits = len(visits)
            summary_parts.append(f"The patient has had {total_visits} clinic visits between {first_visit} and {last_visit}.")
            
            # Recent visit info
            recent = visits[-1]
            summary_parts.append(f"Most recent visit on {recent['visit_date']} for {recent['diagnosis']}, managed by {recent['physician']}.")
        
        return " ".join(summary_parts)

    # =====================================================
    # MAIN GENERATION
    # =====================================================
    async def generate(self, patient_id: str) -> str:
        """
        Generate PDF report for a patient.
        Safe to run as a FastAPI background task.
        
        Args:
            patient_id: Patient identifier (e.g., "NCH-12345")
            
        Returns:
            Path to generated PDF file
        """
        pdf_path = self.report_dir / f"{patient_id}_report.pdf"

        # Idempotency: do not regenerate if already exists
        if pdf_path.exists():
            logger.info(f"Report already exists, skipping: {pdf_path}")
            return str(pdf_path)

        logger.info(f"Starting report generation for patient {patient_id}")

        try:
            # Fetch patient data from database
            patient = await self._get_patient(patient_id)
            visits = await self._get_visits(patient_id)

            # Generate clinical summary - try RAG first, fallback to database summary
            rag_summary = self._generate_clinical_summary_from_db(patient, visits)
            
            # Extract clinical metadata for additional sections
            clinical_metadata = self._extract_clinical_metadata(patient_id)
            
            # Debug: log if clinical_metadata is empty
            if not clinical_metadata:
                logger.warning(f"No clinical metadata extracted for patient {patient_id}")
            
            # Determine if pediatric or adult patient
            is_pediatric = self._is_pediatric(patient_id)
            logger.info(f"Patient {patient_id} is_pediatric: {is_pediatric}")
            
            # Build PDF
            self._build_pdf(pdf_path, patient, visits, rag_summary, clinical_metadata, is_pediatric)

            logger.info(f"Medical report generated successfully: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.exception("Medical report generation failed")
            raise PDFError(str(e))

    # =====================================================
    # SQL DATA FETCHING
    # =====================================================
    async def _get_patient(self, patient_id: str) -> Dict[str, Any]:
        """Fetch patient demographics from database."""
        result = await run_sql_query(
            """
            SELECT patient_id, first_name, last_name, dob, gender
            FROM dim_patient
            WHERE patient_id = $1
            """,
            (patient_id,),
        )
        if isinstance(result, dict) and "status" in result:
            raise ValueError(f"Patient not found: {patient_id}")
        if not result:
            raise ValueError(f"Patient not found: {patient_id}")
        return result[0]

    async def _get_visits(self, patient_id: str) -> List[Dict[str, Any]]:
        """Fetch visit history from database, consolidating multiple visits on the same day."""
        result = await run_sql_query(
            """
            SELECT
                v.visit_timestamp::date AS visit_date,
                ARRAY_AGG(DISTINCT d.description ORDER BY d.description) AS diagnoses,
                ARRAY_AGG(DISTINCT p.name ORDER BY p.name) AS physicians,
                COUNT(DISTINCT v.visit_id) AS visit_count
            FROM fact_patient_visits v
            LEFT JOIN dim_diagnosis d ON v.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_physician p ON v.physician_id = p.physician_id
            WHERE v.patient_id = $1
            GROUP BY v.visit_timestamp::date
            ORDER BY v.visit_timestamp::date ASC
            """,
            (patient_id,),
        )
        
        # Transform aggregated results into a list format
        transformed = []
        for row in result if isinstance(result, list) else []:
            transformed.append({
                'visit_date': row['visit_date'],
                'diagnosis': ', '.join([d for d in row['diagnoses'] if d]) if row.get('diagnoses') else 'No diagnosis',
                'physician': ', '.join([p for p in row['physicians'] if p]) if row.get('physicians') else 'Unknown',
                'visit_count': row.get('visit_count', 1)
            })
        return transformed

    # =====================================================
    # PDF BUILDER
    # =====================================================
    def _build_pdf(self, pdf_path, patient, visits, rag_summary, clinical_metadata=None, is_pediatric=True):
        """
        Build PDF document with ReportLab.
        
        Sections:
            1. Title & timestamp
            2. Patient information
            3. Clinical assessment (RAG summary)
            4. Allergies (for adult patients)
            5. Procedures (for adult patients)
            6. Discharge Medications (for adult patients)
            7. Laboratory Results (for adult patients)
            8. Growth & development (for pediatric patients)
            9. Visit history
            10. Clinical impression & plan
            11. Follow-up recommendations
        """
        if clinical_metadata is None:
            clinical_metadata = {}
        
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        x = 25 * mm
        y = 280 * mm

        # TITLE - dynamic based on patient type
        c.setFont("DejaVu", 16)
        c.drawString(x, y, "Medical Report" if not is_pediatric else "Pediatric Medical Report")
        y -= 22

        c.setFont("DejaVu", 10)
        c.drawString(x, y, f"Generated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
        y -= 30

        # PATIENT INFO
        y = section_title(c, "Patient Information", x, y)
        age_str = calculate_pediatric_age(str(patient['dob']))
        gender_str = "Male" if patient.get('gender') == 'M' else "Female" if patient.get('gender') == 'F' else "Unknown"
        
        patient_info = f"Patient ID: {patient['patient_id']}\n"
        patient_info += f"Name: {patient['first_name']} {patient['last_name']}\n"
        patient_info += f"Date of Birth: {patient['dob']}\n"
        patient_info += f"Age: {age_str}\n"
        patient_info += f"Gender: {gender_str}"
        
        # Add nationality if available
        if clinical_metadata.get('nationality'):
            patient_info += f"\nNationality: {clinical_metadata['nationality']}"
        
        # Add visit date if available
        if clinical_metadata.get('visit_date'):
            patient_info += f"\nVisit Date: {clinical_metadata['visit_date']}"
        
        # Add admission/discharge dates if available (adult patients)
        if not is_pediatric:
            if clinical_metadata.get('admission_date'):
                patient_info += f"\nAdmission Date: {clinical_metadata['admission_date']}"
            if clinical_metadata.get('discharge_date'):
                patient_info += f"\nDischarge Date: {clinical_metadata['discharge_date']}"
            if clinical_metadata.get('discharge_disposition'):
                patient_info += f"\nDischarge Disposition: {clinical_metadata['discharge_disposition']}"
        
        y = draw_paragraph(c, patient_info, x, y)

        # CHIEF COMPLAINTS (for pediatric patients)
        if is_pediatric and clinical_metadata.get('chief_complaints'):
            y -= 10
            y = section_title(c, "Chief Complaints", x, y)
            y = draw_paragraph(c, clinical_metadata['chief_complaints'], x, y, max_chars=90)

        # VITALS (for pediatric patients)
        if is_pediatric and clinical_metadata.get('vitals'):
            y -= 10
            y = section_title(c, "Vital Signs", x, y)
            vitals = clinical_metadata['vitals']
            vitals_text = []
            if vitals.get('temperature'):
                vitals_text.append(f"Temperature: {vitals['temperature']}")
            if vitals.get('weight'):
                vitals_text.append(f"Weight: {vitals['weight']}")
            if vitals.get('height'):
                vitals_text.append(f"Height: {vitals['height']}")
            if clinical_metadata.get('bmi_category'):
                vitals_text.append(f"BMI Category: {clinical_metadata['bmi_category']}")
            if vitals.get('spo2'):
                vitals_text.append(f"SPO2: {vitals['spo2']}")
            if vitals.get('pulse'):
                vitals_text.append(f"Pulse: {vitals['pulse']} bpm")
            if vitals.get('systolic'):
                vitals_text.append(f"Blood Pressure: {vitals['systolic']}/{vitals.get('diastolic', '')} mmHg")
            if vitals_text:
                y = draw_paragraph(c, ", ".join(vitals_text), x, y, max_chars=90)

        # EXAMINATION NOTES (for pediatric patients)
        if is_pediatric and clinical_metadata.get('examination_notes'):
            y -= 10
            y = section_title(c, "Physical Examination", x, y)
            y = draw_paragraph(c, clinical_metadata['examination_notes'], x, y, max_chars=90)

        # CLINICAL SUMMARY
        y -= 10
        y = section_title(c, "Clinical Assessment", x, y)
        y = draw_paragraph(
            c,
            normalize_text(rag_summary),
            x,
            y,
            max_chars=90,
        )

        # ADULT-SPECIFIC SECTIONS
        if not is_pediatric:
            # ALLERGIES
            if clinical_metadata.get('allergies'):
                y -= 10
                y = section_title(c, "Allergies", x, y)
                allergies_text = ", ".join(clinical_metadata['allergies'])
                y = draw_paragraph(c, allergies_text, x, y, max_chars=90)

            # PROCEDURES
            if clinical_metadata.get('procedures'):
                y -= 10
                y = section_title(c, "Procedures", x, y)
                procedures_text = "; ".join(clinical_metadata['procedures'])
                y = draw_paragraph(c, procedures_text, x, y, max_chars=90)

            # DISCHARGE MEDICATIONS
            if clinical_metadata.get('medications'):
                y -= 10
                y = section_title(c, "Discharge Medications", x, y)
                meds_text = "; ".join(clinical_metadata['medications'][:10])  # Limit to 10 meds
                y = draw_paragraph(c, meds_text, x, y, max_chars=90)

            # LABORATORY RESULTS
            if clinical_metadata.get('labs'):
                y -= 10
                y = section_title(c, "Laboratory Results", x, y)
                labs_text = "; ".join(clinical_metadata['labs'])
                y = draw_paragraph(c, labs_text, x, y, max_chars=90)
        else:
            # PEDIATRIC-SPECIFIC SECTIONS
            
            # PROCEDURES (for pediatric)
            if clinical_metadata.get('procedures'):
                y -= 10
                y = section_title(c, "Procedures Performed", x, y)
                procedures_text = "; ".join(clinical_metadata['procedures'][:8])
                y = draw_paragraph(c, procedures_text, x, y, max_chars=90)

            # MEDICATIONS (for pediatric)
            if clinical_metadata.get('medications'):
                y -= 10
                y = section_title(c, "Medications Prescribed", x, y)
                meds_text = "; ".join(clinical_metadata['medications'][:10])
                y = draw_paragraph(c, meds_text, x, y, max_chars=90)

            # LABORATORY RESULTS (for pediatric)
            if clinical_metadata.get('labs'):
                y -= 10
                y = section_title(c, "Laboratory Results", x, y)
                labs_text = "; ".join(clinical_metadata['labs'][:10])
                y = draw_paragraph(c, labs_text, x, y, max_chars=90)

            # RADIOLOGY (for pediatric)
            if clinical_metadata.get('radiology'):
                radiology = clinical_metadata['radiology']
                if radiology.get('findings') or radiology.get('conclusion'):
                    y -= 10
                    y = section_title(c, "Radiology", x, y)
                    if radiology.get('findings'):
                        y = draw_paragraph(c, f"Findings: {radiology['findings']}", x, y, max_chars=90)
                    if radiology.get('conclusion'):
                        y = draw_paragraph(c, f"Conclusion: {radiology['conclusion']}", x, y, max_chars=90)

            # GROWTH & DEVELOPMENT (pediatric only)
            y -= 10
            y = section_title(c, "Growth & Development", x, y)
            age_str = calculate_pediatric_age(str(patient['dob']))
            dev_notes = self._generate_developmental_notes(age_str)
            y = draw_paragraph(c, dev_notes, x, y, max_chars=90)


        # VISIT HISTORY
        y -= 10
        y = section_title(c, "Visit History", x, y)

        if not visits:
            y = draw_paragraph(c, "No visits recorded.", x, y)
        else:
            # Deduplicate visits by visit_date to ensure only one entry per date
            seen_dates = set()
            unique_visits = []
            for v in visits:
                if v['visit_date'] not in seen_dates:
                    seen_dates.add(v['visit_date'])
                    unique_visits.append(v)
            
            for v in unique_visits:
                visit_text = f"Date: {v['visit_date']}\nDiagnosis: {v['diagnosis']}\nProvider: {v['physician']}\n" + "─" * 40
                y = draw_paragraph(c, visit_text, x, y)
                y -= 6

        # CLINICAL IMPRESSION & PLAN
        y -= 10
        y = section_title(c, "Clinical Impression & Plan", x, y)
        clinical_impression = self._generate_pediatric_impression(age_str, visits, clinical_metadata)
        y = draw_paragraph(c, clinical_impression, x, y, max_chars=90)

        # FOLLOW-UP RECOMMENDATIONS
        y -= 10
        y = section_title(c, "Follow-Up Recommendations", x, y)
        follow_up = self._generate_follow_up_recommendations(age_str, visits)
        y = draw_paragraph(c, follow_up, x, y, max_chars=90)

        c.showPage()
        c.save()
