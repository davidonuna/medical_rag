import re
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import hashlib

from app.core.config import get_settings
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_templates import SQL_INTERPRETER_PROMPT
from app.agent.sql_tool import run_sql_query

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

QUARTERS = {
    "q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12),
    "quarter 1": (1, 3), "quarter 2": (4, 6), "quarter 3": (7, 9), "quarter 4": (10, 12),
    "first quarter": (1, 3), "second quarter": (4, 6), "third quarter": (7, 9), "fourth quarter": (10, 12)
}

AGE_GROUP_PATTERNS = {
    "neonate": (0, 28),
    "newborn": (0, 28),
    "infant": (0, 365),
    "toddler": (1, 3),
    "child": (4, 12),
    "adolescent": (13, 18),
    "teenager": (13, 18),
    "adult": (19, 64),
    "elderly": (65, 150),
    "pediatric": (0, 18),
    "senior": (65, 150)
}

settings = get_settings()

# =====================================================
# SQL Security Validator
# =====================================================
class SQLValidator:
    """Validate SQL for security and syntax compliance"""
    
    FORBIDDEN_KEYWORDS = {
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
        'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE', 'UNION'
    }
    
    @staticmethod
    def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax and security"""
        try:
            sql_upper = sql.upper()
            
            for keyword in SQLValidator.FORBIDDEN_KEYWORDS:
                if f" {keyword} " in f" {sql_upper} " or sql_upper.startswith(keyword):
                    return False, f"Dangerous keyword detected: {keyword}"
            
            if not (sql_upper.strip().startswith('SELECT') or sql_upper.strip().startswith('WITH')):
                return False, "Only SELECT queries are allowed"
            
            dangerous_patterns = ["--", "/*", "*/", "xp_", "sp_"]
            sql_lower = sql.lower()
            for pattern in dangerous_patterns:
                if pattern in sql_lower:
                    return False, f"Potentially dangerous pattern detected: {pattern}"
            
            # Allow semicolons at end of query (statement terminator)
            sql_stripped = sql.strip().rstrip(';')
            if ';' in sql:
                if sql.strip() != sql_stripped + ';':
                    return False, f"Potentially dangerous pattern detected: ;"
            
            return True, None
            
        except Exception as e:
            return False, f"SQL validation error: {str(e)}"

# =====================================================
# SQL Auditor for Compliance
# =====================================================
class SQLAuditor:
    """Audit SQL queries for compliance and security"""
    
    def __init__(self):
        self.logger = logging.getLogger("sql_audit")
    
    async def log_query_execution(
        self, 
        nl_query: str, 
        generated_sql: str, 
        execution_time: float, 
        result_count: int,
        user_context: Optional[Dict[str, Any]] = None
    ):
        """Log query execution for audit trail"""
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "natural_language_query": nl_query,
            "generated_sql": generated_sql,
            "execution_time_ms": execution_time * 1000,
            "result_count": result_count,
            "user_context": user_context if user_context is not None else {}
        }
        self.logger.info(f"SQL_AUDIT: {json.dumps(audit_record)}")

# =====================================================
# Enhanced Error Handler
# =====================================================
class SQLErrorHandler:
    """Provide user-friendly error messages for SQL issues"""
    
    @staticmethod
    def provide_user_friendly_error(error: Exception, query: str) -> Dict[str, Any]:
        """Convert technical errors to user-friendly messages"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return {
                "error": "Query took too long to execute",
                "suggestion": "Try narrowing your date range or adding more specific criteria",
                "help": "Consider adding specific time constraints or diagnosis details"
            }
        elif "syntax" in error_str or "invalid" in error_str:
            return {
                "error": "Unable to understand the question",
                "suggestion": "Try rephrasing with simpler medical terms",
                "examples": [
                    "How many patients were diagnosed with diabetes last year?",
                    "Show patients with hypertension",
                    "Count cancer cases this year"
                ]
            }
        elif "relation" in error_str and "does not exist" in error_str:
            return {
                "error": "Referenced medical data not found",
                "suggestion": "Check if the diagnosis or condition exists in the database",
                "help": "Try using more general medical terms"
            }
        else:
            return {
                "error": "Unable to process your request",
                "suggestion": "Please rephrase your question or contact support",
                "technical": str(error)
            }

# =====================================================
# Medical Query Preprocessor
# =====================================================
class MedicalQueryPreprocessor:
    """Enhance medical NLP preprocessing with synonym handling"""
    
    MEDICAL_SYNONYMS = {
        'cancer': ['malignancy', 'malignant tumor', 'carcinoma', 'neoplasm', 'oncologic'],
        'heart attack': ['myocardial infarction', 'cardiac arrest', 'mi', 'acute mi'],
        'stroke': ['cerebral vascular accident', 'cva', 'brain attack', 'cerebrovascular'],
        'diabetes': ['diabetes mellitus', 'dm', 'sugar diabetes', 'type 2 diabetes', 'type 1 diabetes'],
        'hypertension': ['high blood pressure', 'htn', 'elevated blood pressure', 'bp elevation'],
        'pneumonia': ['lung infection', 'chest infection', 'lower respiratory infection'],
        'asthma': ['reactive airway disease', 'rad', 'bronchial asthma'],
        'covid': ['covid-19', 'coronavirus', 'sars-cov-2', 'covid 19'],
        'copd': ['chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
        'chf': ['congestive heart failure', 'heart failure', 'cardiac failure'],
        'aki': ['acute kidney injury', 'kidney failure', 'renal failure', 'acute renal failure'],
        'uti': ['urinary tract infection', 'bladder infection', 'cystitis'],
        'dvt': ['deep vein thrombosis', 'blood clot in leg', 'venous thrombosis'],
        'pe': ['pulmonary embolism', 'lung clot'],
        'arrhythmia': ['irregular heartbeat', 'heart rhythm problem', 'atrial fibrillation'],
        'anemia': ['low blood count', 'low hemoglobin', 'iron deficiency'],
        'depression': ['major depressive disorder', 'clinical depression', 'depressive disorder'],
        'anxiety': ['anxiety disorder', 'generalized anxiety', 'gad'],
        'back pain': ['backache', 'lumbar pain', 'spinal pain', 'dorsalgia'],
        'headache': ['cephalalgia', 'migraine', 'tension headache']
    }
    
    @classmethod
    def normalize_medical_terms(cls, query: str) -> str:
        """Normalize medical terminology using synonyms"""
        normalized = query.lower()
        
        for canonical, synonyms in cls.MEDICAL_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in normalized:
                    normalized = normalized.replace(synonym, canonical)
        
        return normalized
    
    @classmethod
    def extract_medical_conditions(cls, query: str) -> List[str]:
        """Extract all medical conditions from query"""
        normalized = cls.normalize_medical_terms(query.lower())
        conditions = []
        
        for canonical in cls.MEDICAL_SYNONYMS.keys():
            if canonical in normalized:
                conditions.append(canonical)
        
        return conditions

# =====================================================
# Query Cache for Performance
# =====================================================
class QueryCache:
    """Simple query result caching for performance"""
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get_cache_key(self, query: str, params: Optional[tuple] = None) -> str:
        """Generate cache key for query"""
        content = query + (str(params) if params is not None else "")
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result if valid"""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self.ttl:
                return result
            else:
                del self.cache[cache_key]
        return None
    
    def set(self, cache_key: str, result: Any):
        """Cache query result"""
        self.cache[cache_key] = (result, datetime.now().timestamp())

# Global instances
sql_auditor = SQLAuditor()
sql_error_handler = SQLErrorHandler()
query_cache = QueryCache()

# =====================================================
# Enhanced Analytical SQL Generator
# =====================================================
class AnalyticalQueryGenerator:
    """Generate SQL for complex analytical healthcare queries"""
    
    def __init__(self, sql_interpreter):
        self.si = sql_interpreter
    
    def _has_explicit_time_filter(self, q: str) -> bool:
        """Check if query has explicit time references"""
        q_lower = q.lower()
        if re.findall(r"\b20\d{2}\b", q_lower):
            return True
        time_keywords = [
            "last year", "this year", "current year", "last month",
            "last week", "yesterday", "today", "recent"
        ]
        if re.search(r"last \d+ (years?|months?|weeks?|days?)", q_lower):
            return True
        return any(kw in q_lower for kw in time_keywords)

    def _extract_two_months(self, q: str) -> Optional[Tuple[int, int, int]]:
        """Extract two specific months for comparison (e.g., June and July)"""
        q_lower = q.lower()
        month_pattern = re.search(r"between\s+(\w+)\s+and\s+(\w+)\s*(?:of\s+)?(\d{4})?", q_lower)
        if month_pattern:
            month1_name = month_pattern.group(1)
            month2_name = month_pattern.group(2)
            year = int(month_pattern.group(3)) if month_pattern.group(3) else datetime.now().year
            
            month1 = MONTHS.get(month1_name)
            month2 = MONTHS.get(month2_name)
            
            if month1 and month2 and month1 != month2:
                return (month1, month2, year)
        return None

    def _extract_date_range(self, q: str) -> Tuple[str, str, str]:
        """Extract date range from query"""
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        q_lower = q.lower()
        
        years = set(int(y) for y in re.findall(r"\b20\d{2}\b", q))
        
        month_clause = ""
        year_clause = ""
        
        # Check for "between X and Y" patterns
        # Pattern 1: "between June and July 2025" or "between January and December 2025"
        month_pattern = re.search(r"between\s+(\w+)\s+and\s+(\w+)\s*(?:of\s+)?(\d{4})?", q_lower)
        if month_pattern:
            month1_name = month_pattern.group(1)
            month2_name = month_pattern.group(2)
            year_from_pattern = month_pattern.group(3)
            
            month1 = MONTHS.get(month1_name)
            month2 = MONTHS.get(month2_name)
            
            if month1 and month2:
                # Set the year if provided
                if year_from_pattern:
                    years.add(int(year_from_pattern))
                elif "last year" in q_lower:
                    years.add(current_year - 1)
                elif years:
                    # Use the first year found
                    pass
                else:
                    years.add(current_year)
                
                # Build month range clause
                first_year = min(years) if years else current_year
                year_clause = f" AND dd.year = {first_year}"
                month_clause = f" AND dd.month BETWEEN {month1} AND {month2}"
                year_list = str(first_year)
                return year_list, month_clause, year_clause
        
        # Pattern 1b: "June and July last year" or "June-July last year" (without "between")
        month_pattern2 = re.search(r"(\w+)\s+(?:and|-)\s+(\w+)\s+(last\s+year|this\s+year|\d{4})", q_lower)
        if month_pattern2:
            month1_name = month_pattern2.group(1)
            month2_name = month_pattern2.group(2)
            year_str = month_pattern2.group(3)
            
            month1 = MONTHS.get(month1_name)
            month2 = MONTHS.get(month2_name)
            
            if month1 and month2:
                if year_str == "last year":
                    years.add(current_year - 1)
                elif year_str == "this year":
                    years.add(current_year)
                else:
                    years.add(int(year_str))
                
                first_year = min(years) if years else current_year
                year_clause = f" AND dd.year = {first_year}"
                month_clause = f" AND dd.month BETWEEN {month1} AND {month2}"
                year_list = str(first_year)
                return year_list, month_clause, year_clause
        
        # Pattern 2: "between Q1 and Q2 of 2025" or "between quarter 1 and quarter 2 of 2025"
        quarter_pattern = re.search(r"between\s+(q\d|quarter\s*\d|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter)\s+and\s+(q\d|quarter\s*\d|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter)\s*(?:of\s+)?(\d{4})?", q_lower)
        if quarter_pattern:
            q1_name = quarter_pattern.group(1).strip()
            q2_name = quarter_pattern.group(2).strip()
            year_from_quarter = quarter_pattern.group(3)
            
            q1_range = QUARTERS.get(q1_name)
            q2_range = QUARTERS.get(q2_name)
            
            if q1_range and q2_range:
                if year_from_quarter:
                    years.add(int(year_from_quarter))
                elif not years:
                    years.add(current_year)
                
                first_year = min(years) if years else current_year
                year_clause = f" AND dd.year = {first_year}"
                # Get the month range from quarters
                start_month = q1_range[0]
                end_month = q2_range[1]
                month_clause = f" AND dd.month BETWEEN {start_month} AND {end_month}"
                year_list = str(first_year)
                return year_list, month_clause, year_clause
        
        # Pattern 2b: Single quarter - "in Q1 2025" or "Q1 2025" or "for Q2 2025"
        single_quarter_pattern = re.search(r"(?:in|for|during|of)?\s*(q[1-4])\s+(?:of\s+)?(\d{4})", q_lower)
        if single_quarter_pattern:
            q_name = single_quarter_pattern.group(1).strip()
            year_str = single_quarter_pattern.group(2)
            q_range = QUARTERS.get(q_name)
            if q_range:
                years = {int(year_str)}
                year_list = str(int(year_str))
                year_clause = f" AND dd.year = {int(year_str)}"
                month_clause = f" AND dd.month BETWEEN {q_range[0]} AND {q_range[1]}"
                return year_list, month_clause, year_clause
        
        # Pattern 2c: "last quarter 2024" or "quarter 3 2025" (word "quarter" with number)
        # First check for "last quarter YYYY" (no number)
        last_quarter_pattern = re.search(r"last\s+quarter\s+(?:of\s+)?(\d{4})", q_lower)
        if last_quarter_pattern:
            year_str = last_quarter_pattern.group(1)
            q_range = (10, 12)  # Q4 = months 10-12
            years = {int(year_str)}
            year_list = str(int(year_str))
            year_clause = f" AND dd.year = {int(year_str)}"
            month_clause = f" AND dd.month BETWEEN {q_range[0]} AND {q_range[1]}"
            return year_list, month_clause, year_clause
        
        # Then check for "quarter 3 2025" (with number)
        single_quarter_word_pattern = re.search(r"quarter\s*(\d)\s*(?:of\s+)?(\d{4})", q_lower)
        if single_quarter_word_pattern:
            q_num = int(single_quarter_word_pattern.group(1))
            year_str = single_quarter_word_pattern.group(2)
            q_range = ( (q_num - 1) * 3 + 1, q_num * 3 )
            years = {int(year_str)}
            year_list = str(int(year_str))
            year_clause = f" AND dd.year = {int(year_str)}"
            month_clause = f" AND dd.month BETWEEN {q_range[0]} AND {q_range[1]}"
            return year_list, month_clause, year_clause
        
        # Pattern 3: "between 2024 and 2025" (year range)
        year_range_pattern = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", q)
        if year_range_pattern:
            year1 = int(year_range_pattern.group(1))
            year2 = int(year_range_pattern.group(2))
            years = set(range(year1, year2 + 1))
            year_list = ",".join(map(str, sorted(years)))
            return year_list, "", ""
        
        # Pattern 4: Single month with year - "in June 2025" or "June 2025" or "for June 2025"
        single_month_pattern = re.search(r"(?:in|for|during|of)\s+(\w+)\s+(\d{4})", q_lower)
        if not single_month_pattern:
            single_month_pattern = re.search(r"(\w+)\s+(\d{4})\b", q_lower)
        
        if single_month_pattern:
            month_name = single_month_pattern.group(1)
            year_str = single_month_pattern.group(2)
            month_num = MONTHS.get(month_name)
            if month_num:
                years = {int(year_str)}
                year_list = str(int(year_str))
                month_clause = f" AND dd.month = {month_num}"
                year_clause = f" AND dd.year = {int(year_str)}"
                return year_list, month_clause, year_clause
        
        # Existing logic for other patterns
        m = re.search(r"last (\d+) years?", q)
        if m:
            n = int(m.group(1))
            years = set(range(current_year - n + 1, current_year + 1))
        
        if "last year" in q:
            years.add(current_year - 1)
        if "this year" in q or "current year" in q:
            years.add(current_year)
        if "last month" in q:
            month_clause = f" AND dd.month = {current_month - 1 if current_month > 1 else 12}"
            if current_month == 1:
                year_clause = f" AND dd.year = {current_year - 1}"
            else:
                year_clause = f" AND dd.year = {current_year}"
        
        if not years:
            years = {current_year}
        
        year_list = ",".join(map(str, sorted(years)))
        return year_list, month_clause, year_clause
    
    def _is_year_comparison_query(self, q: str) -> bool:
        """Check if query is asking for year-over-year comparison"""
        q_lower = q.lower()
        # Check for "between YYYY and YYYY" pattern (year range without month/quarter)
        year_range_pattern = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", q)
        if year_range_pattern:
            return True
        # Also check for phrases like "year over year", "yoy", "vs year", "year vs year"
        year_keywords = ["year over year", " yoy ", " yoy", "yoy ", "year over", " year to year", 
                         "vs year", "versus year", "compare years", "year vs year", "between 2024", "between 2025", "between 20"]
        if any(x in q_lower for x in year_keywords):
            return True
        return False
    
    def _extract_top_n(self, q: str) -> int:
        """Extract top N from query"""
        m = re.search(r"top (\d+)", q, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+) most common", q, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+) highest", q, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 10
    
    def _extract_comparison_periods(self, q: str) -> List[int]:
        """Extract two periods for comparison"""
        current_year = datetime.now().year
        
        m = re.search(r"(\d{4}) and (\d{4})", q)
        if m:
            return [int(m.group(1)), int(m.group(2))]
        
        m = re.search(r"(\d{4})\s*to\s*(\d{4})", q)
        if m:
            return [int(m.group(1)), int(m.group(2))]
        
        m = re.search(r"(\d{4})\s*(?:vs|versus)\s*(\d{4})", q, re.IGNORECASE)
        if m:
            return [int(m.group(1)), int(m.group(2))]
        
        if "last year vs this year" in q or "year over year" in q or "yoy" in q:
            return [current_year - 1, current_year]
        
        m = re.search(r"last (\d+) years?", q)
        if m:
            n = int(m.group(1))
            return [current_year - n, current_year]
        
        return [current_year - 1, current_year]
    
    def _extract_gender(self, q: str) -> Optional[str]:
        """Extract gender filter from query"""
        if re.search(r'\bfemale\b', q) or re.search(r'\bwomen\b', q) or re.search(r'\bwoman\b', q):
            return "F"
        if re.search(r'\bmale\b', q) or re.search(r'\bmen\b', q) or re.search(r'\bman\b', q):
            return "M"
        return None
    
    def _extract_age_group(self, q: str) -> Optional[Tuple[str, int, int]]:
        """Extract age group filter from query"""
        q_lower = q.lower()
        for group, (min_age, max_age) in AGE_GROUP_PATTERNS.items():
            if group in q_lower:
                return (group, min_age, max_age)
        return None
    
    def detect_intent(self, nl_query: str) -> str:
        """Detect the type of analytical query"""
        q = nl_query.lower()
        
        # Check for diagnosis-specific queries first (e.g., "patients with malaria")
        if ("diagnosed with" in q or "suffering from" in q) and ("patient" in q or "patients" in q):
            diagnosis = self.si._extract_diagnosis(q)
            if diagnosis:
                return "diagnosis_patients"
        
        # Check for "had <diagnosis>" pattern specifically
        had_match = re.search(r"\bhad\s+(\w+)", q)
        if had_match and ("patient" in q or "patients" in q):
            diagnosis_candidate = had_match.group(1)
            invalid_conditions = ["multiple", "several", "many", "few", "more", "less", "other", "any", "no", "some", "most", "all", "various"]
            if diagnosis_candidate.lower() not in invalid_conditions:
                diagnosis = self.si._extract_diagnosis(q)
                if diagnosis:
                    return "diagnosis_patients"
        
        years = re.findall(r"\b20\d{2}\b", q)
        multiple_years = len(set(years)) > 1
        
        is_data_quality = any(x in q for x in [
            "inconsisten", "duplicate", "missing", "without a", "without an", "without",
            "data entry error", "data quality", "abnormal gender",
            "consistent with", "recorded without", "consistency"
        ])
        if is_data_quality:
            if "without any visit" in q or "registered without" in q:
                return "dq_patients_no_visits"
            if "without a diagnosis" in q or "without diagnosis" in q or "without diagnoses" in q or "had no diagnosis" in q or "without any diagnosis" in q:
                return "dq_visits_no_diagnosis"
            if "without a" in q and "physician" in q or "without an assigned" in q or "without assigned" in q:
                return "dq_visits_no_physician"
            if "missing payer" in q or "without" in q and "payer" in q or "without payer" in q:
                return "dq_missing_payer"
            if "age" in q and ("date of birth" in q or "dob" in q or "inconsisten" in q):
                return "dq_age_dob_mismatch"
            if "duplicate" in q:
                return "dq_duplicate_patients"
            if "gender" in q and ("abnormal" in q or "error" in q or "distribution" in q):
                return "dq_gender_distribution"
            if "timestamp" in q and ("consistent" in q or "consistency" in q or "date" in q):
                return "dq_timestamp_consistency"
        
        is_recurrence = any(x in q for x in [
            "recurrence", "recurrent", "recurring", "recur ",
            "repeated", "repeat", "high-risk", "high risk"
        ])
        
        if is_recurrence:
            if re.search(r"(?:between\s+)?[qQ]([12])\s+(?:and|to|vs|versus)\s+[qQ]([12])|[qQ]\s*([12])\s*(?:to|-)\s*[qQ]\s*([12])", q):
                return "recurrence_qoq"
            if re.search(r"\bq[1-4]\b.*\d{4}", q) and ("change" in q or "differ" in q or "compar" in q):
                return "recurrence_qoq"
            if re.search(r"\b20\d{2}\b.*\bq[1-4]\b", q) and ("change" in q or "differ" in q or "compar" in q):
                return "recurrence_qoq"
            # Handle single quarter queries - route to appropriate method with quarter filter
            if re.search(r"\bq[1-4]\b", q) and ("in q" in q or "for q" in q or "during q" in q):
                return "recurrence_highest_rates"
            if "quarter" in q and ("q1" in q or "q2" in q or "q3" in q or "q4" in q) and ("change" in q or "differ" in q or "compar" in q):
                return "recurrence_qoq"
            if "change" in q and ("q1" in q or "q2" in q or "q3" in q or "q4" in q):
                return "recurrence_qoq"
            if "payer" in q or "insurance" in q or "coverage" in q:
                return "recurrence_by_payer"
            if "differ" in q and ("payer" in q or "insurance" in q):
                return "recurrence_by_payer"
            if "physician" in q or "doctor" in q:
                if "proportion" in q or "repeat" in q or "manage" in q:
                    return "physician_repeat_patients"
                return "recurrence_by_physician"
            if "change" in q or "significant" in q or "particular" in q:
                return "diagnosis_yoy"
            # Check for time span BEFORE average
            if "time span" in q or "time between" in q or "duration" in q or "first and last" in q:
                return "recurrence_time_span"
            # Check for age group variation BEFORE highest
            if re.search(r'\bage\b', q) and ("vary" in q or "vary by" in q or "by age" in q):
                return "recurrence_by_age_group"
            if re.search(r'\bage\b', q):
                return "recurrence_by_age_group"
            # Check for high-risk patients BEFORE "how many"
            if "high-risk" in q or "high risk" in q or "classify" in q or "classified" in q:
                return "recurrence_high_risk"
            if "highest" in q or "most" in q:
                return "recurrence_highest_rates"
            if "rate" in q and "payer" not in q and "insurance" not in q:
                return "recurrence_highest_rates"
            if "average" in q or "mean" in q or "avg" in q:
                return "recurrence_avg_count"
            if "how many" in q and "patient" in q:
                return "recurrence_patient_count"
            if "patient" in q and ("repeat" in q or "experience" in q or "same condition" in q):
                return "recurrence_repeated_patients"
            if "physician" in q or "doctor" in q:
                return "recurrence_by_physician"
            if "payer" in q or "insurance" in q:
                return "recurrence_by_payer"
            return "recurrence_highest_rates"
        
        is_retention = any(x in q for x in [
            "retention", "return", "follow-up", "followup", "follow up",
            "come back", "revisit", "re-visit"
        ])
        if not is_retention:
            is_retention = "first-time" in q or "first time" in q
        if not is_retention:
            is_retention = "within 30 day" in q or "within 60 day" in q or "within 90 day" in q
        
        if is_retention:
            if "three-month" in q or "three month" in q or "six-month" in q or "six month" in q or "3-month" in q or "6-month" in q or "3 month" in q or "6 month" in q:
                return "retention_3m_6m"
            if "30 day" in q or "within 30" in q or ("registered" in q and "month" in q):
                return "retention_30day"
            if "how long" in q or "typically return" in q or "after" in q and "diagnosis" in q:
                return "retention_return_gap"
            if "first-time" in q or "first time" in q or "proportion" in q or "percentage" in q:
                return "retention_first_time"
            if "diagnosis" in q or "diagnos" in q or "condition" in q:
                return "retention_by_diagnosis"
            if "payer" in q or "insurance" in q:
                return "retention_by_payer"
            return "retention_first_time"
        
        # Check for location/residential area queries first - must be before ranking
        q_check = q.lower()
        
        is_payer = any(x in q for x in [
            "payer", "insurance", "insured", "self-pay", "selfpay",
            "self pay", "cash pay", "coverage", "private insurance",
            "public insurance"
        ])
        
        if is_payer and not is_recurrence:
            # Check for specific payer queries first (percentage, vs, etc.)
            if "percentage" in q or "proportion" in q:
                return "payer_selfpay_vs_insured"
            if "self" in q and ("insur" in q or "pay" in q):
                return "payer_selfpay_vs_insured"
            if " vs " in q or "versus" in q:
                return "payer_selfpay_vs_insured"
            # Then check for year comparison
            is_payer_comparison = self._is_year_comparison_query(q)
            if is_payer_comparison:
                return "payer_comparison"
            if "over time" in q or "trend" in q or "change" in q or "utilization" in q:
                return "payer_over_time"
            if "diagnos" in q or "condition" in q or "disease" in q or "common" in q:
                return "payer_diagnoses"
            if "private" in q and "public" in q:
                return "payer_comparison"
            if "private" in q or "public" in q:
                return "payer_comparison"
            if "highest" in q or "most" in q or "top" in q:
                return "payer_highest"
            if "distribution" in q or "breakdown" in q or "by payer" in q:
                return "payer"
            if "volume" in q or "vary" in q or "differ" in q or "compare" in q:
                return "payer_comparison"
            return "payer"
        
        is_staffing = any(x in q for x in [
            "staffing", "patient load", "daily load", "capacity",
            "underutiliz", "projected", "projection", "forecast"
        ])
        if not is_staffing:
            is_staffing = "peak" in q and any(x in q for x in ["month", "staff"])
        if not is_staffing:
            is_staffing = "peak" in q and "day" in q and "hour" not in q and "consultation" not in q
        if not is_staffing:
            is_staffing = "hourly" in q and ("visit" in q or "pattern" in q or "staff" in q)
        
        if is_staffing:
            # Check for physician-specific queries first
            if "physician" in q or "doctor" in q:
                if "evenly" in q or "distribution" in q or "spread" in q:
                    return "physician_workload_distribution"
                if "peak" in q or "hour" in q:
                    return "physician_peak_hours"
            if "average" in q or "avg" in q or "daily" in q and "load" in q:
                return "staffing_avg_daily_load"
            if "peak" in q and "month" in q:
                return "staffing_peak_months"
            if "peak" in q and ("day" in q or "days" in q):
                return "staffing_peak_days"
            if "capacity" in q or ("specialty" in q or "specialties" in q) and "highest" in q:
                return "staffing_specialty_capacity"
            if "underutiliz" in q:
                return "staffing_underutilized"
            if "hourly" in q or ("hour" in q and ("visit" in q or "pattern" in q)):
                return "staffing_hourly_patterns"
            if "project" in q or "forecast" in q or "upcoming" in q:
                return "staffing_projected_load"
            return "staffing_avg_daily_load"
        
        if any(re.search(r'\b' + x + r'\b', q_check) for x in ["residential", "residence", "area", "areas", "location", "region", "district", "city", "province"]):
            if "growth" in q or "opportunit" in q:
                return "growth_geographic"
            if "registered" in q or "patient" in q and "diagnos" not in q:
                return "location_patients"
            return "location"
        
        if "chronic" in q_check:
            if "proportion" in q or "workload" in q or "driven" in q:
                return "chronic_workload_proportion"
            return "chronic_conditions"
        
        if "busiest" in q or ("busy" in q and ("day" in q or "month" in q)):
            if "day" in q or "week" in q:
                return "visit_day_of_week"
            if "month" in q:
                return "visit_month"
            return "visit_day_of_week"
        
        is_anomaly = any(x in q for x in [
            "spike", "outbreak", "unusual", "unexpected", "abnormal",
            "anomal", "sudden"
        ])
        if is_anomaly:
            if "pediatric" in q or "child" in q:
                return "anomaly_pediatric"
            if "weekend" in q:
                return "anomaly_weekend"
            if "physician" in q or "doctor" in q:
                return "anomaly_physician"
            if "diagnos" in q or "disease" in q:
                return "anomaly_diagnosis_spikes"
            return "anomaly_diagnosis_spikes"
        
        if "top " in q or "most common" in q or "highest" in q:
            # Check for diagnosis-specific queries first
            diagnosis = self.si._extract_diagnosis(q)
            if diagnosis and ("patient" in q or "patients" in q):
                return "diagnosis_patients"
            if "growth" in q and ("specialty" in q or "specialties" in q):
                return "specialty_growth"
            if "growth" in q and ("diagnos" in q or "diagnosis" in q or "diagnoses" in q):
                return "growth_emerging_diagnoses"
            if "repeat" in q or "recurrent" in q:
                return "physician_repeat_patients"
            if "pediatric" in q or "children" in q or "child" in q:
                return "pediatric_diagnoses"
            if "physician" in q or "doctor" in q:
                return "physician"
            if "visit" in q and ("volume" in q or "month" in q or "day" in q):
                return "visit_count"
            return "ranking"
        
        # Handle "specialty change" or "specialty distribution" queries BEFORE generic period comparison
        if "specialty" in q or "specialties" in q:
            if "change" in q or "differ" in q or "compar" in q or "distribution" in q:
                return "specialty_growth"
            if "growth" in q or "fastest" in q or "greatest" in q:
                return "specialty_growth"
        
        if "percentage" in q or "proportion" in q:
            if "one visit" in q or "single visit" in q or "multiple visit" in q:
                return "visit_single_multiple"
            if "weekend" in q:
                return "visit_weekend"
            if "pediatric" in q or "adult" in q or "elderly" in q:
                return "patient_age_proportion"
            if "workload" in q and "chronic" in q:
                return "chronic_workload_proportion"
        
        if "increasing" in q or "decreasing" in q:
            if "diagnosis" in q or "diagnoses" in q or "disease" in q:
                return "diagnosis_yoy"
        
        if "year over year" in q or "yoy" in q:
            if "diagnosis" in q or "diagnoses" in q or "each diagnosis" in q:
                return "diagnosis_yoy"
            if "visit" in q or "volume" in q:
                return "visit_yoy"
        
        if "visit" in q and "growth rate" in q:
            return "visit_yoy"
        
        if "monthly" in q and "trend" in q and "visit" in q:
            return "visit_month"
        
        if "case mix" in q:
            return "case_mix_yoy"
        
        if "weekend" in q and ("weekday" in q or " vs " in q or "versus" in q):
            return "visit_weekend"
        
        # Check for workload drivers / emerging diagnoses before general period comparison
        if ("workload" in q or "driver" in q or "emerged" in q) and "diagnos" in q:
            return "growth_emerging_diagnoses"
        
        if "vs " in q or "versus" in q or "compared" in q or "year over year" in q or "yoy" in q:
            if any(c in q for c in ["diabetes", "hypertension", "cancer", "heart"]):
                return "comparative"
            return "period_comparison"
        
        # Handle "time between first and last" queries BEFORE period comparison
        if "time between" in q or "first and last" in q or "time span" in q:
            return "recurrence_time_span"
        
        # Handle "X and Y" year patterns
        if multiple_years and (" and " in q or " in " in q):
            return "period_comparison"
        
        # Check for aggregate first (average, mean) before demographic
        if "average" in q or "mean" in q:
            if "visit" in q and "per patient" in q:
                return "aggregate"
            if "age" in q and "visit" not in q:
                return "patient_avg_age"
            if "time between" in q or "first and last" in q or "time span" in q:
                return "recurrence_time_span"
            if "recurrence" in q:
                return "recurrence_avg_count"
            if "daily" in q:
                return "staffing_avg_daily_load"
            return "aggregate"
        
        if ("percentage" in q or "rate" in q) and "recurrence" not in q and "retention" not in q:
            return "aggregate"
        
        if "gender" in q or "by sex" in q or "male" in q or "female" in q or "age group" in q or "by age" in q:
            if "ratio" in q and ("over time" in q or "changing" in q or "trend" in q):
                return "gender_over_time"
            if "diagnosis" in q or "disease" in q or "condition" in q:
                return "disease_demographic"
            if "over time" in q or "changing" in q or "trend" in q:
                return "gender_over_time"
            return "demographic"
        
        if "age" in q:
            return "demographic"
        
        if "physician" in q or "doctor" in q or "provider" in q or "staff" in q:
            if "repeat" in q or "recurrent" in q or "multiple visits" in q:
                return "physician_repeat_patients"
            if "distribution" in q or "evenly" in q or "spread" in q:
                return "physician_workload_distribution"
            if "hour" in q or "peak" in q or "working hour" in q:
                return "physician_peak_hours"
            if "month" in q or "vary" in q:
                return "physician_workload_trend"
            if "register" in q:
                return "staff_registrations"
            return "physician"
        
        if "insurance" in q or "payer" in q or "coverage" in q:
            return "payer"
        
        if "birth" in q or "born" in q:
            return "births"
        
        if ("registered" in q or "registering" in q or "signed up" in q or "joined" in q) and ("each month" in q or "monthly" in q or "each year" in q or "yearly" in q or "per month" in q or "per year" in q or "month" in q or "year" in q):
            return "patient_registration_trend"
        
        if "increasing" in q or "decreasing" in q or "growing" in q:
            if "visit" in q or "volume" in q or "patient" in q or "overall" in q:
                return "visit_yoy"
            if "diagnosis" in q or "diagnoses" in q or "disease" in q:
                return "diagnosis_yoy"
        
        if "growth" in q or "emerging" in q or "expansion" in q:
            if "geographic" in q or re.search(r'\barea\b', q):
                return "growth_geographic"
            if "specialty" in q or "specialties" in q:
                return "specialty_growth"
            if "diagnos" in q or "workload" in q or "contributor" in q:
                return "growth_emerging_diagnoses"
        
        if "trend" in q or "over time" in q:
            if "registration" in q or "registering" in q or ("patient" in q and "visit" not in q and "volume" not in q):
                return "patient_registration_trend"
            if "visit" in q or "volume" in q or "overall" in q:
                return "visit_yoy"
        
        if ("children" in q or "child" in q) and ("under" in q or "over" in q or "age" in q or "years" in q):
            return "patient_count"
        
        if "visit" in q:
            if "growth rate" in q or "growth" in q and "rate" in q:
                return "visit_yoy"
            if "per " in q and any(x in q for x in ["day", "week", "month", "quarter"]):
                return "visit_timeframe"
            if "per specialty" in q or "per" in q and "specialty" in q:
                return "specialty_visits"
            if "total" in q or "how many" in q:
                return "visit_count"
            if "quarter" in q:
                return "visit_quarter_trend"
            if "weekend" in q and ("weekday" in q or " vs " in q or "versus" in q):
                return "visit_weekend"
        
        if "consultation" in q and "hour" in q:
            return "physician_peak_hours"
        
        if "peak" in q and "hour" in q:
            return "physician_peak_hours"
        
        if "quarter" in q and ("volume" in q or "change" in q or "across" in q):
            return "visit_quarter_trend"
        
        if ("multiple visit" in q or "more than one visit" in q) and "patient" in q:
            return "visit_single_multiple"
        
        if ("children" in q or "child" in q) and ("under" in q or "over" in q or "age" in q):
            return "patient_count"
        
        if ("over " in q and "years" in q) or ("under " in q and "years" in q) or ("between " in q and "years" in q) or ("age" in q and ("over" in q or "under" in q or "between" in q)):
            return "patient_count"
        
        if "how many" in q or "count" in q or "number of" in q:
            return "patient_count"
        
        if "icd" in q or "code" in q:
            return "icd10_ranking"
        
        if "trend" in q or "over time" in q:
            if "diagnosis" in q or "diagnoses" in q or "diseases" in q or "condition" in q:
                return "diagnosis_trend"
        
        if "seasonal" in q or "season" in q:
            return "seasonal_pattern"
        
        if "growth" in q and ("each diagnosis" in q or "diagnoses" in q):
            return "diagnosis_yoy"
        
        if "increasing" in q or "decreasing" in q or "increased" in q or "decreased" in q:
            if "diagnosis" in q or "diagnoses" in q or "disease" in q:
                return "diagnosis_yoy"
        
        return "unknown"
    
    def ranking_query(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for ranking queries (top N diagnoses, conditions, etc.)"""
        q = MedicalQueryPreprocessor.normalize_medical_terms(nl_query.lower())
        top_n = self._extract_top_n(nl_query)
        gender = self._extract_gender(q)
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        gender_clause = ""
        if gender:
            gender_clause = f" AND p.gender = '{gender}'"
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
            where_clause = f"WHERE dd.year IN ({year_list}){month_clause}{gender_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
            where_clause = f"WHERE dd.year IN ({periods[0]}, {periods[1]}){gender_clause}"
        else:
            time_filter = ""
            if gender:
                where_clause = f"WHERE 1=1{gender_clause}"
            else:
                where_clause = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {where_clause}
            GROUP BY dd.year, d.description
            ORDER BY d.description, dd.year
            LIMIT {top_n * 2};
            """.strip()
        else:
            sql = f"""
            SELECT 
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {where_clause}
            GROUP BY d.description
            ORDER BY patient_count DESC
            LIMIT {top_n};
            """.strip()
        return sql, []
    
    def location_query(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Generate SQL for location-based queries (residential areas, regions)"""
        q = MedicalQueryPreprocessor.normalize_medical_terms(nl_query.lower())
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        diagnosis = self.si._extract_diagnosis(nl_query)
        
        top_n = self._extract_top_n(nl_query) if "top" in q else 10
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f" AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f" AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        includes_diagnosis = "diagnosis" in q or "diagnoses" in q
        invalid_diagnoses = {"which", "what", "most", "common", "top", "all", "each"}
        if diagnosis and diagnosis.lower() not in invalid_diagnoses and len(diagnosis) > 2:
            base_where = "d.description ILIKE $1"
            if is_comparison:
                sql = f"""
                SELECT 
                    dd.year,
                    COALESCE(p.residence, 'Unknown') AS residence,
                    d.description AS diagnosis,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    COUNT(*) AS visit_count,
                    LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year) AS prev_year_visits,
                    COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year) AS yoy_change
                FROM fact_patient_visits fpv
                JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
                JOIN dim_patient p ON fpv.patient_id = p.patient_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE {base_where}{time_filter}
                GROUP BY dd.year, p.residence, d.description
                ORDER BY p.residence, dd.year
                LIMIT {top_n};
                """.strip()
            else:
                sql = f"""
                SELECT 
                    COALESCE(p.residence, 'Unknown') AS residence,
                    d.description AS diagnosis,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    COUNT(*) AS visit_count
                FROM fact_patient_visits fpv
                JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
                JOIN dim_patient p ON fpv.patient_id = p.patient_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE {base_where}{time_filter}
                GROUP BY p.residence, d.description
                ORDER BY patient_count DESC
                LIMIT {top_n};
                """.strip()
            return sql, [f"%{diagnosis}%"]
        else:
            if time_filter:
                if is_comparison:
                    periods = self._extract_comparison_periods(q)
                    where_clause = f"WHERE dd.year IN ({periods[0]}, {periods[1]})"
                else:
                    year_list = self._extract_date_range(q)[0]
                    where_clause = f"WHERE dd.year IN ({year_list})"
            else:
                where_clause = ""
            
            if includes_diagnosis:
                if is_comparison:
                    # Show top diagnoses per residence with YoY comparison - simplified
                    sql = f"""
                    SELECT 
                        dd.year,
                        COALESCE(p.residence, 'Unknown') AS residence,
                        d.description AS diagnosis,
                        COUNT(DISTINCT fpv.patient_id) AS patient_count,
                        COUNT(*) AS visit_count,
                        LAG(COUNT(*)) OVER (PARTITION BY p.residence, d.description ORDER BY dd.year) AS prev_year_visits,
                        COUNT(*) - COALESCE(LAG(COUNT(*)) OVER (PARTITION BY p.residence, d.description ORDER BY dd.year), 0) AS yoy_change
                    FROM fact_patient_visits fpv
                    JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
                    JOIN dim_patient p ON fpv.patient_id = p.patient_id
                    LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                    {where_clause}
                    GROUP BY dd.year, p.residence, d.description
                    ORDER BY p.residence, dd.year, visit_count DESC
                    LIMIT {top_n};
                    """.strip()
                else:
                    # For each residence, show top diagnoses
                    sql = f"""
                    SELECT 
                        residence,
                        diagnosis,
                        patient_count,
                        visit_count,
                        pct_of_residence
                    FROM (
                        SELECT 
                            COALESCE(p.residence, 'Unknown') AS residence,
                            d.description AS diagnosis,
                            COUNT(DISTINCT fpv.patient_id) AS patient_count,
                            COUNT(*) AS visit_count,
                            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY p.residence), 2) AS pct_of_residence,
                            ROW_NUMBER() OVER (PARTITION BY p.residence ORDER BY COUNT(*) DESC) AS rn
                        FROM fact_patient_visits fpv
                        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
                        JOIN dim_patient p ON fpv.patient_id = p.patient_id
                        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                        {where_clause}
                        GROUP BY p.residence, d.description
                    ) ranked
                    WHERE rn <= {top_n}
                    ORDER BY residence, visit_count DESC;
                    """.strip()
            elif is_comparison:
                sql = f"""
                SELECT 
                    dd.year,
                    COALESCE(p.residence, 'Unknown') AS residence,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    COUNT(*) AS visit_count,
                    LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year) AS prev_year_visits,
                    COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year) AS yoy_change,
                    ROUND(
                        100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year)) / 
                        NULLIF(LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year), 0)
                    , 2) AS yoy_change_pct
                FROM fact_patient_visits fpv
                JOIN dim_patient p ON fpv.patient_id = p.patient_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                {where_clause}
                GROUP BY dd.year, p.residence
                ORDER BY p.residence, dd.year
                LIMIT {top_n};
                """.strip()
            else:
                sql = f"""
                SELECT 
                    COALESCE(p.residence, 'Unknown') AS residence,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    COUNT(*) AS visit_count
                FROM fact_patient_visits fpv
                JOIN dim_patient p ON fpv.patient_id = p.patient_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                {where_clause}
                GROUP BY p.residence
                ORDER BY patient_count DESC
                LIMIT {top_n};
                """.strip()
            return sql, []
    
    def comparative_analysis(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Generate SQL for comparing two conditions or periods"""
        q = MedicalQueryPreprocessor.normalize_medical_terms(nl_query.lower())
        conditions = MedicalQueryPreprocessor.extract_medical_conditions(q)
        
        if len(conditions) >= 2:
            year_list, month_clause, _ = self._extract_date_range(q)
            
            sql = f"""
            SELECT 
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE d.description ILIKE $1 OR d.description ILIKE $2
              AND dd.year IN ({year_list}){month_clause}
            GROUP BY d.description
            ORDER BY patient_count DESC;
            """.strip()
            return sql, [f"%{conditions[0]}%", f"%{conditions[1]}%"]
        
        periods = self._extract_comparison_periods(q)
        if len(periods) == 2:
            sql = f"""
            SELECT 
                dd.year,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year IN ({periods[0]}, {periods[1]})
            GROUP BY dd.year
            ORDER BY dd.year;
            """.strip()
            return sql, []
        
        return None
    
    def diagnosis_trend_analysis(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Generate SQL for diagnosis trends over time"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        diagnosis = self.si._extract_diagnosis(nl_query)
        
        invalid_words = {"how", "do", "specific", "which", "what", "all", "common", "most", "the", "a", "an", "over", "time", "trends", "trend"}
        
        is_valid = False
        if diagnosis:
            diagnosis_words = diagnosis.lower().split()
            is_valid = not any(word in invalid_words for word in diagnosis_words) and len(diagnosis) > 2
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        if is_valid:
            sql = f"""
            SELECT 
                dd.year,
                dd.month,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE d.description ILIKE $1
            GROUP BY dd.year, dd.month, d.description
            ORDER BY dd.year, dd.month;
            """.strip()
            return sql, [f"%{diagnosis}%"]
        else:
            sql = f"""
            SELECT 
                dd.year,
                dd.month,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {time_filter}
            GROUP BY dd.year, dd.month, d.description
            ORDER BY dd.year, dd.month, patient_count DESC
            LIMIT 50;
            """.strip()
            return sql, []
    
    def seasonal_pattern_analysis(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Generate SQL for seasonal patterns in diagnoses"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            where_clause = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            where_clause = ""
        
        sql = f"""
        SELECT 
            dd.month,
            d.description AS diagnosis,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            COUNT(*) AS visit_count
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {where_clause}
        GROUP BY dd.month, d.description
        ORDER BY dd.month, patient_count DESC;
        """.strip()
        return sql, []
    
    def diagnosis_yoy_analysis(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Generate SQL for diagnosis year-over-year changes"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        month_pattern = re.search(r"between\s+(\w+)\s+and\s+(\w+)\s*(?:of\s+)?(\d{4})?", q)
        is_mom_comparison = False
        if month_pattern:
            month1_name = month_pattern.group(1)
            month2_name = month_pattern.group(2)
            year_from_pattern = month_pattern.group(3)
            month1 = MONTHS.get(month1_name)
            month2 = MONTHS.get(month2_name)
            if month1 and month2 and abs(month1 - month2) <= 2:
                is_mom_comparison = True
        
        if has_time and is_mom_comparison:
            year_list, month_clause, _ = self._extract_date_range(q)
            month_range = re.search(r"BETWEEN\s+(\d+)\s+AND\s+(\d+)", month_clause)
            if month_range:
                m1 = int(month_range.group(1))
                m2 = int(month_range.group(2))
                sql = f"""
                SELECT 
                    dd.month,
                    d.description AS diagnosis,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    COUNT(*) AS visit_count,
                    LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.month) AS prev_month_visits,
                    COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.month) AS mom_change,
                    ROUND(
                        100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.month)) / 
                        NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.month), 0)
                    , 2) AS mom_change_pct
                FROM fact_patient_visits fpv
                JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE dd.year IN ({year_list}) AND dd.month BETWEEN {m1} AND {m2}
                GROUP BY dd.month, d.description
                ORDER BY d.description, dd.month;
                """.strip()
                return sql, []
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"WHERE dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            dd.year,
            d.description AS diagnosis,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            COUNT(*) AS visit_count,
            LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year) AS prev_year_visits,
            COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year) AS yoy_change,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year)) / 
                NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year), 0)
            , 2) AS yoy_change_pct
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY dd.year, d.description
        ORDER BY d.description, dd.year;
        """.strip()
        return sql, []
    
    def specialty_growth_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for specialty growth over time"""
        q = nl_query.lower()
        
        two_months = self._extract_two_months(q)
        
        if two_months:
            month1, month2, year = two_months
            
            sql = f"""
            SELECT 
                ph.specialty,
                {month1} AS month_from,
                {month2} AS month_to,
                {year} AS year,
                SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS visits_month_from,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) AS visits_month_to,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                    SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS absolute_change,
                ROUND(
                    100.0 * (
                        SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                        SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END)
                    ) / NULLIF(SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END), 0)
                , 2) AS mom_growth_pct
            FROM fact_patient_visits fpv
            JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year = {year} AND dd.month IN ({month1}, {month2})
            GROUP BY ph.specialty
            HAVING SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) > 0
            ORDER BY mom_growth_pct DESC NULLS LAST;
            """.strip()
            return sql, []
        
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        quarter_match = re.search(r"(q\d|quarter\s*\d)", q)
        if quarter_match:
            sql = f"""
            SELECT 
                ph.specialty,
                dd.quarter,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count
            FROM fact_patient_visits fpv
            JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {time_filter}
            GROUP BY ph.specialty, dd.quarter
            ORDER BY ph.specialty, dd.quarter
            LIMIT 50;
            """.strip()
        else:
            sql = f"""
            SELECT 
                ph.specialty,
                dd.year,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                LAG(COUNT(*)) OVER (PARTITION BY ph.specialty ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY ph.specialty ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY ph.specialty ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY ph.specialty ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {time_filter}
            GROUP BY ph.specialty, dd.year
            ORDER BY ph.specialty, dd.year
            LIMIT 50;
            """.strip()
        return sql, []
    
    def pediatric_diagnoses_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for pediatric diagnoses"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        top_n = self._extract_top_n(nl_query)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            where_clause = f"WHERE dd.year IN ({year_list}){month_clause}\n          AND EXTRACT(YEAR FROM AGE(p.dob)) < 18"
        else:
            where_clause = "WHERE EXTRACT(YEAR FROM AGE(p.dob)) < 18"
        
        sql = f"""
        SELECT 
            d.description AS diagnosis,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            COUNT(*) AS visit_count
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        JOIN dim_patient p ON fpv.patient_id = p.patient_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {where_clause}
        GROUP BY d.description
        ORDER BY visit_count DESC
        LIMIT {top_n};
        """.strip()
        return sql, []
    
    def icd10_ranking_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for ICD-10 code frequency ranking"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        top_n = self._extract_top_n(nl_query)
        
        if "change" in q and is_comparison:
            periods = self._extract_comparison_periods(q)
            sql = f"""
            SELECT 
                d.icd10_code,
                d.description AS diagnosis,
                dd.year,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                LAG(COUNT(*)) OVER (PARTITION BY d.icd10_code ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.icd10_code ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.icd10_code ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.icd10_code ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year IN ({periods[0]}, {periods[1]})
            GROUP BY d.icd10_code, d.description, dd.year
            ORDER BY d.icd10_code, dd.year
            LIMIT {top_n};
            """.strip()
        elif has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            where_clause = f"WHERE dd.year IN ({year_list}){month_clause}"
            sql = f"""
            SELECT 
                d.icd10_code,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {where_clause}
            GROUP BY d.icd10_code, d.description
            ORDER BY visit_count DESC
            LIMIT {top_n};
            """.strip()
        else:
            sql = f"""
            SELECT 
                d.icd10_code,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            GROUP BY d.icd10_code, d.description
            ORDER BY visit_count DESC
            LIMIT {top_n};
            """.strip()
        return sql, []
    
    def disease_demographic_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for disease prevalence by gender or age group"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        by_age = "age" in q
        by_gender = "gender" in q or "sex" in q
        
        if by_age and not by_gender:
            sql = f"""
            SELECT 
                CASE
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                    ELSE 'Senior (65+)'
                END AS age_group,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY age_group, d.description
            ORDER BY age_group, patient_count DESC;
            """.strip()
        elif by_gender and not by_age:
            sql = f"""
            SELECT 
                CASE p.gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' ELSE 'Unknown' END AS gender,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY p.gender, d.description
            ORDER BY p.gender, patient_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                CASE p.gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' ELSE 'Unknown' END AS gender,
                CASE
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 18 THEN 'Pediatric'
                    WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 18 AND 64 THEN 'Adult'
                    ELSE 'Senior (65+)'
                END AS age_group,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY p.gender, age_group, d.description
            ORDER BY p.gender, age_group, patient_count DESC;
            """.strip()
        return sql, []
    
    def chronic_conditions_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for most common chronic conditions"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        top_n = self._extract_top_n(nl_query)
        is_comparison = self._is_year_comparison_query(nl_query)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        chronic_icd_prefixes = [
            'E10', 'E11', 'E13', 'E14',
            'I10', 'I11', 'I12', 'I13', 'I15',
            'J44', 'J45',
            'E66',
            'I20', 'I21', 'I22', 'I23', 'I24', 'I25',
            'I50',
            'N18',
            'M05', 'M06', 'M15', 'M16', 'M17', 'M19',
            'G20',
            'G30',
            'K21', 'K25', 'K26', 'K29',
            'L40',
            'M32',
            'G40',
            'J31', 'J32', 'J37', 'J41', 'J42', 'J43',
            'D50', 'D51', 'D52', 'D53', 'D56', 'D57', 'D58', 'D59',
            'E00', 'E01', 'E02', 'E03', 'E04', 'E05',
            'F32', 'F33', 'F41',
            'E55',
            'K58',
            'M54'
        ]
        
        prefix_conditions = " OR ".join([f"d.icd10_code LIKE '{p}%'" for p in chronic_icd_prefixes])
        
        if is_comparison:
            periods = self._extract_comparison_periods(nl_query)
            sql = f"""
            SELECT 
                dd.year,
                d.icd10_code,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE ({prefix_conditions}) AND dd.year IN ({periods[0]}, {periods[1]})
            GROUP BY dd.year, d.icd10_code, d.description
            ORDER BY d.description, dd.year
            LIMIT {top_n};
            """.strip()
        else:
            sql = f"""
            SELECT 
                d.icd10_code,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE ({prefix_conditions}) {time_filter}
            GROUP BY d.icd10_code, d.description
            ORDER BY patient_count DESC
            LIMIT {top_n};
            """.strip()
        return sql, []
    
    def demographic_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for demographic breakdowns"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        is_registration_query = "register" in q or "registrat" in q or "signed up" in q or "joined" in q
        
        if is_registration_query:
            if has_time:
                year_list, month_clause, _ = self._extract_date_range(q)
                month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
                month_num_end = re.search(r"AND (\d+)", month_clause)
                if month_num_start and month_num_end:
                    start_month = month_num_start.group(1)
                    end_month = month_num_end.group(1)
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND EXTRACT(MONTH FROM created_at)::int BETWEEN {start_month} AND {end_month}"
                else:
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list}"
            else:
                time_filter = ""
            
            if "age" in q:
                sql = f"""
                SELECT 
                    CASE
                        WHEN EXTRACT(YEAR FROM AGE(dob)) < 1 THEN 'Infant (0-1)'
                        WHEN EXTRACT(YEAR FROM AGE(dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                        WHEN EXTRACT(YEAR FROM AGE(dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                        WHEN EXTRACT(YEAR FROM AGE(dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                        WHEN EXTRACT(YEAR FROM AGE(dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                        WHEN EXTRACT(YEAR FROM AGE(dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                        ELSE 'Senior (65+)'
                    END AS age_group,
                    COUNT(*) AS patient_count
                FROM dim_patient
                {time_filter}
                GROUP BY age_group
                ORDER BY patient_count DESC;
                """.strip()
            else:
                sql = f"""
                SELECT 
                    CASE gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' ELSE 'Unknown' END AS gender,
                    COUNT(*) AS patient_count
                FROM dim_patient
                {time_filter}
                GROUP BY gender
                ORDER BY patient_count DESC;
                """.strip()
            return sql, []
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if "age" in q:
            if is_comparison:
                sql = f"""
                SELECT 
                    dd.year,
                    CASE
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                        ELSE 'Senior (65+)'
                    END AS age_group,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    COUNT(*) AS visit_count,
                    LAG(COUNT(*)) OVER (PARTITION BY 
                        CASE
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                            ELSE 'Senior (65+)'
                        END ORDER BY dd.year) AS prev_year_visits,
                    COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY 
                        CASE
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                            WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                            ELSE 'Senior (65+)'
                        END ORDER BY dd.year) AS yoy_change,
                    ROUND(
                        100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY 
                            CASE
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                                ELSE 'Senior (65+)'
                            END ORDER BY dd.year)) / 
                        NULLIF(LAG(COUNT(*)) OVER (PARTITION BY 
                            CASE
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                                ELSE 'Senior (65+)'
                            END ORDER BY dd.year), 0)
                    , 2) AS yoy_change_pct
                FROM fact_patient_visits fpv
                JOIN dim_patient p ON fpv.patient_id = p.patient_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE 1=1 {time_filter}
                GROUP BY dd.year, age_group
                ORDER BY age_group, dd.year;
                """.strip()
            else:
                sql = f"""
                SELECT 
                    CASE
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                        WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                        ELSE 'Senior (65+)'
                    END AS age_group,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    COUNT(*) AS visit_count
                FROM fact_patient_visits fpv
                JOIN dim_patient p ON fpv.patient_id = p.patient_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE 1=1 {time_filter}
                GROUP BY age_group
                ORDER BY patient_count DESC;
                """.strip()
        else:
            sql = f"""
            SELECT 
                CASE p.gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' ELSE 'Unknown' END AS gender,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY p.gender
            ORDER BY patient_count DESC;
            """.strip()
        return sql, []
    
    def period_comparison(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Generate SQL for period comparison"""
        q = nl_query.lower()
        periods = self._extract_comparison_periods(q)
        
        is_registration_query = ("register" in q or "registrat" in q or "signed up" in q or "joined" in q) and "volume" in q
        is_gender_ratio_query = "gender" in q and ("ratio" in q or "change" in q) and ("register" in q or "registrat" in q or "signed up" in q or "joined" in q)
        is_quarterly_query = "quarter" in q.lower() or re.search(r"\bq[1-4]\b", q.lower()) is not None
        is_icd_query = "icd" in q
        is_physician_query = "physician" in q or "doctor" in q
        is_visit_growth_rate = ("visit" in q or "volume" in q) and "growth rate" in q
        
        is_monthly_growth = ("month" in q and "growth" in q) or ("month" in q and "rate" in q) or ("trend" in q and "month" in q)
        is_specialty_query = "specialty" in q
        is_seasonal = "season" in q
        is_time_span = "time between" in q or "average time" in q
        
        if is_quarterly_query:
            year_list, month_clause, _ = self._extract_date_range(q)
            if "specialty" in q:
                if year_list and "," not in year_list:
                    sql = f"""
                    SELECT 
                        ph.specialty,
                        dd.quarter,
                        COUNT(*) AS visit_count,
                        COUNT(DISTINCT fpv.patient_id) AS patient_count
                    FROM fact_patient_visits fpv
                    JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
                    LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                    WHERE dd.year = {year_list}
                    GROUP BY ph.specialty, dd.quarter
                    ORDER BY ph.specialty, dd.quarter
                    LIMIT 50;
                    """.strip()
                else:
                    sql = f"""
                    SELECT 
                        ph.specialty,
                        dd.quarter,
                        COUNT(*) AS visit_count,
                        COUNT(DISTINCT fpv.patient_id) AS patient_count,
                        LAG(COUNT(*)) OVER (PARTITION BY ph.specialty ORDER BY dd.quarter) AS prev_quarter_visits,
                        COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY ph.specialty ORDER BY dd.quarter) AS qoq_change
                    FROM fact_patient_visits fpv
                    JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
                    LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                    WHERE dd.year IN ({year_list})
                    GROUP BY ph.specialty, dd.quarter
                    ORDER BY ph.specialty, dd.quarter
                    LIMIT 50;
                    """.strip()
                return sql, []
            
            if year_list and "," not in year_list:
                sql = f"""
                SELECT 
                    dd.year,
                    dd.quarter,
                    COUNT(*) AS visit_count,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count
                FROM fact_patient_visits fpv
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE dd.year = {year_list}
                GROUP BY dd.year, dd.quarter
                ORDER BY dd.year, dd.quarter
                LIMIT 50;
                """.strip()
            elif year_list and "," in year_list:
                sql = f"""
                SELECT 
                    dd.year,
                    dd.quarter,
                    COUNT(*) AS visit_count,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count,
                    LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year) AS prev_year_visits,
                    COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year) AS yoy_change,
                    ROUND(
                        100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year)) / 
                        NULLIF(LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year), 0)
                    , 2) AS yoy_change_pct
                FROM fact_patient_visits fpv
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE dd.year IN ({year_list})
                GROUP BY dd.year, dd.quarter
                ORDER BY dd.quarter, dd.year
                LIMIT 50;
                """.strip()
            else:
                sql = """
                SELECT 
                    dd.year,
                    dd.quarter,
                    COUNT(*) AS visit_count,
                    COUNT(DISTINCT fpv.patient_id) AS patient_count
                FROM fact_patient_visits fpv
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                GROUP BY dd.year, dd.quarter
                ORDER BY dd.year, dd.quarter
                LIMIT 50;
                """.strip()
            return sql, []
        
        if is_time_span:
            return self.recurrence_time_span(nl_query)
        
        if is_seasonal:
            return self.seasonal_pattern_analysis(nl_query)
        
        if is_specialty_query:
            if "change" in q or "growth" in q:
                return self.specialty_growth_analysis(nl_query)
            return self.specialty_visits_analysis(nl_query)
        
        if is_visit_growth_rate:
            periods = self._extract_comparison_periods(q)
            return self.visit_yoy_analysis(nl_query, periods)
        
        if is_monthly_growth:
            return self.visit_month_analysis(nl_query)
        
        if is_physician_query and "change" in q:
            return self.physician_workload_trend_analysis(nl_query)
        
        if is_icd_query:
            return self.icd10_ranking_analysis(nl_query)
        
        if is_gender_ratio_query:
            sql = f"""
            SELECT 
                year,
                SUM(CASE WHEN gender = 'F' THEN patient_count ELSE 0 END) AS female_patients,
                SUM(CASE WHEN gender = 'M' THEN patient_count ELSE 0 END) AS male_patients,
                ROUND(
                    SUM(CASE WHEN gender = 'F' THEN patient_count ELSE 0 END)::numeric / 
                    NULLIF(SUM(CASE WHEN gender = 'M' THEN patient_count ELSE 0 END), 0), 2
                ) AS female_to_male_ratio
            FROM (
                SELECT 
                    EXTRACT(YEAR FROM created_at)::int AS year,
                    gender,
                    COUNT(*) AS patient_count
                FROM dim_patient
                WHERE EXTRACT(YEAR FROM created_at)::int IN ({periods[0]}, {periods[1]})
                GROUP BY EXTRACT(YEAR FROM created_at)::int, gender
            ) t
            GROUP BY year
            ORDER BY year;
            """.strip()
            return sql, []
        
        if is_registration_query:
            sql = f"""
            SELECT 
                EXTRACT(YEAR FROM created_at)::int AS year,
                COUNT(*) AS new_patients,
                LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int) AS prev_year_patients,
                COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int)) / 
                    NULLIF(LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int), 0)
                , 2) AS yoy_change_pct
            FROM dim_patient
            WHERE EXTRACT(YEAR FROM created_at)::int IN ({periods[0]}, {periods[1]})
            GROUP BY EXTRACT(YEAR FROM created_at)::int
            ORDER BY year;
            """.strip()
        else:
            sql = f"""
            SELECT 
                dd.year,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                LAG(COUNT(*)) OVER (ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year IN ({periods[0]}, {periods[1]})
            GROUP BY dd.year
            ORDER BY dd.year;
            """.strip()
        return sql, []
    
    def aggregate_statistics(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for aggregate statistics"""
        q = nl_query.lower()
        
        if "per month" in q and "average" in q:
            return self.avg_visits_per_month(nl_query)
        
        if ("registrat" in q or "register" in q) and ("growth" in q or "rate" in q):
            return self.patient_registration_trend(nl_query)
        
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(DISTINCT fpv.patient_id) AS total_patients,
            COUNT(*) AS total_visits,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT fpv.patient_id), 0), 2) AS avg_visits_per_patient
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter};
        """.strip()
        return sql, []
    
    def avg_visits_per_month(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for average visits per month"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            dd.year,
            dd.month,
            COUNT(*) AS visit_count,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT fpv.patient_id), 0), 2) AS avg_visits_per_patient
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY dd.year, dd.month
        ORDER BY dd.year, dd.month;
        """.strip()
        return sql, []
    
    def physician_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for physician analysis"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        top_n = self._extract_top_n(nl_query)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        q_lower = nl_query.lower()
        if "patient" in q_lower and "highest" in q_lower:
            order_by = "patient_count DESC"
        else:
            order_by = "visit_count DESC"
        
        sql = f"""
        SELECT 
            ph.name AS physician,
            ph.specialty,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            COUNT(*) AS visit_count
        FROM fact_patient_visits fpv
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY ph.name, ph.specialty
        ORDER BY {order_by}
        LIMIT {top_n};
        """.strip()
        return sql, []
    
    def patient_count_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for patient count queries"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        is_registration_query = "register" in q or "registrat" in q or "signed up" in q or "joined" in q
        
        age_filter = self._extract_age_group(q)
        age_clause = ""
        
        match_over = re.search(r"\bover\s+(\d+)\b", q)
        if not match_over:
            match_over = re.search(r"\bover\s+(\d+)\s+years?\b", q)
        if match_over:
            min_age = int(match_over.group(1))
            age_clause = f"AND EXTRACT(YEAR FROM AGE(p.dob)) > {min_age}"
        else:
            match_under = re.search(r"\bunder\s+(\d+)\b", q)
            if not match_under:
                match_under = re.search(r"\bunder\s+(\d+)\s+years?\b", q)
            if not match_under and ("children" in q or "child" in q or "infant" in q or "toddler" in q):
                match_under = re.search(r"\bunder\s+(\d+)\b", q)
                if not match_under:
                    if "children under 5" in q or "child under 5" in q or "infant" in q:
                        age_clause = "AND EXTRACT(YEAR FROM AGE(p.dob)) < 5"
                    elif "children under 10" in q or "child under 10" in q:
                        age_clause = "AND EXTRACT(YEAR FROM AGE(p.dob)) < 10"
                    elif "children under 18" in q or "child under 18" in q:
                        age_clause = "AND EXTRACT(YEAR FROM AGE(p.dob)) < 18"
            if match_under:
                max_age = int(match_under.group(1))
                age_clause = f"AND EXTRACT(YEAR FROM AGE(p.dob)) < {max_age}"
            else:
                match_between = re.search(r"\bbetween\s+(\d+)\s+and\s+(\d+)\b", q)
                if match_between:
                    min_age = int(match_between.group(1))
                    max_age = int(match_between.group(2))
                    age_clause = f"AND EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN {min_age} AND {max_age}"
                elif age_filter:
                    _, min_age, max_age = age_filter
                    age_clause = f"AND EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN {min_age} AND {max_age}"
        
        visit_count_match = re.search(r"\bmore than\s+(\d+)\b", q)
        if not visit_count_match:
            visit_count_match = re.search(r"\bover\s+(\d+)\s+times?\b", q)
        if visit_count_match:
            min_visits = int(visit_count_match.group(1))
            visit_count_having = f"HAVING COUNT(*) > {min_visits}"
        elif "multiple visit" in q or "more than one visit" in q:
            visit_count_having = "HAVING COUNT(*) > 1"
        else:
            visit_count_having = ""
        
        is_from_location = re.search(r"\bfrom\s+(\w+)\b", q)
        if is_from_location:
            location = is_from_location.group(1)
            location_clause = f"AND p.residence ILIKE '{location}%'"
        else:
            location_clause = ""
        
        if is_registration_query:
            if has_time:
                year_list, month_clause, _ = self._extract_date_range(q)
                year_clause = year_list
                if month_clause:
                    month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
                    month_num_end = re.search(r"AND (\d+)", month_clause)
                    if month_num_start and month_num_end:
                        start_month = month_num_start.group(1)
                        end_month = month_num_end.group(1)
                        time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_clause} AND EXTRACT(MONTH FROM created_at)::int BETWEEN {start_month} AND {end_month}"
                    else:
                        time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_clause}"
                else:
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_clause}"
            else:
                time_filter = ""
            
            sql = f"""
            SELECT 
                COUNT(*) AS total_patients
            FROM dim_patient
            {time_filter};
            """.strip()
            return sql, []
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(DISTINCT fpv.patient_id) AS total_patients,
            COUNT(*) AS total_visits
        FROM fact_patient_visits fpv
        JOIN dim_patient p ON fpv.patient_id = p.patient_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter} {age_clause} {location_clause}
        {visit_count_having};
        """.strip()
        return sql, []
    
    def diagnosis_patients_count(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for diagnosis-specific patient count queries"""
        q = nl_query.lower()
        diagnosis = self.si._extract_diagnosis(q)
        
        if not diagnosis:
            return self.patient_count_analysis(nl_query)
        
        has_time = self._has_explicit_time_filter(q)
        gender = self._extract_gender(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            year_list = f"{datetime.now().year}"
            time_filter = f"AND dd.year IN ({year_list})"
        
        gender_clause = ""
        if gender:
            gender_clause = f"AND p.gender = '{gender}'"
        
        sql = f"""
        SELECT 
            COUNT(DISTINCT fpv.patient_id) AS total_patients,
            COUNT(*) AS total_visits
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        JOIN dim_patient p ON fpv.patient_id = p.patient_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE d.description ILIKE $1
        {time_filter} {gender_clause};
        """.strip()
        
        return sql, [f"%{diagnosis}%"]
    
    def visit_count_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for visit count queries"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            year_list = ""
            month_clause = ""
            time_filter = ""
        
        # Check if query is asking for breakdown by month
        if "month" in q and ("highest" in q or "busiest" in q or "most" in q or "trend" in q):
            sql = f"""
            SELECT 
                dd.month,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS unique_patients
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.month
            ORDER BY visit_count DESC;
            """.strip()
        elif "day" in q and ("highest" in q or "busiest" in q or "most" in q):
            sql = f"""
            SELECT 
                dd.day_of_week,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS unique_patients
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.day_of_week
            ORDER BY visit_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                COUNT(*) AS total_visits,
                COUNT(DISTINCT fpv.patient_id) AS unique_patients
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter};
            """.strip()
        return sql, []
    
    def physician_workload_trend_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for physician workload trends"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if is_comparison:
            periods = self._extract_comparison_periods(q)
            sql = f"""
            SELECT 
                ph.name AS physician,
                ph.specialty,
                dd.year,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                LAG(COUNT(*)) OVER (PARTITION BY ph.physician_id ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY ph.physician_id ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY ph.physician_id ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY ph.physician_id ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year IN ({periods[0]}, {periods[1]})
            GROUP BY ph.physician_id, ph.name, ph.specialty, dd.year
            ORDER BY ph.name, dd.year;
            """.strip()
        elif has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
            sql = f"""
            SELECT 
                ph.name AS physician,
                ph.specialty,
                dd.year,
                dd.month,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count
            FROM fact_patient_visits fpv
            JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {time_filter}
            GROUP BY ph.name, ph.specialty, dd.year, dd.month
            ORDER BY ph.name, dd.year, dd.month;
            """.strip()
        else:
            sql = """
            SELECT 
                ph.name AS physician,
                ph.specialty,
                dd.year,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count
            FROM fact_patient_visits fpv
            JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            GROUP BY ph.name, ph.specialty, dd.year
            ORDER BY ph.name, dd.year;
            """.strip()
        return sql, []
    
    def physician_repeat_patients_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for physicians with most repeat patients"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        top_n = self._extract_top_n(nl_query)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            ph.name AS physician,
            ph.specialty,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients,
            COUNT(*) AS total_visits,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT fpv.patient_id), 0), 2) AS visits_per_patient,
            COUNT(*) - COUNT(DISTINCT fpv.patient_id) AS repeat_visits
        FROM fact_patient_visits fpv
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY ph.name, ph.specialty
        HAVING COUNT(*) > COUNT(DISTINCT fpv.patient_id)
        ORDER BY visits_per_patient DESC
        LIMIT {top_n};
        """.strip()
        return sql, []
    
    def physician_workload_distribution_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for physician workload distribution"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            ph.name AS physician,
            ph.specialty,
            COUNT(*) AS visit_count,
            ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS pct_of_total,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT fpv.patient_id), 0), 2) AS avg_visits_per_patient,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients
        FROM fact_patient_visits fpv
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY ph.name, ph.specialty
        ORDER BY visit_count DESC;
        """.strip()
        return sql, []
    
    def physician_peak_hours_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for physician peak hours"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_peak = "peak" in q or "busiest" in q or "most" in q
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        if is_peak:
            order_by = "visit_count DESC"
        else:
            order_by = "fpv.visit_hour"
        
        sql = f"""
        SELECT 
            fpv.visit_hour,
            COUNT(*) AS visit_count
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY fpv.visit_hour
        ORDER BY {order_by};
        """.strip()
        return sql, []
    
    def visit_yoy_analysis(self, nl_query: str, periods: Optional[List[int]] = None) -> Tuple[str, List]:
        """Generate SQL for visit year-over-year analysis"""
        q = nl_query.lower()
        
        two_months = self._extract_two_months(q)
        
        if two_months:
            month1, month2, year = two_months
            
            sql = f"""
            SELECT 
                {month1} AS month_from,
                {month2} AS month_to,
                {year} AS year,
                SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS visits_month_from,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) AS visits_month_to,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                    SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS absolute_change,
                ROUND(
                    100.0 * (
                        SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                        SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END)
                    ) / NULLIF(SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END), 0)
                , 2) AS mom_growth_pct
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year = {year} AND dd.month IN ({month1}, {month2});
            """.strip()
            return sql, []
        
        has_time = self._has_explicit_time_filter(q)
        
        if periods is None:
            periods = self._extract_comparison_periods(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            if month_clause:
                time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
            else:
                time_filter = f"WHERE dd.year IN ({year_list})"
        else:
            if periods and len(periods) == 2:
                time_filter = f"WHERE dd.year IN ({periods[0]}, {periods[1]})"
            else:
                time_filter = ""
        
        sql = f"""
        SELECT 
            dd.year,
            COUNT(*) AS visit_count,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            LAG(COUNT(*)) OVER (ORDER BY dd.year) AS prev_year_visits,
            COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY dd.year) AS yoy_change,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY dd.year)) / 
                NULLIF(LAG(COUNT(*)) OVER (ORDER BY dd.year), 0)
            , 2) AS yoy_change_pct
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY dd.year
        ORDER BY dd.year;
        """.strip()
        return sql, []
    
    def visit_timeframe_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for visits per timeframe"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            if month_clause:
                time_filter = f"AND dd.year IN ({year_list}){month_clause}"
            else:
                time_filter = f"AND dd.year IN ({year_list})"
        else:
            time_filter = ""
        
        if "week" in q:
            group_by = "dd.year, dd.week_of_month"
            select_cols = "dd.year, dd.week_of_month"
            order_by = "dd.year, dd.week_of_month"
        elif "quarter" in q:
            group_by = "dd.year, dd.quarter"
            select_cols = "dd.year, dd.quarter"
            order_by = "dd.year, dd.quarter"
        elif "day" in q:
            group_by = "dd.calendar_date"
            select_cols = "dd.calendar_date"
            order_by = "dd.calendar_date"
        else:
            group_by = "dd.year, dd.month"
            select_cols = "dd.year, dd.month"
            order_by = "dd.year, dd.month"
        
        sql = f"""
        SELECT 
            {select_cols},
            COUNT(*) AS visit_count,
            COUNT(DISTINCT fpv.patient_id) AS patient_count
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY {group_by}
        ORDER BY {order_by};
        """.strip()
        return sql, []
    
    def visit_single_multiple_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for single vs multiple visit patients"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        is_more_than_one = "more than one" in q or "more than 1" in q
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        if is_more_than_one:
            sql = f"""
            SELECT 
                ROUND(100.0 * COUNT(CASE WHEN visit_count > 1 THEN 1 END) / NULLIF(COUNT(*), 0), 2) AS pct_patients_with_multiple_visits,
                COUNT(CASE WHEN visit_count > 1 THEN 1 END) AS patients_with_multiple_visits,
                COUNT(*) AS total_patients
            FROM (
                SELECT fpv.patient_id, COUNT(*) AS visit_count
                FROM fact_patient_visits fpv
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE 1=1 {time_filter}
                GROUP BY fpv.patient_id
            ) t;
            """.strip()
        else:
            sql = f"""
            SELECT 
                CASE WHEN visit_count = 1 THEN 'Single Visit' ELSE 'Multiple Visits' END AS visit_type,
                COUNT(*) AS patient_count,
                ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS percentage
            FROM (
                SELECT fpv.patient_id, COUNT(*) AS visit_count
                FROM fact_patient_visits fpv
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE 1=1 {time_filter}
                GROUP BY fpv.patient_id
            ) t
            GROUP BY CASE WHEN visit_count = 1 THEN 'Single Visit' ELSE 'Multiple Visits' END;
            """.strip()
        return sql, []
    
    def visit_day_of_week_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for visits by day of week"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            CASE dd.day_of_week
                WHEN 0 THEN 'Monday'
                WHEN 1 THEN 'Tuesday'
                WHEN 2 THEN 'Wednesday'
                WHEN 3 THEN 'Thursday'
                WHEN 4 THEN 'Friday'
                WHEN 5 THEN 'Saturday'
                WHEN 6 THEN 'Sunday'
            END AS day_of_week,
            COUNT(*) AS visit_count
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY dd.day_of_week
        ORDER BY visit_count DESC;
        """.strip()
        return sql, []
    
    def visit_weekend_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for weekend vs weekday visits"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            CASE WHEN dd.is_weekend THEN 'Weekend' ELSE 'Weekday' END AS day_type,
            COUNT(*) AS visit_count,
            ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS percentage
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY dd.is_weekend;
        """.strip()
        return sql, []
    
    def visit_month_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for visits by month"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        # Check if query asks about "busiest" or "which months" (general question)
        is_general_monthly = "busiest" in q or "which month" in q or "month" in q and not has_time
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            if month_clause:
                time_filter = f"AND dd.year IN ({year_list}){month_clause}"
            else:
                time_filter = f"AND dd.year IN ({year_list})"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                dd.month,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                LAG(COUNT(*)) OVER (PARTITION BY dd.month ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.month ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.month ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY dd.month ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, dd.month
            ORDER BY dd.month, dd.year;
            """.strip()
        elif is_general_monthly and not has_time:
            # Show aggregated by month across all years
            sql = """
            SELECT 
                dd.month,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1
            GROUP BY dd.month
            ORDER BY visit_count DESC;
            """.strip()
        elif has_time and ("busiest" in q or "which month" in q):
            # Show months sorted by visit count for a specific year
            sql = f"""
            SELECT 
                dd.year,
                dd.month,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_year
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, dd.month
            ORDER BY visit_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                dd.year,
                dd.month,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, dd.month
            ORDER BY dd.year, dd.month;
            """.strip()
        return sql, []
    
    def visit_quarter_trend_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for visit trends by quarter"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        is_comparison = self._is_year_comparison_query(q)
        
        if is_comparison:
            periods = self._extract_comparison_periods(q)
            sql = f"""
            SELECT 
                dd.year,
                dd.quarter,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year IN ({periods[0]}, {periods[1]})
            GROUP BY dd.year, dd.quarter
            ORDER BY dd.quarter, dd.year;
            """.strip()
        elif has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
            sql = f"""
            SELECT 
                dd.year,
                dd.quarter,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year) AS prev_year_visits,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, dd.quarter
            ORDER BY dd.quarter, dd.year;
            """.strip()
        else:
            sql = """
            SELECT 
                dd.year,
                dd.quarter,
                COUNT(*) AS visit_count,
                COUNT(DISTINCT fpv.patient_id) AS patient_count
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            GROUP BY dd.year, dd.quarter
            ORDER BY dd.year, dd.quarter;
            """.strip()
        return sql, []
    
    def visit_hour_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for visits by hour"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            fpv.visit_hour,
            COUNT(*) AS visit_count
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY fpv.visit_hour
        ORDER BY fpv.visit_hour;
        """.strip()
        return sql, []
    
    def specialty_visits_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for visits per specialty"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            ph.specialty,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            COUNT(*) AS visit_count
        FROM fact_patient_visits fpv
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY ph.specialty
        ORDER BY visit_count DESC;
        """.strip()
        return sql, []
    
    def patient_registration_trend(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for patient registration trend"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_growth = "growth" in q or "rate" in q
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            if month_clause:
                month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
                month_num_end = re.search(r"AND (\d+)", month_clause)
                if month_num_start and month_num_end:
                    start_month = month_num_start.group(1)
                    end_month = month_num_end.group(1)
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND EXTRACT(MONTH FROM created_at)::int BETWEEN {start_month} AND {end_month}"
                else:
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list}"
            else:
                time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int IN ({year_list})"
        else:
            time_filter = ""
        
        if is_growth and "month" in q:
            sql = f"""
            SELECT 
                EXTRACT(YEAR FROM created_at)::int AS year,
                EXTRACT(MONTH FROM created_at)::int AS month,
                COUNT(*) AS new_patients,
                LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int, EXTRACT(MONTH FROM created_at)::int) AS prev_month_patients,
                COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int, EXTRACT(MONTH FROM created_at)::int) AS mom_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int, EXTRACT(MONTH FROM created_at)::int)) / 
                    NULLIF(LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int, EXTRACT(MONTH FROM created_at)::int), 0)
                , 2) AS monthly_growth_rate_pct
            FROM dim_patient
            {time_filter}
            GROUP BY EXTRACT(YEAR FROM created_at)::int, EXTRACT(MONTH FROM created_at)::int
            ORDER BY year, month;
            """.strip()
        elif "year" in q and not ("month" in q):
            sql = f"""
            SELECT 
                EXTRACT(YEAR FROM created_at)::int AS year,
                COUNT(*) AS new_patients,
                LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int) AS prev_year_patients,
                COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int)) / 
                    NULLIF(LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM created_at)::int), 0)
                , 2) AS yoy_growth_rate_pct
            FROM dim_patient
            {time_filter}
            GROUP BY EXTRACT(YEAR FROM created_at)::int
            ORDER BY year;
            """.strip()
        else:
            sql = f"""
            SELECT 
                EXTRACT(YEAR FROM created_at)::int AS year,
                EXTRACT(MONTH FROM created_at)::int AS month,
                COUNT(*) AS new_patients
            FROM dim_patient
            {time_filter}
            GROUP BY EXTRACT(YEAR FROM created_at)::int, EXTRACT(MONTH FROM created_at)::int
            ORDER BY year, month;
            """.strip()
        return sql, []
    
    def recurrence_qoq_analysis(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Generate SQL for recurrence quarter-over-quarter comparison"""
        q = nl_query.lower()
        
        quarter_match = re.search(r"(?:between\s+)?q([12])\s+(?:and|to|vs|versus)\s+q([12])\s*(?:of\s+)?(\d{4})?", q)
        
        if quarter_match:
            q1 = int(quarter_match.group(1))
            q2 = int(quarter_match.group(2))
            year = quarter_match.group(3) or "2025"
            
            sql = f"""
            SELECT 
                d.description AS diagnosis,
                COUNT(DISTINCT CASE WHEN dd.quarter = {q1} THEN fra.patient_id END) AS q{q1}_patients,
                SUM(CASE WHEN dd.quarter = {q1} THEN fra.recurrence_count ELSE 0 END) AS q{q1}_recurrences,
                COUNT(DISTINCT CASE WHEN dd.quarter = {q2} THEN fra.patient_id END) AS q{q2}_patients,
                SUM(CASE WHEN dd.quarter = {q2} THEN fra.recurrence_count ELSE 0 END) AS q{q2}_recurrences,
                COUNT(DISTINCT CASE WHEN dd.quarter = {q2} THEN fra.patient_id END) - COUNT(DISTINCT CASE WHEN dd.quarter = {q1} THEN fra.patient_id END) AS patient_change,
                SUM(CASE WHEN dd.quarter = {q2} THEN fra.recurrence_count ELSE 0 END) - SUM(CASE WHEN dd.quarter = {q1} THEN fra.recurrence_count ELSE 0 END) AS recurrence_change,
                ROUND(
                    100.0 * (SUM(CASE WHEN dd.quarter = {q2} THEN fra.recurrence_count ELSE 0 END) - SUM(CASE WHEN dd.quarter = {q1} THEN fra.recurrence_count ELSE 0 END)) /
                    NULLIF(SUM(CASE WHEN dd.quarter = {q1} THEN fra.recurrence_count ELSE 0 END), 0)
                , 2) AS change_pct
            FROM fact_recurrence_analysis fra
            JOIN dim_diagnosis d ON fra.diagnosis_id = d.diagnosis_id
            JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id AND fra.diagnosis_id = fpv.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year = {year}
              AND dd.quarter IN ({q1}, {q2})
            GROUP BY d.description
            ORDER BY change_pct DESC NULLS LAST;
            """.strip()
            return sql, []
        
        return None
    
    def recurrence_highest_rates(self, nl_query: str) -> Tuple[str, List]:
        """Diagnoses with highest recurrence rates"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        top_n = self._extract_top_n(nl_query)
        sql = f"""
        SELECT 
            d.description AS diagnosis,
            d.icd10_code,
            COUNT(DISTINCT fra.patient_id) AS patient_count,
            ROUND(AVG(fra.recurrence_count), 2) AS avg_recurrence,
            MAX(fra.recurrence_count) AS max_recurrence,
            SUM(fra.recurrence_count) AS total_recurrences
        FROM fact_recurrence_analysis fra
        JOIN dim_diagnosis d ON fra.diagnosis_id = d.diagnosis_id
        LEFT JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id AND fra.diagnosis_id = fpv.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fra.recurrence_count > 1 {time_filter}
        GROUP BY d.description, d.icd10_code
        ORDER BY avg_recurrence DESC
        LIMIT {top_n};
        """.strip()
        return sql, []
    
    def recurrence_repeated_patients(self, nl_query: str) -> Tuple[str, List]:
        """Patients with repeated occurrences of the same condition"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            fra.patient_id,
            p.first_name,
            p.last_name,
            d.description AS diagnosis,
            fra.recurrence_count,
            fra.first_occurrence_date,
            fra.last_occurrence_date,
            (fra.last_occurrence_date - fra.first_occurrence_date) AS days_span
        FROM fact_recurrence_analysis fra
        JOIN dim_patient p ON fra.patient_id = p.patient_id
        JOIN dim_diagnosis d ON fra.diagnosis_id = d.diagnosis_id
        LEFT JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id AND fra.diagnosis_id = fpv.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fra.recurrence_count > 1 {time_filter}
        ORDER BY fra.recurrence_count DESC
        LIMIT 50;
        """.strip()
        return sql, []
    
    def recurrence_avg_count(self, nl_query: str) -> Tuple[str, List]:
        """Average recurrence count per diagnosis"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            d.description AS diagnosis,
            d.icd10_code,
            COUNT(DISTINCT fra.patient_id) AS patient_count,
            ROUND(AVG(fra.recurrence_count), 2) AS avg_recurrence,
            MIN(fra.recurrence_count) AS min_recurrence,
            MAX(fra.recurrence_count) AS max_recurrence
        FROM fact_recurrence_analysis fra
        JOIN dim_diagnosis d ON fra.diagnosis_id = d.diagnosis_id
        LEFT JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id AND fra.diagnosis_id = fpv.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY d.description, d.icd10_code
        ORDER BY avg_recurrence DESC
        LIMIT 20;
        """.strip()
        return sql, []
    
    def recurrence_time_span(self, nl_query: str) -> Tuple[str, List]:
        """Time span between first and last occurrence for recurring conditions"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            d.description AS diagnosis,
            d.icd10_code,
            COUNT(DISTINCT fra.patient_id) AS patient_count,
            ROUND(AVG(fra.last_occurrence_date - fra.first_occurrence_date)) AS avg_days_span,
            MIN(fra.last_occurrence_date - fra.first_occurrence_date) AS min_days_span,
            MAX(fra.last_occurrence_date - fra.first_occurrence_date) AS max_days_span
        FROM fact_recurrence_analysis fra
        JOIN dim_diagnosis d ON fra.diagnosis_id = d.diagnosis_id
        LEFT JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id AND fra.diagnosis_id = fpv.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fra.recurrence_count > 1 {time_filter}
        GROUP BY d.description, d.icd10_code
        ORDER BY avg_days_span DESC
        LIMIT 20;
        """.strip()
        return sql, []
    
    def recurrence_patient_count(self, nl_query: str) -> Tuple[str, List]:
        """Count of patients with at least one recurring diagnosis"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(DISTINCT fra.patient_id) AS patients_with_recurrence,
            (SELECT COUNT(DISTINCT patient_id) FROM dim_patient) AS total_patients,
            ROUND(100.0 * COUNT(DISTINCT fra.patient_id) / 
                NULLIF((SELECT COUNT(DISTINCT patient_id) FROM dim_patient), 0), 2) AS recurrence_pct
        FROM fact_recurrence_analysis fra
        LEFT JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id AND fra.diagnosis_id = fpv.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fra.recurrence_count > 1 {time_filter};
        """.strip()
        return sql, []
    
    def recurrence_by_age_group(self, nl_query: str) -> Tuple[str, List]:
        """Recurrence patterns by age group"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            CASE
                WHEN EXTRACT(YEAR FROM AGE(p.dob)) < 1 THEN 'Infant (0-1)'
                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 1 AND 4 THEN 'Toddler (1-4)'
                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 5 AND 12 THEN 'Child (5-12)'
                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 13 AND 18 THEN 'Adolescent (13-18)'
                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 19 AND 39 THEN 'Young Adult (19-39)'
                WHEN EXTRACT(YEAR FROM AGE(p.dob)) BETWEEN 40 AND 64 THEN 'Middle Age (40-64)'
                ELSE 'Senior (65+)'
            END AS age_group,
            COUNT(DISTINCT fra.patient_id) AS patient_count,
            ROUND(AVG(fra.recurrence_count), 2) AS avg_recurrence,
            SUM(fra.recurrence_count) AS total_recurrences
        FROM fact_recurrence_analysis fra
        JOIN dim_patient p ON fra.patient_id = p.patient_id
        LEFT JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id AND fra.diagnosis_id = fpv.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fra.recurrence_count > 1 {time_filter}
        GROUP BY age_group
        ORDER BY total_recurrences DESC;
        """.strip()
        return sql, []
    
    def recurrence_by_physician(self, nl_query: str) -> Tuple[str, List]:
        """Recurrence patterns by physician"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            ph.name AS physician,
            ph.specialty,
            COUNT(DISTINCT fra.patient_id) AS patient_count,
            ROUND(AVG(fra.recurrence_count), 2) AS avg_recurrence,
            SUM(fra.recurrence_count) AS total_recurrences
        FROM fact_recurrence_analysis fra
        JOIN dim_patient p ON fra.patient_id = p.patient_id
        JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id 
            AND fra.diagnosis_id = fpv.diagnosis_id
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fra.recurrence_count > 1 {time_filter}
        GROUP BY ph.name, ph.specialty
        ORDER BY total_recurrences DESC
        LIMIT 20;
        """.strip()
        return sql, []
    
    def recurrence_by_payer(self, nl_query: str) -> Tuple[str, List]:
        """Recurrence patterns by payer type"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_name,
                py.payer_type,
                COUNT(DISTINCT fra.patient_id) AS patient_count,
                ROUND(AVG(fra.recurrence_count), 2) AS avg_recurrence,
                SUM(fra.recurrence_count) AS total_recurrences
            FROM fact_recurrence_analysis fra
            JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id 
                AND fra.diagnosis_id = fpv.diagnosis_id
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE fra.recurrence_count > 1 {time_filter}
            GROUP BY dd.year, py.payer_name, py.payer_type
            ORDER BY dd.year, total_recurrences DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                py.payer_name,
                py.payer_type,
                COUNT(DISTINCT fra.patient_id) AS patient_count,
                ROUND(AVG(fra.recurrence_count), 2) AS avg_recurrence,
                SUM(fra.recurrence_count) AS total_recurrences
            FROM fact_recurrence_analysis fra
            JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id 
                AND fra.diagnosis_id = fpv.diagnosis_id
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE fra.recurrence_count > 1 {time_filter}
            GROUP BY py.payer_name, py.payer_type
            ORDER BY total_recurrences DESC;
            """.strip()
        return sql, []
    
    def recurrence_high_risk(self, nl_query: str) -> Tuple[str, List]:
        """Patients classified as high-risk based on recurrence frequency"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        time_filter = ""
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f" AND dd.year IN ({year_list}){month_clause}"
        
        sql = f"""
        SELECT 
            fra.patient_id,
            p.first_name,
            p.last_name,
            p.gender,
            EXTRACT(YEAR FROM AGE(p.dob))::int AS age,
            COUNT(DISTINCT fra.diagnosis_id) AS distinct_recurring_conditions,
            SUM(fra.recurrence_count) AS total_recurrences,
            MAX(fra.recurrence_count) AS max_single_recurrence
        FROM fact_recurrence_analysis fra
        JOIN dim_patient p ON fra.patient_id = p.patient_id
        LEFT JOIN fact_patient_visits fpv ON fra.patient_id = fpv.patient_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fra.recurrence_count > 1 {time_filter}
        GROUP BY fra.patient_id, p.first_name, p.last_name, p.gender, p.dob
        HAVING SUM(fra.recurrence_count) >= 5
        ORDER BY total_recurrences DESC
        LIMIT 50;
        """.strip()
        return sql, []
    
    def payer_analysis(self, nl_query: str) -> Tuple[str, List]:
        """Generate SQL for payer/insurance analysis"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_name,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_total
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, py.payer_name, py.payer_type
            ORDER BY dd.year, visit_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                py.payer_name,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY py.payer_name, py.payer_type
            ORDER BY visit_count DESC;
            """.strip()
        return sql, []
    
    def payer_highest(self, nl_query: str) -> Tuple[str, List]:
        """Payer with highest number of visits"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_name,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_total
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, py.payer_name, py.payer_type
            ORDER BY dd.year, visit_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                py.payer_name,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY py.payer_name, py.payer_type
            ORDER BY visit_count DESC;
            """.strip()
        return sql, []
    
    def payer_selfpay_vs_insured(self, nl_query: str) -> Tuple[str, List]:
        """Percentage of visits self-pay vs insured"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_total_visits
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, py.payer_type
            ORDER BY dd.year, visit_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total_visits
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY py.payer_type
            ORDER BY visit_count DESC;
            """.strip()
        return sql, []
    
    def payer_over_time(self, nl_query: str) -> Tuple[str, List]:
        """Payer utilization trends over time"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, py.payer_type
            ORDER BY py.payer_type, dd.year;
            """.strip()
        else:
            sql = f"""
            SELECT 
                dd.year,
                dd.month,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, dd.month, py.payer_type
            ORDER BY dd.year, dd.month, py.payer_type;
            """.strip()
        return sql, []
    
    def payer_diagnoses(self, nl_query: str) -> Tuple[str, List]:
        """Most common diagnoses per payer type"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        top_n = self._extract_top_n(nl_query)
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_type,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, py.payer_type, d.description
            ORDER BY dd.year, py.payer_type, visit_count DESC
            LIMIT {top_n * 2};
            """.strip()
        else:
            sql = f"""
            SELECT 
                py.payer_type,
                d.description AS diagnosis,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY py.payer_type, d.description
            ORDER BY py.payer_type, visit_count DESC
            LIMIT {top_n * 2};
            """.strip()
        return sql, []
    
    def payer_comparison(self, nl_query: str) -> Tuple[str, List]:
        """Compare visit volume between payer types"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        elif is_comparison:
            periods = self._extract_comparison_periods(q)
            time_filter = f"AND dd.year IN ({periods[0]}, {periods[1]})"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT fpv.patient_id), 0), 2) AS avg_visits_per_patient,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_year_visits,
                LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year) AS prev_year_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year) AS yoy_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year)) / 
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY py.payer_type ORDER BY dd.year), 0)
                , 2) AS yoy_change_pct
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, py.payer_type
            ORDER BY py.payer_type, dd.year;
            """.strip()
        elif has_time:
            sql = f"""
            SELECT 
                dd.year,
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT fpv.patient_id), 0), 2) AS avg_visits_per_patient,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_year_visits
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, py.payer_type
            ORDER BY dd.year, visit_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                py.payer_type,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT fpv.patient_id), 0), 2) AS avg_visits_per_patient,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total_visits
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY py.payer_type
            ORDER BY visit_count DESC;
            """.strip()
        return sql, []
    
    def staffing_avg_daily_load(self, nl_query: str) -> Tuple[str, List]:
        """Average daily patient load"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            ROUND(AVG(daily_patients), 2) AS avg_daily_patients,
            ROUND(AVG(daily_visits), 2) AS avg_daily_visits,
            MIN(daily_patients) AS min_daily_patients,
            MAX(daily_patients) AS max_daily_patients,
            MIN(daily_visits) AS min_daily_visits,
            MAX(daily_visits) AS max_daily_visits
        FROM (
            SELECT 
                dd.calendar_date,
                COUNT(DISTINCT fpv.patient_id) AS daily_patients,
                COUNT(*) AS daily_visits
            FROM fact_patient_visits fpv
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            {time_filter}
            GROUP BY dd.calendar_date
        ) daily;
        """.strip()
        return sql, []
    
    def staffing_peak_months(self, nl_query: str) -> Tuple[str, List]:
        """Peak months requiring increased staffing"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            dd.month,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients,
            COUNT(*) AS total_visits,
            COUNT(DISTINCT dd.calendar_date) AS active_days,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT dd.calendar_date), 0), 2) AS avg_daily_visits
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY dd.month
        ORDER BY total_visits DESC;
        """.strip()
        return sql, []
    
    def staffing_peak_days(self, nl_query: str) -> Tuple[str, List]:
        """Peak days of the week requiring increased staffing"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            dd.day_of_week,
            CASE dd.day_of_week
                WHEN 0 THEN 'Monday'
                WHEN 1 THEN 'Tuesday'
                WHEN 2 THEN 'Wednesday'
                WHEN 3 THEN 'Thursday'
                WHEN 4 THEN 'Friday'
                WHEN 5 THEN 'Saturday'
                WHEN 6 THEN 'Sunday'
            END AS day_name,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients,
            COUNT(*) AS total_visits,
            COUNT(DISTINCT dd.calendar_date) AS weeks_counted,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT dd.calendar_date), 0), 2) AS avg_daily_visits
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY dd.day_of_week
        ORDER BY total_visits DESC;
        """.strip()
        return sql, []
    
    def staffing_specialty_capacity(self, nl_query: str) -> Tuple[str, List]:
        """Specialties operating at highest capacity"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            ph.specialty,
            COUNT(DISTINCT ph.physician_id) AS physician_count,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients,
            COUNT(*) AS total_visits,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT ph.physician_id), 0), 2) AS visits_per_physician,
            ROUND(COUNT(DISTINCT fpv.patient_id)::numeric / NULLIF(COUNT(DISTINCT ph.physician_id), 0), 2) AS patients_per_physician
        FROM fact_patient_visits fpv
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY ph.specialty
        ORDER BY visits_per_physician DESC;
        """.strip()
        return sql, []
    
    def staffing_underutilized(self, nl_query: str) -> Tuple[str, List]:
        """Underutilized physicians or specialties"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
            time_filter_subq = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
            time_filter_subq = ""
        
        sql = f"""
        SELECT 
            ph.name AS physician,
            ph.specialty,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients,
            COUNT(*) AS total_visits,
            COUNT(DISTINCT dd.calendar_date) AS days_worked,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT dd.calendar_date), 0), 2) AS avg_visits_per_day
        FROM fact_patient_visits fpv
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE 1=1 {time_filter}
        GROUP BY ph.name, ph.specialty
        HAVING COUNT(DISTINCT dd.calendar_date) < 10
        ORDER BY total_visits ASC;
        """.strip()
        return sql, []
    
    def staffing_hourly_patterns(self, nl_query: str) -> Tuple[str, List]:
        """Hourly visit patterns for staffing adjustment"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            fpv.visit_hour,
            COUNT(*) AS total_visits,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients,
            COUNT(DISTINCT dd.calendar_date) AS days_with_visits,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT dd.calendar_date), 0), 2) AS avg_visits_per_day
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY fpv.visit_hour
        ORDER BY fpv.visit_hour;
        """.strip()
        return sql, []
    
    def staffing_projected_load(self, nl_query: str) -> Tuple[str, List]:
        """Projected patient load based on historical quarterly trends"""
        sql = """
        SELECT 
            dd.year,
            dd.quarter,
            COUNT(DISTINCT fpv.patient_id) AS unique_patients,
            COUNT(*) AS total_visits,
            ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT dd.calendar_date), 0), 2) AS avg_daily_visits,
            LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year) AS prev_year_same_qtr_visits,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year)) /
                NULLIF(LAG(COUNT(*)) OVER (PARTITION BY dd.quarter ORDER BY dd.year), 0)
            , 2) AS yoy_growth_pct
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        GROUP BY dd.year, dd.quarter
        ORDER BY dd.year, dd.quarter;
        """.strip()
        return sql, []
    
    def retention_30day(self, nl_query: str) -> Tuple[str, List]:
        """Patients registered in each month who return within 30 days"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
            month_num_end = re.search(r"AND (\d+)", month_clause)
            if month_num_start and month_num_end:
                start_month = int(month_num_start.group(1))
                end_month = int(month_num_end.group(1))
                time_filter = f"WHERE fv.reg_year = {year_list} AND fv.reg_month BETWEEN {start_month} AND {end_month}"
            else:
                time_filter = f"WHERE fv.reg_year = {year_list}"
        else:
            time_filter = ""
        
        sql = f"""
        WITH first_visits AS (
            SELECT 
                patient_id,
                MIN(visit_timestamp::date) AS first_visit_date,
                EXTRACT(YEAR FROM MIN(visit_timestamp))::int AS reg_year,
                EXTRACT(MONTH FROM MIN(visit_timestamp))::int AS reg_month
            FROM fact_patient_visits
            GROUP BY patient_id
        ),
        return_visits AS (
            SELECT 
                fv.patient_id,
                fv.reg_year AS rv_reg_year,
                fv.reg_month AS rv_reg_month,
                MIN(fpv.visit_timestamp::date) AS next_visit_date
            FROM first_visits fv
            JOIN fact_patient_visits fpv ON fv.patient_id = fpv.patient_id
                AND fpv.visit_timestamp::date > fv.first_visit_date
            GROUP BY fv.patient_id, fv.reg_year, fv.reg_month
        )
        SELECT 
            fv.reg_year AS year,
            fv.reg_month AS month,
            COUNT(DISTINCT fv.patient_id) AS registered_patients,
            COUNT(DISTINCT CASE WHEN rv.next_visit_date - fv.first_visit_date <= 30 THEN rv.patient_id END) AS returned_within_30_days,
            ROUND(100.0 * COUNT(DISTINCT CASE WHEN rv.next_visit_date - fv.first_visit_date <= 30 THEN rv.patient_id END) / NULLIF(COUNT(DISTINCT fv.patient_id), 0), 2) AS return_rate_pct
        FROM first_visits fv
        LEFT JOIN return_visits rv ON fv.patient_id = rv.patient_id AND fv.reg_year = rv.rv_reg_year AND fv.reg_month = rv.rv_reg_month
        {time_filter}
        GROUP BY fv.reg_year, fv.reg_month
        ORDER BY fv.reg_year, fv.reg_month;
        """.strip()
        return sql, []
    
    def retention_3m_6m(self, nl_query: str) -> Tuple[str, List]:
        """Three-month and six-month retention rates"""
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
            month_num_end = re.search(r"AND (\d+)", month_clause)
            if month_num_start and month_num_end:
                start_month = int(month_num_start.group(1))
                end_month = int(month_num_end.group(1))
                time_filter = f"WHERE fv.reg_year = {year_list} AND fv.reg_month BETWEEN {start_month} AND {end_month}"
            else:
                time_filter = f"WHERE fv.reg_year = {year_list}"
        else:
            time_filter = ""
        
        sql = f"""
        WITH first_visits AS (
            SELECT 
                patient_id,
                MIN(visit_timestamp::date) AS first_visit_date,
                EXTRACT(YEAR FROM MIN(visit_timestamp))::int AS reg_year,
                EXTRACT(MONTH FROM MIN(visit_timestamp))::int AS reg_month
            FROM fact_patient_visits
            GROUP BY patient_id
        ),
        return_info AS (
            SELECT 
                fv.patient_id,
                fv.first_visit_date,
                MIN(fpv.visit_timestamp::date) AS next_visit_date,
                MAX(fpv.visit_timestamp::date) AS latest_visit_date
            FROM first_visits fv
            JOIN fact_patient_visits fpv ON fv.patient_id = fpv.patient_id
                AND fpv.visit_timestamp::date > fv.first_visit_date
            GROUP BY fv.patient_id, fv.first_visit_date
        )
        SELECT 
            COUNT(DISTINCT fv.patient_id) AS total_patients,
            COUNT(DISTINCT fv.patient_id) FILTER (
                WHERE ri.next_visit_date IS NOT NULL
            ) AS patients_who_returned,
            COUNT(DISTINCT fv.patient_id) FILTER (
                WHERE ri.next_visit_date - fv.first_visit_date <= 90
            ) AS returned_within_3_months,
            ROUND(100.0 * COUNT(DISTINCT fv.patient_id) FILTER (
                WHERE ri.next_visit_date - fv.first_visit_date <= 90
            ) / NULLIF(COUNT(DISTINCT fv.patient_id), 0), 2) AS three_month_retention_pct,
            COUNT(DISTINCT fv.patient_id) FILTER (
                WHERE ri.next_visit_date - fv.first_visit_date <= 180
            ) AS returned_within_6_months,
            ROUND(100.0 * COUNT(DISTINCT fv.patient_id) FILTER (
                WHERE ri.next_visit_date - fv.first_visit_date <= 180
            ) / NULLIF(COUNT(DISTINCT fv.patient_id), 0), 2) AS six_month_retention_pct
        FROM first_visits fv
        LEFT JOIN return_info ri ON fv.patient_id = ri.patient_id
        {time_filter};
        """.strip()
        return sql, []
    
    def retention_return_gap(self, nl_query: str) -> Tuple[str, List]:
        """How long after initial visit do patients typically return"""
        sql = """
        WITH first_visits AS (
            SELECT 
                patient_id,
                MIN(visit_timestamp::date) AS first_visit_date
            FROM fact_patient_visits
            GROUP BY patient_id
        ),
        return_gap AS (
            SELECT 
                fv.patient_id,
                MIN(fpv.visit_timestamp::date) - fv.first_visit_date AS days_to_return
            FROM first_visits fv
            JOIN fact_patient_visits fpv ON fv.patient_id = fpv.patient_id
                AND fpv.visit_timestamp::date > fv.first_visit_date
            GROUP BY fv.patient_id, fv.first_visit_date
        )
        SELECT 
            COUNT(*) AS patients_who_returned,
            ROUND(AVG(days_to_return)::numeric, 1) AS avg_days_to_return,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_to_return)::numeric, 1) AS median_days_to_return,
            MIN(days_to_return) AS min_days,
            MAX(days_to_return) AS max_days,
            COUNT(*) FILTER (WHERE days_to_return <= 7) AS returned_within_7_days,
            COUNT(*) FILTER (WHERE days_to_return <= 30) AS returned_within_30_days,
            COUNT(*) FILTER (WHERE days_to_return <= 90) AS returned_within_90_days
        FROM return_gap;
        """.strip()
        return sql, []
    
    def retention_first_time(self, nl_query: str) -> Tuple[str, List]:
        """Proportion of first-time patients who return for follow-up"""
        sql = """
        WITH patient_visits AS (
            SELECT 
                patient_id,
                COUNT(DISTINCT visit_timestamp::date) AS distinct_visit_dates
            FROM fact_patient_visits
            GROUP BY patient_id
        )
        SELECT 
            COUNT(*) AS total_patients,
            COUNT(*) FILTER (WHERE distinct_visit_dates = 1) AS one_visit_only,
            COUNT(*) FILTER (WHERE distinct_visit_dates > 1) AS returned_for_followup,
            ROUND(100.0 * COUNT(*) FILTER (WHERE distinct_visit_dates > 1) / NULLIF(COUNT(*), 0), 2) AS followup_return_pct
        FROM patient_visits;
        """.strip()
        return sql, []
    
    def retention_by_diagnosis(self, nl_query: str) -> Tuple[str, List]:
        """Retention rates by diagnosis"""
        sql = """
        WITH first_diagnosis_visit AS (
            SELECT 
                fpv.patient_id,
                fpv.diagnosis_id,
                d.description AS diagnosis,
                MIN(fpv.visit_timestamp::date) AS first_visit_date
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            GROUP BY fpv.patient_id, fpv.diagnosis_id, d.description
        ),
        return_info AS (
            SELECT 
                fdv.patient_id,
                fdv.diagnosis_id,
                fdv.diagnosis,
                MIN(fpv.visit_timestamp::date) AS next_visit_date,
                MIN(fpv.visit_timestamp::date) - fdv.first_visit_date AS days_to_return
            FROM first_diagnosis_visit fdv
            JOIN fact_patient_visits fpv ON fdv.patient_id = fpv.patient_id
                AND fpv.diagnosis_id = fdv.diagnosis_id
                AND fpv.visit_timestamp::date > fdv.first_visit_date
            GROUP BY fdv.patient_id, fdv.diagnosis_id, fdv.diagnosis, fdv.first_visit_date
        )
        SELECT 
            fdv.diagnosis,
            COUNT(DISTINCT fdv.patient_id) AS total_patients,
            COUNT(DISTINCT ri.patient_id) AS patients_who_returned,
            ROUND(100.0 * COUNT(DISTINCT ri.patient_id) / NULLIF(COUNT(DISTINCT fdv.patient_id), 0), 2) AS retention_pct,
            ROUND(AVG(ri.days_to_return)::numeric, 1) AS avg_days_to_return
        FROM first_diagnosis_visit fdv
        LEFT JOIN return_info ri ON fdv.patient_id = ri.patient_id AND fdv.diagnosis_id = ri.diagnosis_id
        GROUP BY fdv.diagnosis
        HAVING COUNT(DISTINCT fdv.patient_id) >= 5
        ORDER BY retention_pct DESC
        LIMIT 20;
        """.strip()
        return sql, []
    
    def retention_by_payer(self, nl_query: str) -> Tuple[str, List]:
        """Retention rates by payer type"""
        sql = """
        WITH patient_payer AS (
            SELECT 
                fpv.patient_id,
                py.payer_type,
                MIN(fpv.visit_timestamp::date) AS first_visit_date
            FROM fact_patient_visits fpv
            JOIN dim_payer py ON fpv.payer_id = py.payer_id
            GROUP BY fpv.patient_id, py.payer_type
        ),
        return_info AS (
            SELECT 
                pp.patient_id,
                pp.payer_type,
                MIN(fpv.visit_timestamp::date) AS next_visit_date,
                MIN(fpv.visit_timestamp::date) - pp.first_visit_date AS days_to_return
            FROM patient_payer pp
            JOIN fact_patient_visits fpv ON pp.patient_id = fpv.patient_id
                AND fpv.visit_timestamp::date > pp.first_visit_date
            JOIN dim_payer py ON fpv.payer_id = py.payer_id AND py.payer_type = pp.payer_type
            GROUP BY pp.patient_id, pp.payer_type, pp.first_visit_date
        )
        SELECT 
            pp.payer_type,
            COUNT(DISTINCT pp.patient_id) AS total_patients,
            COUNT(DISTINCT ri.patient_id) AS patients_who_returned,
            ROUND(100.0 * COUNT(DISTINCT ri.patient_id) / NULLIF(COUNT(DISTINCT pp.patient_id), 0), 2) AS retention_pct,
            ROUND(AVG(ri.days_to_return)::numeric, 1) AS avg_days_to_return
        FROM patient_payer pp
        LEFT JOIN return_info ri ON pp.patient_id = ri.patient_id AND pp.payer_type = ri.payer_type
        GROUP BY pp.payer_type
        ORDER BY retention_pct DESC;
        """.strip()
        return sql, []
    
    def patient_avg_age(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
            sql = f"""
            SELECT 
                ROUND(AVG(EXTRACT(YEAR FROM AGE(p.dob)))::numeric, 1) AS avg_age,
                MIN(EXTRACT(YEAR FROM AGE(p.dob)))::int AS min_age,
                MAX(EXTRACT(YEAR FROM AGE(p.dob)))::int AS max_age,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(YEAR FROM AGE(p.dob)))::numeric, 1) AS median_age,
                COUNT(DISTINCT fpv.patient_id) AS total_patients
            FROM fact_patient_visits fpv
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter};
            """.strip()
        else:
            sql = """
            SELECT 
                ROUND(AVG(EXTRACT(YEAR FROM AGE(dob)))::numeric, 1) AS avg_age,
                MIN(EXTRACT(YEAR FROM AGE(dob)))::int AS min_age,
                MAX(EXTRACT(YEAR FROM AGE(dob)))::int AS max_age,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(YEAR FROM AGE(dob)))::numeric, 1) AS median_age,
                COUNT(*) AS total_patients
            FROM dim_patient;
            """.strip()
        return sql, []
    
    def patient_age_proportion(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            if month_clause:
                month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
                month_num_end = re.search(r"AND (\d+)", month_clause)
                if month_num_start and month_num_end:
                    start_month = month_num_start.group(1)
                    end_month = month_num_end.group(1)
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND EXTRACT(MONTH FROM created_at)::int BETWEEN {start_month} AND {end_month}"
                else:
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int IN ({year_list})"
            else:
                time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int IN ({year_list})"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            CASE
                WHEN EXTRACT(YEAR FROM AGE(dob)) < 18 THEN 'Pediatric (0-17)'
                WHEN EXTRACT(YEAR FROM AGE(dob)) BETWEEN 18 AND 64 THEN 'Adult (18-64)'
                ELSE 'Elderly (65+)'
            END AS age_category,
            COUNT(*) AS patient_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
        FROM dim_patient
        {time_filter}
        GROUP BY age_category
        ORDER BY patient_count DESC;
        """.strip()
        return sql, []
    
    def gender_over_time(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        is_registration_query = "register" in q or "registrat" in q or "signed up" in q or "joined" in q
        
        if is_registration_query:
            sql = """
            SELECT 
                EXTRACT(YEAR FROM created_at)::int AS year,
                CASE gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' ELSE 'Unknown' END AS gender,
                COUNT(*) AS patient_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY EXTRACT(YEAR FROM created_at)::int), 2) AS pct_of_year,
                ROUND(
                    COUNT(*) FILTER (WHERE gender = 'F')::numeric / 
                    NULLIF(COUNT(*) FILTER (WHERE gender = 'M'), 0), 2
                ) AS female_to_male_ratio
            FROM dim_patient
            GROUP BY EXTRACT(YEAR FROM created_at)::int, gender
            ORDER BY year, gender;
            """.strip()
        else:
            sql = """
            SELECT 
                dd.year,
                CASE p.gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' ELSE 'Unknown' END AS gender,
                COUNT(DISTINCT fpv.patient_id) AS patient_count,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(DISTINCT fpv.patient_id) / SUM(COUNT(DISTINCT fpv.patient_id)) OVER (PARTITION BY dd.year), 2) AS pct_of_year,
                ROUND(
                    COUNT(*) FILTER (WHERE p.gender = 'F')::numeric / 
                    NULLIF(COUNT(*) FILTER (WHERE p.gender = 'M'), 0), 2
                ) AS female_to_male_ratio
            FROM fact_patient_visits fpv
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            GROUP BY dd.year, p.gender
            ORDER BY dd.year, p.gender;
            """.strip()
        return sql, []
    
    def staff_registrations(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            if month_clause:
                month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
                month_num_end = re.search(r"AND (\d+)", month_clause)
                if month_num_start and month_num_end:
                    start_month = month_num_start.group(1)
                    end_month = month_num_end.group(1)
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND EXTRACT(MONTH FROM created_at)::int BETWEEN {start_month} AND {end_month} AND registered_by IS NOT NULL"
                else:
                    month_num = re.search(r"=\s*(\d+)", month_clause)
                    if month_num:
                        time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND EXTRACT(MONTH FROM created_at)::int = {month_num.group(1)} AND registered_by IS NOT NULL"
                    else:
                        time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND registered_by IS NOT NULL"
            else:
                time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND registered_by IS NOT NULL"
        else:
            time_filter = "WHERE registered_by IS NOT NULL"
        
        sql = f"""
        SELECT 
            COALESCE(registered_by, 'Unknown') AS staff_member,
            COUNT(*) AS patients_registered
        FROM dim_patient
        {time_filter}
        GROUP BY registered_by
        ORDER BY patients_registered DESC
        LIMIT 20;
        """.strip()
        return sql, []
    
    def location_patients(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            if month_clause:
                month_num_start = re.search(r"BETWEEN (\d+)", month_clause)
                month_num_end = re.search(r"AND (\d+)", month_clause)
                if month_num_start and month_num_end:
                    start_month = month_num_start.group(1)
                    end_month = month_num_end.group(1)
                    time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND EXTRACT(MONTH FROM created_at)::int BETWEEN {start_month} AND {end_month}"
                else:
                    month_num = re.search(r"=\s*(\d+)", month_clause)
                    if month_num:
                        time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list} AND EXTRACT(MONTH FROM created_at)::int = {month_num.group(1)}"
                    else:
                        time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int = {year_list}"
            else:
                time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int IN ({year_list})"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COALESCE(residence, 'Unknown') AS residence,
            COUNT(*) AS patient_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
        FROM dim_patient
        {time_filter}
        GROUP BY residence
        ORDER BY patient_count DESC;
        """.strip()
        return sql, []
    
    def growth_geographic(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        two_months = self._extract_two_months(q)
        
        if two_months:
            month1, month2, year = two_months
            
            sql = f"""
            SELECT 
                COALESCE(p.residence, 'Unknown') AS residence,
                {month1} AS month_from,
                {month2} AS month_to,
                {year} AS year,
                SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS visits_month_from,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) AS visits_month_to,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                    SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS absolute_change,
                ROUND(
                    100.0 * (
                        SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                        SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END)
                    ) / NULLIF(SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END), 0)
                , 2) AS mom_growth_pct
            FROM fact_patient_visits fpv
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year = {year} AND dd.month IN ({month1}, {month2})
            GROUP BY p.residence
            HAVING SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) > 0
            ORDER BY mom_growth_pct DESC NULLS LAST;
            """.strip()
            return sql, []
        
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COALESCE(p.residence, 'Unknown') AS residence,
            dd.year,
            COUNT(DISTINCT fpv.patient_id) AS patient_count,
            COUNT(*) AS visit_count,
            LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year) AS prev_year_visits,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year)) /
                NULLIF(LAG(COUNT(*)) OVER (PARTITION BY p.residence ORDER BY dd.year), 0)
            , 2) AS yoy_growth_pct
        FROM fact_patient_visits fpv
        JOIN dim_patient p ON fpv.patient_id = p.patient_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY p.residence, dd.year
        ORDER BY p.residence, dd.year;
        """.strip()
        return sql, []
    
    def growth_emerging_diagnoses(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        
        two_months = self._extract_two_months(q)
        
        if two_months:
            month1, month2, year = two_months
            
            sql = f"""
            SELECT 
                d.description AS diagnosis,
                {month1} AS month_from,
                {month2} AS month_to,
                {year} AS year,
                SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS visits_month_from,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) AS visits_month_to,
                SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                    SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) AS absolute_change,
                ROUND(
                    100.0 * (
                        SUM(CASE WHEN dd.month = {month2} THEN 1 ELSE 0 END) - 
                        SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END)
                    ) / NULLIF(SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END), 0)
                , 2) AS mom_growth_pct
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE dd.year = {year} AND dd.month IN ({month1}, {month2})
            GROUP BY d.description
            HAVING SUM(CASE WHEN dd.month = {month1} THEN 1 ELSE 0 END) > 0
            ORDER BY mom_growth_pct DESC NULLS LAST;
            """.strip()
            return sql, []
        
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            d.description AS diagnosis,
            dd.year,
            COUNT(*) AS visit_count,
            LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year) AS prev_year_visits,
            COUNT(*) - COALESCE(LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year), 0) AS absolute_change,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year)) /
                NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.year), 0)
            , 2) AS yoy_growth_pct
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter}
        GROUP BY d.description, dd.year
        HAVING COUNT(*) > 10
        ORDER BY absolute_change DESC NULLS LAST, dd.year;
        """.strip()
        return sql, []
    
    def case_mix_yoy(self, nl_query: str) -> Tuple[str, List]:
        sql = """
        SELECT 
            dd.year,
            d.description AS diagnosis,
            COUNT(*) AS visit_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_year
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        GROUP BY dd.year, d.description
        HAVING COUNT(*) > 20
        ORDER BY dd.year, visit_count DESC;
        """.strip()
        return sql, []
    
    def chronic_workload_proportion(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        is_comparison = self._is_year_comparison_query(nl_query)
        
        chronic_icd_prefixes = [
            'E10', 'E11', 'E13', 'E14', 'I10', 'I11', 'I12', 'I13', 'I15',
            'J44', 'J45', 'E66', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25',
            'I50', 'N18', 'M05', 'M06', 'M15', 'M16', 'M17', 'M19', 'G20',
            'G30', 'K21', 'K25', 'K26', 'K29', 'L40', 'M32', 'G40',
            'J31', 'J32', 'J37', 'J41', 'J42', 'J43',
            'D50', 'D51', 'D52', 'D53', 'D56', 'D57', 'D58', 'D59',
            'E00', 'E01', 'E02', 'E03', 'E04', 'E05',
            'F32', 'F33', 'F41', 'E55', 'K58', 'M54'
        ]
        prefix_conditions = " OR ".join([f"d.icd10_code LIKE '{p}%'" for p in chronic_icd_prefixes])
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        if is_comparison:
            sql = f"""
            SELECT 
                dd.year,
                CASE WHEN ({prefix_conditions}) THEN 'Chronic' ELSE 'Non-Chronic' END AS condition_type,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY dd.year), 2) AS pct_of_total
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY dd.year, condition_type
            ORDER BY dd.year, visit_count DESC;
            """.strip()
        else:
            sql = f"""
            SELECT 
                CASE WHEN ({prefix_conditions}) THEN 'Chronic' ELSE 'Non-Chronic' END AS condition_type,
                COUNT(*) AS visit_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE 1=1 {time_filter}
            GROUP BY condition_type
            ORDER BY visit_count DESC;
            """.strip()
        return sql, []
    
    def anomaly_diagnosis_spikes(self, nl_query: str) -> Tuple[str, List]:
        sql = """
        SELECT 
            d.description AS diagnosis,
            dd.year,
            dd.month,
            COUNT(*) AS visit_count,
            LAG(COUNT(*)) OVER (PARTITION BY d.description, dd.month ORDER BY dd.year) AS same_month_prev_year,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description, dd.month ORDER BY dd.year)) /
                NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description, dd.month ORDER BY dd.year), 0)
            , 2) AS pct_change
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        GROUP BY d.description, dd.year, dd.month
        HAVING COUNT(*) > 5
        ORDER BY pct_change DESC NULLS LAST
        LIMIT 30;
        """.strip()
        return sql, []
    
    def anomaly_pediatric(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        
        # Check for quarter comparison
        is_quarter = "q1" in q or "q2" in q or "quarter" in q
        
        # Check for respiratory specific
        is_respiratory = "respiratory" in q
        
        if is_quarter:
            # Q1 vs Q2 comparison within the same year
            sql = """
            SELECT 
                d.description AS diagnosis,
                dd.quarter,
                COUNT(*) AS visit_count,
                LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.quarter) AS prev_quarter_visits,
                COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.quarter) AS absolute_change,
                ROUND(
                    100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.quarter)) /
                    NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description ORDER BY dd.quarter), 0)
                , 2) AS pct_change
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            JOIN dim_patient p ON fpv.patient_id = p.patient_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE EXTRACT(YEAR FROM AGE(p.dob)) < 18
            """
            
            if is_respiratory:
                sql += """ AND (d.description ILIKE '%respiratory%' OR d.description ILIKE '%bronchitis%' OR 
                      d.description ILIKE '%pneumonia%' OR d.description ILIKE '%asthma%' OR 
                      d.description ILIKE '%influenza%' OR d.description ILIKE '%flu%' OR
                      d.description ILIKE '%cough%' OR d.description ILIKE '%lung%')"""
            
            sql += """
              AND dd.quarter IN (1, 2)
            GROUP BY d.description, dd.quarter
            HAVING COUNT(*) > 3
            ORDER BY pct_change DESC NULLS LAST
            LIMIT 30;
            """.strip()
            return sql, []
        
        # Default: year-over-year monthly comparison
        sql = """
        SELECT 
            d.description AS diagnosis,
            dd.year,
            dd.month,
            COUNT(*) AS visit_count,
            LAG(COUNT(*)) OVER (PARTITION BY d.description, dd.month ORDER BY dd.year) AS same_month_prev_year,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY d.description, dd.month ORDER BY dd.year)) /
                NULLIF(LAG(COUNT(*)) OVER (PARTITION BY d.description, dd.month ORDER BY dd.year), 0)
            , 2) AS pct_change
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        JOIN dim_patient p ON fpv.patient_id = p.patient_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE EXTRACT(YEAR FROM AGE(p.dob)) < 18
        """
        
        if is_respiratory:
            sql += """ AND (d.description ILIKE '%respiratory%' OR d.description ILIKE '%bronchitis%' OR 
                  d.description ILIKE '%pneumonia%' OR d.description ILIKE '%asthma%' OR 
                  d.description ILIKE '%influenza%' OR d.description ILIKE '%flu%' OR
                  d.description ILIKE '%cough%' OR d.description ILIKE '%lung%')"""
        
        sql += """
        GROUP BY d.description, dd.year, dd.month
        HAVING COUNT(*) > 3
        ORDER BY pct_change DESC NULLS LAST
        LIMIT 30;
        """.strip()
        return sql, []
    
    def anomaly_weekend(self, nl_query: str) -> Tuple[str, List]:
        sql = """
        SELECT 
            dd.year,
            dd.month,
            COUNT(*) FILTER (WHERE dd.is_weekend) AS weekend_visits,
            COUNT(*) FILTER (WHERE NOT dd.is_weekend) AS weekday_visits,
            ROUND(100.0 * COUNT(*) FILTER (WHERE dd.is_weekend) / NULLIF(COUNT(*), 0), 2) AS weekend_pct,
            LAG(ROUND(100.0 * COUNT(*) FILTER (WHERE dd.is_weekend) / NULLIF(COUNT(*), 0), 2))
                OVER (PARTITION BY dd.month ORDER BY dd.year) AS prev_year_weekend_pct
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        GROUP BY dd.year, dd.month
        ORDER BY dd.year, dd.month;
        """.strip()
        return sql, []
    
    def anomaly_physician(self, nl_query: str) -> Tuple[str, List]:
        sql = """
        SELECT 
            ph.name AS physician,
            ph.specialty,
            dd.year,
            dd.month,
            COUNT(*) AS visit_count,
            LAG(COUNT(*)) OVER (PARTITION BY ph.name, dd.month ORDER BY dd.year) AS same_month_prev_year,
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY ph.name, dd.month ORDER BY dd.year)) /
                NULLIF(LAG(COUNT(*)) OVER (PARTITION BY ph.name, dd.month ORDER BY dd.year), 0)
            , 2) AS pct_change
        FROM fact_patient_visits fpv
        JOIN dim_physician ph ON fpv.physician_id = ph.physician_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        GROUP BY ph.name, ph.specialty, dd.year, dd.month
        ORDER BY pct_change DESC NULLS LAST
        LIMIT 30;
        """.strip()
        return sql, []
    
    def dq_patients_no_visits(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND EXTRACT(YEAR FROM p.created_at)::int IN ({year_list})"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(*) AS patients_without_visits
        FROM dim_patient p
        LEFT JOIN fact_patient_visits fpv ON p.patient_id = fpv.patient_id
        WHERE fpv.visit_id IS NULL {time_filter};
        """.strip()
        return sql, []
    
    def dq_visits_no_diagnosis(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(*) AS visits_without_diagnosis
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fpv.diagnosis_id IS NULL {time_filter};
        """.strip()
        return sql, []
    
    def dq_visits_no_physician(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(*) AS visits_without_physician
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fpv.physician_id IS NULL {time_filter};
        """.strip()
        return sql, []
    
    def dq_missing_payer(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(*) AS visits_without_payer
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE fpv.payer_id IS NULL {time_filter};
        """.strip()
        return sql, []
    
    def dq_age_dob_mismatch(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"AND EXTRACT(YEAR FROM created_at)::int IN ({year_list})"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(*) AS mismatched_records,
            COUNT(*) FILTER (WHERE ABS(age - EXTRACT(YEAR FROM AGE(dob))::int) > 1) AS significant_mismatches
        FROM dim_patient
        WHERE age IS NOT NULL AND dob IS NOT NULL {time_filter};
        """.strip()
        return sql, []
    
    def dq_duplicate_patients(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, _, _ = self._extract_date_range(q)
            time_filter = f"AND EXTRACT(YEAR FROM a.created_at)::int IN ({year_list})"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            a.patient_id AS patient_id_1,
            b.patient_id AS patient_id_2,
            a.first_name,
            a.last_name,
            a.dob,
            b.first_name AS first_name_2,
            b.last_name AS last_name_2,
            b.dob AS dob_2,
            SIMILARITY(a.full_name, b.full_name) AS name_similarity
        FROM dim_patient a
        JOIN dim_patient b ON a.patient_id < b.patient_id
            AND SIMILARITY(a.full_name, b.full_name) > 0.6
        WHERE 1=1 {time_filter}
        ORDER BY name_similarity DESC
        LIMIT 50;
        """.strip()
        return sql, []
    
    def dq_gender_distribution(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE EXTRACT(YEAR FROM created_at)::int IN ({year_list})"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            CASE gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' ELSE 'Unknown/Missing' END AS gender,
            COUNT(*) AS patient_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
        FROM dim_patient
        {time_filter}
        GROUP BY gender
        ORDER BY patient_count DESC;
        """.strip()
        return sql, []
    
    def dq_timestamp_consistency(self, nl_query: str) -> Tuple[str, List]:
        q = nl_query.lower()
        has_time = self._has_explicit_time_filter(q)
        
        if has_time:
            year_list, month_clause, _ = self._extract_date_range(q)
            time_filter = f"WHERE dd.year IN ({year_list}){month_clause}"
        else:
            time_filter = ""
        
        sql = f"""
        SELECT 
            COUNT(*) AS total_visits,
            COUNT(*) FILTER (WHERE fpv.visit_timestamp::date = dd.calendar_date) AS consistent,
            COUNT(*) FILTER (WHERE fpv.visit_timestamp::date != dd.calendar_date) AS inconsistent,
            ROUND(100.0 * COUNT(*) FILTER (WHERE fpv.visit_timestamp::date = dd.calendar_date) / NULLIF(COUNT(*), 0), 2) AS consistency_pct
        FROM fact_patient_visits fpv
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        {time_filter};
        """.strip()
        return sql, []
    
    def generate_analytical_sql(self, nl_query: str) -> Optional[Tuple[str, List]]:
        """Main entry point for analytical SQL generation"""
        intent = self.detect_intent(nl_query)
        
        if intent == "ranking":
            return self.ranking_query(nl_query)
        elif intent == "location":
            return self.location_query(nl_query)
        elif intent == "comparative":
            return self.comparative_analysis(nl_query)
        elif intent == "period_comparison":
            return self.period_comparison(nl_query)
        elif intent == "demographic":
            return self.demographic_analysis(nl_query)
        elif intent == "aggregate":
            return self.aggregate_statistics(nl_query)
        elif intent == "physician":
            return self.physician_analysis(nl_query)
        
        elif intent == "diagnosis_patients":
            return self.diagnosis_patients_count(nl_query)
        
        elif intent == "patient_count":
            return self.patient_count_analysis(nl_query)
        
        elif intent == "visit_count":
            return self.visit_count_analysis(nl_query)
        
        elif intent == "icd10_ranking":
            return self.icd10_ranking_analysis(nl_query)
        
        elif intent == "disease_demographic":
            return self.disease_demographic_analysis(nl_query)
        
        elif intent == "diagnosis_trend":
            return self.diagnosis_trend_analysis(nl_query)
        
        elif intent == "seasonal_pattern":
            return self.seasonal_pattern_analysis(nl_query)
        
        elif intent == "diagnosis_yoy":
            return self.diagnosis_yoy_analysis(nl_query)
        
        elif intent == "physician_workload_trend":
            return self.physician_workload_trend_analysis(nl_query)
        
        elif intent == "physician_repeat_patients":
            return self.physician_repeat_patients_analysis(nl_query)
        
        elif intent == "physician_workload_distribution":
            return self.physician_workload_distribution_analysis(nl_query)
        
        elif intent == "physician_peak_hours":
            return self.physician_peak_hours_analysis(nl_query)
        
        elif intent == "visit_yoy":
            periods = self._extract_comparison_periods(nl_query)
            return self.visit_yoy_analysis(nl_query, periods)
        
        elif intent == "visit_timeframe":
            return self.visit_timeframe_analysis(nl_query)
        
        elif intent == "visit_single_multiple":
            return self.visit_single_multiple_analysis(nl_query)
        
        elif intent == "visit_day_of_week":
            return self.visit_day_of_week_analysis(nl_query)
        
        elif intent == "visit_month":
            return self.visit_month_analysis(nl_query)
        
        elif intent == "visit_quarter_trend":
            return self.visit_quarter_trend_analysis(nl_query)
        
        elif intent == "visit_weekend":
            return self.visit_weekend_analysis(nl_query)
        
        elif intent == "visit_hour":
            return self.visit_hour_analysis(nl_query)
        
        elif intent == "specialty_visits":
            return self.specialty_visits_analysis(nl_query)
        
        elif intent == "specialty_growth":
            return self.specialty_growth_analysis(nl_query)
        
        elif intent == "pediatric_diagnoses":
            return self.pediatric_diagnoses_analysis(nl_query)
        
        elif intent == "chronic_conditions":
            return self.chronic_conditions_analysis(nl_query)
        
        elif intent == "recurrence_qoq":
            result = self.recurrence_qoq_analysis(nl_query)
            if result:
                return result
            return self.recurrence_highest_rates(nl_query)
        
        elif intent == "recurrence_highest_rates":
            return self.recurrence_highest_rates(nl_query)
        
        elif intent == "recurrence_repeated_patients":
            return self.recurrence_repeated_patients(nl_query)
        
        elif intent == "recurrence_avg_count":
            return self.recurrence_avg_count(nl_query)
        
        elif intent == "recurrence_time_span":
            return self.recurrence_time_span(nl_query)
        
        elif intent == "recurrence_patient_count":
            return self.recurrence_patient_count(nl_query)
        
        elif intent == "recurrence_by_age_group":
            return self.recurrence_by_age_group(nl_query)
        
        elif intent == "recurrence_by_physician":
            return self.recurrence_by_physician(nl_query)
        
        elif intent == "recurrence_by_payer":
            return self.recurrence_by_payer(nl_query)
        
        elif intent == "recurrence_high_risk":
            return self.recurrence_high_risk(nl_query)
        
        elif intent == "payer":
            return self.payer_analysis(nl_query)
        
        elif intent == "payer_highest":
            return self.payer_highest(nl_query)
        
        elif intent == "payer_selfpay_vs_insured":
            return self.payer_selfpay_vs_insured(nl_query)
        
        elif intent == "payer_over_time":
            return self.payer_over_time(nl_query)
        
        elif intent == "payer_diagnoses":
            return self.payer_diagnoses(nl_query)
        
        elif intent == "payer_comparison":
            return self.payer_comparison(nl_query)
        
        elif intent == "staffing_avg_daily_load":
            return self.staffing_avg_daily_load(nl_query)
        
        elif intent == "staffing_peak_months":
            return self.staffing_peak_months(nl_query)
        
        elif intent == "staffing_peak_days":
            return self.staffing_peak_days(nl_query)
        
        elif intent == "staffing_specialty_capacity":
            return self.staffing_specialty_capacity(nl_query)
        
        elif intent == "staffing_underutilized":
            return self.staffing_underutilized(nl_query)
        
        elif intent == "staffing_hourly_patterns":
            return self.staffing_hourly_patterns(nl_query)
        
        elif intent == "staffing_projected_load":
            return self.staffing_projected_load(nl_query)
        
        elif intent == "retention_30day":
            return self.retention_30day(nl_query)
        
        elif intent == "retention_3m_6m":
            return self.retention_3m_6m(nl_query)
        
        elif intent == "retention_return_gap":
            return self.retention_return_gap(nl_query)
        
        elif intent == "retention_first_time":
            return self.retention_first_time(nl_query)
        
        elif intent == "retention_by_diagnosis":
            return self.retention_by_diagnosis(nl_query)
        
        elif intent == "retention_by_payer":
            return self.retention_by_payer(nl_query)
        
        elif intent == "patient_avg_age":
            return self.patient_avg_age(nl_query)
        
        elif intent == "patient_age_proportion":
            return self.patient_age_proportion(nl_query)
        
        elif intent == "gender_over_time":
            return self.gender_over_time(nl_query)
        
        elif intent == "staff_registrations":
            return self.staff_registrations(nl_query)
        
        elif intent == "growth_geographic":
            return self.growth_geographic(nl_query)
        
        elif intent == "location_patients":
            return self.location_patients(nl_query)
        
        elif intent == "growth_emerging_diagnoses":
            return self.growth_emerging_diagnoses(nl_query)
        
        elif intent == "case_mix_yoy":
            return self.case_mix_yoy(nl_query)
        
        elif intent == "chronic_workload_proportion":
            return self.chronic_workload_proportion(nl_query)
        
        elif intent == "anomaly_diagnosis_spikes":
            return self.anomaly_diagnosis_spikes(nl_query)
        
        elif intent == "anomaly_pediatric":
            return self.anomaly_pediatric(nl_query)
        
        elif intent == "anomaly_weekend":
            return self.anomaly_weekend(nl_query)
        
        elif intent == "anomaly_physician":
            return self.anomaly_physician(nl_query)
        
        elif intent == "dq_patients_no_visits":
            return self.dq_patients_no_visits(nl_query)
        
        elif intent == "dq_visits_no_diagnosis":
            return self.dq_visits_no_diagnosis(nl_query)
        
        elif intent == "dq_visits_no_physician":
            return self.dq_visits_no_physician(nl_query)
        
        elif intent == "dq_missing_payer":
            return self.dq_missing_payer(nl_query)
        
        elif intent == "dq_age_dob_mismatch":
            return self.dq_age_dob_mismatch(nl_query)
        
        elif intent == "dq_duplicate_patients":
            return self.dq_duplicate_patients(nl_query)
        
        elif intent == "dq_gender_distribution":
            return self.dq_gender_distribution(nl_query)
        
        elif intent == "dq_timestamp_consistency":
            return self.dq_timestamp_consistency(nl_query)
        
        elif intent == "patient_registration_trend":
            return self.patient_registration_trend(nl_query)
        
        return None

# =====================================================
# Enhanced SQL Interpreter
# =====================================================
class SQLInterpreter:
    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.OLLAMA_MODEL
        self.client = OllamaClient(host=settings.OLLAMA_URL)
        self.analytical_generator = AnalyticalQueryGenerator(self)
    
    # =====================================================
    # Diagnosis extraction
    # =====================================================
    def _extract_diagnosis(self, q: str) -> Optional[str]:
        # Character class for diagnosis names: letters, numbers, spaces, hyphens, periods
        diag_chars = r"[a-zA-Z0-9\s\-\.]"
        
        # Try 'diagnosed with X' - capture multiple words, stop at common query-ending words
        m = re.search(
            rf"diagnosed\s+with\s+({diag_chars}+?)(?:\s+(?:in|on|for|to|with|from|and|or|the|a|an)\s|\s*\?|\s*$)",
            q, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        
        # Try 'suffering from X'
        m = re.search(
            rf"suffering\s+from\s+({diag_chars}+?)(?:\s+(?:in|on|for|to|with|from|and|or|the|a|an)\s|\s*\?|\s*$)",
            q, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()

        # Try 'had X' - more flexible
        m = re.search(
            rf"\bhad\s+({diag_chars}+?)(?:\s+(?:in|on|for|to|with|from|and|or|the|a|an)\s|\s*\?|\s*$)",
            q, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        
        # Fallback to simple 'with X' pattern for simple queries
        skip_words = {"patients", "people", "individuals", "children", "adults", "men", "women", "the"}
        m = re.search(
            rf"\bwith\s+({diag_chars}+?)(?:\s+(?:in|on|for|to|and|or|the|a|an)\s|\s*\?|\s*$)",
            q, re.IGNORECASE
        )
        if m:
            diagnosis_candidate = m.group(1).strip().lower()
            if diagnosis_candidate not in skip_words:
                return m.group(1).strip()
        
        m = re.search(
            r"(?:of\s+)?([a-zA-Z0-9\s]+?)\s+(diagnoses|diagnosis|cases)",
            q
        )
        if not m:
            return None

        phrase = m.group(1).strip()
        if " of " in phrase:
            phrase = phrase.split(" of ")[-1]
        return phrase.strip() or None

    # =====================================================
    # Safe Analytics SQL generation
    # =====================================================
    def _diagnosis_analytics(self, nl_query: str) -> Optional[str]:
        """Handle diagnosis-specific analytics queries"""
        q = MedicalQueryPreprocessor.normalize_medical_terms(nl_query.lower())
        current_year = datetime.now().year
        current_month = datetime.now().month

        diagnosis = self._extract_diagnosis(q)
        if not diagnosis:
            return None

        base_where = "d.description ILIKE $1"

        years = set(int(y) for y in re.findall(r"\b20\d{2}\b", q))
        m = re.search(r"last (\d+) years?", q)
        if m:
            n = int(m.group(1))
            years = set(range(current_year - n + 1, current_year + 1))
        if "last year" in q:
            years.add(current_year - 1)
        if "this year" in q or "current year" in q:
            years.add(current_year)
        if not years:
            years = {current_year}
        years = sorted(years)
        year_list = ",".join(map(str, years))

        months = [num for m, num in MONTHS.items() if m in q]
        m = re.search(r"last (\d+) months?", q)
        if m:
            n = int(m.group(1))
            parts = []
            for i in range(n):
                y = current_year
                mo = current_month - i
                if mo <= 0:
                    mo += 12
                    y -= 1
                parts.append(f"(dd.year={y} AND dd.month={mo})")
            month_clause = f" AND ({' OR '.join(parts)})"
        elif months:
            month_clause = f" AND dd.month IN ({','.join(map(str, months))})"
        else:
            month_clause = ""

        # Build year filter that handles LEFT JOIN (NULL dates should be excluded for year filters)
        year_filter = f"dd.year IN ({year_list}){month_clause}"

        and_intent = "both" in q
        trend_intent = any(x in q for x in ["trend", "over time", "monthly"])
        per_year_intent = any(x in q for x in ["each year", "per year", "yearly"])

        if and_intent and len(years) > 1:
            sql = f"""
            SELECT COUNT(*) AS patient_count
            FROM (
                SELECT fpv.patient_id
                FROM fact_patient_visits fpv
                JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
                LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
                WHERE {base_where}
                  AND {year_filter}
                GROUP BY fpv.patient_id
                HAVING COUNT(DISTINCT dd.year) = {len(years)}
            ) t;
            """.strip()
            return self._add_sql_params(sql, [f"%{diagnosis}%"])

        if trend_intent:
            sql = f"""
            SELECT
                dd.year_month,
                COUNT(DISTINCT fpv.patient_id) AS patient_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE {base_where}
              AND {year_filter}
            GROUP BY dd.year_month
            ORDER BY dd.year_month;
            """.strip()
            return self._add_sql_params(sql, [f"%{diagnosis}%"])

        if per_year_intent:
            sql = f"""
            SELECT
                dd.year,
                COUNT(DISTINCT fpv.patient_id) AS patient_count
            FROM fact_patient_visits fpv
            JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
            LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
            WHERE {base_where}
            GROUP BY dd.year
            ORDER BY dd.year;
            """.strip()
            return self._add_sql_params(sql, [f"%{diagnosis}%"])

        sql = f"""
        SELECT COUNT(DISTINCT fpv.patient_id) AS patient_count
        FROM fact_patient_visits fpv
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE {base_where}
          AND {year_filter};
        """.strip()
        return self._add_sql_params(sql, [f"%{diagnosis}%"])

    # =====================================================
    # Patient listing
    # =====================================================
    def list_patients_for_diagnosis(self, nl_query: str) -> str:
        q = nl_query.lower()
        
        # Extract diagnosis
        diagnosis = self._extract_diagnosis(q)
        if not diagnosis:
            raise RuntimeError("Unable to parse diagnosis query")
        
        # Check if query has explicit year
        has_year = bool(re.search(r"\b20\d{2}\b", q))
        
        # Build WHERE clause
        base_where = f"d.description ILIKE $1"
        
        if has_year:
            # Use _diagnosis_analytics for year-filtered queries
            sql = self._diagnosis_analytics(nl_query)
            if not sql:
                raise RuntimeError("Unable to parse diagnosis query")
            
            if "GROUP BY dd.year_month" in sql or "GROUP BY dd.year" in sql:
                return sql
            
            match = re.search(r"WHERE (.*?)(GROUP BY|;)", sql, re.DOTALL)
            if not match:
                raise RuntimeError("Failed to extract WHERE clause")
            where_clause = match.group(1).strip()
        else:
            # No year specified - search all years (last 3 years as reasonable default)
            # This handles queries like "list patients with diabetes" without year
            year_list = "2024, 2025, 2026"  # Adjust based on your data range
            # Handle LEFT JOIN - use OR with NULL check
            where_clause = f"{base_where} AND (dd.year IS NULL OR dd.year IN ({year_list}))"
        
        return f"""
        SELECT DISTINCT ON (fpv.patient_id)
            fpv.patient_id,
            p.first_name,
            p.last_name,
            p.gender,
            p.dob,
            d.description AS diagnosis,
            dd.year,
            dd.month
        FROM fact_patient_visits fpv
        JOIN dim_patient p ON fpv.patient_id = p.patient_id
        JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id
        LEFT JOIN dim_date dd ON fpv.date_id = dd.date_id
        WHERE {where_clause}
        ORDER BY fpv.patient_id, dd.year DESC, dd.month DESC;
        """.strip()

    def _add_sql_params(self, sql: str, params: list) -> str:
        """Helper method to store SQL with parameters for later execution"""
        if not hasattr(self, '_sql_params'):
            self._sql_params = {}
        
        sql_key = hashlib.md5(sql.encode()).hexdigest()
        self._sql_params[sql_key] = params
        
        return sql

    async def interpret(self, nl_query: str) -> Dict[str, Any]:
        """
        Enhanced interpret method with analytical query support
        Returns: {"sql": str, "params": list, "cache_key": str}
        """
        start_time = datetime.now()
        
        try:
            cache_key = query_cache.get_cache_key(nl_query)
            cached_result = query_cache.get(cache_key)
            if cached_result:
                await sql_auditor.log_query_execution(
                    nl_query, cached_result["sql"], 
                    0.0, 0, {"cache_hit": True}
                )
                return cached_result

            # Check for period comparison keywords first - these take priority
            q_lower = nl_query.lower()
            is_period_query = any(kw in q_lower for kw in [
                "year over year", "yoy", " vs ", "versus", "compared to", "compared with",
                "change in", "difference in", "growth in", "increase in", "decrease in",
                "monthly growth", "growth rate"
            ])
            
            # Check for data quality queries - handle these explicitly
            is_data_quality_query = any(x in q_lower for x in [
                "inconsisten", "duplicate", "missing", "without a", "without an",
                "data entry error", "data quality", "abnormal gender",
                "consistent with", "recorded without", "consistency", "timestamp"
            ])
            
            sql = None
            params = []
            
            # For queries with period keywords, use analytical generator first
            if is_period_query or is_data_quality_query:
                analytical_intent = self.analytical_generator.detect_intent(nl_query)
                if analytical_intent != "unknown":
                    result = self.analytical_generator.generate_analytical_sql(nl_query)
                    if result:
                        sql, params = result
            
            # Also try analytical generator for other analytical queries (ranking, etc.)
            if not sql:
                analytical_intent = self.analytical_generator.detect_intent(nl_query)
                if analytical_intent != "unknown":
                    result = self.analytical_generator.generate_analytical_sql(nl_query)
                    if result:
                        sql, params = result
            
            # Skip LLM fallback for data quality queries - return error instead
            if not sql and is_data_quality_query:
                return {
                    "error": {
                        "error": "Unable to process data quality query",
                        "suggestion": "Please rephrase your question",
                        "technical": "No SQL generated for data quality query"
                    },
                    "sql": None
                }
            
            # Fall back to rule-based diagnosis analytics
            if not sql or sql is None:
                sql = self._diagnosis_analytics(nl_query)
                # _diagnosis_analytics uses _add_sql_params method internally
            
            # Fall back to LLM if no rule-based SQL generated
            if not sql:
                prompt = SQL_INTERPRETER_PROMPT.replace("{nl_query}", nl_query)
                sql = await self.client.generate(self.model, prompt, max_tokens=256)
                sql = sql.strip()

            is_valid, error_msg = SQLValidator.validate_sql(sql)
            if not is_valid:
                error_response = sql_error_handler.provide_user_friendly_error(
                    Exception(f"SQL validation failed: {error_msg}"), nl_query
                )
                return {"error": error_response, "sql": None}

            # If params weren't set from analytical generator, check _sql_params (for _diagnosis_analytics)
            if not params and hasattr(self, '_sql_params'):
                sql_key = hashlib.md5(sql.encode()).hexdigest()
                params = self._sql_params.get(sql_key, [])

            result = {"sql": sql, "params": params, "cache_key": cache_key}
            query_cache.set(cache_key, result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            await sql_auditor.log_query_execution(
                nl_query, sql, execution_time, 0, {"method": "rule-based"}
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            await sql_auditor.log_query_execution(
                nl_query, "ERROR", execution_time, 0, {"error": str(e)}
            )
            
            error_response = sql_error_handler.provide_user_friendly_error(e, nl_query)
            return {"error": error_response, "sql": None}
