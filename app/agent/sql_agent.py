"""
app/agent/sql_agent.py

LLM-powered SQL Generation Agent using LangGraph.
Dynamically generates SQL queries from natural language questions.
"""

import re
import json
import logging
import hashlib
import asyncio
from datetime import datetime
from typing import TypedDict, Optional, List, Dict, Any, Annotated
from functools import lru_cache

from app.core.config import get_settings
from app.llm.ollama_client import OllamaClient
from app.agent.sql_tool import run_sql_query
from app.agent.sql_interpreter import SQLValidator

settings = get_settings()
logger = logging.getLogger("sql_agent")

# =====================================================
# DATABASE SCHEMA
# =====================================================
DATABASE_SCHEMA = """
Tables:
1. dim_patient
   - patient_id (VARCHAR, PRIMARY KEY) - Format: NCH-XXXXX
   - first_name (VARCHAR)
   - last_name (VARCHAR)
   - gender (CHAR) - 'M' or 'F'
   - dob (DATE) - Date of birth
   - age (INT)
   - contact (VARCHAR)
   - residence (VARCHAR) - Patient's residential area
   - registered_by (VARCHAR)
   - created_at (TIMESTAMP)
   - full_name (TEXT, generated) - lowercase first_name + last_name

2. dim_physician
   - physician_id (INT, PRIMARY KEY)
   - name (VARCHAR)
   - specialty (VARCHAR)

3. dim_diagnosis
   - diagnosis_id (SERIAL, PRIMARY KEY)
   - icd10_code (VARCHAR)
   - description (VARCHAR) - Diagnosis description

4. dim_payer
   - payer_id (SERIAL, PRIMARY KEY)
   - payer_name (VARCHAR)
   - payer_type (VARCHAR)

5. dim_date
   - date_id (SERIAL, PRIMARY KEY)
   - calendar_date (DATE)
   - year (INT)
   - month (INT)
   - day (INT)
   - day_of_week (INT) - 1=Monday, 7=Sunday
   - week_of_month (INT)
   - is_weekend (BOOLEAN)
   - year_month (VARCHAR) - Format: YYYY-MM
   - quarter (INT) - 1-4

6. fact_patient_visits
   - visit_id (SERIAL, PRIMARY KEY)
   - patient_id (VARCHAR) - FK to dim_patient
   - physician_id (INT) - FK to dim_physician
   - diagnosis_id (INT) - FK to dim_diagnosis
   - payer_id (INT) - FK to dim_payer
   - date_id (INT) - FK to dim_date
   - visit_timestamp (TIMESTAMP)
   - visit_hour (INT) - Hour of visit (0-23)
   - created_at (TIMESTAMP)

7. fact_recurrence_analysis
   - patient_id (VARCHAR) - FK to dim_patient
   - diagnosis_id (INT) - FK to dim_diagnosis
   - recurrence_count (INT) - Number of times condition recurred
   - first_occurrence_date (DATE)
   - last_occurrence_date (DATE)
   - created_at (TIMESTAMP)
   - PRIMARY KEY (patient_id, diagnosis_id)

Relationships:
- fact_patient_visits.patient_id -> dim_patient.patient_id
- fact_patient_visits.physician_id -> dim_physician.physician_id
- fact_patient_visits.diagnosis_id -> dim_diagnosis.diagnosis_id
- fact_patient_visits.payer_id -> dim_payer.payer_id
- fact_patient_visits.date_id -> dim_date.date_id
"""

SQL_GENERATION_SYSTEM_PROMPT = f"""You are an expert SQL query generator for a medical analytics database.

Your task is to convert natural language questions into accurate SQL SELECT queries.

Database Schema:
{DATABASE_SCHEMA}

IMPORTANT RULES:
1. Only generate SELECT queries - NEVER generate INSERT, UPDATE, DELETE, DROP, or any modification queries
2. Always use proper JOINs to connect fact tables with dimension tables
3. Use appropriate aggregations (COUNT, SUM, AVG, MAX, MIN) with GROUP BY
4. Format dates properly for the database
5. Use ILIKE for case-insensitive text matching
6. Always include ORDER BY and LIMIT when appropriate
7. Use COALESCE to handle NULL values gracefully

Common Query Patterns:
- Counting patients: COUNT(DISTINCT patient_id)
- Counting visits: COUNT(*) or COUNT(DISTINCT visit_id)
- Date filtering: WHERE dd.year = 2024, dd.month BETWEEN 1 AND 6
- Age calculation: For age groups, use: CASE WHEN EXTRACT(YEAR FROM age(dob)) BETWEEN 0 AND 5 THEN '0-5' END
- Residential area: p.residence
- Gender: p.gender IN ('M', 'F')
- Physician specialty: ph.specialty
- Payer types: pay.payer_name, pay.payer_type

Return ONLY the JSON object, no additional text.
"""

SQL_GENERATION_USER_PROMPT = """Generate a SQL query to answer this question:
{question}

CRITICAL - Query Type Detection (MUST FOLLOW):
- Words like: "list", "show", "display", "get all", "retrieve", "what are", "who were", "find patients", "show patients", "list all" → SELECT actual ROWS (patient details, visit details, etc.)
- Words like: "how many", "count", "total", "number of", "summarize", "statistics" → SELECT with COUNT

EXAMPLES (FOLLOW THESE PATTERNS):
Q: list patients with diabetes
A: {{"sql": "SELECT p.patient_id, p.first_name, p.last_name, p.dob, p.gender, d.description as diagnosis FROM fact_patient_visits fpv JOIN dim_patient p ON fpv.patient_id = p.patient_id JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id WHERE d.description ILIKE $1", "params": ["%diabetes%"], "explanation": "Lists patient details with diabetes"}}

Q: how many patients have diabetes
A: {{"sql": "SELECT COUNT(DISTINCT p.patient_id) as total_patients FROM fact_patient_visits fpv JOIN dim_patient p ON fpv.patient_id = p.patient_id JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id WHERE d.description ILIKE $1", "params": ["%diabetes%"], "explanation": "Counts patients with diabetes"}}

Q: show all visits from 2024
A: {{"sql": "SELECT p.patient_id, p.first_name, p.last_name, v.visit_timestamp, d.description FROM fact_patient_visits v JOIN dim_patient p ON v.patient_id = p.patient_id JOIN dim_diagnosis d ON v.diagnosis_id = d.diagnosis_id JOIN dim_date dd ON v.date_id = dd.date_id WHERE dd.year = 2024", "params": [], "explanation": "Shows all visits from 2024"}}

Q: top 5 diagnoses
A: {{"sql": "SELECT d.description, COUNT(*) as visit_count FROM fact_patient_visits fpv JOIN dim_diagnosis d ON fpv.diagnosis_id = d.diagnosis_id GROUP BY d.description ORDER BY visit_count DESC LIMIT 5", "params": [], "explanation": "Shows top 5 diagnoses by visit count"}}

Return ONLY valid JSON with this exact format:
{{
    "sql": "SELECT ...",
    "params": [],
    "explanation": "..."
}}
"""

# =====================================================
# STATE DEFINITION
# =====================================================
class SQLAgentState(TypedDict):
    """State for the SQL Agent LangGraph."""
    question: str
    generated_sql: Optional[str]
    params: List[Any]
    explanation: Optional[str]
    result: Optional[List[Dict]]
    error: Optional[str]
    validation_error: Optional[str]
    retry_count: int


# =====================================================
# SQL GENERATION NODE
# =====================================================
async def generate_sql_node(state: SQLAgentState) -> SQLAgentState:
    """Generate SQL query from natural language using LLM."""
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    
    # Check cache first
    cache_key = get_cache_key(question)
    cached = query_cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for query: {question[:50]}...")
        return {
            "generated_sql": cached["sql"],
            "params": cached.get("params", []),
            "explanation": cached.get("explanation", ""),
            "result": None,
            "error": None,
            "retry_count": retry_count
        }
    
    ollama = OllamaClient(settings.OLLAMA_URL)
    
    prompt = SQL_GENERATION_SYSTEM_PROMPT + "\n\n" + SQL_GENERATION_USER_PROMPT.format(question=question)
    
    try:
        response = await ollama.generate(
            model=settings.OLLAMA_MODEL,
            prompt=prompt,
            max_tokens=1024,
        )
        
        # Parse JSON from response
        parsed = parse_sql_response(response)
        
        if not parsed or not parsed.get("sql"):
            state["error"] = "Failed to generate SQL from LLM response"
            state["retry_count"] = retry_count + 1
            return state
        
        sql = parsed["sql"].strip()
        
        # Store in cache
        query_cache.set(cache_key, {
            "sql": sql,
            "params": parsed.get("params", []),
            "explanation": parsed.get("explanation", "")
        })
        
        state["generated_sql"] = sql
        state["params"] = parsed.get("params", [])
        state["explanation"] = parsed.get("explanation", "")
        state["error"] = None
        state["retry_count"] = retry_count
        
        return state
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        state["error"] = f"SQL generation failed: {str(e)}"
        state["retry_count"] = retry_count + 1
        return state


def parse_sql_response(response: str) -> Optional[Dict]:
    """Parse SQL query from LLM response."""
    try:
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?", "", response)
            response = re.sub(r"```$", "", response)
        
        # Find JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None


# =====================================================
# SQL VALIDATION NODE
# =====================================================
def validate_sql_node(state: SQLAgentState) -> SQLAgentState:
    """Validate generated SQL for safety and syntax."""
    sql = state.get("generated_sql")
    
    if not sql:
        state["validation_error"] = "No SQL to validate"
        return state
    
    # Use existing validator
    is_valid, error_msg = SQLValidator.validate_sql(sql)
    
    if not is_valid:
        state["validation_error"] = error_msg
        state["generated_sql"] = None  # Clear invalid SQL
    else:
        state["validation_error"] = None
    
    return state


# =====================================================
# SQL EXECUTION NODE
# =====================================================
async def execute_sql_node(state: SQLAgentState) -> SQLAgentState:
    """Execute the validated SQL query."""
    sql = state.get("generated_sql")
    params = state.get("params", [])
    validation_error = state.get("validation_error")
    
    if validation_error or not sql:
        state["error"] = state.get("error") or validation_error or "Invalid SQL"
        state["result"] = []
        return state
    
    try:
        if params:
            result = await run_sql_query(sql, tuple(params))
        else:
            result = await run_sql_query(sql)
        
        # Handle non-list results
        if not isinstance(result, list):
            if isinstance(result, dict):
                result = [result]
            else:
                result = []
        
        state["result"] = result
        state["error"] = None
        
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        state["error"] = f"SQL execution failed: {str(e)}"
        state["result"] = []
    
    return state


# =====================================================
# ERROR HANDLING / RETRY NODE
# =====================================================
def handle_error_node(state: SQLAgentState) -> SQLAgentState:
    """Handle errors and decide retry strategy."""
    retry_count = state.get("retry_count", 0)
    error = state.get("error")
    
    if retry_count < 3 and error:
        # Add retry context to help the LLM
        state["question"] = f"{state['question']} (Previous attempt failed: {error})"
        logger.info(f"Retrying SQL generation, attempt {retry_count + 1}")
    
    return state


# =====================================================
# RESULT FORMATTING NODE
# =====================================================
def format_result_node(state: SQLAgentState) -> SQLAgentState:
    """Format results for display."""
    result = state.get("result", [])
    question = state.get("question", "")
    
    if not result:
        return state
    
    # Add summary for large result sets
    if len(result) > 100:
        # Calculate summary statistics
        try:
            if result and isinstance(result[0], dict):
                # Try to find numeric columns for summary
                numeric_cols = []
                for key in result[0].keys():
                    if any(n in key.lower() for n in ["count", "total", "number", "sum", "avg", "age", "year"]):
                        numeric_cols.append(key)
                
                if numeric_cols:
                    summary = f"Showing {len(result)} results. "
                    for col in numeric_cols[:3]:
                        values = [r.get(col, 0) for r in result if isinstance(r.get(col), (int, float))]
                        if values:
                            summary += f"{col}: total={sum(values)}, avg={sum(values)/len(values):.1f}. "
                    state["result"] = result[:100]
                    state["result"].append({"_summary": summary})
        except Exception as e:
            logger.warning(f"Could not generate summary: {e}")
    
    return state


# =====================================================
# CACHE IMPLEMENTATION
# =====================================================
class QueryCache:
    """Simple in-memory query cache."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.ttl = ttl_seconds
        self.timestamps: Dict[str, float] = {}
    
    def get_cache_key(self, query: str) -> str:
        """Generate cache key."""
        content = query.lower().strip()
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Dict]:
        """Get cached result if valid."""
        if cache_key in self.cache:
            timestamp = self.timestamps.get(cache_key, 0)
            if (datetime.now().timestamp() - timestamp) < self.ttl:
                return self.cache[cache_key]
            else:
                del self.cache[cache_key]
                del self.timestamps[cache_key]
        return None
    
    def set(self, cache_key: str, result: Dict):
        """Cache query result."""
        self.cache[cache_key] = result
        self.timestamps[cache_key] = datetime.now().timestamp()
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.timestamps.clear()


def get_cache_key(query: str) -> str:
    """Generate cache key for a query."""
    content = query.lower().strip()
    return hashlib.md5(content.encode()).hexdigest()


query_cache = QueryCache(ttl_seconds=300)


# =====================================================
# LANGGRAPH WORKFLOW
# =====================================================
def create_sql_agent_graph():
    """Create the LangGraph workflow for SQL generation."""
    try:
        from langgraph.graph import StateGraph, END
        
        graph = StateGraph(SQLAgentState)
        
        # Add nodes
        graph.add_node("generate", generate_sql_node)
        graph.add_node("validate", validate_sql_node)
        graph.add_node("execute", execute_sql_node)
        graph.add_node("format", format_result_node)
        graph.add_node("handle_error", handle_error_node)
        
        # Set entry point
        graph.set_entry_point("generate")
        
        # Add edges
        graph.add_edge("generate", "validate")
        
        # Conditional edge from validate
        def validate_decision(state: SQLAgentState) -> str:
            if state.get("validation_error"):
                if state.get("retry_count", 0) < 3:
                    return "handle_error"
                else:
                    return END
            return "execute"
        
        graph.add_conditional_edges("validate", validate_decision)
        
        # Error handling loop
        graph.add_edge("handle_error", "generate")
        
        # Execution to format
        graph.add_edge("execute", "format")
        graph.add_edge("format", END)
        
        return graph.compile()
        
    except ImportError:
        logger.warning("LangGraph not installed, using fallback implementation")
        return None


# =====================================================
# FALLBACK (Non-LangGraph) IMPLEMENTATION
# =====================================================
async def run_sql_agent_fallback(question: str) -> Dict[str, Any]:
    """Fallback implementation without LangGraph."""
    state: SQLAgentState = {
        "question": question,
        "generated_sql": None,
        "params": [],
        "explanation": None,
        "result": None,
        "error": None,
        "validation_error": None,
        "retry_count": 0
    }
    
    # Generate SQL (with retries)
    max_retries = 3
    for attempt in range(max_retries):
        state = await generate_sql_node(state)
        
        if state.get("error"):
            if attempt < max_retries - 1:
                state = handle_error_node(state)
                continue
            break
        
        # Validate
        state = validate_sql_node(state)
        
        if state.get("validation_error"):
            if attempt < max_retries - 1:
                state["retry_count"] = attempt + 1
                state = handle_error_node(state)
                continue
            break
        
        # Execute
        state = await execute_sql_node(state)
        
        if not state.get("error"):
            break
    
    # Format result
    state = format_result_node(state)
    
    return {
        "sql": state.get("generated_sql"),
        "result": state.get("result", []),
        "explanation": state.get("explanation"),
        "error": state.get("error"),
        "validation_error": state.get("validation_error")
    }


# =====================================================
# MAIN AGENT CLASS
# =====================================================
class SQLAgent:
    """
    LLM-powered SQL Agent with CPU optimization.
    
    Uses smaller model with few-shot prompting for accurate SQL generation.
    Falls back to legacy interpreter on timeout.
    """
    
    def __init__(self):
        self.use_llm = False  # Default to legacy (LLM too slow on CPU)
        self.timeout = 15  # 15 second timeout for CPU
    
    async def run(self, question: str) -> Dict[str, Any]:
        """
        Run the SQL agent on a question.
        
        Args:
            question: Natural language question about the data
            
        Returns:
            Dict with sql, result, explanation, and error keys
        """
        if self.use_llm:
            try:
                # Try LLM-based generation with timeout
                result = await asyncio.wait_for(
                    self._run_llm(question),
                    timeout=self.timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"LLM generation timed out, falling back to legacy")
                return await self._run_legacy(question)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return await self._run_legacy(question)
        else:
            return await self._run_legacy(question)
    
    async def _run_llm(self, question: str) -> Dict[str, Any]:
        """Run LLM-based SQL generation."""
        from app.llm.ollama_client import OllamaClient
        from app.core.config import get_settings
        from app.agent.sql_tool import run_sql_query
        
        settings = get_settings()
        ollama = OllamaClient(settings.OLLAMA_URL)
        
        prompt = SQL_GENERATION_SYSTEM_PROMPT + "\n\n" + SQL_GENERATION_USER_PROMPT.format(question=question)
        
        response = await ollama.generate(
            model=settings.OLLAMA_MODEL,
            prompt=prompt,
            max_tokens=512,  # Smaller for faster generation
        )
        
        parsed = parse_sql_response(response)
        
        if not parsed or not parsed.get("sql"):
            raise ValueError("Failed to parse SQL from LLM response")
        
        sql = parsed["sql"].strip()
        params = parsed.get("params", [])
        
        # Execute query
        if params:
            query_result = await run_sql_query(sql, tuple(params))
        else:
            query_result = await run_sql_query(sql)
        
        if not isinstance(query_result, list):
            query_result = []
        
        return {
            "sql": sql,
            "result": query_result,
            "explanation": parsed.get("explanation", "Generated via LLM"),
            "error": None
        }
    
    async def _run_legacy(self, question: str) -> Dict[str, Any]:
        """Fallback to legacy interpreter with list query detection."""
        from app.agent.sql_interpreter import SQLInterpreter
        from app.core.config import get_settings
        
        settings = get_settings()
        interpreter = SQLInterpreter(model=settings.OLLAMA_MODEL)
        
        # Detect list queries
        q_lower = question.lower()
        is_list_query = any(kw in q_lower for kw in [
            "list ", "show me", "display", "get all", "find patients", 
            "show patients", "list all", "who were"
        ])
        
        try:
            if is_list_query:
                # Use list_patients_for_diagnosis for list queries
                sql = interpreter.list_patients_for_diagnosis(question)
                if sql:
                    # Extract parameters - handle diagnosis extraction
                    import re
                    # Try 'diagnosed with X' pattern first
                    diagnosis_match = re.search(
                        r'diagnosed\s+with\s+([a-zA-Z0-9\s\-]+?)(?:\s+in|\s+for|\s+on|\s+last|\s+this|\s+\d|\?|$)', 
                        q_lower
                    )
                    if not diagnosis_match:
                        # Try 'with X' pattern - skip 'patients' to get diagnosis
                        diagnosis_match = re.search(
                            r'with\s+([a-zA-Z0-9\s\-]+?)(?:\s+in|\s+for|\s+on|\s+last|\s+this|\s+\d|\?|$)', 
                            q_lower
                        )
                        if diagnosis_match:
                            # Verify it's not a non-diagnosis word
                            captured = diagnosis_match.group(1).strip().lower()
                            skip_words = {"patients", "people", "individuals", "children", "the", "all", "any", "new"}
                            if captured in skip_words or len(captured) < 2:
                                diagnosis_match = None
                    if not diagnosis_match:
                        # Fall back to 'had X' or 'suffering from X'
                        diagnosis_match = re.search(
                            r'(?:had|suffering)\s+(?:from\s+)?([a-zA-Z0-9\s\-]+?)(?:\s+in|\s+on|\s+for|\s+last|\s+this|\s+\d|\?|$)', 
                            q_lower
                        )
                    params = ()
                    if diagnosis_match:
                        diagnosis = diagnosis_match.group(1).strip()
                        params = (f"%{diagnosis}%",)
                    
                    from app.agent.sql_tool import run_sql_query
                    result = await run_sql_query(sql, params) if params else await run_sql_query(sql)
                    
                    return {
                        "sql": sql,
                        "result": result if isinstance(result, list) else [],
                        "explanation": "Listed patient records",
                        "error": None
                    }
            
            # Use standard interpret for other queries
            result = await interpreter.interpret(question)
            
            if "error" in result:
                return {
                    "sql": None,
                    "result": [],
                    "explanation": result.get("error"),
                    "error": result.get("error")
                }
            
            sql = result.get("sql", "")
            params = result.get("params", [])
            
            from app.agent.sql_tool import run_sql_query
            if params:
                query_result = await run_sql_query(sql, tuple(params))
            else:
                query_result = await run_sql_query(sql)
            
            if not isinstance(query_result, list):
                query_result = []
            
            return {
                "sql": sql,
                "result": query_result,
                "explanation": "Generated via legacy interpreter (LLM fallback)",
                "error": None
            }
        except Exception as e:
            logger.error(f"SQL agent error: {e}")
            return {
                "sql": None,
                "result": [],
                "explanation": str(e),
                "error": str(e)
            }
    
    def clear_cache(self):
        """Clear the query cache."""
        query_cache.clear()


# =====================================================
# CONVENIENCE FUNCTION
# =====================================================
async def ask_database(question: str) -> Dict[str, Any]:
    """
    Convenience function to ask the database a question.
    
    Args:
        question: Natural language question
        
    Returns:
        Dict with sql, result, explanation, and error keys
    """
    agent = SQLAgent()
    return await agent.run(question)
