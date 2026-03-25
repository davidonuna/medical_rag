# app/agent/patient_detection.py

"""
Patient detection and identification module.
Provides 3-stage identification: regex, trigram search, LLM extraction.
"""

import re
import asyncio
from typing import Optional, Dict, List

from app.llm.prompt_templates import PATIENT_DETECTION_PROMPT
from app.llm.ollama_client import OllamaClient
from app.agent.sql_tool import run_sql_query, get_patient_info_sync

# Regex pattern for patient IDs (format: NCH- followed by digits)
PATIENT_REGEX = re.compile(r"NCH-\d{1,10}", re.IGNORECASE)


def normalize(text: str) -> str:
    """
    Normalize text for searching.
    Removes extra whitespace and converts to lowercase.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize_name(text: str) -> List[str]:
    """
    Tokenize a name into searchable terms.
    Filters out single characters.
    """
    return [t for t in normalize(text).split(" ") if len(t) > 1]


def validate_patient_exists(patient_id: str) -> bool:
    """
    Check if a patient ID exists in the database.
    
    Args:
        patient_id: Patient identifier to validate
        
    Returns:
        True if patient exists, False otherwise
    """
    return bool(get_patient_info_sync(patient_id))


async def trigram_patient_search_async(query: str, limit: int = 5) -> List[Dict]:
    """
    Async trigram-based patient search using PostgreSQL similarity.
    
    Search strategies (in order):
        1. Exact full_name match
        2. Single token exact match on first_name OR last_name
        3. First name + partial last name match
        4. Last name contains all tokens
        5. Fuzzy trigram similarity (threshold >= 0.4)
    
    Args:
        query: Search query (patient name or ID)
        limit: Maximum results to return
        
    Returns:
        List of patient matches with scores
    """
    tokens = tokenize_name(query)
    if not tokens:
        return []

    # First try exact match on full_name
    if len(tokens) >= 2:
        exact_sql = """
            SELECT patient_id, first_name, last_name, full_name, 1.0 AS score
            FROM dim_patient
            WHERE LOWER(full_name) = LOWER($1)
            LIMIT 1
        """
        full_name_query = " ".join(tokens)
        exact_result = await run_sql_query(exact_sql, (full_name_query,))
        if isinstance(exact_result, list) and exact_result:
            return exact_result

    # Try single token exact match on first_name OR last_name
    if len(tokens) == 1:
        single_token = tokens[0]
        single_sql = """
            SELECT patient_id, first_name, last_name, full_name, 1.0 AS score
            FROM dim_patient
            WHERE LOWER(first_name) = LOWER($1) OR LOWER(last_name) = LOWER($1)
            ORDER BY 
                CASE WHEN LOWER(last_name) = LOWER($1) THEN 0 ELSE 1 END,
                first_name
            LIMIT 5
        """
        single_result = await run_sql_query(single_sql, (single_token,))
        if isinstance(single_result, list) and single_result:
            return single_result

    # Try exact match on first_name + last_name combination
    if len(tokens) >= 2:
        # first_name=token1, last_name contains token2
        first_name_match_sql = """
            SELECT patient_id, first_name, last_name, full_name, 1.0 AS score
            FROM dim_patient
            WHERE LOWER(first_name) = LOWER($1) AND LOWER(last_name) LIKE LOWER('%' || $2 || '%')
            LIMIT 1
        """
        first_result = await run_sql_query(first_name_match_sql, (tokens[0], " ".join(tokens[1:])))
        if isinstance(first_result, list) and first_result:
            return first_result
        
        # Check if last_name contains all tokens
        all_tokens_in_lastname = " AND ".join([f"LOWER(last_name) LIKE LOWER('%' || ${i+1} || '%')" for i in range(len(tokens))])
        lastname_contains_all_sql = f"""
            SELECT patient_id, first_name, last_name, full_name, 1.0 AS score
            FROM dim_patient
            WHERE {all_tokens_in_lastname}
            ORDER BY LENGTH(last_name)
            LIMIT 1
        """
        lastname_result = await run_sql_query(lastname_contains_all_sql, tuple(tokens))
        if isinstance(lastname_result, list) and lastname_result:
            return lastname_result

    # Fall back to trigram similarity
    tokens_placeholder = ",".join([f"${i+1}" for i in range(len(tokens))])
    sql = f"""
        SELECT
            patient_id,
            first_name,
            last_name,
            full_name,
            GREATEST(
                MAX(similarity(first_name, t.token)),
                MAX(similarity(last_name, t.token)),
                MAX(similarity(full_name, t.token))
            ) AS score
        FROM dim_patient
        CROSS JOIN unnest(ARRAY[{tokens_placeholder}]) AS t(token)
        GROUP BY patient_id, first_name, last_name, full_name
        HAVING GREATEST(
            MAX(similarity(first_name, t.token)),
            MAX(similarity(last_name, t.token)),
            MAX(similarity(full_name, t.token))
        ) >= 0.4
        ORDER BY score DESC
        LIMIT {limit};
    """

    result = await run_sql_query(sql, tuple(tokens))
    return result if isinstance(result, list) else []


def trigram_patient_search(query: str, limit: int = 5) -> List[Dict]:
    """
    Synchronous trigram patient search.
    Wraps async version for non-async contexts.
    """
    tokens = tokenize_name(query)
    if not tokens:
        return []

    sql = f"""
        SELECT
            patient_id,
            first_name,
            last_name,
            full_name,
            GREATEST(
                MAX(similarity(first_name, t.token)),
                MAX(similarity(last_name, t.token)),
                MAX(similarity(full_name, t.token))
            ) AS score
        FROM dim_patient
        CROSS JOIN unnest(%s::text[]) AS t(token)
        GROUP BY patient_id, first_name, last_name, full_name
        HAVING GREATEST(
            MAX(similarity(first_name, t.token)),
            MAX(similarity(last_name, t.token)),
            MAX(similarity(full_name, t.token))
        ) >= 0.4
        ORDER BY score DESC
        LIMIT {limit};
    """

    async def _run():
        return await run_sql_query(sql, (tokens,))
    
    return asyncio.run(_run())


def extract_by_llm(text: str) -> Optional[str]:
    """
    Extract patient ID using LLM as fallback.
    Uses prompt template to extract NCH-XXXXX pattern.
    
    Args:
        text: Input text potentially containing patient ID
        
    Returns:
        Extracted patient ID or None
    """
    prompt = PATIENT_DETECTION_PROMPT.format(text=text)
    response = OllamaClient().generate(prompt).strip().upper()

    if PATIENT_REGEX.fullmatch(response):
        return response
    return None


def detect_patient(text: str) -> Dict:
    """
    Synchronous patient detection with 3-stage fallback.
    
    Stages:
        1. Regex: Direct NCH-XXXX pattern match
        2. Trigram: PostgreSQL similarity search
        3. LLM: Ollama extraction
    
    Args:
        text: Input text containing patient info
        
    Returns:
        Dict with patient_id, confidence, suggestions, source
    """
    # Stage 1: Regex match
    pid = PATIENT_REGEX.search(text)
    if pid:
        pid = pid.group(0).upper()
        patient_info = get_patient_info_sync(pid)
        if patient_info:
            return {
                "patient_id": pid,
                "confidence": 1.0,
                "suggestions": [],
                "source": "regex",
                "first_name": patient_info.get("first_name"),
                "last_name": patient_info.get("last_name"),
                "full_name": patient_info.get("full_name")
            }
        else:
            return {
                "patient_id": pid,
                "confidence": 1.0,
                "suggestions": [],
                "source": "regex"
            }

    # Stage 2: Trigram search
    query = normalize(text)
    matches = trigram_patient_search(query)

    suggestions = [
        {
            "patient_id": m["patient_id"],
            "first_name": m["first_name"],
            "last_name": m["last_name"],
            "confidence": round(float(m["score"]), 3)
        }
        for m in matches
    ]

    if suggestions:
        top = suggestions[0]
        if top["confidence"] >= 0.75:
            return {
                "patient_id": top["patient_id"],
                "confidence": top["confidence"],
                "suggestions": suggestions,
                "source": "trigram"
            }
        return {
            "patient_id": None,
            "confidence": None,
            "suggestions": suggestions,
            "source": "trigram"
        }

    # Stage 3: LLM extraction
    pid = extract_by_llm(text)
    if pid and validate_patient_exists(pid):
        return {
            "patient_id": pid,
            "confidence": 0.4,
            "suggestions": [],
            "source": "llm"
        }

    return {
        "patient_id": None,
        "confidence": None,
        "suggestions": [],
        "source": None
    }


async def detect_patient_async(text: str) -> Dict:
    """
    Async version of detect_patient that returns full patient info.
    
    Args:
        text: Input text containing patient info
        
    Returns:
        Dict with patient_id, confidence, suggestions, source, name fields
    """
    from app.agent.sql_tool import get_patient_info
    
    # Stage 1: Regex match
    pid = PATIENT_REGEX.search(text)
    if pid:
        pid = pid.group(0).upper()
        patient_info = await get_patient_info(pid)
        if patient_info:
            return {
                "patient_id": pid,
                "confidence": 1.0,
                "suggestions": [],
                "source": "regex",
                "first_name": patient_info.get("first_name"),
                "last_name": patient_info.get("last_name"),
                "full_name": patient_info.get("full_name")
            }
        else:
            return {
                "patient_id": pid,
                "confidence": 1.0,
                "suggestions": [],
                "source": "regex"
            }
    
    # Stage 2: Async trigram search
    query = normalize(text)
    search_tokens = set(tokenize_name(text))
    matches = await trigram_patient_search_async(query)

    suggestions = [
        {
            "patient_id": m["patient_id"],
            "first_name": m["first_name"],
            "last_name": m["last_name"],
            "confidence": round(float(m["score"]), 3)
        }
        for m in matches
    ]

    # Find exact match (all tokens match)
    exact_match = None
    for s in suggestions:
        full_name_tokens = set(tokenize_name(f"{s.get('first_name', '')} {s.get('last_name', '')}"))
        if search_tokens == full_name_tokens or search_tokens == set(tokenize_name(s.get('first_name', ''))) or search_tokens == set(tokenize_name(s.get('last_name', ''))):
            exact_match = s
            break
    
    if exact_match:
        return {
            "patient_id": exact_match["patient_id"],
            "confidence": exact_match["confidence"],
            "suggestions": suggestions,
            "source": "trigram",
            "first_name": exact_match.get("first_name"),
            "last_name": exact_match.get("last_name")
        }
    
    if suggestions:
        top = suggestions[0]
        if top["confidence"] >= 0.75:
            patient_info = await get_patient_info(top["patient_id"])
            if patient_info:
                top["first_name"] = patient_info.get("first_name")
                top["last_name"] = patient_info.get("last_name")
            return {
                "patient_id": top["patient_id"],
                "confidence": top["confidence"],
                "suggestions": suggestions,
                "source": "trigram",
                "first_name": top.get("first_name"),
                "last_name": top.get("last_name")
            }
        return {
            "patient_id": None,
            "confidence": None,
            "suggestions": suggestions,
            "source": "trigram"
        }
    
    return {
        "patient_id": None,
        "confidence": None,
        "suggestions": [],
        "source": None
    }
