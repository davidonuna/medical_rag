"""
app/agent/sql_tool.py

Async PostgreSQL query execution helpers.
Provides safe SQL execution with timeout and parameterization.
"""

from typing import Optional, List, Dict, Any, Union
import asyncpg

from app.core.config import get_settings
from app.core.errors import DatabaseError
from app.core.database import get_db_pool

settings = get_settings()

# Default statement timeout in milliseconds (30 seconds)
# Prevents runaway queries from blocking the pool
SQL_STATEMENT_TIMEOUT_MS = 30_000


# -------------------------------------------------
# Core SQL execution
# -------------------------------------------------
async def run_sql_query(
    sql_query: str,
    params: Optional[tuple] = None
) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Execute SQL safely with optional parameters and per-statement timeout.
    
    Args:
        sql_query: SQL string with optional $1, $2 placeholders.
        params: Optional tuple of parameters for parameterized queries.
        
    Returns:
        - list[dict] for SELECT queries (list of row dictionaries)
        - {"status": "success"} for non-SELECT queries
        
    Raises:
        DatabaseError: On timeout, connection, or query failure.
    """
    pool = get_db_pool()
    if pool is None:
        raise DatabaseError("Database pool is not initialized")

    try:
        async with pool.acquire() as conn:
            # Enforce per-statement timeout in milliseconds
            # Prevents long-running queries from blocking resources
            await conn.execute(f"SET statement_timeout = {SQL_STATEMENT_TIMEOUT_MS}")

            if params:
                # Parameterized query - prevents SQL injection
                stmt = await conn.prepare(sql_query)
                records = await stmt.fetch(*params)
            else:
                records = await conn.fetch(sql_query)

            # SELECT queries return results
            if records:
                return [dict(r) for r in records]

            # Non-SELECT queries return success status
            return {"status": "success"}

    except asyncpg.exceptions.QueryCanceledError as e:
        raise DatabaseError(f"SQL execution timeout: {e}")
    except asyncpg.PostgresError as e:
        raise DatabaseError(f"SQL execution error: {e}")
    except Exception as e:
        raise DatabaseError(f"Database failure: {e}")


# -------------------------------------------------
# Convenience helpers
# -------------------------------------------------
async def get_patient_info(patient_id: str) -> Dict[str, Any]:
    """
    Retrieve patient info by patient_id.
    
    Args:
        patient_id: The patient's unique identifier (e.g., "NCH-12345")
        
    Returns:
        Dictionary with patient fields, or empty dict if not found.
    """
    sql_query = """
        SELECT *
        FROM dim_patient
        WHERE patient_id = $1
    """
    result = await run_sql_query(sql_query, (patient_id,))
    if isinstance(result, list) and result:
        return result[0]
    return {}


def get_patient_info_sync(patient_id: str) -> Dict[str, Any]:
    """
    Synchronous version of get_patient_info.
    Useful for non-async contexts.
    
    Args:
        patient_id: The patient's unique identifier
        
    Returns:
        Dictionary with patient fields, or empty dict if not found.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        raise RuntimeError("Cannot use sync function in async context")
    
    sql_query = """
        SELECT *
        FROM dim_patient
        WHERE patient_id = $1
    """
    from app.core.database import get_db_pool
    
    async def _query():
        pool = get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.fetch(sql_query, patient_id)
            return dict(result[0]) if result else {}
    
    if loop:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _query())
            return future.result()
    else:
        return asyncio.run(_query())


async def get_test_results(patient_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all test results for a patient.
    
    Args:
        patient_id: The patient's unique identifier
        
    Returns:
        List of test result records, or empty list on error.
    """
    sql_query = """
        SELECT *
        FROM fact_test_results
        WHERE patient_id = $1
    """
    try:
        result = await run_sql_query(sql_query, (patient_id,))
        return result if isinstance(result, list) else []
    except DatabaseError:
        return []


async def get_medications(patient_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all medication records for a patient.
    
    Args:
        patient_id: The patient's unique identifier
        
    Returns:
        List of medication records, or empty list on error.
    """
    sql_query = """
        SELECT *
        FROM fact_medications
        WHERE patient_id = $1
    """
    try:
        result = await run_sql_query(sql_query, (patient_id,))
        return result if isinstance(result, list) else []
    except DatabaseError:
        return []
