"""
app/core/database.py

Async PostgreSQL database helper providing connection pooling
and async query execution using asyncpg.
"""

import asyncpg
from app.core.config import get_settings

settings = get_settings()

# Database connection parameters from settings
POSTGRES_USER: str = settings.POSTGRES_USER
POSTGRES_PASSWORD: str = settings.POSTGRES_PASSWORD
POSTGRES_DB: str = settings.POSTGRES_DB
DB_HOST: str = settings.DB_HOST
DB_PORT: int = settings.DB_PORT

# Global connection pool - initialized once
DB_POOL: asyncpg.Pool | None = None


async def init_db_pool() -> None:
    """
    Initialize the async PostgreSQL connection pool.
    Creates a pool with 1-10 connections for concurrent queries.
    Must be called before any database operations.
    """
    global DB_POOL

    if DB_POOL is not None:
        return

    DB_POOL = await asyncpg.create_pool(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
        host=DB_HOST,
        port=DB_PORT,
        min_size=1,
        max_size=10,
        command_timeout=60,
    )


async def close_db_pool() -> None:
    """
    Close the database connection pool gracefully.
    Should be called on application shutdown.
    """
    global DB_POOL

    if DB_POOL is not None:
        await DB_POOL.close()
        DB_POOL = None


def get_db_pool() -> asyncpg.Pool:
    """
    Get the current database connection pool.
    
    Returns:
        asyncpg.Pool: The active connection pool
        
    Raises:
        RuntimeError: If pool hasn't been initialized
    """
    if DB_POOL is None:
        raise RuntimeError("Database pool is not initialized")
    return DB_POOL



