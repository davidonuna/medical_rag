"""
app/core/config.py

Application configuration management using Pydantic settings.
Loads configuration from environment variables with validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All required fields must be present in .env file.
    """
    
    # PostgreSQL / Data Warehouse
    POSTGRES_USER: str = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(..., env="POSTGRES_DB")
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: int = Field(..., env="DB_PORT")

    # Ollama / LLM
    OLLAMA_URL: str = Field(..., env="OLLAMA_URL")
    OLLAMA_MODEL: str = Field(..., env="OLLAMA_MODEL")

    # SerpAPI (optional)
    SERPAPI_API_KEY: str = Field("", env="SERPAPI_API_KEY")

    # File storage paths
    PDF_UPLOAD_PATH: Path = Field(default=Path("./data/pdfs"), env="PDF_UPLOAD_PATH")
    VECTORSTORE_PATH: Path = Field(default=Path("./data/vectorstore"), env="VECTORSTORE_PATH")
    REPORT_PATH: Path = Field(default=Path("./data/reports"), env="REPORT_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global singleton instance - initialized once
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings singleton.
    Creates the Settings instance on first call.
    
    Returns:
        Settings: Application configuration instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
