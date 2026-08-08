from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CAREER_MATCH_", extra="ignore")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:4b"
    ollama_timeout_seconds: float = Field(default=300.0, gt=0)
    max_cv_text_characters: int = Field(default=30_000, ge=1)
    max_hiring_agent_report_bytes: int = Field(default=1_048_576, ge=1024)

    job_provider: str = "arbeitnow"
    arbeitnow_base_url: str = "https://www.arbeitnow.com"
    arbeitnow_timeout_seconds: float = Field(default=20.0, gt=0)
    arbeitnow_max_pages: int = Field(default=3, ge=1, le=10)
    http_user_agent: str = "career-match-agent/0.1.0"

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=16, ge=1, le=128)

    job_evaluation_model: str = "gemma3:4b"
    job_evaluation_timeout_seconds: float = Field(default=300.0, gt=0)
    maximum_evaluation_jobs: int = Field(default=5, ge=1, le=20)

    search_planner_model: str = "gemma3:4b"
    search_planner_timeout_seconds: float = Field(default=300.0, gt=0)

@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
