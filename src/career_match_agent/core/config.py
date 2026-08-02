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

@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
