from pathlib import Path
from functools import lru_cache
from typing import Literal
from pydantic import (
    Field,
    SecretStr
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CAREER_MATCH_", extra="ignore")

    # ------------------------------------------------------------------
    # LLM configuration
    # ------------------------------------------------------------------

    llm_provider: Literal["ollama", "openai", "gemini"] = "ollama"
    llm_model: str = "gemma3:4b"
    llm_timeout_seconds: float = Field(default=1200.0, gt=0)

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    ollama_base_url: str = "http://127.0.0.1:11434"

    # ------------------------------------------------------------------
    # Candidate / assessment limits
    # ------------------------------------------------------------------

    max_cv_text_characters: int = Field(default=30_000, ge=1)
    max_hiring_agent_report_bytes: int = Field(default=1_048_576, ge=1024)
    max_preferences_text_characters: int = Field(default=5_000, ge=1, le=20_000)

    # ------------------------------------------------------------------
    # Job provider
    # ------------------------------------------------------------------

    job_providers: list[str] = Field(default_factory=lambda: ["arbeitnow"])

    # Arbeitnow
    arbeitnow_base_url: str = ("https://www.arbeitnow.com")
    arbeitnow_timeout_seconds: float = Field(default=1200.0, gt=0)
    arbeitnow_max_pages: int = Field(default=3, ge=1, le=10)
    http_user_agent: str = ("career-match-agent/0.1.0")

    # Adzuna

    adzuna_base_url: str = "https://api.adzuna.com"
    adzuna_country: str = "de"
    adzuna_app_id: str | None = None
    adzuna_app_key: str | None = None
    adzuna_timeout_seconds: float = Field(default=20.0, gt=0)
    adzuna_results_per_page: int = Field(default=20, ge=1, le=50)
    adzuna_max_requests_per_search: int = Field(default=6, ge=1, le=20)

    # Jooble

    jooble_base_url: str = "https://de.jooble.org"
    jooble_api_key: str | None = None
    jooble_default_location: str = "Germany"
    jooble_timeout_seconds: float = Field(default=20.0, gt=0)
    jooble_results_per_page: int = Field(default=20, ge=1, le=50)
    jooble_max_requests_per_search: int = Field(default=4, ge=1, le=10)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    embedding_provider: str = ("sentence_transformers")
    embedding_model: str = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=16, ge=1, le=128)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    maximum_evaluation_jobs: int = Field(default=5, ge=1, le=20,)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    workflow_artifacts_directory: Path = Field(default=(PROJECT_ROOT/"data"/"workflow_artifacts"))
    workflow_artifacts_enabled: bool = False
    maximum_prepared_workflow_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)

@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
