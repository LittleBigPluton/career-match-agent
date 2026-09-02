from enum import StrEnum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl
)
from career_match_agent.models.job import JobPosting


class WebJobModel(BaseModel):
    """Base model for web-job ingestion."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class WebJobSource(StrEnum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    STEPSTONE = "stepstone"
    GLASSDOOR = "glassdoor"

class WebJobExtractionStrategy(StrEnum):
    JSON_LD = "json_ld"
    DOM_FALLBACK = "dom_fallback"

class WebJobParseMetadata(WebJobModel):
    """Provenance information for one parsed web job."""
    source: WebJobSource
    source_url: HttpUrl
    extraction_strategy: WebJobExtractionStrategy
    parser_version: str = Field(min_length=1)
    content_sha256: str = Field(min_length=64, max_length=64)
    warnings: list[str] = Field(default_factory=list)

class WebJobParseResponse(WebJobModel):
    """Normalized job and parsing provenance."""
    job: JobPosting
    metadata: WebJobParseMetadata
