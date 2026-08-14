from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator
)

from career_match_agent.models.candidate import EmploymentType


def clean_string_list(values: list[str]) -> list[str]:
    """Remove blank values and case-insensitive duplicates."""
    cleaned_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        cleaned_value = value.strip()
        if not cleaned_value:
            continue

        comparison_value = cleaned_value.casefold()
        if comparison_value in seen_values:
            continue

        seen_values.add(comparison_value)
        cleaned_values.append(cleaned_value)

    return cleaned_values


class JobModel(BaseModel):
    """Base configuration for job-related models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class JobSearchMatchScope(StrEnum):
    BROAD = "broad"
    TITLE_AND_TAGS = "title_and_tags"

class JobPosting(JobModel):
    """Provider-independent representation of a job posting."""
    source_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    description: str = Field(min_length=1)
    location: str | None = None
    remote: bool | None = None
    visa_sponsorship: bool | None = None
    employment_types: list[EmploymentType] = Field(default_factory=list)
    raw_employment_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    url: HttpUrl
    posted_at: datetime | None = None
    fingerprint: str = Field(min_length=64, max_length=64)

    @field_validator("raw_employment_types", "tags")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)


class JobSearchQuery(JobModel):
    """Provider-independent job search request."""
    keywords: list[str] = Field(min_length=1)
    locations: list[str] = Field(default_factory=list)
    remote_only: bool = False
    visa_sponsorship: bool | None = None
    employment_types: list[EmploymentType] = Field(default_factory=list)
    maximum_results: int = Field(default=20, ge=1, le=100)
    max_pages: int = Field(default=1, ge=1, le=5)
    match_scope: JobSearchMatchScope = (JobSearchMatchScope.BROAD)

    @field_validator("keywords")
    @classmethod
    def clean_and_require_keywords(cls, values: list[str]) -> list[str]:
        cleaned_values = clean_string_list(values)
        if not cleaned_values:
            raise ValueError("At least one non-empty keyword is required.")

        return cleaned_values

    @field_validator("locations")
    @classmethod
    def clean_locations(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)

class JobProviderSearchResult(JobModel):
    """Raw normalized result returned by a provider."""
    provider: str = Field(min_length=1)
    jobs: list[JobPosting]
    pages_fetched: int = Field(ge=0)
    received_count: int = Field(ge=0)
    skipped_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def clean_warnings(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)

class JobSearchStatistics(JobModel):
    """Counts describing the complete search pipeline."""
    pages_fetched: int = Field(ge=0)
    received_count: int = Field(ge=0)
    normalized_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)

class JobSearchResponse(JobModel):
    """Final result returned by the job-search API."""
    provider: str
    query: JobSearchQuery
    jobs: list[JobPosting]
    statistics: JobSearchStatistics
    warnings: list[str] = Field(default_factory=list)
