from datetime import UTC, datetime

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    ValidationError
)

from career_match_agent.models.job import (
    JobPosting,
    JobProviderSearchResult,
    JobSearchQuery
)
from career_match_agent.providers.base import (
    JobProviderResponseError,
    JobProviderUnavailableError
)
from career_match_agent.services.job_normalizer import (
    create_job_fingerprint,
    html_to_plain_text,
    normalize_employment_types
)

HTTP_URL_ADAPTER: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)

class ArbeitnowResponseModel(BaseModel):
    """Base configuration for Arbeitnow responses."""
    model_config = ConfigDict(extra="ignore")

class ArbeitnowRawJob(ArbeitnowResponseModel):
    """One raw Arbeitnow job entry."""
    slug: str
    company_name: str
    title: str
    description: str
    remote: bool = False
    url: str
    tags: list[str] = Field(default_factory=list)
    job_types: list[str] = Field(default_factory=list)
    location: str | None = None
    created_at: int | float | None = None

class ArbeitnowLinks(ArbeitnowResponseModel):
    """Pagination links returned by Arbeitnow."""
    next: str | None = None

class ArbeitnowApiResponse(ArbeitnowResponseModel):
    """Validated Arbeitnow API response."""
    data: list[ArbeitnowRawJob]
    links: ArbeitnowLinks = Field(default_factory=ArbeitnowLinks)

def parse_posted_at(timestamp: int | float | None) -> datetime | None:
    """Convert a Unix timestamp to a UTC datetime."""
    if timestamp is None:
        return None

    try:
        return datetime.fromtimestamp(timestamp, tz=UTC)

    except (OSError, OverflowError, ValueError):
        return None


def normalize_arbeitnow_job(raw_job: ArbeitnowRawJob, *, visa_sponsorship_filter: bool | None) -> JobPosting:
    """Convert an Arbeitnow job into the common schema."""
    title = raw_job.title.strip()
    company = raw_job.company_name.strip()
    description = html_to_plain_text(raw_job.description)
    location = (raw_job.location.strip() if raw_job.location else None)
    url = HTTP_URL_ADAPTER.validate_python(raw_job.url)
    if not title:
        raise ValueError("The job title is empty.")

    if not company:
        raise ValueError("The company name is empty.")

    if not description:
        raise ValueError("The job description is empty.")

    return JobPosting(source_id=f"arbeitnow:{raw_job.slug}",
                      provider="arbeitnow",
                      external_id=raw_job.slug,
                      title=title,
                      company=company,
                      description=description,
                      location=location,
                      remote=raw_job.remote,
                      visa_sponsorship=visa_sponsorship_filter,
                      employment_types=normalize_employment_types(raw_job.job_types),
                      raw_employment_types=raw_job.job_types,
                      tags=raw_job.tags,
                      url=url,
                      posted_at=parse_posted_at(raw_job.created_at),
                      fingerprint=create_job_fingerprint(title=title, company=company, location=location))

class ArbeitnowJobProvider:
    """Retrieve public jobs from Arbeitnow."""
    provider_name = "arbeitnow"

    def __init__(self, *, base_url: str, timeout_seconds: float, maximum_pages: int, user_agent: str, client: httpx.AsyncClient | None = None) -> None:
        self.maximum_pages = maximum_pages
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, headers={"Accept": "application/json", "User-Agent": user_agent})

    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        """Retrieve and normalize Arbeitnow pages."""
        requested_pages = min(query.max_pages, self.maximum_pages)
        normalized_jobs: list[JobPosting] = []
        received_count = 0
        skipped_count = 0
        pages_fetched = 0
        has_next_page = True
        for page in range(1, requested_pages + 1):
            if not has_next_page:
                break

            parameters: dict[str, str | int] = {"page": page}
            if query.visa_sponsorship is not None:
                parameters["visa_sponsorship"] = str(query.visa_sponsorship).lower()

            try:
                response = await self._client.get("/api/job-board-api", params=parameters)
                response.raise_for_status()

            except httpx.RequestError as error:
                raise JobProviderUnavailableError("Arbeitnow could not be reached.") from error

            except httpx.HTTPStatusError as error:
                raise JobProviderResponseError(f"Arbeitnow returned HTTP status {error.response.status_code}.") from error

            try:
                response_payload = (ArbeitnowApiResponse.model_validate(response.json()))

            except (ValueError, ValidationError) as error:
                raise JobProviderResponseError("Arbeitnow returned an invalid response.") from error

            pages_fetched += 1
            received_count += len(response_payload.data)
            for raw_job in response_payload.data:
                try:
                    normalized_job = (normalize_arbeitnow_job(raw_job, visa_sponsorship_filter=(query.visa_sponsorship)))

                except (ValueError, ValidationError):
                    skipped_count += 1
                    continue

                normalized_jobs.append(normalized_job)

            has_next_page = (response_payload.links.next is not None)

        warnings: list[str] = []
        if skipped_count:
            warnings.append(f"{skipped_count} provider entries could not be normalized.")

        return JobProviderSearchResult(provider=self.provider_name,
                                       jobs=normalized_jobs,
                                       pages_fetched=pages_fetched,
                                       received_count=received_count,
                                       skipped_count=skipped_count,
                                       warnings=warnings)

    async def aclose(self) -> None:
        """Close an internally created HTTP client."""
        if self._owns_client:
            await self._client.aclose()
