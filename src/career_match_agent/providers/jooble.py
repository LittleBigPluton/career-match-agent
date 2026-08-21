import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    HttpUrl,
    TypeAdapter
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
    normalize_employment_types,
    parse_iso_datetime
)

HTTP_URL_ADAPTER: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)


class JoobleResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class JoobleRawJob(JoobleResponseModel):
    id: str | int
    title: str
    location: str | None = None
    snippet: str = ""
    salary: str | None = None
    source: str | None = None
    type: str | None = None
    link: str
    company: str | None = None
    updated: str | None = None


class JoobleApiResponse(JoobleResponseModel):
    totalCount: int = 0
    jobs: list[JoobleRawJob] = Field(default_factory=list)

def normalize_jooble_job(raw_job: JoobleRawJob) -> JobPosting:
    title = html_to_plain_text(raw_job.title).strip()
    company = (raw_job.company.strip() if raw_job.company else "Unknown company")
    description = html_to_plain_text(raw_job.snippet).strip()
    url = HTTP_URL_ADAPTER.validate_python(raw_job.link)

    if not description:
        description = title

    location = (raw_job.location.strip() if raw_job.location else None)
    raw_employment_types = ([raw_job.type] if raw_job.type else [])
    external_id = str(raw_job.id)

    return JobPosting(source_id=f"jooble:{external_id}",
                      provider="jooble",
                      external_id=external_id,
                      title=title,
                      company=company,
                      description=description,
                      location=location,
                      remote=None,
                      visa_sponsorship=None,
                      employment_types=(normalize_employment_types(raw_employment_types)),
                      raw_employment_types=(raw_employment_types),
                      tags=[],
                      url=url,
                      posted_at=parse_iso_datetime(raw_job.updated),
                      fingerprint=create_job_fingerprint(title=title, company=company, location=location))


class JoobleJobProvider:
    provider_name = "jooble"

    def __init__(self, *, base_url: str, api_key: str, default_location: str, timeout_seconds: float, results_per_page: int,
                 maximum_requests: int, user_agent: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.default_location = default_location
        self.results_per_page = results_per_page
        self.maximum_requests = maximum_requests
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, headers={"Accept": "application/json", "User-Agent": user_agent})

    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        jobs: list[JobPosting] = []
        received_count = 0
        skipped_count = 0
        requests_made = 0
        locations = (query.locations if query.locations else [self.default_location])
        exhausted_queries: set[tuple[str, str]] = set()

        for page in range(1, query.max_pages + 1):
            for keyword in query.keywords:
                for location in locations:
                    query_key = (keyword, location)
                    if query_key in exhausted_queries:
                        continue

                    if (requests_made >= self.maximum_requests):
                        break

                    request_payload = {"keywords": keyword, "location": location, "page": page, "ResultOnPage": (self.results_per_page), "companysearch": False}
                    try:
                        response = (await self._client.post(f"/api/{self.api_key}", json=request_payload))
                        response.raise_for_status()

                    except httpx.RequestError as error:
                        raise (JobProviderUnavailableError("Jooble could not be reached.")) from error

                    except httpx.HTTPStatusError as error:
                        raise JobProviderResponseError(f"Jooble returned HTTP {error.response.status_code}.") from error

                    requests_made += 1
                    try:
                        payload = (JoobleApiResponse.model_validate(response.json()))

                    except (ValueError, ValidationError) as error:
                        raise JobProviderResponseError("Jooble returned an invalid response.") from error

                    received_count += len(payload.jobs)
                    if not payload.jobs:
                        exhausted_queries.add(query_key)

                    for raw_job in payload.jobs:
                        try:
                            jobs.append(normalize_jooble_job(raw_job))

                        except (ValueError, ValidationError):
                            skipped_count += 1

                if (requests_made >= self.maximum_requests):
                    break

            if (requests_made >= self.maximum_requests):
                break

        warnings: list[str] = []
        if (requests_made >= self.maximum_requests):
            warnings.append("Jooble request budget was reached.")

        if skipped_count:
            warnings.append(f"{skipped_count} Jooble jobs could not be normalized.")

        return JobProviderSearchResult(provider=self.provider_name,
                                       jobs=jobs,
                                       pages_fetched=requests_made,
                                       received_count=received_count,
                                       skipped_count=skipped_count,
                                       warnings=warnings)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
