import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    HttpUrl,
    TypeAdapter
)

from career_match_agent.models.candidate import EmploymentType
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

class AdzunaResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AdzunaCompany(AdzunaResponseModel):
    display_name: str | None = None


class AdzunaLocation(AdzunaResponseModel):
    display_name: str | None = None
    area: list[str] = Field(default_factory=list)


class AdzunaCategory(AdzunaResponseModel):
    label: str | None = None
    tag: str | None = None


class AdzunaRawJob(AdzunaResponseModel):
    id: str | int
    title: str
    description: str
    redirect_url: str
    company: AdzunaCompany | None = None
    location: AdzunaLocation | None = None
    category: AdzunaCategory | None = None
    created: str | None = None
    contract_time: str | None = None
    contract_type: str | None = None


class AdzunaApiResponse(AdzunaResponseModel):
    results: list[AdzunaRawJob] = Field(default_factory=list)
    count: int | None = None


def normalize_adzuna_job(raw_job: AdzunaRawJob) -> JobPosting:
    title = html_to_plain_text(raw_job.title).strip()
    description = html_to_plain_text(raw_job.description).strip()
    company = (raw_job.company.display_name.strip() if (raw_job.company and raw_job.company.display_name) else "Unknown company")
    location = (raw_job.location.display_name.strip() if (raw_job.location and raw_job.location.display_name) else None)
    raw_employment_types = [value for value in [raw_job.contract_time, raw_job.contract_type] if value]
    url = HTTP_URL_ADAPTER.validate_python(raw_job.redirect_url)
    tags = []
    if raw_job.category:
        if raw_job.category.label:
            tags.append(raw_job.category.label)

        if raw_job.category.tag:
            tags.append(raw_job.category.tag)

    external_id = str(raw_job.id)
    return JobPosting(source_id=f"adzuna:{external_id}",
                      provider="adzuna",
                      external_id=external_id,
                      title=title,
                      company=company,
                      description=description,
                      location=location,
                      remote=None,
                      visa_sponsorship=None,
                      employment_types=(normalize_employment_types(raw_employment_types)),
                      raw_employment_types=(raw_employment_types),
                      tags=tags,
                      url=url,
                      posted_at=parse_iso_datetime(raw_job.created),
                      fingerprint=create_job_fingerprint(title=title, company=company, location=location))


class AdzunaJobProvider:
    provider_name = "adzuna"

    def __init__(self, *, base_url: str, country: str, app_id: str, app_key: str, timeout_seconds: float,
                 results_per_page: int, maximum_requests: int, user_agent: str, client: httpx.AsyncClient | None = None) -> None:
        self.country = country
        self.app_id = app_id
        self.app_key = app_key
        self.results_per_page = results_per_page
        self.maximum_requests = maximum_requests
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, headers={"Accept": "application/json", "User-Agent": user_agent})

    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        jobs: list[JobPosting] = []

        received_count = 0
        skipped_count = 0
        requests_made = 0

        locations: list[str | None] = (list(query.locations) if query.locations else [None])

        exhausted_queries: set[tuple[str, str | None]] = set()

        for page in range(1, query.max_pages + 1):
            for keyword in query.keywords:
                for location in locations:
                    query_key = (keyword, location,)

                    if query_key in exhausted_queries:
                        continue

                    if (requests_made >= self.maximum_requests):
                        break

                    params: dict[str, str | int] = {"app_id": self.app_id,
                                                    "app_key": self.app_key,
                                                    "what": keyword,
                                                    "results_per_page": (self.results_per_page), "content-type": "application/json"}

                    if location:
                        params["where"] = location

                    if query.employment_types == [EmploymentType.FULL_TIME]:
                        params["full_time"] = 1

                    elif query.employment_types == [EmploymentType.PART_TIME]:
                        params["part_time"] = 1

                    try:
                        response = (await self._client.get((f"/v1/api/jobs/{self.country}/search/{page}"), params=params))
                        response.raise_for_status()

                    except httpx.RequestError as error:
                        raise (JobProviderUnavailableError("Adzuna could not be reached.")) from error

                    except httpx.HTTPStatusError as error:
                        raise JobProviderResponseError(f"Adzuna returned HTTP {error.response.status_code}.") from error

                    requests_made += 1
                    try:
                        payload = (AdzunaApiResponse.model_validate(response.json()))

                    except (ValueError, ValidationError) as error:
                        raise JobProviderResponseError("Adzuna returned an invalid response.") from error

                    received_count += len(payload.results)

                    if not payload.results:
                        exhausted_queries.add(query_key)
                        continue

                    for raw_job in payload.results:
                        try:
                            jobs.append(normalize_adzuna_job(raw_job))
                        except (ValueError, ValidationError):
                            skipped_count += 1

                if (requests_made >= self.maximum_requests):
                    break

            if (requests_made >= self.maximum_requests):
                break

        warnings: list[str] = []
        if (requests_made >= self.maximum_requests):
            warnings.append("Adzuna request budget was reached before all keyword/location combinations were searched.")

        if skipped_count:
            warnings.append(f"{skipped_count} Adzuna jobs could not be normalized.")

        return JobProviderSearchResult(provider=self.provider_name,
                                       jobs=jobs,
                                       pages_fetched=requests_made,
                                       received_count=received_count,
                                       skipped_count=skipped_count,
                                       warnings=warnings)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
