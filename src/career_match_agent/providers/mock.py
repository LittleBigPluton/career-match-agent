from career_match_agent.models.job import (
    JobPosting,
    JobProviderSearchResult,
    JobSearchQuery
)


class MockJobProvider:
    """In-memory provider used by tests and development."""
    provider_name = "mock"

    def __init__(self, jobs: list[JobPosting] | None = None) -> None:
        self.jobs = jobs or []

    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        del query
        return JobProviderSearchResult(provider=self.provider_name, jobs=self.jobs, pages_fetched=1, received_count=len(self.jobs), skipped_count=0, warnings=[])

    async def aclose(self) -> None:
        """No resources need to be released."""
