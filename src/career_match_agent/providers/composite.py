import asyncio

from career_match_agent.models.job import (
    JobProviderSearchResult,
    JobSearchQuery
)
from career_match_agent.providers.base import (
    JobProvider,
    JobProviderError,
    JobProviderUnavailableError
)


class CompositeJobProvider:
    """Fan out one search across multiple job providers."""

    provider_name = "multi"

    def __init__(self, providers: list[JobProvider]) -> None:
        if not providers:
            raise ValueError("At least one job provider is required.")

        self.providers = providers

    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        results = await asyncio.gather(*[provider.search(query) for provider in self.providers], return_exceptions=True)
        jobs = []
        pages_fetched = 0
        received_count = 0
        skipped_count = 0
        warnings: list[str] = []
        successful_provider_count = 0
        for provider, result in zip(self.providers, results, strict=True):
            if isinstance(result, BaseException):
                warnings.append(f"{provider.provider_name}: {result}")
                continue

            successful_provider_count += 1
            jobs.extend(result.jobs)
            pages_fetched += (result.pages_fetched)
            received_count += (result.received_count)
            skipped_count += (result.skipped_count)
            warnings.extend(f"{provider.provider_name}: {warning}" for warning in result.warnings)

        if successful_provider_count == 0:
            raise JobProviderUnavailableError("All configured job providers failed.")

        return JobProviderSearchResult(provider=self.provider_name,
                                       jobs=jobs,
                                       pages_fetched=pages_fetched,
                                       received_count=received_count,
                                       skipped_count=skipped_count,
                                       warnings=warnings)

    async def aclose(self) -> None:
        await asyncio.gather(*[provider.aclose() for provider in self.providers], return_exceptions=True)
