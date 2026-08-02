from typing import Protocol

from career_match_agent.models.job import (
    JobProviderSearchResult,
    JobSearchQuery
)


class JobProviderError(RuntimeError):
    """Base error raised by job providers."""

class JobProviderUnavailableError(JobProviderError):
    """Raised when a provider cannot be reached."""

class JobProviderResponseError(JobProviderError):
    """Raised when a provider returns an unusable response."""

class JobProvider(Protocol):
    """Interface implemented by every job provider."""
    provider_name: str
    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        """Retrieve and normalize jobs."""

    async def aclose(self) -> None:
        """Release provider resources."""
