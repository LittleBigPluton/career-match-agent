from career_match_agent.models.job import (
    JobPosting,
    JobSearchQuery,
    JobSearchResponse,
    JobSearchStatistics
)
from career_match_agent.providers.base import JobProvider
from career_match_agent.services.job_normalizer import (
    deduplicate_jobs,
    normalize_for_matching
)


def phrase_matches_text(phrase: str, text: str) -> bool:
    """Match a phrase directly or by all normalized tokens."""
    normalized_phrase = normalize_for_matching(phrase)
    normalized_text = normalize_for_matching(text)

    if not normalized_phrase:
        return False

    if normalized_phrase in normalized_text:
        return True

    phrase_tokens = normalized_phrase.split()
    return bool(phrase_tokens) and all(token in normalized_text for token in phrase_tokens)


def job_matches_keywords(job: JobPosting, keywords: list[str]) -> bool:
    """Return whether a job matches at least one keyword."""
    searchable_text = " ".join([job.title, job.company, job.location or "", " ".join(job.tags), job.description])
    return any(phrase_matches_text(keyword, searchable_text) for keyword in keywords)

def job_matches_locations(job: JobPosting, locations: list[str]) -> bool:
    """Return whether a job matches a requested location."""
    if not locations:
        return True

    if not job.location:
        return False

    return any(phrase_matches_text(location, job.location)for location in locations)

def job_matches_query(job: JobPosting, query: JobSearchQuery) -> bool:
    """Apply provider-independent local filters."""
    if not job_matches_keywords(job, query.keywords):
        return False

    if not job_matches_locations(job, query.locations):
        return False

    if query.remote_only and job.remote is not True:
        return False

    if (query.visa_sponsorship is not None and job.visa_sponsorship is not query.visa_sponsorship):
        return False

    if query.employment_types:
        requested_types = set(query.employment_types)
        job_types = set(job.employment_types)
        if not requested_types.intersection(job_types):
            return False

    return True

def filter_jobs(jobs: list[JobPosting], query: JobSearchQuery) -> list[JobPosting]:
    """Apply all local search filters."""
    return [job for job in jobs if job_matches_query(job, query)]

class JobSearchService:
    """Search, filter and deduplicate provider jobs."""

    def __init__(self, provider: JobProvider) -> None:
        self.provider = provider

    async def search(self, query: JobSearchQuery) -> JobSearchResponse:
        provider_result = await self.provider.search(query)
        matched_jobs = filter_jobs(provider_result.jobs, query)
        deduplication_result = deduplicate_jobs(matched_jobs)
        returned_jobs = deduplication_result.jobs[: query.maximum_results]
        return JobSearchResponse(provider=provider_result.provider,
                                 query=query,
                                 jobs=returned_jobs,
                                 statistics=JobSearchStatistics(pages_fetched=(provider_result.pages_fetched),
                                 received_count=(provider_result.received_count),
                                 normalized_count=len(provider_result.jobs),
                                 skipped_count=(provider_result.skipped_count),
                                 matched_count=len(matched_jobs),
                                 duplicate_count=(deduplication_result.duplicate_count),
                                 returned_count=len(returned_jobs)), warnings=provider_result.warnings)
