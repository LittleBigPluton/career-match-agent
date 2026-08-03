from fastapi import APIRouter

from career_match_agent.models.matching import (
    JobFilteringRequest,
    JobFilteringResponse
)
from career_match_agent.services.job_filter import filter_jobs_for_candidate


router = APIRouter(prefix="/matching", tags=["matching"])


@router.post("/filter", response_model=JobFilteringResponse)
def filter_candidate_jobs(request: JobFilteringRequest) -> JobFilteringResponse:
    """Apply deterministic candidate suitability filters."""
    return filter_jobs_for_candidate(request)
