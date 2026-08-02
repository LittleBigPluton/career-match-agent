from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status
)

from career_match_agent.api.dependencies import get_job_provider
from career_match_agent.models.job import (
    JobSearchQuery,
    JobSearchResponse
)
from career_match_agent.providers.base import (
    JobProvider,
    JobProviderResponseError,
    JobProviderUnavailableError
)
from career_match_agent.services.job_search import JobSearchService


router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.post("/search", response_model=JobSearchResponse)
async def search_jobs(query: JobSearchQuery, provider: Annotated[JobProvider, Depends(get_job_provider)]) -> JobSearchResponse:
    """Search and normalize jobs through the configured provider."""
    try:
        service = JobSearchService(provider)
        return await service.search(query)

    except JobProviderUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    except JobProviderResponseError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    finally:
        await provider.aclose()
