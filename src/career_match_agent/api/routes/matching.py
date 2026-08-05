from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status
)

from career_match_agent.models.matching import (
    JobFilteringRequest,
    JobFilteringResponse
)
from career_match_agent.services.job_filter import filter_jobs_for_candidate

from career_match_agent.api.dependencies import get_embedding_provider
from career_match_agent.models.ranking import (
    HybridRankingRequest,
    HybridRankingResponse
)
from career_match_agent.services.embedding import (
    EmbeddingModelUnavailableError,
    EmbeddingProvider,
    InvalidEmbeddingResponseError
)
from career_match_agent.services.semantic_ranker import HybridJobRankingService

router = APIRouter(prefix="/matching", tags=["matching"])


@router.post("/filter", response_model=JobFilteringResponse)
def filter_candidate_jobs(request: JobFilteringRequest) -> JobFilteringResponse:
    """Apply deterministic candidate suitability filters."""
    return filter_jobs_for_candidate(request)
    
@router.post("/rank", response_model=HybridRankingResponse)
async def rank_candidate_jobs(request: HybridRankingRequest, embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)]) -> HybridRankingResponse:
    """Rank accepted jobs using hybrid semantic scoring."""
    try:
        ranking_service = HybridJobRankingService(embedding_provider)
        return await ranking_service.rank(request)

    except EmbeddingModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    except InvalidEmbeddingResponseError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
