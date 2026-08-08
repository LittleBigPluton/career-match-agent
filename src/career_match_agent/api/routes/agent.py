from typing import Annotated, cast

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status
)
from langgraph.errors import GraphRecursionError

from career_match_agent.api.dependencies import (
    get_embedding_provider,
    get_job_provider,
    get_job_report_generator,
    get_search_planner
)
from career_match_agent.core.config import (
    Settings,
    get_settings
)
from career_match_agent.graphs.job_search_graph import (
    CareerMatchGraphDependencies,
    CareerMatchGraphState,
    build_agent_search_response,
    build_career_match_graph
)
from career_match_agent.models.agent import (
    AgentSearchRequest,
    AgentSearchResponse
)
from career_match_agent.providers.base import (
    JobProvider,
    JobProviderResponseError,
    JobProviderUnavailableError
)
from career_match_agent.services.embedding import (
    EmbeddingModelUnavailableError,
    EmbeddingProvider,
    InvalidEmbeddingResponseError
)
from career_match_agent.services.job_evaluator import (
    JobEvaluationModelUnavailableError,
    JobEvaluationResponseError,
    JobReportGenerator
)
from career_match_agent.services.search_planner import (
    SearchPlanner,
    SearchPlannerResponseError,
    SearchPlannerUnavailableError
)


router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/search", response_model=AgentSearchResponse)
async def run_agentic_job_search(request: AgentSearchRequest,
                                 planner: Annotated[SearchPlanner, Depends(get_search_planner)],
                                 provider: Annotated[JobProvider, Depends(get_job_provider)],
                                 embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
                                 report_generator: Annotated[JobReportGenerator, Depends(get_job_report_generator)],
                                 settings: Annotated[Settings, Depends(get_settings)]) -> AgentSearchResponse:
    """Run the complete bounded CareerMatch agent workflow."""
    dependencies = CareerMatchGraphDependencies(search_planner=planner, job_provider=provider, embedding_provider=embedding_provider,
                                                report_generator=report_generator, maximum_evaluation_jobs=(settings.maximum_evaluation_jobs))

    graph = build_career_match_graph(dependencies)

    try:
        raw_state = await graph.ainvoke({"request": request, "trace": []}, config={"recursion_limit": 25})
        final_state = cast(CareerMatchGraphState, raw_state)
        return build_agent_search_response(final_state)

    except (SearchPlannerUnavailableError, JobProviderUnavailableError, EmbeddingModelUnavailableError, JobEvaluationModelUnavailableError) as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    except (SearchPlannerResponseError, JobProviderResponseError, InvalidEmbeddingResponseError, JobEvaluationResponseError) as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    except GraphRecursionError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=("The agent workflow exceeded its maximum execution depth.")) from error

    finally:
        await provider.aclose()
