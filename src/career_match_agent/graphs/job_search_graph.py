import operator
from dataclasses import dataclass
from typing import Annotated, Literal, Required

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from career_match_agent.models.agent import (
    AgentSearchPlan,
    AgentSearchRequest,
    AgentSearchResponse,
    AgentTraceEntry
)
from career_match_agent.models.evaluation import (
    JobEvaluationRequest,
    JobEvaluationResponse
)
from career_match_agent.models.job import JobSearchResponse
from career_match_agent.models.matching import (
    JobFilteringRequest,
    JobFilteringResponse
)
from career_match_agent.models.ranking import (
    HybridRankingRequest,
    HybridRankingResponse
)
from career_match_agent.providers.base import JobProvider
from career_match_agent.services.embedding import EmbeddingProvider
from career_match_agent.services.job_evaluator import (
    JobEvaluationService,
    JobReportGenerator
)
from career_match_agent.services.job_filter import filter_jobs_for_candidate
from career_match_agent.services.job_search import JobSearchService
from career_match_agent.services.search_planner import (
    SearchPlanner,
    build_job_search_query,
    normalize_broadened_plan
)
from career_match_agent.services.semantic_ranker import HybridJobRankingService

#############
### DEBUG ###
#############

from collections import Counter


class CareerMatchGraphState(TypedDict, total=False):
    """Mutable state carried through the LangGraph workflow."""
    request: Required[AgentSearchRequest]
    search_plan: AgentSearchPlan
    search_attempt: int
    job_search: JobSearchResponse
    filtering: JobFilteringResponse
    ranking: HybridRankingResponse
    evaluation: JobEvaluationResponse
    trace: Annotated[list[AgentTraceEntry], operator.add]

class CareerMatchGraphUpdate(TypedDict, total=False):
    """Partial state update returned by graph nodes."""
    search_plan: AgentSearchPlan
    search_attempt: int
    job_search: JobSearchResponse
    filtering: JobFilteringResponse
    ranking: HybridRankingResponse
    evaluation: JobEvaluationResponse
    trace: list[AgentTraceEntry]

@dataclass(frozen=True)
class CareerMatchGraphDependencies:
    """Runtime services used by graph nodes."""
    search_planner: SearchPlanner
    job_provider: JobProvider
    embedding_provider: EmbeddingProvider
    report_generator: JobReportGenerator
    maximum_evaluation_jobs: int

async def create_initial_search_plan(state: CareerMatchGraphState, *, dependencies: CareerMatchGraphDependencies) -> CareerMatchGraphUpdate:
    """Create the first LLM-generated retrieval strategy."""
    request = state["request"]
    configuration = request.configuration
    attempt = 1
    plan = await dependencies.search_planner.plan(profile=request.profile, preferences=request.preferences, attempt=attempt,
                                                  maximum_attempts=(configuration.maximum_search_attempts), previous_plan=None, accepted_count=None)

    plan = normalize_broadened_plan(plan, previous_plan=None)
    return {"search_plan": plan,"search_attempt": attempt, "trace": [AgentTraceEntry(step="plan_search", attempt=attempt, message=(f"Created initial search plan: {plan.rationale}"),)]}

async def search_jobs_node(state: CareerMatchGraphState, *, dependencies: CareerMatchGraphDependencies) -> CareerMatchGraphUpdate:
    """Retrieve jobs using the current agent plan."""
    request = state["request"]
    plan = state["search_plan"]
    query = build_job_search_query(plan=plan, preferences=request.preferences, policy=request.configuration.filter_policy, visa_sponsorship=(request.configuration.visa_sponsorship))
    service = JobSearchService(dependencies.job_provider)
    result = await service.search(query)
    return {"job_search": result, "trace": [AgentTraceEntry(step="search_jobs", attempt=state["search_attempt"],
            message=(f"Query keywords={query.keywords}; "
                     f"locations={query.locations}; "
                     f"employment_types="
                     f"{[value.value for value in query.employment_types]}; "
                     f"scope={query.match_scope.value}. "
                     f"Retrieved {result.statistics.received_count} "
                     f"provider jobs, matched "
                     f"{result.statistics.matched_count}, and returned "
                     f"{result.statistics.returned_count}."))]}

async def filter_jobs_node(state: CareerMatchGraphState) -> CareerMatchGraphUpdate:
    """Apply deterministic hard suitability filters."""
    request = state["request"]
    search_response = state["job_search"]
    filtering_response = filter_jobs_for_candidate(JobFilteringRequest(profile=request.profile, preferences=request.preferences, jobs=search_response.jobs, policy=request.configuration.filter_policy))
    rejection_counts = Counter(reason.code.value for decision in filtering_response.rejected_jobs for reason in decision.rejection_reasons)
    rejection_summary = ", ".join(f"{reason}={count}" for reason, count in rejection_counts.most_common())

    if not rejection_summary:
        rejection_summary = "none"

    return {"filtering":filtering_response, "trace":[AgentTraceEntry(step="filter_jobs", attempt=state["search_attempt"],
            message=(f"Accepted "
                     f"{filtering_response.statistics.accepted_count} "
                     f"jobs and rejected "
                     f"{filtering_response.statistics.rejected_count}. "
                     f"Rejection reasons: {rejection_summary}."))]}


def route_after_filtering(state: CareerMatchGraphState) -> Literal["broaden_search","rank_jobs"]:
    """Decide whether the agent should retry or continue."""
    request = state["request"]
    configuration = request.configuration
    accepted_count = (state["filtering"].statistics.accepted_count)
    attempt = state["search_attempt"]
    if (accepted_count < configuration.minimum_accepted_jobs and attempt < configuration.maximum_search_attempts):
        return "broaden_search"

    return "rank_jobs"


async def broaden_search_node(state: CareerMatchGraphState, *, dependencies: CareerMatchGraphDependencies) -> CareerMatchGraphUpdate:
    """Ask the planner to broaden retrieval after weak results."""
    request = state["request"]
    previous_plan = state["search_plan"]
    previous_filtering = state["filtering"]
    next_attempt = state["search_attempt"] + 1
    new_plan = await dependencies.search_planner.plan(profile=request.profile, preferences=request.preferences, attempt=next_attempt,
                                                      maximum_attempts=(request.configuration.maximum_search_attempts),previous_plan=previous_plan,
                                                    accepted_count=(previous_filtering.statistics.accepted_count))

    broadened_plan = normalize_broadened_plan(new_plan, previous_plan=previous_plan)
    return {"search_plan": broadened_plan, "search_attempt": next_attempt,"trace": [AgentTraceEntry(step="broaden_search", attempt=next_attempt,
                message=(f"Too few suitable jobs were found. \n Broadened retrieval strategy: {broadened_plan.rationale}"))]}

async def rank_jobs_node(state: CareerMatchGraphState, *, dependencies: CareerMatchGraphDependencies) -> CareerMatchGraphUpdate:
    """Semantically rank all jobs surviving hard filtering."""
    request = state["request"]
    filtering_response = state["filtering"]
    ranking_service = HybridJobRankingService(dependencies.embedding_provider)
    ranking_response = await ranking_service.rank(HybridRankingRequest(profile=request.profile, preferences=request.preferences, accepted_jobs=(filtering_response.accepted_jobs),
                                                                       evidence_signals=request.evidence_signals, configuration=(request.configuration.ranking)))

    return {"ranking": ranking_response, "trace": [AgentTraceEntry(step="rank_jobs", attempt=state["search_attempt"],
            message=(f"Ranked {ranking_response.statistics.ranked_count} suitable jobs semantically."))]}

async def evaluate_jobs_node(state: CareerMatchGraphState, *, dependencies: CareerMatchGraphDependencies) -> CareerMatchGraphUpdate:
    """Generate grounded reports for top-ranked vacancies."""
    request = state["request"]
    ranking_response = state["ranking"]
    evaluation_service = JobEvaluationService(dependencies.report_generator,maximum_jobs=(dependencies.maximum_evaluation_jobs))
    evaluation_response = (await evaluation_service.evaluate(JobEvaluationRequest(profile=request.profile, preferences=request.preferences, ranked_jobs=(ranking_response.ranked_jobs),
                                                                                  evidence_signals=(request.evidence_signals), configuration=(request.configuration.evaluation),)))

    return {"evaluation": evaluation_response, "trace": [AgentTraceEntry(step="evaluate_jobs", attempt=state["search_attempt"],
             message=(f"Generated {evaluation_response.statistics.completed_count} grounded suitability reports."))]}

def build_career_match_graph(dependencies: CareerMatchGraphDependencies):  # type: ignore[no-untyped-def]
    """Build the bounded CareerMatch LangGraph workflow."""
    builder = StateGraph(CareerMatchGraphState)

    async def plan_search(state: CareerMatchGraphState) -> CareerMatchGraphUpdate:
        return await create_initial_search_plan(state, dependencies=dependencies)

    async def search_jobs(state: CareerMatchGraphState) -> CareerMatchGraphUpdate:
        return await search_jobs_node(state, dependencies=dependencies)

    async def broaden_search(state: CareerMatchGraphState) -> CareerMatchGraphUpdate:
        return await broaden_search_node(state, dependencies=dependencies)

    async def rank_jobs(state: CareerMatchGraphState) -> CareerMatchGraphUpdate:
        return await rank_jobs_node(state, dependencies=dependencies)

    async def evaluate_jobs(state: CareerMatchGraphState) -> CareerMatchGraphUpdate:
        return await evaluate_jobs_node(state, dependencies=dependencies)

    builder.add_node("plan_search", plan_search)
    builder.add_node("search_jobs", search_jobs)
    builder.add_node("filter_jobs", filter_jobs_node)
    builder.add_node("broaden_search", broaden_search)
    builder.add_node("rank_jobs", rank_jobs)
    builder.add_node("evaluate_jobs", evaluate_jobs)
    builder.add_edge(START, "plan_search")
    builder.add_edge("plan_search", "search_jobs")
    builder.add_edge("search_jobs", "filter_jobs")
    builder.add_conditional_edges("filter_jobs", route_after_filtering, {"broaden_search": "broaden_search", "rank_jobs": "rank_jobs"})
    builder.add_edge("broaden_search", "search_jobs")
    builder.add_edge("rank_jobs", "evaluate_jobs")
    builder.add_edge("evaluate_jobs", END)
    return builder.compile()

def build_agent_search_response(state: CareerMatchGraphState) -> AgentSearchResponse:
    """Convert the completed graph state to the public response."""
    return AgentSearchResponse(final_search_plan=state["search_plan"], search_attempts=state["search_attempt"], search_statistics=(state["job_search"].statistics),
                               filtering_statistics=(state["filtering"].statistics), ranking=state["ranking"], evaluation=state["evaluation"], trace=state.get("trace", []))
