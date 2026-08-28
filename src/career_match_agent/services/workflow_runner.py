from dataclasses import dataclass
from typing import cast

from career_match_agent.graphs.job_search_graph import (
    CareerMatchGraphDependencies,
    CareerMatchGraphState,
    build_agent_search_response,
    build_career_match_graph
)
from career_match_agent.models.agent import (
    AgentSearchRequest,
    AgentSearchResponse,
    AgentWorkflowConfiguration
)
from career_match_agent.models.hiring_agent import HiringAgentAssessment
from career_match_agent.models.workflow import (
    AutomatedWorkflowResponse,
    PreparedWorkflowState,
    WorkflowLLMMetadata
)
from career_match_agent.providers.base import JobProvider
from career_match_agent.services.candidate_enrichment import build_hiring_agent_evidence_signals
from career_match_agent.services.embedding import EmbeddingProvider
from career_match_agent.services.job_evaluator import JobReportGenerator
from career_match_agent.services.preference_extractor import PreferenceExtractor
from career_match_agent.services.profile_extractor import CandidateProfileExtractor
from career_match_agent.services.search_planner import SearchPlanner


@dataclass(frozen=True)
class AutomatedWorkflowDependencies:
    """Services required by the automated user workflow."""
    profile_extractor: CandidateProfileExtractor
    preference_extractor: PreferenceExtractor
    search_planner: SearchPlanner
    job_provider: JobProvider
    embedding_provider: EmbeddingProvider
    report_generator: JobReportGenerator
    maximum_evaluation_jobs: int


class AutomatedCareerMatchWorkflow:
    """Prepare user input and run the existing agent graph."""

    def __init__(self, dependencies: AutomatedWorkflowDependencies) -> None:
        self.dependencies = dependencies

    async def prepare_request(self, *, cv_text: str, preference_text: str, hiring_agent_assessment: (HiringAgentAssessment | None),
                              configuration: AgentWorkflowConfiguration) -> tuple[AgentSearchRequest, PreparedWorkflowState]:
        """Convert raw candidate input into reusable graph input."""
        profile = (await self.dependencies.profile_extractor.extract(cv_text))
        preferences = (await self.dependencies.preference_extractor.extract(preference_text=preference_text, profile=profile))
        evidence_signals = (build_hiring_agent_evidence_signals(hiring_agent_assessment) if hiring_agent_assessment is not None else [])
        request = AgentSearchRequest(profile=profile, preferences=preferences, evidence_signals=evidence_signals, configuration=configuration)
        prepared_state = PreparedWorkflowState(source_llm=WorkflowLLMMetadata(provider=(self.dependencies.profile_extractor.provider_name),
                                                                              model=(self.dependencies.profile_extractor.model_name)),
                                               agent_request=request,
                                               hiring_agent_assessment=(hiring_agent_assessment))

        return (request, prepared_state)

    async def execute_request(self, *, request: AgentSearchRequest) -> AgentSearchResponse:
        """Run an already prepared AgentSearchRequest."""
        graph_dependencies = (CareerMatchGraphDependencies(search_planner=(self.dependencies.search_planner),
                                                           job_provider=(self.dependencies.job_provider),
                                                           embedding_provider=(self.dependencies.embedding_provider),
                                                           report_generator=(self.dependencies.report_generator),
                                                           maximum_evaluation_jobs=(self.dependencies.maximum_evaluation_jobs)))
        graph = build_career_match_graph(graph_dependencies)
        raw_state = await graph.ainvoke({"request": request, "trace": []}, config={"recursion_limit": 25})
        final_state = cast(CareerMatchGraphState, raw_state)
        return build_agent_search_response(final_state)

    async def run(self, *, cv_text: str, preference_text: str, hiring_agent_assessment: (HiringAgentAssessment | None),
                  configuration: AgentWorkflowConfiguration) -> AutomatedWorkflowResponse:
        request, prepared_state = (await self.prepare_request(cv_text=cv_text, preference_text=preference_text, hiring_agent_assessment=(hiring_agent_assessment),
                                                              configuration=configuration))
        agent_response = (await self.execute_request(request=request))

        return AutomatedWorkflowResponse(llm=WorkflowLLMMetadata(provider=(self.dependencies.profile_extractor.provider_name),
                                                                 model=(self.dependencies.profile_extractor.model_name)),
                                         profile=request.profile,
                                         preferences=request.preferences,
                                         hiring_agent_assessment=(hiring_agent_assessment),
                                         evidence_signal_count=len(request.evidence_signals),
                                         prepared_state=prepared_state,
                                         agent_request=request,
                                         agent=agent_response)
