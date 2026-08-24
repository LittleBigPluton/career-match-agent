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
    AgentWorkflowConfiguration
)
from career_match_agent.models.hiring_agent import HiringAgentAssessment
from career_match_agent.models.workflow import (
    AutomatedWorkflowResponse,
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
    """Prepare raw user input and run the existing agent graph."""

    def __init__(self, dependencies: AutomatedWorkflowDependencies) -> None:
        self.dependencies = dependencies

    async def run(self, *, cv_text: str, preference_text: str, hiring_agent_assessment: HiringAgentAssessment | None, configuration: AgentWorkflowConfiguration) -> AutomatedWorkflowResponse:
        profile = (await self.dependencies.profile_extractor.extract(cv_text))
        preferences = (await self.dependencies.preference_extractor.extract(preference_text=preference_text, profile=profile))
        evidence_signals = (build_hiring_agent_evidence_signals(hiring_agent_assessment) if hiring_agent_assessment is not None else [])
        request = AgentSearchRequest(profile=profile, preferences=preferences, evidence_signals=evidence_signals, configuration=configuration)
        graph_dependencies = CareerMatchGraphDependencies(search_planner=(self.dependencies.search_planner),
                                                          job_provider=(self.dependencies.job_provider),
                                                          embedding_provider=(self.dependencies.embedding_provider),
                                                          report_generator=(self.dependencies.report_generator),
                                                          maximum_evaluation_jobs=(self.dependencies.maximum_evaluation_jobs))

        graph = build_career_match_graph(graph_dependencies)
        raw_state = await graph.ainvoke({"request": request, "trace": []},config={"recursion_limit": 25})
        final_state = cast(CareerMatchGraphState, raw_state)
        agent_response = (build_agent_search_response(final_state))

        return AutomatedWorkflowResponse(
            llm=WorkflowLLMMetadata(provider=(self.dependencies.profile_extractor.provider_name), model=(self.dependencies.profile_extractor.model_name)),
            profile=profile,
            preferences=preferences,
            hiring_agent_assessment=(hiring_agent_assessment),
            evidence_signal_count=len(evidence_signals), agent=agent_response)
