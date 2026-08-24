import asyncio
import math

from career_match_agent.graphs.job_search_graph import (
    CareerMatchGraphDependencies,
    CareerMatchGraphState,
    build_agent_search_response,
    build_career_match_graph
)
from career_match_agent.models.agent import (
    AgentSearchPlan,
    AgentSearchRequest,
    AgentWorkflowConfiguration
)
from career_match_agent.models.candidate import (
    CandidateProfile,
    EmploymentType,
    JobPreferences,
    SeniorityLevel,
    WorkMode
)
from career_match_agent.models.evaluation import (
    EvaluationConfidence,
    EvidenceScope,
    GroundedFinding,
    GroundedStatement,
    GroundingEvidenceItem,
    JobRecommendation,
    JobSuitabilityReportDraft
)
from career_match_agent.models.job import JobPosting
from career_match_agent.providers.mock import (
    MockJobProvider
)
from career_match_agent.services.job_normalizer import create_job_fingerprint


class FakeSearchPlanner:
    provider_name = "fake"
    model_name = "fake-planner"
    prompt_version = "planner-test-v1"

    def __init__(self) -> None:
        self.call_count = 0

    async def plan(self, *, profile, preferences, attempt, maximum_attempts, previous_plan, accepted_count) -> AgentSearchPlan:
        del (profile, preferences, maximum_attempts, accepted_count)
        self.call_count += 1
        if attempt == 1:
            assert previous_plan is None
            return AgentSearchPlan(keywords=["Machine Learning Engineer"], max_pages=1, maximum_results=100, rationale="Start precisely.")

        return AgentSearchPlan(keywords=["AI Engineer"], max_pages=1, maximum_results=100, rationale=("Broaden to an equivalent AI engineering title."))


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-embedding"
    dimension: int | None = 3
    def vector(self, text: str) -> list[float]:
        lowered_text = text.casefold()
        vector = [1.0 if ("python" in lowered_text or "ai" in lowered_text or "machine" in lowered_text) else 0.1, 0.1,0.1]
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

class FakeJobReportGenerator:
    provider_name = "fake"
    model_name = "fake-report-model"
    prompt_version = "report-test-v1"
    async def generate(self, *, source_id: str, evidence_items: list[GroundingEvidenceItem]) -> JobSuitabilityReportDraft:
        candidate_evidence = next(item.evidence_id for item in evidence_items if item.scope == EvidenceScope.CANDIDATE)
        job_evidence = next(item.evidence_id for item in evidence_items if item.scope == EvidenceScope.JOB)
        return JobSuitabilityReportDraft(source_id=source_id, recommendation=JobRecommendation.MATCH, confidence=EvaluationConfidence.HIGH,
                                         summary=GroundedStatement(text=("The candidate has relevant experience for the role."),evidence_ids=[candidate_evidence, job_evidence]),
                                         strengths=[GroundedFinding(title="Relevant technical background", explanation=("Candidate and job evidence align."),
                                                                    evidence_ids=[candidate_evidence, job_evidence])])


def create_ai_job() -> JobPosting:
    title = "Junior AI Engineer"
    company = "Example AI GmbH"
    location = "Berlin"
    return JobPosting(source_id="mock:ai-engineer", provider="mock", external_id="ai-engineer", title=title, company=company,
                      description=("Build Python AI systems and production APIs."),location=location, remote=False, employment_types=[EmploymentType.FULL_TIME],
                      tags=["Python", "AI"], url="https://example.com/jobs/ai-engineer", fingerprint=create_job_fingerprint( title=title, company=company, location=location))


def create_agent_request() -> AgentSearchRequest:
    return AgentSearchRequest(profile=CandidateProfile(professional_summary=("Machine-learning engineer with Python experience."), skills=["Python", "PyTorch"]),
                                                       preferences=JobPreferences(roles=["Machine Learning Engineer"],
                                                                                  locations=["Berlin"],
                                                                                  work_modes=[WorkMode.ON_SITE],
                                                                                  employment_types=[EmploymentType.FULL_TIME],
                                                                                  seniority_levels=[SeniorityLevel.JUNIOR]),
                                                       configuration=AgentWorkflowConfiguration(minimum_accepted_jobs=2, maximum_search_attempts=2))

def test_agent_replans_when_initial_search_is_too_narrow() -> None:
    async def run_test() -> None:
        planner = FakeSearchPlanner()
        dependencies = CareerMatchGraphDependencies(search_planner=planner, job_provider=MockJobProvider(jobs=[create_ai_job()]),
                                                    embedding_provider=(FakeEmbeddingProvider()), report_generator=(FakeJobReportGenerator()),
                                                    maximum_evaluation_jobs=5)

        graph = build_career_match_graph(dependencies)
        raw_state = await graph.ainvoke({"request": create_agent_request(),"trace": []}, config={"recursion_limit": 25})
        state = CareerMatchGraphState(**raw_state)
        response = build_agent_search_response(state)
        assert planner.call_count == 2
        assert response.search_attempts == 2
        assert "AI Engineer" in (response.final_search_plan.keywords)
        assert (response.filtering_statistics.accepted_count == 1)
        assert (response.ranking.statistics.ranked_count == 1)
        assert (response.evaluation.statistics.completed_count == 1)
        trace_steps = [entry.step for entry in response.trace]
        assert trace_steps == ["plan_search", "search_jobs", "filter_jobs", "broaden_search", "search_jobs", "filter_jobs", "rank_jobs", "evaluate_jobs"]

    asyncio.run(run_test())
