import asyncio
import math
import pytest

from career_match_agent.models.candidate import (
    CandidateProfile,
    EmploymentType,
    JobPreferences
)
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import JobFilterDecision
from career_match_agent.models.ranking import HybridRankingRequest
from career_match_agent.services.job_normalizer import (
    create_job_fingerprint,
    normalize_for_matching
)
from career_match_agent.services.semantic_ranker import (
    HybridJobRankingService,
    calculate_available_weighted_score
)

class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-semantic-model"
    dimension: int | None = 3

    def create_vector(self, text: str) -> list[float]:
        normalized_text = normalize_for_matching(text)
        vector = [0.05, 0.05, 0.05]
        machine_learning_terms = {"machine learning", "pytorch", "python", "data scientist", "ml engineer"}
        frontend_terms = {"frontend", "javascript", "react","css",}
        sales_terms = {"sales", "account manager","business development",}

        if any(term in normalized_text for term in machine_learning_terms):
            vector[0] += 1.0

        if any(term in normalized_text for term in frontend_terms):
            vector[1] += 1.0

        if any(term in normalized_text for term in sales_terms):
            vector[2] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self.create_vector(text) for text in texts]

    async def embed_documents(self, texts: list[str],) -> list[list[float]]:
        return [self.create_vector(text) for text in texts]


def make_job(*, external_id: str, title: str, description: str, tags: list[str]) -> JobPosting:
    company = "Example GmbH"
    location = "Berlin"
    return JobPosting(source_id=f"mock:{external_id}", provider="mock", external_id=external_id, title=title,
                      company=company, description=description, location=location, remote=False,
                      employment_types=[EmploymentType.FULL_TIME], tags=tags, url=f"https://example.com/jobs/{external_id}",
                      fingerprint=create_job_fingerprint(title=title, company=company, location=location))


def make_decision(job: JobPosting) -> JobFilterDecision:
    return JobFilterDecision(job=job, accepted=True, matched_roles=["Machine Learning Engineer"], matched_required_keywords=["Python"])


def test_semantic_ranker_prioritizes_ml_job() -> None:
    async def run_test() -> None:
        machine_learning_job = make_job(external_id="1", title="Machine Learning Engineer",
                                        description=("Develop Python and PyTorch machine-learning models and deploy inference services."),
                                        tags=["Python", "PyTorch", "Machine Learning"])

        frontend_job = make_job(external_id="2", title="Frontend Software Engineer",
                                description=("Build React and JavaScript user interfaces with CSS."), tags=["React", "JavaScript"])

        request = HybridRankingRequest(profile=CandidateProfile(professional_summary=("Machine-learning engineer with model development experience."),
                                                                              skills=["Python", "PyTorch", "scikit-learn"]),
                                                                preferences=JobPreferences(roles=["Machine Learning Engineer"], required_keywords=["Python"]),
                                                                accepted_jobs=[make_decision(frontend_job), make_decision(machine_learning_job)])

        service = HybridJobRankingService(FakeEmbeddingProvider())
        response = await service.rank(request)
        assert len(response.ranked_jobs) == 2
        assert (response.ranked_jobs[0].decision.job.title == "Machine Learning Engineer")
        assert (response.ranked_jobs[0].hybrid_score > response.ranked_jobs[1].hybrid_score)
        assert (response.ranked_jobs[0].score_breakdown.semantic_score > response.ranked_jobs[1].score_breakdown.semantic_score)

    asyncio.run(run_test())


def test_semantic_ranker_returns_empty_response() -> None:
    async def run_test() -> None:
        request = HybridRankingRequest(profile=CandidateProfile(skills=["Python"]), preferences=JobPreferences(roles=["Machine Learning Engineer"]), accepted_jobs=[])
        service = HybridJobRankingService(FakeEmbeddingProvider())
        response = await service.rank(request)
        assert response.ranked_jobs == []
        assert response.statistics.received_count == 0
        assert response.statistics.returned_count == 0

    asyncio.run(run_test())


def test_ranking_response_does_not_expose_embeddings() -> None:
    async def run_test() -> None:
        job = make_job(external_id="3", title="Machine Learning Engineer", description="Build Python ML models.", tags=["Python"])
        request = HybridRankingRequest(profile=CandidateProfile(skills=["Python"]),
                                       preferences=JobPreferences(roles=["Machine Learning Engineer"]), accepted_jobs=[make_decision(job)])
        service = HybridJobRankingService(FakeEmbeddingProvider())
        response = await service.rank(request)
        payload = response.model_dump()
        assert "embeddings" not in str(payload)
        assert payload["embedding"]["dimension"] == 3

    asyncio.run(run_test())

def test_weighted_score_does_not_exceed_one_hundred() -> None:
    score, weights, contributions = (calculate_available_weighted_score(components={"semantic": 100.0, "skill_overlap": 100.0, "required_keywords": 100.0,
                                                                                    "role_alignment": 100.0, "warning_quality": 100.0},
                                                                        configured_weights={"semantic": 0.60, "skill_overlap": 0.20, "required_keywords": 0.10,
                                                                                            "role_alignment": 0.05, "warning_quality": 0.05}))

    assert score == 100.0
    assert score <= 100.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)
    assert sum(contributions.values()) == pytest.approx(score, abs=0.01)
