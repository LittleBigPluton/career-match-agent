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
from career_match_agent.models.ranking import (
    HybridRankingRequest,
    HybridRankingConfiguration
)
from career_match_agent.services.job_normalizer import (
    create_job_fingerprint,
    normalize_for_matching
)
from career_match_agent.services.semantic_ranker import (
    SemanticTextChunk,
    average_top_similarities,
    build_candidate_chunks,
    build_job_chunks,
    calculate_semantic_match,
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

def test_job_description_chunks_do_not_repeat_header() -> None:
    job = make_job(external_id="semantic-test", title="Machine Learning Engineer", description=("Support customer workshops and prepare solution demonstrations."),
                    tags=["Python", "PyTorch", "Machine Learning"])
    configuration = (HybridRankingConfiguration())
    chunks = build_job_chunks(job, configuration=configuration)
    header_chunks = [chunk for chunk in chunks if chunk.kind == "job_header"]
    description_chunks = [chunk for chunk in chunks if chunk.kind == "job_description"]
    assert len(header_chunks) == 1
    assert description_chunks

    for chunk in description_chunks:
        assert ("Job title:" not in chunk.text)
        assert ("Tags:" not in chunk.text)

def test_average_top_similarities_uses_strongest_matches() -> None:
    result = average_top_similarities([0.82, 0.76, 0.71, 0.30, 0.20], evidence_count=3)
    assert result == pytest.approx((0.82 + 0.76 + 0.71) / 3)

def test_average_top_similarities_handles_empty_input() -> None:
    assert (average_top_similarities([], evidence_count=3) == 0.0)

def test_average_top_similarities_handles_fewer_matches() -> None:
    result = average_top_similarities([0.8, 0.6], evidence_count=3)
    assert result == pytest.approx(0.7)

def test_candidate_chunks_do_not_repeat_technology_metadata() -> None:
    profile = make_profile()
    chunks = build_candidate_chunks(profile=profile, preferences=make_preferences(), evidence_signals=[], configuration=HybridRankingConfiguration())
    project_chunks = [chunk for chunk in chunks if chunk.kind == "project"]
    experience_chunks = [chunk for chunk in chunks if chunk.kind == "experience"]
    assert project_chunks
    assert experience_chunks
    assert all("Technologies:" not in chunk.text for chunk in project_chunks)
    assert all("Technologies:" not in chunk.text for chunk in experience_chunks)

def test_candidate_chunks_include_education_details() -> None:
    profile = make_profile()
    chunks = build_candidate_chunks(profile=profile, preferences=make_preferences(), evidence_signals=[], configuration=HybridRankingConfiguration())
    education_chunks = [chunk for chunk in chunks if chunk.kind == "education"]
    assert education_chunks

    education_text = " ".join(chunk.text for chunk in education_chunks)
    assert "scientific computing" in education_text.lower()

def test_semantic_match_ignores_target_role_chunks() -> None:
    candidate_chunks = [SemanticTextChunk(identifier="target-roles", kind="target_roles", text="Machine Learning Engineer"),
                        SemanticTextChunk(identifier="project-0", kind="project", text="Built and evaluated machine-learning models.")]

    job_chunks = [SemanticTextChunk(identifier="job:description:0", kind="job_description", text="Develop predictive machine-learning systems.")]
    candidate_embeddings = [[1.0, 0.0],[0.8, 0.6]]
    job_embeddings = [[1.0, 0.0]]
    similarity, evidence = calculate_semantic_match(candidate_chunks=candidate_chunks, candidate_embeddings=candidate_embeddings, job_chunks=job_chunks, job_embeddings=job_embeddings, evidence_count=3)
    assert similarity == pytest.approx(0.8)
    assert len(evidence) == 1
    assert evidence[0].candidate_chunk_kind == "project"

def make_profile() -> CandidateProfile:
    return CandidateProfile.model_validate({"full_name": None,
                                            "location": "Germany",
                                            "professional_summary": ("Early-career machine-learning engineer with experience in scientific computing and API development."),
                                            "skills": ["Python", "PyTorch", "FastAPI"],
                                            "skill_evidence": [],
                                            "experience": [{"job_title": "AI/ML Engineer Intern", "organization": "ML Team", "location": None, "start_date": None, "end_date": None,
                                                            "is_current": False, "highlights": ["Fine-tuned and evaluated machine-learning models."], "technologies": ["Python", "PyTorch"],"evidence": [],}],
                                            "projects": [{"name": "CareerMatch Agent", "summary": ("Built an agentic job-search system with semantic ranking and grounded evaluation."),
                                                          "technologies": ["Python", "FastAPI"],"highlights": ["Implemented semantic ranking and API services."],"evidence": []}],
                                            "education": [{"degree": "M.Sc.", "field_of_study": "Computational Science", "institution": "University", "location": "Germany", "start_date": None, "end_date": None,
                                                           "details": ["Graduate work included machine learning and scientific computing."], "evidence": []}], "languages": [], "certifications": [],
                                                           "years_of_experience": None})

def make_preferences() -> JobPreferences:
    return JobPreferences.model_validate({"roles": ["Machine Learning Engineer", "Data Scientist",],
                                          "locations": ["Berlin", "Munich"],
                                          "work_modes": ["remote", "hybrid", "on_site"],
                                          "employment_types": ["full_time"],
                                          "seniority_levels": ["entry_level", "junior"],
                                          "required_keywords": [],
                                          "excluded_keywords": [],
                                          "preferred_languages": ["English"],
                                          "maximum_results": 20})
