import asyncio

from career_match_agent.models.candidate import (
    CandidateProfile,
    EmploymentType,
    JobPreferences
)
from career_match_agent.models.evaluation import (
    EvidenceScope,
    GroundedFinding,
    GroundedStatement,
    JobEvaluationRequest,
    JobRecommendation,
    JobSuitabilityReportDraft,
    EvaluationConfidence,
    GroundingEvidenceItem
)
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import JobFilterDecision
from career_match_agent.models.ranking import (
    HybridScoreBreakdown,
    RankedJob,
    SemanticMatchEvidence
)
from career_match_agent.services.job_evaluator import (
    JobEvaluationService,
    build_job_evidence_bundle
)
from career_match_agent.services.job_normalizer import create_job_fingerprint


class FakeJobReportGenerator:
    provider_name = "fake"
    model_name = "fake-report-model"
    prompt_version = "job-report-test-v1"

    async def generate(self, *, source_id: str, evidence_items: list[GroundingEvidenceItem]) -> JobSuitabilityReportDraft:
        candidate_id = next(item.evidence_id for item in evidence_items if item.scope == EvidenceScope.CANDIDATE)
        job_id = next(item.evidence_id for item in evidence_items if item.scope == EvidenceScope.JOB)
        return JobSuitabilityReportDraft(source_id=source_id, recommendation=JobRecommendation.STRONG_MATCH, confidence=EvaluationConfidence.HIGH,
                                         summary=GroundedStatement(text=("The candidate has relevant skills for the advertised role."),
                                                                    evidence_ids=[candidate_id, job_id]),
                                       strengths=[GroundedFinding(title="Relevant technical stack",explanation=("The candidate's Python experience matches the job requirements."),
                                                                  evidence_ids=[candidate_id,job_id])],
                                            gaps=[],
                                           risks=[],
                                 interview_focus=["Discuss production deployment experience."])


class InvalidCitationGenerator(FakeJobReportGenerator):
    async def generate(self, *, source_id: str, evidence_items: list[GroundingEvidenceItem], previous_report: JobSuitabilityReportDraft | None = None,
    validation_feedback: str | None = None) -> JobSuitabilityReportDraft:
        del evidence_items

        return JobSuitabilityReportDraft(source_id=source_id, recommendation=JobRecommendation.MATCH, confidence=EvaluationConfidence.MEDIUM,
                        summary=GroundedStatement(text="The candidate appears suitable.", evidence_ids=["candidate:invented", "job:invented"]),
                        strengths=[GroundedFinding(title="Potential match", explanation="Some skills may align.", evidence_ids=["candidate:invented"],)])


def make_ranked_job() -> RankedJob:
    title = "Machine Learning Engineer"
    company = "Example AI GmbH"
    location = "Berlin"

    job = JobPosting(source_id="mock:1", provider="mock", external_id="1", title=title, company=company,
                    description=("Develop Python and PyTorch machine-learning models and deploy inference APIs."),
                    location=location, remote=False, employment_types=[EmploymentType.FULL_TIME],
                    tags=["Python", "PyTorch"], url="https://example.com/jobs/1",
                    fingerprint=create_job_fingerprint(title=title,company=company,location=location))

    decision = JobFilterDecision(job=job,accepted=True,matched_roles=["Machine Learning Engineer"], matched_required_keywords=["Python"])

    return RankedJob(rank=1, hybrid_score=88.0, decision=decision,
                    score_breakdown=HybridScoreBreakdown(semantic_score=85.0,
                                                         skill_overlap_score=100.0,
                                                         required_keyword_score=100.0,
                                                         role_alignment_score=100.0,
                                                         warning_quality_score=100.0,
                                                         matched_skills=["Python", "PyTorch"],
                                                         missing_skills=[],
                                                         component_weights={"semantic": 0.6,
                                                                            "skill_overlap": 0.2,
                                                                            "required_keywords": 0.1,
                                                                            "role_alignment": 0.05,
                                                                            "warning_quality": 0.05,},
                                                         component_contributions={"semantic": 51.0,
                                                                                  "skill_overlap": 20.0,
                                                                                  "required_keywords": 10.0,
                                                                                  "role_alignment": 5.0,
                                                                                  "warning_quality": 5.0,}),
                     semantic_matches=[SemanticMatchEvidence(candidate_chunk_kind="profile_overview",
                                                             candidate_excerpt=("Technical skills: Python, PyTorch"),
                                                             job_excerpt=("Develop Python and PyTorch models"),
                                                             similarity=0.91)])


def make_request() -> JobEvaluationRequest:
    return JobEvaluationRequest(profile=CandidateProfile(professional_summary=("Machine-learning engineer with Python and PyTorch experience."),
                                skills=["Python", "PyTorch"]), preferences=JobPreferences(roles=["Machine Learning Engineer"], required_keywords=["Python"]),
                                ranked_jobs=[make_ranked_job()])


def test_evidence_bundle_contains_all_scopes() -> None:
    request = make_request()
    evidence_items = build_job_evidence_bundle(profile=request.profile, preferences=request.preferences, evidence_signals=[], ranked_job=request.ranked_jobs[0],
                                               maximum_candidate_evidence=40, maximum_description_chunks=8, description_chunk_characters=700)

    scopes = {evidence_item.scope for evidence_item in evidence_items}
    assert EvidenceScope.CANDIDATE in scopes
    assert EvidenceScope.JOB in scopes
    assert EvidenceScope.COMPARISON in scopes


def test_job_evaluation_service_returns_grounded_report() -> None:
    async def run_test() -> None:
        service = JobEvaluationService(FakeJobReportGenerator(), maximum_jobs=5)
        response = await service.evaluate(make_request())
        assert response.statistics.completed_count == 1
        assert response.statistics.failed_count == 0

        evaluated_report = response.reports[0]
        assert evaluated_report.source_id == "mock:1"
        assert evaluated_report.report.recommendation == JobRecommendation.STRONG_MATCH
        assert evaluated_report.grounding.candidate_citation_count >= 1
        assert evaluated_report.grounding.job_citation_count >= 1
        assert evaluated_report.cited_evidence

    asyncio.run(run_test())


def test_invalid_citation_creates_failure() -> None:
    async def run_test() -> None:
        service = JobEvaluationService(InvalidCitationGenerator(), maximum_jobs=5)
        response = await service.evaluate(make_request())
        assert response.statistics.completed_count == 0
        assert response.statistics.failed_count == 1
        assert "unknown evidence IDs" in response.failures[0].error

    asyncio.run(run_test())
