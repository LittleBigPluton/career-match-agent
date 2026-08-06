import pytest
from pydantic import ValidationError

from career_match_agent.models.evaluation import (
    GroundedFinding,
    JobEvaluationConfiguration,
    JobSuitabilityReportDraft
)


def test_grounded_finding_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        GroundedFinding(title="Python alignment", explanation=("The candidate has relevant Python experience."), evidence_ids=[])


def test_evaluation_configuration_limits_jobs() -> None:
    with pytest.raises(ValidationError):
        JobEvaluationConfiguration(maximum_jobs=21)


def test_report_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        JobSuitabilityReportDraft.model_validate(
            {"source_id": "mock:1", "recommendation": "match", "confidence": "high",
             "summary": {"text": "Relevant candidate.","evidence_ids": ["candidate:skills", "job:identity"]},
             "strengths": [{"title": "Skill match","explanation": "Python aligns.","evidence_ids": ["candidate:skills","job:description:0",]}],
             "gaps": [], "risks": [], "interview_focus": [], "unexpected_field": "invalid"})
