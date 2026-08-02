import json

import pytest

from pathlib import Path
from career_match_agent.models.hiring_agent import HiringAgentReportFormat
from career_match_agent.services.hiring_agent_parser import (
    InvalidHiringAgentReportError,
    parse_hiring_agent_report
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIRECTORY = PROJECT_ROOT / "data" / "sample" / "hiring_agent_report.txt"
SAMPLE_TEXT_REPORT = SAMPLE_DATA_DIRECTORY.read_text(encoding="utf-8")
SAMPLE_REPORT_BYTES = SAMPLE_TEXT_REPORT.encode("utf-8")

def test_parse_hiring_agent_text_report() -> None:
    assessment = parse_hiring_agent_report(SAMPLE_REPORT_BYTES, source_filename="hiring_agent_report.txt", role_name="software_engineering_intern")
    assert assessment.candidate_name == "B. Bunny"
    assert assessment.role_name == "software_engineering_intern"
    assert assessment.source_format == HiringAgentReportFormat.TEXT
    assert assessment.source_filename == "hiring_agent_report.txt"
    assert assessment.reported_overall_score == -2
    assert assessment.base_max_score == 100
    assert len(assessment.categories) == 4

    open_source = assessment.categories[0]
    assert open_source.key == "open_source"
    assert open_source.label == "Open Source"
    assert open_source.score == 0
    assert open_source.max_score == 35
    assert open_source.capped_score == 0
    assert "No evidence of GitHub profile" in open_source.evidence

    self_projects = assessment.categories[1]
    assert self_projects.key == "self_projects"
    assert self_projects.score == 0
    assert self_projects.max_score == 30
    assert "Projects lack technical software development" in (self_projects.evidence)

    production_experience = assessment.categories[2]
    assert production_experience.key == "production_experience"
    assert production_experience.score == 3
    assert production_experience.max_score == 25
    assert "routing algorithm" in production_experience.evidence

    technical_skills = assessment.categories[3]
    assert technical_skills.key == "technical_skills"
    assert technical_skills.score == 0
    assert technical_skills.max_score == 10
    assert "No relevant software programming languages" in (technical_skills.evidence)
    assert assessment.category_total == 3
    assert assessment.bonus_points.total == 1
    assert assessment.deductions.total == 6
    assert assessment.computed_overall_score == -2
    assert assessment.score_difference == 0
    assert assessment.warnings == []
    assert assessment.bonus_points.breakdown == ("+1 point for providing a LinkedIn profile link.")
    assert assessment.deductions.reasons == ("Applied -3 points deduction for each of the 2 listed projects lacking GitHub repository links or active URLs.")
    assert assessment.key_strengths == [("Algorithmic concepts referenced in work experience (routing algorithms)"),"Strong crisis management and adaptability under pressure"]
    assert assessment.areas_for_improvement == ["Add proficiency in mainstream software engineering programming languages and developer frameworks",
                                                "Include links to code repositories, open source contributions, or live demos for projects",
                                                "Gain production software engineering experience and technical development skills"]


def test_parse_dynamic_json_categories() -> None:
    report_payload = {"scores": {"applied_research": {"score": 18, "max": 25, "evidence": ("The candidate completed multiple research-oriented ML projects.")},
                      "ml_engineering": {"score": 31, "max": 40, "evidence": ("The candidate used Python, PyTorch, Docker and AWS.")}},
                      "bonus_points": {"total": 3, "breakdown": "Technical writing and public projects."},
                      "deductions": {"total": 1, "reasons": "Limited monitoring experience."},
                      "key_strengths": ["Strong experimentation background"],
                      "areas_for_improvement": ["Add production monitoring evidence"]}

    assessment = parse_hiring_agent_report(json.dumps(report_payload).encode(), source_filename="evaluation.json", role_name="machine_learning_engineer")
    assert assessment.source_format == HiringAgentReportFormat.JSON
    assert assessment.role_name == "machine_learning_engineer"
    assert [category.key for category in assessment.categories] == ["applied_research", "ml_engineering"]
    assert assessment.base_max_score == 65
    assert assessment.category_total == 49
    assert assessment.bonus_points.total == 3
    assert assessment.deductions.total == 1
    assert assessment.computed_overall_score == 51
    assert assessment.reported_overall_score == 51
    assert assessment.score_difference == 0

def test_report_detects_inconsistent_overall_score() -> None:
    inconsistent_report = SAMPLE_TEXT_REPORT.replace("OVERALL SCORE: -2.0/100", "OVERALL SCORE: 5.0/100")
    assessment = parse_hiring_agent_report(inconsistent_report.encode(), source_filename="hiring_agent_report.txt")
    assert assessment.reported_overall_score == 5
    assert assessment.computed_overall_score == -2
    assert assessment.score_difference == 7
    assert any("differs from the score recomputed" in warning for warning in assessment.warnings)

def test_report_rejects_missing_categories() -> None:
    with pytest.raises(InvalidHiringAgentReportError, match="category scores"):
        parse_hiring_agent_report(b"OVERALL SCORE: 50/100", source_filename="hiring_agent_report.txt")

def test_report_rejects_invalid_json() -> None:
    with pytest.raises(InvalidHiringAgentReportError, match="JSON is invalid"):
        parse_hiring_agent_report(b'{"scores": invalid}', source_filename="evaluation.json")

def test_report_rejects_json_without_scores() -> None:
    report_payload = {"scores": {}, "bonus_points": {"total": 0,"breakdown": ""}, "deductions": {"total": 0, "reasons": ""}}
    with pytest.raises(InvalidHiringAgentReportError, match="contains no score categories"):
        parse_hiring_agent_report(json.dumps(report_payload).encode(), source_filename="evaluation.json")
