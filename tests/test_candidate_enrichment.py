from career_match_agent.models.candidate import CandidateProfile
from career_match_agent.models.hiring_agent import (
    EvidencePolarity,
    EvidenceSignalType
)
from career_match_agent.services.candidate_enrichment import build_candidate_evidence_context
from career_match_agent.services.hiring_agent_parser import parse_hiring_agent_report
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIRECTORY = PROJECT_ROOT / "data" / "sample" / "hiring_agent_report.txt"
SAMPLE_TEXT_REPORT = SAMPLE_DATA_DIRECTORY.read_text(encoding="utf-8")
SAMPLE_REPORT_BYTES = SAMPLE_TEXT_REPORT.encode("utf-8")

def test_build_candidate_evidence_context() -> None:
    profile = CandidateProfile(skills=["Python", "PyTorch"])
    assessment = parse_hiring_agent_report(SAMPLE_TEXT_REPORT.encode(), source_filename="hiring_agent_report.txt", role_name="software_engineering_intern")
    context = build_candidate_evidence_context(profile, assessment)
    assert context.profile.skills == ["Python", "PyTorch"]
    assert len(context.assessments) == 1
    category_signals = [signal for signal in context.evidence_signals if signal.signal_type == EvidenceSignalType.CATEGORY]
    strength_signals = [signal for signal in context.evidence_signals if signal.signal_type == EvidenceSignalType.STRENGTH]
    improvement_signals = [signal for signal in context.evidence_signals if signal.signal_type == EvidenceSignalType.IMPROVEMENT]
    assert len(category_signals) == 4
    assert len(strength_signals) == 2
    assert len(improvement_signals) == 3
    assert all(signal.polarity == EvidencePolarity.NEUTRAL for signal in category_signals)
    assert all(signal.polarity == EvidencePolarity.POSITIVE for signal in strength_signals)
    assert all(signal.polarity == EvidencePolarity.DEVELOPMENT for signal in improvement_signals)
