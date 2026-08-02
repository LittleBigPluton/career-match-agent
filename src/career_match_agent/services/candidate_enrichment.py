from career_match_agent.models.candidate import CandidateProfile
from career_match_agent.models.hiring_agent import (
    CandidateEvidenceContext,
    CandidateEvidenceSignal,
    EvidencePolarity,
    EvidenceSignalType,
    HiringAgentAssessment
)


def build_hiring_agent_evidence_signals(assessment: HiringAgentAssessment) -> list[CandidateEvidenceSignal]:
    """Convert an external assessment into reusable evidence signals."""
    signals: list[CandidateEvidenceSignal] = []
    for category in assessment.categories:
        signals.append(CandidateEvidenceSignal(signal_type=EvidenceSignalType.CATEGORY, title=f"{category.label} assessment", evidence=category.evidence,
                                               polarity=EvidencePolarity.NEUTRAL, category_key=category.key, source_score=category.capped_score, source_max_score=category.max_score))

    for strength in assessment.key_strengths:
        signals.append(CandidateEvidenceSignal(signal_type=EvidenceSignalType.STRENGTH,title="Hiring-agent strength", evidence=strength, polarity=EvidencePolarity.POSITIVE))

    for improvement in assessment.areas_for_improvement:
        signals.append(CandidateEvidenceSignal(signal_type=EvidenceSignalType.IMPROVEMENT, title="Hiring-agent improvement area", evidence=improvement, polarity=EvidencePolarity.DEVELOPMENT))

    if (assessment.bonus_points.total > 0 and assessment.bonus_points.breakdown):
        signals.append(CandidateEvidenceSignal(signal_type=EvidenceSignalType.BONUS, title="Hiring-agent bonus evidence", evidence=assessment.bonus_points.breakdown,
                                               polarity=EvidencePolarity.POSITIVE, source_score=assessment.bonus_points.total))

    if (assessment.deductions.total > 0 and assessment.deductions.reasons):
        signals.append(CandidateEvidenceSignal(signal_type=EvidenceSignalType.DEDUCTION, title="Hiring-agent deduction evidence",evidence=assessment.deductions.reasons,
                                               polarity=EvidencePolarity.DEVELOPMENT, source_score=assessment.deductions.total))

    return signals


def build_candidate_evidence_context(profile: CandidateProfile, assessment: HiringAgentAssessment) -> CandidateEvidenceContext:
    """Combine candidate facts and external assessment evidence."""
    return CandidateEvidenceContext(profile=profile, assessments=[assessment], evidence_signals=build_hiring_agent_evidence_signals(assessment))
