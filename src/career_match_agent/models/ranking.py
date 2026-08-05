from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator
)

from career_match_agent.models.candidate import (
    CandidateProfile,
    JobPreferences
)
from career_match_agent.models.hiring_agent import CandidateEvidenceSignal
from career_match_agent.models.matching import JobFilterDecision


class RankingModel(BaseModel):
    """Base configuration for ranking-related models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class HybridRankingWeights(RankingModel):
    """Configured component weights for hybrid ranking."""
    semantic: float = Field(default=0.60, ge=0, le=1)
    skill_overlap: float = Field(default=0.20, ge=0, le=1)
    required_keywords: float = Field(default=0.10, ge=0, le=1)
    role_alignment: float = Field(default=0.05, ge=0, le=1)
    warning_quality: float = Field(default=0.05, ge=0, le=1)

    @model_validator(mode="after")
    def require_normalized_weights(self) -> Self:
        total = (self.semantic + self.skill_overlap + self.required_keywords + self.role_alignment + self.warning_quality)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Hybrid ranking weights must sum to 1.0.")

        return self

class HybridRankingConfiguration(RankingModel):
    """Controls chunking, scoring and response size."""
    weights: HybridRankingWeights = Field(default_factory=HybridRankingWeights)
    top_k: int = Field(default=20, ge=1, le=100)
    semantic_evidence_count: int = Field(default=3, ge=1, le=10)
    chunk_max_characters: int = Field(default=700, ge=300, le=2000)
    maximum_candidate_chunks: int = Field(default=24, ge=1, le=100)
    maximum_job_chunks: int = Field(default=12, ge=1, le=50)
    warning_penalty: float = Field(default=0.10, ge=0, le=1)

class HybridRankingRequest(RankingModel):
    """Accepted jobs and candidate data submitted for ranking."""
    profile: CandidateProfile
    preferences: JobPreferences
    accepted_jobs: list[JobFilterDecision]
    evidence_signals: list[CandidateEvidenceSignal] = Field(default_factory=list)
    configuration: HybridRankingConfiguration = Field(default_factory=HybridRankingConfiguration)

    @model_validator(mode="after")
    def require_accepted_decisions(self) -> Self:
        invalid_decisions = [decision for decision in self.accepted_jobs if not decision.accepted]
        if invalid_decisions:
            raise ValueError("Hybrid ranking accepts only jobs that passed deterministic filtering.")

        return self

class EmbeddingMetadata(RankingModel):
    """Information about the embedding implementation."""
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimension: int | None = Field(default=None, ge=1)

class SemanticMatchEvidence(RankingModel):
    """One strong semantic relationship used during ranking."""
    candidate_chunk_kind: str = Field(min_length=1)
    candidate_excerpt: str = Field(min_length=1)
    job_excerpt: str = Field(min_length=1)
    similarity: float = Field(ge=0, le=1)

class HybridScoreBreakdown(RankingModel):
    """Transparent components contributing to the final score."""
    semantic_score: float = Field(ge=0, le=100)
    skill_overlap_score: float | None = Field(default=None, ge=0, le=100)
    required_keyword_score: float | None = Field(default=None, ge=0, le=100)
    role_alignment_score: float | None = Field(default=None, ge=0, le=100)
    warning_quality_score: float = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    component_weights: dict[str, float]
    component_contributions: dict[str, float]

class RankedJob(RankingModel):
    """One accepted job with its semantic and hybrid ranking."""
    rank: int = Field(ge=1)
    hybrid_score: float = Field(ge=0, le=100)
    decision: JobFilterDecision
    score_breakdown: HybridScoreBreakdown
    semantic_matches: list[SemanticMatchEvidence]

class HybridRankingStatistics(RankingModel):
    """Summary information for one ranking operation."""
    received_count: int = Field(ge=0)
    ranked_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    candidate_chunk_count: int = Field(ge=0)
    job_chunk_count: int = Field(ge=0)

class HybridRankingResponse(RankingModel):
    """Complete hybrid-ranking response."""
    embedding: EmbeddingMetadata
    ranked_jobs: list[RankedJob]
    statistics: HybridRankingStatistics
