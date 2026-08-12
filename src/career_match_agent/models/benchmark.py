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
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import JobFilterReasonCode
from career_match_agent.models.ranking import (
    HybridRankingConfiguration,
    SemanticMatchEvidence
)


class BenchmarkModel(BaseModel):
    """Base configuration for benchmark models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class BenchmarkJobCase(BenchmarkModel):
    """Ground-truth label for one benchmark vacancy."""
    job: JobPosting
    expected_accept: bool
    relevance_grade: int = Field(ge=0, le=3)
    expected_rejection_reasons: list[JobFilterReasonCode] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_expected_reasons(self) -> "BenchmarkJobCase":
        if (self.expected_accept and self.expected_rejection_reasons):
            raise ValueError("Accepted benchmark jobs cannot contain expected rejection reasons.")

        return self

class JobMatchingBenchmarkDataset(BenchmarkModel):
    """Versioned labelled dataset for job matching."""
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    profile: CandidateProfile
    preferences: JobPreferences
    evidence_signals: list[CandidateEvidenceSignal] = Field(default_factory=list)
    jobs: list[BenchmarkJobCase] = Field(min_length=1)

class BinaryClassificationMetrics(BenchmarkModel):
    """Binary accept/reject filtering metrics."""
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    false_acceptance_rate: float = Field(ge=0, le=1)
    false_rejection_rate: float = Field(ge=0, le=1)

class ReasonCodeMetrics(BenchmarkModel):
    """Accuracy of deterministic rejection explanations."""
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    expected_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)


class RankingAtKMetrics(BenchmarkModel):
    """Ranking quality for one cutoff."""
    k: int = Field(ge=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    ndcg: float = Field(ge=0, le=1)

class RankingBenchmarkMetrics(BenchmarkModel):
    """Overall ranking-quality metrics."""
    at_k: list[RankingAtKMetrics]
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    relevant_job_count: int = Field(ge=0)

class EvaluationBenchmarkMetrics(BenchmarkModel):
    """Operational metrics for grounded LLM reports."""
    attempted_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    average_cited_evidence: float = Field(ge=0)
    candidate_and_job_scope_rate: float = Field(ge=0, le=1)

class BenchmarkLatency(BenchmarkModel):
    """Measured service latency."""
    filtering_ms: float = Field(ge=0)
    ranking_ms: float = Field(ge=0)
    evaluation_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)

class RankingJobDiagnostic(BenchmarkModel):
    """Per-job diagnostic values for ranking analysis."""
    source_id: str
    expected_relevance_grade: int = Field(ge=0, le=3)
    rank: int = Field(ge=1)
    hybrid_score: float = Field(ge=0, le=100)
    semantic_score: float = Field(ge=0, le=100)
    skill_overlap_score: float | None = Field(default=None, ge=0, le=100)
    required_keyword_score: float | None = Field(default=None, ge=0, le=100)
    role_alignment_score: float | None = Field(default=None, ge=0, le=100)
    warning_quality_score: float = Field(ge=0, le=100)
    semantic_matches: list[SemanticMatchEvidence]

class JobMatchingBenchmarkResult(BenchmarkModel):
    """Complete result from one benchmark configuration."""
    dataset_name: str
    dataset_version: str
    configuration_name: str
    filtering: BinaryClassificationMetrics
    reason_codes: ReasonCodeMetrics
    ranking: RankingBenchmarkMetrics
    evaluation: EvaluationBenchmarkMetrics | None = None
    latency: BenchmarkLatency
    ranked_source_ids: list[str]
    ranking_configuration: HybridRankingConfiguration
    ranking_diagnostics: list[RankingJobDiagnostic]
