from time import perf_counter

from career_match_agent.models.benchmark import (
    BenchmarkLatency,
    EvaluationBenchmarkMetrics,
    JobMatchingBenchmarkDataset,
    JobMatchingBenchmarkResult,
    RankingJobDiagnostic,
)
from career_match_agent.models.evaluation import (
    JobEvaluationConfiguration,
    JobEvaluationRequest
)
from career_match_agent.models.matching import JobFilteringRequest
from career_match_agent.models.ranking import (
    HybridRankingConfiguration,
    HybridRankingRequest
)
from career_match_agent.services.benchmark_metrics import (
    calculate_binary_metrics,
    calculate_ranking_metrics,
    calculate_reason_code_metrics,
    safe_divide
)
from career_match_agent.services.embedding import EmbeddingProvider
from career_match_agent.services.job_evaluator import (
    JobEvaluationService,
    JobReportGenerator
)
from career_match_agent.services.job_filter import filter_jobs_for_candidate
from career_match_agent.services.semantic_ranker import HybridJobRankingService
from career_match_agent.models.ranking import (
    HybridRankingWeights,
    SemanticMatchEvidence
)

class JobMatchingBenchmarkRunner:
    """Evaluate filtering, ranking and grounded reports."""
    def __init__(self, *, embedding_provider: EmbeddingProvider, report_generator: JobReportGenerator | None = None, maximum_evaluation_jobs: int = 5) -> None:
        self.embedding_provider = (embedding_provider)
        self.report_generator = (report_generator)
        self.maximum_evaluation_jobs = (maximum_evaluation_jobs)

    async def run(self, *, dataset: JobMatchingBenchmarkDataset, configuration_name: str, ranking_configuration: HybridRankingConfiguration) -> JobMatchingBenchmarkResult:
        """Execute one reproducible benchmark configuration."""
        total_start = perf_counter()
        jobs = [benchmark_case.job for benchmark_case in dataset.jobs]
        filter_start = perf_counter()
        filtering_response = (filter_jobs_for_candidate(JobFilteringRequest(profile=dataset.profile, preferences=dataset.preferences, jobs=jobs)))
        filtering_ms = (perf_counter() - filter_start) * 1000
        all_decisions = [*filtering_response.accepted_jobs, *filtering_response.rejected_jobs]
        expected_accept = {benchmark_case.job.source_id: benchmark_case.expected_accept for benchmark_case in dataset.jobs}
        predicted_accept = {decision.job.source_id: decision.accepted for decision in all_decisions}
        filtering_metrics = (calculate_binary_metrics(expected=expected_accept, predicted=predicted_accept))
        expected_reasons = {benchmark_case.job.source_id: {reason.value for reason in benchmark_case.expected_rejection_reasons} for benchmark_case in dataset.jobs}
        reason_metrics = (calculate_reason_code_metrics(expected=expected_reasons, decisions=all_decisions))
        rank_start = perf_counter()
        ranking_service = HybridJobRankingService(self.embedding_provider)
        ranking_response = await ranking_service.rank(HybridRankingRequest(profile=dataset.profile, preferences=dataset.preferences, accepted_jobs=filtering_response.accepted_jobs,
                                                                           evidence_signals=dataset.evidence_signals, configuration=ranking_configuration))

        ranking_ms = (perf_counter() - rank_start) * 1000
        relevance_by_source_id = {benchmark_case.job.source_id: benchmark_case.relevance_grade for benchmark_case in dataset.jobs}
        ranking_diagnostics = [RankingJobDiagnostic(source_id=(ranked_job.decision.job.source_id),
                                                    expected_relevance_grade=(relevance_by_source_id[ranked_job.decision.job.source_id]),
                                                    rank=ranked_job.rank,
                                                    hybrid_score=ranked_job.hybrid_score,
                                                    semantic_score=(ranked_job.score_breakdown.semantic_score),
                                                    skill_overlap_score=(ranked_job.score_breakdown.skill_overlap_score),
                                                    required_keyword_score=(ranked_job.score_breakdown.required_keyword_score),
                                                    role_alignment_score=(ranked_job.score_breakdown.role_alignment_score),
                                                    warning_quality_score=(ranked_job.score_breakdown.warning_quality_score),
                                                    semantic_matches=(ranked_job.semantic_matches)) for ranked_job in ranking_response.ranked_jobs]

        ranked_source_ids = [ranked_job.decision.job.source_id for ranked_job in ranking_response.ranked_jobs]
        ranking_metrics = calculate_ranking_metrics(ranked_source_ids=ranked_source_ids, relevance_by_source_id=(relevance_by_source_id))
        evaluation_metrics = None
        evaluation_ms: float | None = None

        if self.report_generator is not None:
            evaluation_start = perf_counter()
            evaluation_service = (JobEvaluationService(self.report_generator, maximum_jobs=(self.maximum_evaluation_jobs)))
            evaluation_response = (await evaluation_service.evaluate(JobEvaluationRequest(profile=dataset.profile,
                                                                                                  preferences=(dataset.preferences),
                                                                                                  ranked_jobs=(ranking_response.ranked_jobs),
                                                                                                  evidence_signals=(dataset.evidence_signals),
                                                                                                  configuration=(JobEvaluationConfiguration(maximum_jobs=(self.maximum_evaluation_jobs))))))

            evaluation_ms = (perf_counter() - evaluation_start) * 1000
            completed_reports = (evaluation_response.reports)
            average_cited_evidence = (safe_divide(sum(report.grounding.cited_evidence_count for report in completed_reports), len(completed_reports)))
            candidate_and_job_scope_count = sum((report.grounding.candidate_citation_count > 0 and report.grounding.job_citation_count > 0) for report in completed_reports)
            evaluation_metrics = (EvaluationBenchmarkMetrics(attempted_count=(evaluation_response.statistics.attempted_count),
                                                             completed_count=(evaluation_response.statistics.completed_count),
                                                             failed_count=(evaluation_response.statistics.failed_count),
                                                             success_rate=safe_divide(evaluation_response.statistics.completed_count,
                                                             evaluation_response.statistics.attempted_count),
                                                             average_cited_evidence=(average_cited_evidence),
                                                             candidate_and_job_scope_rate=(safe_divide(candidate_and_job_scope_count, len(completed_reports)))))

        total_ms = (perf_counter() - total_start) * 1000
        return JobMatchingBenchmarkResult(dataset_name=dataset.name,
                                          dataset_version=dataset.version,
                                          configuration_name=(configuration_name),
                                          filtering=filtering_metrics,
                                          reason_codes=reason_metrics,
                                          ranking=ranking_metrics,
                                          evaluation=evaluation_metrics,
                                          latency=BenchmarkLatency(filtering_ms=round(filtering_ms, 2),
                                                                   ranking_ms=round(ranking_ms, 2),
                                                                   evaluation_ms=(round(evaluation_ms, 2) if evaluation_ms is not None else None),
                                                                   total_ms=round(total_ms, 2)),
                                          ranked_source_ids=(ranked_source_ids),
                                          ranking_configuration=(ranking_configuration),
                                          ranking_diagnostics=(ranking_diagnostics))

def create_ranking_ablation_configurations(
) -> dict[str, HybridRankingConfiguration]:
    """Return standard ranking configurations for comparison."""
    return {"hybrid_default": (HybridRankingConfiguration()),
            "semantic_only": (HybridRankingConfiguration(weights=HybridRankingWeights(semantic=1.0, skill_overlap=0.0, required_keywords=0.0, role_alignment=0.0, warning_quality=0.0))),
            "deterministic_only": (HybridRankingConfiguration(weights=HybridRankingWeights(semantic=0.0, skill_overlap=0.50, required_keywords=0.25, role_alignment=0.15, warning_quality=0.10)))}
