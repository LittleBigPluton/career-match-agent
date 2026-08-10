import argparse
import asyncio
from pathlib import Path

from career_match_agent.core.config import get_settings
from career_match_agent.models.benchmark import JobMatchingBenchmarkDataset
from career_match_agent.services.benchmark_runner import (
    JobMatchingBenchmarkRunner,
    create_ranking_ablation_configurations
)
from career_match_agent.services.embedding import SentenceTransformerEmbeddingProvider
from career_match_agent.services.job_evaluator import OllamaJobReportGenerator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Benchmark CareerMatch filtering, ranking and report grounding."))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--with-llm", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_arguments()
    settings = get_settings()
    dataset = (JobMatchingBenchmarkDataset.model_validate_json(args.dataset.read_text(encoding="utf-8")))
    embedding_provider = (SentenceTransformerEmbeddingProvider(model_name=(settings.embedding_model), device=settings.embedding_device, batch_size=(settings.embedding_batch_size)))
    report_generator = None

    if args.with_llm:
        report_generator = (OllamaJobReportGenerator(base_url=(settings.ollama_base_url), model_name=(settings.job_evaluation_model), timeout_seconds=(settings.job_evaluation_timeout_seconds)))

    runner = JobMatchingBenchmarkRunner(embedding_provider=(embedding_provider), report_generator=report_generator, maximum_evaluation_jobs=(settings.maximum_evaluation_jobs))
    results = []

    for (configuration_name, configuration) in (create_ranking_ablation_configurations().items()):
        result = await runner.run(dataset=dataset, configuration_name=(configuration_name), ranking_configuration=(configuration))
        results.append(result.model_dump(mode="json"))
        print(f"\n === {configuration_name} ===")
        print(f"Filtering F1: {result.filtering.f1:.3f}",)
        for metric in result.ranking.at_k:
            print(f"P@{metric.k}: "
                  f"{metric.precision:.3f} | "
                  f"R@{metric.k}: "
                  f"{metric.recall:.3f} | "
                  f"nDCG@{metric.k}: "
                  f"{metric.ndcg:.3f}")
        print("MRR:",f"{result.ranking.mean_reciprocal_rank:.3f}")
        print(f"Ranking latency: {result.latency.ranking_ms:.1f} ms")

    if args.output is not None:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8",)


if __name__ == "__main__":
    asyncio.run(main())
