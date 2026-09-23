import argparse
import asyncio
import json
from pathlib import Path

from career_match_agent.services.embedding import SentenceTransformerEmbeddingProvider
from career_match_agent.services.benchmark_aggregation import summarize_benchmark_suite
from career_match_agent.core.config import get_settings
from career_match_agent.models.benchmark import (
    JobMatchingBenchmarkDataset,
    JobMatchingBenchmarkSuite
)
from career_match_agent.services.benchmark_runner import (
    JobMatchingBenchmarkRunner,
    JobMatchingBenchmarkSuiteRunner,
    create_ranking_ablation_configurations
)


DEVELOPMENT_DATASETS = (
    "ml_junior",
    "backend_mid",
    "data_junior",
    "career_switcher",
)

DATA_DIRECTORY = Path("data/benchmarks/development")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete development benchmark suite."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/development_suite_v1.json"),
    )

    return parser.parse_args()


def load_development_suite() -> JobMatchingBenchmarkSuite:
    datasets = []

    for name in DEVELOPMENT_DATASETS:
        path = DATA_DIRECTORY / f"{name}.json"

        dataset = JobMatchingBenchmarkDataset.model_validate_json(
            path.read_text(encoding="utf-8")
        )

        if dataset.name != name:
            raise ValueError(
                f"Expected dataset {name}, got {dataset.name}"
            )

        if dataset.version != "1.0.0":
            raise ValueError(
                f"{name} is not frozen at version 1.0.0"
            )

        datasets.append(dataset)

    return JobMatchingBenchmarkSuite(
        name="career_match_development",
        version="1.0.0",
        split="development",
        description="Four frozen synthetic development scenarios.",
        datasets=datasets,
    )


async def main() -> None:
    args = parse_arguments()
    settings = get_settings()

    suite = load_development_suite()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    benchmark_runner = JobMatchingBenchmarkRunner(
        embedding_provider=embedding_provider,
        report_generator=None,
        maximum_evaluation_jobs=settings.maximum_evaluation_jobs,
    )

    suite_runner = JobMatchingBenchmarkSuiteRunner(
        benchmark_runner=benchmark_runner
    )

    suite_result = await suite_runner.run(
        suite=suite,
        ranking_configurations=create_ranking_ablation_configurations(),
    )

    expected_count = (
        len(suite.datasets)
        * len(create_ranking_ablation_configurations())
    )

    if len(suite_result.results) != expected_count:
        raise ValueError("Incomplete development-suite results.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        json.dumps(
            suite_result.model_dump(mode="json"),
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = summarize_benchmark_suite(suite_result)

    summary_path = args.output.with_name(
        f"{args.output.stem}_summary.json"
    )
    
    summary_path.write_text(
        json.dumps(
            summary.model_dump(mode="json"),
            indent=2,
        ),
        encoding="utf-8",
    )
    
    print(f"\nAggregated summary saved to: {summary_path}")

    for result in suite_result.results:
        print(
            f"\n{result.dataset_name} — "
            f"{result.configuration_name}"
        )

        print(f"Filtering F1: {result.filtering.f1:.3f}")
        print(f"Reason-code F1: {result.reason_codes.f1:.3f}")

        for metric in result.ranking.at_k:
            print(
                f"nDCG@{metric.k}: {metric.ndcg:.3f}"
            )

    print(f"\nSaved {len(suite_result.results)} results")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
