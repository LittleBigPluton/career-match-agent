import argparse
import asyncio
import json
import hashlib
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


DATASETS_BY_SPLIT = {"development": ("ml_junior", "backend_mid", "data_junior", "career_switcher"),
                     "holdout": ("nlp_junior", "ml_platform_junior")}

HOLDOUT_CHECKSUMS = {"nlp_junior": ("2378d70ba57e69eac0ccdf8510e199ac9f5fe219d28ddbb1e0ef77b342c1b4bc"),
                     "ml_platform_junior": ("2591e130f467cea0245f7fb28523ce6f8bda3dc2f4d97da31bb4d6748962b29d")}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete development benchmark suite.")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_suite(split: str) -> JobMatchingBenchmarkSuite:
    data_directory = Path("data/benchmarks") / split
    dataset_names = DATASETS_BY_SPLIT[split]

    datasets = []

    for name in dataset_names:
        path = data_directory / f"{name}.json"
        raw_data = path.read_bytes()

        if split == "holdout":
            actual_checksum = hashlib.sha256(raw_data).hexdigest()
            expected_checksum = HOLDOUT_CHECKSUMS[name]

            if actual_checksum != expected_checksum:
                raise ValueError(f"Holdout checksum mismatch for {name}. Do not evaluate modified holdout data.")

        dataset = JobMatchingBenchmarkDataset.model_validate_json(raw_data)
        if dataset.name != name:
            raise ValueError(f"Expected dataset {name}, got {dataset.name}")

        if dataset.version != "1.0.0":
            raise ValueError(f"{name} must be frozen at version 1.0.0")

        datasets.append(dataset)

    return JobMatchingBenchmarkSuite(name=f"career_match_{split}",
                                     version="1.0.0",
                                     split=split,
                                     description=f"Frozen {split} benchmark scenarios.",
                                     datasets=datasets)


async def main() -> None:
    args = parse_arguments()
    settings = get_settings()
    suite = load_suite(args.split)
    embedding_provider = SentenceTransformerEmbeddingProvider(model_name=settings.embedding_model,
                                                              device=settings.embedding_device,
                                                              batch_size=settings.embedding_batch_size)

    benchmark_runner = JobMatchingBenchmarkRunner(embedding_provider=embedding_provider,
                                                  report_generator=None,
                                                  maximum_evaluation_jobs=settings.maximum_evaluation_jobs)

    suite_runner = JobMatchingBenchmarkSuiteRunner(benchmark_runner=benchmark_runner)
    suite_result = await suite_runner.run(suite=suite, ranking_configurations=create_ranking_ablation_configurations())
    expected_count = (len(suite.datasets) * len(create_ranking_ablation_configurations()))
    if len(suite_result.results) != expected_count:
        raise ValueError("Incomplete development-suite results.")

    output_path = args.output or Path(f"benchmark_results/{args.split}_suite_v1.json")
    args.output.write_text(json.dumps(suite_result.model_dump(mode="json"), indent=2), encoding="utf-8")
    summary = summarize_benchmark_suite(suite_result)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"\nAggregated summary saved to: {summary_path}")
    for result in suite_result.results:
        print(f"\n{result.dataset_name} — {result.configuration_name}")
        print(f"Filtering F1: {result.filtering.f1:.3f}")
        print(f"Reason-code F1: {result.reason_codes.f1:.3f}")
        for metric in result.ranking.at_k:
            print(f"nDCG@{metric.k}: {metric.ndcg:.3f}")

    print(f"\nSaved {len(suite_result.results)} results")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
