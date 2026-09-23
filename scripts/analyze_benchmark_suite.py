import argparse
import json
from itertools import product
from math import ceil, floor
from pathlib import Path
from statistics import mean

CONFIGURATION_PAIRS = (("hybrid_default", "deterministic_only"), ("hybrid_default", "semantic_only"))
CUTOFFS = (5, 10)


def percentile(sorted_values: list[float], proportion: float) -> float:
    """Linearly interpolated percentile of nonempty sorted values."""
    if not sorted_values or not 0 <= proportion <= 1:
        raise ValueError("Expected nonempty sorted values and percentile in [0, 1].")

    index = (len(sorted_values) - 1) * proportion
    left, right = floor(index), ceil(index)
    fraction = index - left
    return sorted_values[left] * (1 - fraction) + sorted_values[right] * fraction


def exact_paired_bootstrap_interval(differences: list[float]) -> tuple[float, float]:
    """Exact percentile interval over all ordered scenario resamples."""
    if not differences:
        raise ValueError("At least one paired scenario is required.")
    # Four scenarios produce 256 resamples; cap exhaustive enumeration to avoid
    # exponential growth if this script is reused with larger suites.
    if len(differences) > 7:
        raise ValueError("Exact bootstrap supports at most seven scenarios.")

    distribution = sorted(mean(resample) for resample in product(differences, repeat=len(differences)))
    return percentile(distribution, 0.025), percentile(distribution, 0.975)


def analyze(suite: dict) -> dict:
    """Require paired coverage and calculate each configuration contrast."""
    if suite.get("split") != "development":
        raise ValueError("Only the development split should be analyzed here.")

    indexed: dict[tuple[str, str], dict[int, float]] = {}
    scenario_names: set[str] = set()
    configuration_names: set[str] = set()

    for result in suite["results"]:
        scenario = result["dataset_name"]
        config = result["configuration_name"]
        pair = (scenario, config)
        if pair in indexed:
            raise ValueError(f"Duplicate result: {pair}")
        metrics = {item["k"]: item["ndcg"] for item in result["ranking"]["at_k"]}
        if not set(CUTOFFS).issubset(metrics):
            raise ValueError(f"Missing ranking cutoff for {pair}")
        indexed[pair] = metrics
        scenario_names.add(scenario)
        configuration_names.add(config)

    required_configs = {name for pair in CONFIGURATION_PAIRS for name in pair}
    if not required_configs.issubset(configuration_names):
        raise ValueError("Missing one or more required ranking configurations.")
    scenarios = sorted(scenario_names)
    if len(scenarios) != 4:
        raise ValueError(f"Expected four development scenarios; found {len(scenarios)}.")
    expected_pairs = {(scenario, config) for scenario in scenarios for config in required_configs}
    if set(indexed) != expected_pairs:
        raise ValueError(f"Incomplete paired coverage. Missing: {expected_pairs - set(indexed)}; unexpected: {set(indexed) - expected_pairs}")

    comparisons = []
    for left, right in CONFIGURATION_PAIRS:
        for k in CUTOFFS:
            per_scenario = {scenario: indexed[(scenario, left)][k] - indexed[(scenario, right)][k] for scenario in scenarios}
            low, high = exact_paired_bootstrap_interval(list(per_scenario.values()))
            comparisons.append({"contrast": f"{left} - {right}",
                                "metric": f"nDCG@{k}",
                                "scenario_differences": per_scenario,
                                "mean_paired_difference": mean(per_scenario.values()),
                                "exploratory_95pct_bootstrap_interval": [low, high]})

    return {"suite_name": suite["suite_name"],
            "suite_version": suite["suite_version"],
            "scenario_count": len(scenarios),
            "method": "Exact scenario bootstrap (all 4^4 ordered resamples)",
            "caution": ("Exploratory intervals from four synthetic calibrated scenarios; not evidence of real-world statistical significance."),
            "comparisons": comparisons}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmark_results/development_suite_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/development_suite_v1_paired.json"))
    args = parser.parse_args()
    suite = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze(suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for comparison in report["comparisons"]:
        low, high = comparison["exploratory_95pct_bootstrap_interval"]
        print(f"\n{comparison['contrast']} — {comparison['metric']}")
        for scenario, difference in comparison["scenario_differences"].items():
            print(f"  {scenario:<19} {difference:+.4f}")
        print(f"  Mean paired difference: {comparison['mean_paired_difference']:+.4f}")
        print(f"  Exploratory 95% bootstrap interval: [{low:+.4f}, {high:+.4f}]")

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
