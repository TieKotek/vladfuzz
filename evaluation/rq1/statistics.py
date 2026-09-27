from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Sequence, Tuple


def linear_weighted_kappa(first: Sequence[int], second: Sequence[int], categories: Sequence[int] = (0, 1, 2)) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Kappa requires two non-empty score sequences of equal length.")
    category_index = {value: index for index, value in enumerate(categories)}
    if any(value not in category_index for value in list(first) + list(second)):
        raise ValueError("Scores contain a category not included in categories.")
    denominator = max(len(categories) - 1, 1)
    observed_disagreement = sum(
        abs(category_index[a] - category_index[b]) / denominator
        for a, b in zip(first, second)
    ) / len(first)
    first_counts = {value: first.count(value) / len(first) for value in categories}
    second_counts = {value: second.count(value) / len(second) for value in categories}
    expected_disagreement = sum(
        first_counts[a] * second_counts[b] * abs(category_index[a] - category_index[b]) / denominator
        for a in categories
        for b in categories
    )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def exact_mcnemar(*, prior_only: int, no_prior_only: int) -> float:
    if prior_only < 0 or no_prior_only < 0:
        raise ValueError("Discordant counts must be non-negative.")
    discordant = prior_only + no_prior_only
    if discordant == 0:
        return 1.0
    lower = min(prior_only, no_prior_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _risk_difference(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        raise ValueError("At least one paired row is required.")
    return sum(bool(row["route_prior"]) - bool(row["no_prior"]) for row in rows) / len(rows)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_risk_difference(
    pairs: Sequence[Mapping[str, object]],
    *,
    iterations: int = 10000,
    random_seed: int = 0,
) -> Tuple[float, float, float]:
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in pairs:
        grouped.setdefault(str(row["scenario_name"]), []).append(row)
    if not grouped:
        raise ValueError("At least one paired row is required.")
    clusters = sorted(grouped)
    rng = random.Random(random_seed)
    estimates = []
    for _ in range(iterations):
        sampled_rows = []
        for cluster in rng.choices(clusters, k=len(clusters)):
            sampled_rows.extend(grouped[cluster])
        estimates.append(_risk_difference(sampled_rows))
    return _risk_difference(pairs), _percentile(estimates, 0.025), _percentile(estimates, 0.975)
