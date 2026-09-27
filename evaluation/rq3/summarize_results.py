#!/usr/bin/env python3
"""Summarize Full and available RQ3 ablation results."""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


CONFIGURATIONS = ("Full", "Global-only", "Local-only", "Random-scenario Local")
OPTIONAL_CONFIGURATIONS = ("Random-scenario Local",)
CONFIG_METHODS = {
    "Full": "vlad_fuzz_cmd1",
    "Global-only": "global_only",
    "Local-only": "local_only",
    "Random-scenario Local": "random_scenario_local",
}
CONFIG_ROOTS = {
    "Full": ("rq2", "runs", "vladfuzz"),
    "Global-only": ("rq3", "runs", "global_only"),
    "Local-only": ("rq3", "runs", "local_only"),
    "Random-scenario Local": ("rq3", "runs", "random_scenario_local"),
}
CORE_FAILURE_CATEGORIES = (
    "Collision",
    "Lane Invasion",
    "Out of Bounds",
    "Timeout/Stuck",
    "Speed Limit Exceeded",
    "Maintain Distance Failed",
)
EXTRA_FAILURE_CATEGORIES = (
    "Other Oracle Failure",
    "Execution Error",
    "Unhandled Exception",
    "Unknown",
)
FAILURE_CATEGORIES = CORE_FAILURE_CATEGORIES + EXTRA_FAILURE_CATEGORIES


@dataclass(frozen=True)
class RunRecord:
    configuration: str
    model: str
    manifest_id: str
    run_dir: Path
    sort_time: float
    executions: int
    failures: int
    failure_counts: Dict[str, int]
    hourly_checkpoints: List[Dict]
    metadata: Dict

    @property
    def failure_rate(self) -> float:
        return self.failures / self.executions if self.executions else 0.0

    @property
    def category_coverage(self) -> int:
        return sum(self.failure_counts.get(category, 0) > 0 for category in FAILURE_CATEGORIES)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_manifest_ids(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            manifest_id = row.get("id")
            if not manifest_id:
                raise ValueError(f"Manifest row {line_number} has no id: {path}")
            ids.append(str(manifest_id))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Manifest contains duplicate ids: {path}")
    return ids


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _timestamp(metadata: Dict, run_dir: Path) -> float:
    for key in ("start_time", "end_time"):
        value = metadata.get(key)
        if value:
            try:
                return datetime.fromisoformat(str(value)).timestamp()
            except ValueError:
                pass
    return run_dir.stat().st_mtime


def _normalize_category(value: str) -> str:
    compact = " ".join(str(value).strip().replace("_", " ").replace("-", " ").lower().split())
    if "collision" in compact:
        return "Collision"
    if "lane invasion" in compact:
        return "Lane Invasion"
    if "out of bounds" in compact:
        return "Out of Bounds"
    if "timeout" in compact or "stuck" in compact or "target not reached" in compact:
        return "Timeout/Stuck"
    if "speed limit exceeded" in compact or ("max speed" in compact and "limit" in compact):
        return "Speed Limit Exceeded"
    if "maintain distance failed" in compact or ("min distance" in compact and "target" in compact):
        return "Maintain Distance Failed"
    if "execution error" in compact or "carla execution error" in compact:
        return "Execution Error"
    if "unhandled exception" in compact:
        return "Unhandled Exception"
    if compact == "other" or "other oracle" in compact:
        return "Other Oracle Failure"
    return "Unknown"


def _reason_categories(reason: Optional[str]) -> Set[str]:
    if not reason:
        return {"Unknown"}
    parts = str(reason).replace(";", ",").split(",")
    categories = {_normalize_category(part) for part in parts if part.strip()}
    return categories or {"Unknown"}


def _result_failure_categories(result: Dict) -> Set[str]:
    execution = result.get("execution_result") or {}
    oracle = execution.get("oracle_events") or {}
    triggered = oracle.get("triggered_checks") or []
    categories: Set[str] = set()
    for check in triggered:
        categories.update(_reason_categories(str(check)))
    if not categories or categories == {"Unknown"}:
        categories = _reason_categories(
            result.get("semantic_failure_reason") or result.get("failure_reason")
        )
    return categories


def _failure_breakdown(run_dir: Path, metadata: Dict) -> Dict[str, int]:
    counts = Counter()
    result_paths = sorted((run_dir / "failures").glob("*/result.json"))
    if result_paths:
        for result_path in result_paths:
            try:
                result = _load_json(result_path)
            except (OSError, json.JSONDecodeError):
                counts["Unknown"] += 1
                continue
            for category in _result_failure_categories(result):
                counts[category] += 1
    else:
        for raw_category, count in (metadata.get("failure_analysis") or {}).items():
            counts[_normalize_category(raw_category)] += _safe_int(count)
    return {category: counts.get(category, 0) for category in FAILURE_CATEGORIES}


def _configured_budget_seconds(metadata: Dict) -> Optional[float]:
    config = metadata.get("configuration") or {}
    value = (
        config.get("time_budget_seconds")
        or config.get("target_duration_seconds")
        or metadata.get("target_duration_seconds")
    )
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

def _collect_runs(
    results_root: Path,
    manifest_ids: Set[str],
    expected_budget_seconds: Optional[float],
) -> Tuple[List[RunRecord], List[Dict]]:
    runs: List[RunRecord] = []
    warnings: List[Dict] = []
    for configuration in CONFIGURATIONS:
        root = results_root.joinpath(*CONFIG_ROOTS[configuration])
        if not root.exists():
            continue
        for metadata_path in sorted(root.glob("*/*/metadata.json")):
            if "infrastructure_failures" in metadata_path.parts:
                continue
            try:
                metadata = _load_json(metadata_path)
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append({
                    "warning_type": "invalid_metadata",
                    "model": "",
                    "scenario_seed": "",
                    "configuration": configuration,
                    "details": f"{metadata_path}: {exc}",
                })
                continue
            config = metadata.get("configuration") or {}
            if config.get("method") != CONFIG_METHODS[configuration]:
                continue
            manifest_id = str(config.get("manifest_id") or "")
            if manifest_id not in manifest_ids:
                continue
            model = str(config.get("vla_model") or metadata_path.parent.parent.name)
            run_dir = metadata_path.parent
            configured_budget = _configured_budget_seconds(metadata)
            if (
                expected_budget_seconds is not None
                and configured_budget is not None
                and not math.isclose(configured_budget, expected_budget_seconds, rel_tol=0.0, abs_tol=1.0)
            ):
                warnings.append({
                    "warning_type": "budget_mismatch_ignored",
                    "model": model,
                    "scenario_seed": manifest_id,
                    "configuration": configuration,
                    "details": (
                        f"configured_budget_seconds={configured_budget}; "
                        f"expected_budget_seconds={expected_budget_seconds}; run={run_dir}"
                    ),
                })
                continue
            runs.append(RunRecord(
                configuration=configuration,
                model=model,
                manifest_id=manifest_id,
                run_dir=run_dir,
                sort_time=_timestamp(metadata, run_dir),
                executions=_safe_int(metadata.get("total_simulations_executed")),
                failures=_safe_int(metadata.get("total_failures_detected")),
                failure_counts=_failure_breakdown(run_dir, metadata),
                hourly_checkpoints=list(metadata.get("hourly_checkpoints") or []),
                metadata=metadata,
            ))
    return runs, warnings


def _latest_runs(runs: Iterable[RunRecord], warnings: List[Dict]) -> List[RunRecord]:
    latest: Dict[Tuple[str, str, str], RunRecord] = {}
    for run in runs:
        key = (run.model, run.manifest_id, run.configuration)
        current = latest.get(key)
        if current is None or run.sort_time >= current.sort_time:
            ignored, kept = current, run
            latest[key] = run
        else:
            ignored, kept = run, current
        if ignored is not None:
            warnings.append({
                "warning_type": "duplicate_run_ignored",
                "model": run.model,
                "scenario_seed": run.manifest_id,
                "configuration": run.configuration,
                "details": f"ignored={ignored.run_dir}; kept={kept.run_dir}",
            })
    return list(latest.values())


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _wilcoxon_signed_rank(differences: Sequence[float]) -> Tuple[float, float]:
    nonzero = [(abs(value), 1 if value > 0 else -1) for value in differences if value != 0]
    if not nonzero:
        return 1.0, 0.0
    rank_by_value: Dict[float, float] = {}
    next_rank = 1
    for value in sorted({value for value, _ in nonzero}):
        count = sum(candidate == value for candidate, _ in nonzero)
        rank_by_value[value] = (next_rank + next_rank + count - 1) / 2.0
        next_rank += count
    ranks = [rank_by_value[value] for value, _ in nonzero]
    observed_positive = sum(rank for rank, (_, sign) in zip(ranks, nonzero) if sign > 0)
    total = sum(ranks)
    observed_extreme = min(observed_positive, total - observed_positive)
    extreme_count = 0
    for choices in itertools.product((False, True), repeat=len(ranks)):
        positive = sum(rank for rank, selected in zip(ranks, choices) if selected)
        if min(positive, total - positive) <= observed_extreme + 1e-12:
            extreme_count += 1
    p_value = min(1.0, extreme_count / (2 ** len(ranks)))
    rank_biserial = (2 * observed_positive - total) / total
    return p_value, rank_biserial


def _write_csv(path: Path, rows: List[Dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: float) -> float:
    return round(float(value), 6)


def _per_seed_rows(runs: Sequence[RunRecord], manifest_order: Dict[str, int]) -> List[Dict]:
    return [{
        "scenario_seed": run.manifest_id,
        "configuration": run.configuration,
        "executions": run.executions,
        "failures": run.failures,
        "failure_rate": _format_number(run.failure_rate),
        "failure_category_coverage": run.category_coverage,
        "run_dir": str(run.run_dir),
    } for run in sorted(
        runs,
        key=lambda item: (
            manifest_order[item.manifest_id],
            CONFIGURATIONS.index(item.configuration),
        ),
    )]


def _aggregate_rows(runs: Sequence[RunRecord]) -> List[Dict]:
    rows = []
    for configuration in CONFIGURATIONS:
        selected = [run for run in runs if run.configuration == configuration]
        if not selected:
            continue
        executions = sum(run.executions for run in selected)
        failures = sum(run.failures for run in selected)
        rates = [run.failure_rate for run in selected]
        rows.append({
            "configuration": configuration,
            "seed_count": len(selected),
            "executions": executions,
            "failures": failures,
            "pooled_failure_rate": _format_number(failures / executions if executions else 0.0),
            "median_failure_rate": _format_number(statistics.median(rates)),
            "failure_rate_q1": _format_number(_quantile(rates, 0.25)),
            "failure_rate_q3": _format_number(_quantile(rates, 0.75)),
            "failure_category_coverage": len({
                category
                for run in selected
                for category, count in run.failure_counts.items()
                if count > 0
            }),
            "mean_seed_category_coverage": _format_number(
                statistics.mean(run.category_coverage for run in selected)
            ),
        })
    return rows


def _category_rows(runs: Sequence[RunRecord]) -> List[Dict]:
    rows = []
    for configuration in CONFIGURATIONS:
        selected = [run for run in runs if run.configuration == configuration]
        if not selected:
            continue
        failures = sum(run.failures for run in selected)
        for category in FAILURE_CATEGORIES:
            count = sum(run.failure_counts.get(category, 0) for run in selected)
            rows.append({
                "configuration": configuration,
                "failure_category": category,
                "count": count,
                "failure_share": _format_number(count / failures if failures else 0.0),
                "seeds_with_category": sum(run.failure_counts.get(category, 0) > 0 for run in selected),
            })
    return rows


def _hourly_rows(runs: Sequence[RunRecord]) -> List[Dict]:
    totals: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0, 0])
    for run in runs:
        for checkpoint in run.hourly_checkpoints:
            hour = _safe_int(checkpoint.get("hour"))
            if hour <= 0:
                continue
            bucket = totals[(run.configuration, hour)]
            bucket[0] += _safe_int(checkpoint.get("executions"))
            bucket[1] += _safe_int(checkpoint.get("failures"))
            bucket[2] += 1
    rows = []
    for configuration in CONFIGURATIONS:
        for (candidate, hour), (executions, failures, seeds) in sorted(totals.items(), key=lambda item: item[0][1]):
            if candidate != configuration:
                continue
            rows.append({
                "configuration": configuration,
                "hour": hour,
                "seed_count": seeds,
                "executions": executions,
                "failures": failures,
                "failure_rate": _format_number(failures / executions if executions else 0.0),
            })
    return rows


def _paired_rows(runs: Sequence[RunRecord]) -> List[Dict]:
    by_key = {(run.manifest_id, run.configuration): run for run in runs}
    rows = []
    comparisons = [
        configuration
        for configuration in CONFIGURATIONS[1:]
        if any(run.configuration == configuration for run in runs)
    ]
    for comparison in comparisons:
        paired = [
            (by_key[(manifest_id, "Full")], by_key[(manifest_id, comparison)])
            for manifest_id in sorted({run.manifest_id for run in runs})
            if (manifest_id, "Full") in by_key and (manifest_id, comparison) in by_key
        ]
        for metric, getter in (
            ("failures", lambda run: float(run.failures)),
            ("failure_rate", lambda run: run.failure_rate),
        ):
            differences = [getter(full) - getter(ablation) for full, ablation in paired]
            p_value, effect = _wilcoxon_signed_rank(differences)
            rows.append({
                "comparison": f"Full - {comparison}",
                "metric": metric,
                "paired_seeds": len(differences),
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
                "median_difference": _format_number(statistics.median(differences) if differences else 0.0),
                "wilcoxon_p_value": _format_number(p_value),
                "rank_biserial_effect": _format_number(effect),
                "holm_adjusted_p_value": 0.0,
            })
    for metric in ("failures", "failure_rate"):
        metric_rows = [row for row in rows if row["metric"] == metric]
        ordered = sorted(metric_rows, key=lambda row: row["wilcoxon_p_value"])
        running = 0.0
        for index, row in enumerate(ordered):
            adjusted = min(1.0, row["wilcoxon_p_value"] * (len(ordered) - index))
            running = max(running, adjusted)
            row["holm_adjusted_p_value"] = _format_number(running)
    return rows


def _write_markdown(path: Path, model: str, aggregates: List[Dict], categories: List[Dict], paired: List[Dict]) -> None:
    lines = [
        f"# RQ3 Ablation Summary: {model}",
        "",
        "## Aggregate Results",
        "",
        "| Configuration | Executions | Failures | Failure Rate | Median Rate | Error Types |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['configuration']} | {row['executions']} | {row['failures']} | "
            f"{100 * row['pooled_failure_rate']:.2f}% | {100 * row['median_failure_rate']:.2f}% | "
            f"{row['failure_category_coverage']} |"
        )
    lines.extend([
        "",
        "## Failure Categories",
        "",
        "| Configuration | Category | Count | Failure Share | Seeds |",
        "|---|---|---:|---:|---:|",
    ])
    for row in categories:
        if row["count"]:
            lines.append(
                f"| {row['configuration']} | {row['failure_category']} | {row['count']} | "
                f"{100 * row['failure_share']:.2f}% | {row['seeds_with_category']} |"
            )
    lines.extend([
        "",
        "## Paired Statistics",
        "",
        "| Comparison | Metric | Pairs | W/T/L | Median Difference | Wilcoxon p | Holm p | Effect |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired:
        difference = row["median_difference"] * 100 if row["metric"] == "failure_rate" else row["median_difference"]
        suffix = " pp" if row["metric"] == "failure_rate" else ""
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['paired_seeds']} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {difference:.2f}{suffix} | "
            f"{row['wilcoxon_p_value']:.4f} | {row['holm_adjusted_p_value']:.4f} | "
            f"{row['rank_biserial_effect']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_rq3(
    results_root: Path,
    manifest_path: Path,
    output_dir: Path,
    expected_budget_seconds: Optional[float] = 14400.0,
) -> Dict[str, List[Dict]]:
    manifest_ids = _load_manifest_ids(manifest_path)
    manifest_order = {manifest_id: index for index, manifest_id in enumerate(manifest_ids)}
    collected, warnings = _collect_runs(results_root, set(manifest_ids), expected_budget_seconds)
    runs = _latest_runs(collected, warnings)
    models = sorted({run.model for run in runs})
    active_configurations = [
        configuration
        for configuration in CONFIGURATIONS
        if (
            configuration not in OPTIONAL_CONFIGURATIONS
            or results_root.joinpath(*CONFIG_ROOTS[configuration]).exists()
        )
    ]
    for model in models:
        for manifest_id in manifest_ids:
            for configuration in active_configurations:
                if not any(
                    run.model == model and run.manifest_id == manifest_id and run.configuration == configuration
                    for run in runs
                ):
                    warnings.append({
                        "warning_type": "missing_run",
                        "model": model,
                        "scenario_seed": manifest_id,
                        "configuration": configuration,
                        "details": "No matching valid metadata found",
                    })

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, List[Dict]] = {}
    for model in models:
        model_runs = [run for run in runs if run.model == model]
        per_seed = _per_seed_rows(model_runs, manifest_order)
        aggregate = _aggregate_rows(model_runs)
        categories = _category_rows(model_runs)
        hourly = _hourly_rows(model_runs)
        paired = _paired_rows(model_runs)
        outputs[model] = per_seed
        prefix = f"rq3_{model}"
        _write_csv(output_dir / f"{prefix}_per_seed.csv", per_seed, (
            "scenario_seed", "configuration", "executions", "failures", "failure_rate",
            "failure_category_coverage", "run_dir",
        ))
        _write_csv(output_dir / f"{prefix}_aggregate.csv", aggregate, (
            "configuration", "seed_count", "executions", "failures", "pooled_failure_rate",
            "median_failure_rate", "failure_rate_q1", "failure_rate_q3",
            "failure_category_coverage", "mean_seed_category_coverage",
        ))
        _write_csv(output_dir / f"{prefix}_failure_categories.csv", categories, (
            "configuration", "failure_category", "count", "failure_share", "seeds_with_category",
        ))
        _write_csv(output_dir / f"{prefix}_hourly.csv", hourly, (
            "configuration", "hour", "seed_count", "executions", "failures", "failure_rate",
        ))
        _write_csv(output_dir / f"{prefix}_paired_statistics.csv", paired, (
            "comparison", "metric", "paired_seeds", "wins", "ties", "losses",
            "median_difference", "wilcoxon_p_value", "holm_adjusted_p_value", "rank_biserial_effect",
        ))
        _write_markdown(output_dir / f"{prefix}_summary.md", model, aggregate, categories, paired)

    _write_csv(output_dir / "warnings.csv", warnings, (
        "warning_type", "model", "scenario_seed", "configuration", "details",
    ))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RQ3 VLAD-Fuzz ablation results.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--manifest", default="configs/rq3_manifest.jsonl")
    parser.add_argument("--output-dir", default="results/rq3/summary")
    parser.add_argument("--expected-budget-minutes", type=float, default=240.0, help="Ignore runs with a different declared budget; use a negative value to disable filtering.")
    args = parser.parse_args()
    expected_budget = None if args.expected_budget_minutes < 0 else args.expected_budget_minutes * 60.0
    outputs = summarize_rq3(Path(args.results_root), Path(args.manifest), Path(args.output_dir), expected_budget_seconds=expected_budget)
    print(f"Built RQ3 summaries for {len(outputs)} model(s): {', '.join(sorted(outputs)) or 'none'}")
    print(f"Output directory: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
