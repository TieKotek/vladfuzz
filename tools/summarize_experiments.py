import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from vladfuzz_runtime.failure_analysis import FAILURE_CATEGORIES


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _row_from_metadata(run_dir: Path, method: str, metadata: Dict) -> Dict:
    config = metadata.get("configuration", {})
    simulations = int(metadata.get("total_simulations_executed", 0))
    failures = int(metadata.get("total_failures_detected", 0))
    failure_analysis = metadata.get("failure_analysis", {})
    duration = metadata.get("total_duration_seconds", metadata.get("actual_duration_seconds", 0.0))

    row = {
        "run_dir": str(run_dir),
        "method": config.get("method", method),
        "model": config.get("vla_model", "unknown"),
        "manifest_id": config.get("manifest_id"),
        "instruction_source": config.get("instruction_source"),
        "max_simulations": config.get("max_simulations"),
        "random_seed": config.get("random_seed"),
        "mutation_depth": config.get("mutation_depth"),
        "operators": ",".join(config.get("operators") or []),
        "static_scenario": config.get("static_scenario"),
        "seed_scenario": config.get("seed_scenario"),
        "resolved_static_scenario": config.get("resolved_static_scenario", config.get("static_scenario")),
        "resolved_seed_scenario": config.get("resolved_seed_scenario", config.get("seed_scenario")),
        "duration_seconds": round(float(duration), 2),
        "simulations": simulations,
        "failures": failures,
        "failure_rate": _rate(failures, simulations),
        "failures_per_100_simulations": round(_rate(failures, simulations) * 100.0, 2),
    }

    for category in FAILURE_CATEGORIES:
        row[category] = int(failure_analysis.get(category, 0))

    other_count = 0
    known = set(FAILURE_CATEGORIES)
    for category, count in failure_analysis.items():
        if category not in known:
            other_count += int(count)
    row["Other"] = other_count
    return row


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    groups: Dict[tuple, Dict] = {}
    for row in rows:
        key = (
            row.get("method"),
            row.get("model"),
            row.get("instruction_source"),
            row.get("resolved_static_scenario"),
        )
        if key not in groups:
            groups[key] = {
                "method": row.get("method"),
                "model": row.get("model"),
                "instruction_source": row.get("instruction_source"),
                "static_scenario": row.get("resolved_static_scenario"),
                "runs": 0,
                "duration_seconds": 0.0,
                "simulations": 0,
                "failures": 0,
            }
            for category in FAILURE_CATEGORIES:
                groups[key][category] = 0
            groups[key]["Other"] = 0

        group = groups[key]
        group["runs"] += 1
        group["duration_seconds"] += float(row.get("duration_seconds", 0.0))
        group["simulations"] += int(row.get("simulations", 0))
        group["failures"] += int(row.get("failures", 0))
        for category in FAILURE_CATEGORIES:
            group[category] += int(row.get(category, 0))
        group["Other"] += int(row.get("Other", 0))

    aggregated = []
    for group in groups.values():
        group["duration_seconds"] = round(group["duration_seconds"], 2)
        group["failure_rate"] = _rate(group["failures"], group["simulations"])
        group["failures_per_100_simulations"] = round(group["failure_rate"] * 100.0, 2)
        aggregated.append(group)
    return sorted(
        aggregated,
        key=lambda row: (
            str(row.get("static_scenario")),
            str(row.get("model")),
            str(row.get("instruction_source")),
            str(row.get("method")),
        ),
    )


def collect_rows(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    experiment_root = root / "results"
    if experiment_root.exists():
        for metadata_path in sorted(experiment_root.rglob("metadata.json")):
            run_dir = metadata_path.parent
            metadata = _load_json(metadata_path)
            configuration = metadata.get("configuration", {})
            if not configuration.get("vla_model") or not configuration.get("method"):
                continue
            method = configuration["method"]
            rows.append(_row_from_metadata(run_dir, method, metadata))

    for run_dir in sorted(root.glob("optimization_*")):
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            rows.append(_row_from_metadata(run_dir, "vlad-fuzz", _load_json(metadata_path)))

        baseline_metadata_path = run_dir / "baseline" / "metadata.json"
        if baseline_metadata_path.exists():
            rows.append(_row_from_metadata(run_dir / "baseline", "baseline", _load_json(baseline_metadata_path)))

    return rows


def write_csv(rows: Iterable[Dict], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: List[Dict], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append("" if value is None or value == "None" else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(rows: List[Dict], aggregated_rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Experiment Summary\n\n"
    if rows:
        detail_columns = [
            "method",
            "model",
            "manifest_id",
            "instruction_source",
            "simulations",
            "failures",
            "failure_rate",
            "Collision",
            "Timeout/Stuck",
            "Speed Limit Exceeded",
            "Maintain Distance Failed",
        ]
        aggregate_columns = [
            "method",
            "model",
            "instruction_source",
            "static_scenario",
            "runs",
            "simulations",
            "failures",
            "failure_rate",
            "failures_per_100_simulations",
        ]
        text += "## Aggregated\n\n"
        text += _markdown_table(aggregated_rows, aggregate_columns)
        text += "\n## Runs\n\n"
        text += _markdown_table(rows, detail_columns)
    else:
        text += "No optimization_* metadata files found.\n"
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize optimization_* runs and matched baseline metadata.")
    parser.add_argument("--root", default=".", help="Repository root or directory containing optimization_* runs.")
    parser.add_argument("--csv", default="paper/experiment_summary.csv", help="Output CSV path.")
    parser.add_argument("--aggregate-csv", default="paper/experiment_summary_aggregated.csv", help="Output aggregated CSV path.")
    parser.add_argument("--md", default="paper/experiment_summary.md", help="Output Markdown path.")
    args = parser.parse_args()

    rows = collect_rows(Path(args.root))
    aggregated_rows = aggregate_rows(rows)
    write_csv(rows, Path(args.csv))
    write_csv(aggregated_rows, Path(args.aggregate_csv))
    write_markdown(rows, aggregated_rows, Path(args.md))
    print(f"Collected {len(rows)} rows")
    print(f"Collected {len(aggregated_rows)} aggregated rows")
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote aggregate CSV: {args.aggregate_csv}")
    print(f"Wrote Markdown: {args.md}")


if __name__ == "__main__":
    main()
