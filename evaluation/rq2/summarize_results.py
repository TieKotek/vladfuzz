import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vladfuzz_runtime.failure_analysis import FAILURE_CATEGORIES, categorize_failure_reason


DRIVEFUZZ_OUT_RE = re.compile(r"^out-drivefuzz(?:-selected)?-(?P<model>[^-]+)-(?P<suffix>.+)$")
DRIVEFUZZ_TIME_RE = re.compile(r"_(?P<timestamp>\d+(?:\.\d+)?)\.json$")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(rows: Iterable[Dict], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_failure_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value[: len(datetime.now().strftime(fmt))], fmt)
        except ValueError:
            continue
    return _parse_iso_timestamp(value)


def _command_id_from_method(method: Optional[str]) -> str:
    if not method:
        return ""
    match = re.search(r"_cmd(\d+)$", method)
    return match.group(1) if match else ""


def _increment_failure(stats: Dict[str, int], category: str) -> None:
    stats[category] = stats.get(category, 0) + 1


def _empty_failure_stats() -> Dict[str, int]:
    stats = {category: 0 for category in FAILURE_CATEGORIES}
    stats.update({
        "Red Light": 0,
        "Speeding": 0,
        "Unknown": 0,
    })
    return stats


def _local_failure_breakdown(failure_dir: Path) -> Tuple[Dict[str, int], Optional[float]]:
    stats = _empty_failure_stats()
    first_failure_epoch = None
    for result_path in sorted(failure_dir.glob("*/result.json")):
        try:
            result = _load_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue

        execution = result.get("execution_result") or {}
        oracle = execution.get("oracle_events") or {}
        triggered = oracle.get("triggered_checks") or []
        if triggered:
            for check in triggered:
                if check == "collision":
                    _increment_failure(stats, "Collision")
                elif check == "stuck":
                    _increment_failure(stats, "Timeout/Stuck")
                elif check == "lane_invasion":
                    _increment_failure(stats, "Lane Invasion")
                elif check == "speeding":
                    _increment_failure(stats, "Speeding")
                elif check == "timeout":
                    _increment_failure(stats, "Timeout/Stuck")
                elif check == "out_of_bounds":
                    _increment_failure(stats, "Out of Bounds")
                elif check == "other":
                    _increment_failure(stats, "Other Oracle Failure")
                else:
                    _increment_failure(stats, "Unknown")
        else:
            category = categorize_failure_reason(result.get("semantic_failure_reason"))
            _increment_failure(stats, category if category in stats else "Unknown")

        timestamp = _parse_failure_timestamp(result.get("timestamp"))
        if timestamp is not None:
            epoch = timestamp.timestamp()
            first_failure_epoch = epoch if first_failure_epoch is None else min(first_failure_epoch, epoch)

    return stats, first_failure_epoch


def _local_method_family(method: str) -> Optional[str]:
    if method.startswith("instruction_counterfactual"):
        return "instruction_counterfactual"
    if "vlad" in method or method == "full":
        return "vladfuzz"
    if "random" in method:
        return "random"
    return None


def _collect_local_runs(root: Path, method_filter: Optional[str] = None) -> List[Dict]:
    rows = []
    if not root.exists():
        return rows
    for metadata_path in sorted(root.rglob("metadata.json")):
        run_dir = metadata_path.parent
        metadata = _load_json(metadata_path)
        config = metadata.get("configuration", {})
        method = config.get("method", "")
        method_family = _local_method_family(method)
        if not config.get("vla_model") or method_family is None:
            continue
        if method_filter and method_filter not in method:
            continue

        simulations = int(metadata.get("total_simulations_executed", 0))
        failures = int(metadata.get("total_failures_detected", 0))
        start_time = _parse_iso_timestamp(metadata.get("start_time"))
        failure_stats, first_failure_epoch = _local_failure_breakdown(run_dir / "failures")
        if first_failure_epoch is not None and start_time is not None:
            ttf = max(0.0, first_failure_epoch - start_time.timestamp())
        else:
            ttf = None

        row = {
            "method": method,
            "method_family": method_family,
            "model": config.get("vla_model", "unknown"),
            "manifest_id": config.get("manifest_id"),
            "command_id": _command_id_from_method(method),
            "run_dir": str(run_dir),
            "duration_seconds": metadata.get("total_duration_seconds", metadata.get("actual_duration_seconds", "")),
            "budget_minutes": _budget_minutes(config, metadata),
            "executions": simulations,
            "failures": failures,
            "failure_rate": _rate(failures, simulations),
            "time_to_first_failure_seconds": "" if ttf is None else round(ttf, 2),
        }
        row.update(failure_stats)
        rows.append(row)
    return rows


def _budget_minutes(config: Dict, metadata: Optional[Dict] = None) -> str:
    metadata = metadata or {}
    seconds = (
        config.get("time_budget_seconds")
        or config.get("target_duration_seconds")
        or metadata.get("target_duration_seconds")
    )
    if seconds in (None, ""):
        return ""
    try:
        return round(float(seconds) / 60.0, 2)
    except (TypeError, ValueError):
        return ""


def _drivefuzz_timestamp(path: Path) -> Optional[float]:
    match = DRIVEFUZZ_TIME_RE.search(path.name)
    if not match:
        return None
    try:
        return float(match.group("timestamp"))
    except ValueError:
        return None


def _drivefuzz_failure_breakdown(error_paths: List[Path]) -> Dict[str, int]:
    stats = _empty_failure_stats()
    for error_path in error_paths:
        try:
            data = _load_json(error_path)
        except (OSError, json.JSONDecodeError):
            _increment_failure(stats, "Unknown")
            continue
        events = data.get("events") or {}
        matched = False
        if events.get("crash"):
            _increment_failure(stats, "Collision")
            matched = True
        if events.get("stuck"):
            _increment_failure(stats, "Timeout/Stuck")
            matched = True
        if events.get("lane_invasion"):
            _increment_failure(stats, "Lane Invasion")
            matched = True
        if events.get("red"):
            _increment_failure(stats, "Red Light")
            matched = True
        if events.get("out_of_bounds"):
            _increment_failure(stats, "Out of Bounds")
            matched = True
        if events.get("speeding"):
            _increment_failure(stats, "Speeding")
            matched = True
        if events.get("other") == "timeout":
            _increment_failure(stats, "Timeout/Stuck")
            matched = True
        elif events.get("other"):
            _increment_failure(stats, "Other Oracle Failure")
            matched = True
        if not matched:
            _increment_failure(stats, "Unknown")
    return stats


def _drivefuzz_path_has_failure(path: Path) -> bool:
    try:
        data = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    events = data.get("events") or {}
    return any([
        bool(events.get("crash")),
        bool(events.get("stuck")),
        bool(events.get("lane_invasion")),
        bool(events.get("red")),
        bool(events.get("out_of_bounds")),
        bool(events.get("speeding")),
        bool(events.get("other")),
    ])


def _drivefuzz_mapping_for_out_dir(seed_roots: List[Path], out_dir: Path) -> Dict:
    match = DRIVEFUZZ_OUT_RE.match(out_dir.name)
    if not match:
        return {}
    for seed_root in seed_roots:
        candidates = []
        if out_dir.parent.name:
            candidates.append(seed_root / out_dir.parent.name)
        candidates.append(seed_root)
        for candidate_root in candidates:
            for prefix in ("seed-vlad", "seed-vlad-selected"):
                seed_dir = candidate_root / f"{prefix}-{match.group('suffix')}"
                rows = _load_jsonl(seed_dir / "mapping.jsonl")
                if rows:
                    return rows[0]
    return {}


def _collect_drivefuzz_runs(drivefuzz_src: Path, drivefuzz_results_root: Optional[Path] = None) -> List[Dict]:
    rows = []
    run_roots = []
    seed_roots = []
    if drivefuzz_results_root is not None:
        run_roots.append(drivefuzz_results_root / "runs")
        seed_roots.append(drivefuzz_results_root / "seeds")
    run_roots.append(drivefuzz_src)
    seed_roots.append(drivefuzz_src)

    out_dirs = []
    for run_root in run_roots:
        if run_root.exists():
            out_dirs.extend(sorted(run_root.glob("out-drivefuzz-*")))
            out_dirs.extend(sorted(run_root.glob("*/out-drivefuzz-*")))

    for out_dir in out_dirs:
        if not out_dir.is_dir():
            continue
        match = DRIVEFUZZ_OUT_RE.match(out_dir.name)
        if not match:
            continue
        queue_paths = sorted((out_dir / "queue").glob("*.json"))
        error_paths = sorted((out_dir / "errors").glob("*.json"))
        if not error_paths:
            error_paths = [path for path in queue_paths if _drivefuzz_path_has_failure(path)]
        queue_times = [value for value in (_drivefuzz_timestamp(path) for path in queue_paths) if value is not None]
        error_times = [value for value in (_drivefuzz_timestamp(path) for path in error_paths) if value is not None]
        first_queue = min(queue_times) if queue_times else None
        first_error = min(error_times) if error_times else None
        ttf = None
        if first_queue is not None and first_error is not None:
            ttf = max(0.0, first_error - first_queue)

        mapping = _drivefuzz_mapping_for_out_dir(seed_roots, out_dir)
        failure_stats = _drivefuzz_failure_breakdown(error_paths)
        executions = len(queue_paths)
        failures = len(error_paths)
        suffix = match.group("suffix")
        command_match = re.search(r"-cmd(\d+)$", suffix)
        command_id = command_match.group(1) if command_match else ""
        row = {
            "method": f"drivefuzz_cmd{command_id}" if command_id else "drivefuzz",
            "method_family": "drivefuzz",
            "model": match.group("model"),
            "manifest_id": mapping.get("manifest_id"),
            "command_id": command_id,
            "run_id": out_dir.parent.name if out_dir.parent.name != "runs" else "",
            "run_dir": str(out_dir),
            "duration_seconds": "" if not queue_times else round(max(queue_times) - min(queue_times), 2),
            "budget_minutes": "",
            "executions": executions,
            "failures": failures,
            "failure_rate": _rate(failures, executions),
            "time_to_first_failure_seconds": "" if ttf is None else round(ttf, 2),
        }
        row.update(failure_stats)
        rows.append(row)
    return rows


def _aggregate(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str], Dict] = {}
    for row in rows:
        key = (row.get("method_family", ""), row.get("model", ""))
        if key not in groups:
            groups[key] = {
                "method_family": key[0],
                "model": key[1],
                "runs": 0,
                "seeds": set(),
                "executions": 0,
                "failures": 0,
                "time_to_first_failure_values": [],
            }
            for category in _empty_failure_stats():
                groups[key][category] = 0
        group = groups[key]
        group["runs"] += 1
        if row.get("manifest_id"):
            group["seeds"].add(row["manifest_id"])
        group["executions"] += int(row.get("executions", 0))
        group["failures"] += int(row.get("failures", 0))
        ttf = row.get("time_to_first_failure_seconds")
        if ttf not in ("", None):
            group["time_to_first_failure_values"].append(float(ttf))
        for category in _empty_failure_stats():
            group[category] += int(row.get(category, 0))

    aggregated = []
    for group in groups.values():
        ttf_values = group.pop("time_to_first_failure_values")
        seeds = group.pop("seeds")
        group["seeds"] = len(seeds)
        group["failure_rate"] = _rate(group["failures"], group["executions"])
        group["median_time_to_first_failure_seconds"] = (
            "" if not ttf_values else round(sorted(ttf_values)[len(ttf_values) // 2], 2)
        )
        aggregated.append(group)
    return sorted(aggregated, key=lambda row: (row["model"], row["method_family"]))


def _failure_breakdown(rows: List[Dict]) -> List[Dict]:
    breakdown = []
    for row in _aggregate(rows):
        entry = {
            "method_family": row["method_family"],
            "model": row["model"],
            "failures": row["failures"],
        }
        for category in _empty_failure_stats():
            entry[category] = row.get(category, 0)
        breakdown.append(entry)
    return breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RQ2 baseline and VLAD-Fuzz experiment results.")
    parser.add_argument("--experiment-root", default="results/rq2/runs", help="Root containing VLAD-Fuzz/random metadata runs.")
    parser.add_argument("--drivefuzz-src", default="baselines/drivefuzz/src", help="DriveFuzz src directory containing out-drivefuzz-* runs.")
    parser.add_argument("--drivefuzz-results-root", default="results/rq2/runs/drivefuzz", help="Root containing DriveFuzz runs/ and seeds/ directories.")
    parser.add_argument("--output-dir", default="results/rq2/summary", help="Output directory for CSV summaries.")
    args = parser.parse_args()

    rows = []
    rows.extend(_collect_local_runs(Path(args.experiment_root)))
    rows.extend(_collect_drivefuzz_runs(Path(args.drivefuzz_src), Path(args.drivefuzz_results_root)))
    rows = sorted(rows, key=lambda row: (str(row.get("model")), str(row.get("manifest_id")), str(row.get("method_family")), str(row.get("command_id"))))

    output_dir = Path(args.output_dir)
    _write_csv(rows, output_dir / "rq2_runs.csv")
    _write_csv(_aggregate(rows), output_dir / "rq2_aggregated.csv")
    _write_csv(_failure_breakdown(rows), output_dir / "rq2_failure_breakdown.csv")

    print(f"Collected {len(rows)} RQ2 run rows")
    print(f"Wrote {output_dir / 'rq2_runs.csv'}")
    print(f"Wrote {output_dir / 'rq2_aggregated.csv'}")
    print(f"Wrote {output_dir / 'rq2_failure_breakdown.csv'}")


if __name__ == "__main__":
    main()
