#!/usr/bin/env python3
"""Build paper-oriented RQ2 tables from all configured testing methods."""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BASELINE_ORDER = {
    "VLAD-Fuzz": 0,
    "Random": 1,
    "DriveFuzz": 2,
    "Instruction-CF": 3,
}
DRIVEFUZZ_OUT_RE = re.compile(r"^out-drivefuzz(?:-selected)?-(?P<model>[^-]+)-(?P<suffix>.+)$")
RUN_TIMESTAMP_RE = re.compile(r"_(?P<timestamp>\d{8}_\d{6})$")


@dataclass(frozen=True)
class RunRow:
    model: str
    baseline: str
    scenario_seed: str
    executions: int
    failures: int
    failure_rate: float
    run_dir: str
    sort_time: str


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rate(failures: int, executions: int) -> float:
    if executions <= 0:
        return 0.0
    return round(failures / executions, 4)


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _baseline_from_method(method: str, root_name: str) -> str:
    method = method or ""
    if "vlad" in method or root_name == "vladfuzz":
        return "VLAD-Fuzz"
    if "random" in method or root_name == "random":
        return "Random"
    if "instruction_counterfactual" in method or root_name == "instruction_counterfactual":
        return "Instruction-CF"
    return root_name


def _sort_time_from_local_run(run_dir: Path, metadata: Dict) -> str:
    start_time = metadata.get("start_time") or metadata.get("end_time")
    if start_time:
        return str(start_time)
    match = RUN_TIMESTAMP_RE.search(run_dir.name)
    if match:
        return match.group("timestamp")
    return str(run_dir.stat().st_mtime)


def _collect_local_runs(results_root: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    for root_name in ("vladfuzz", "random", "instruction_counterfactual"):
        root = results_root / root_name
        if not root.exists():
            continue
        for metadata_path in sorted(root.glob("*/*/metadata.json")):
            run_dir = metadata_path.parent
            try:
                metadata = _load_json(metadata_path)
            except (OSError, json.JSONDecodeError):
                continue
            config = metadata.get("configuration") or {}
            model = str(config.get("vla_model") or run_dir.parent.name)
            scenario_seed = str(config.get("manifest_id") or "")
            if not scenario_seed:
                continue
            baseline = _baseline_from_method(str(config.get("method") or ""), root_name)
            executions = _safe_int(metadata.get("total_simulations_executed"))
            failures = _safe_int(metadata.get("total_failures_detected"))
            rows.append(
                RunRow(
                    model=model,
                    baseline=baseline,
                    scenario_seed=scenario_seed,
                    executions=executions,
                    failures=failures,
                    failure_rate=_rate(failures, executions),
                    run_dir=str(run_dir),
                    sort_time=_sort_time_from_local_run(run_dir, metadata),
                )
            )
    return rows


def _drivefuzz_mapping(results_root: Path, run_id: str, suffix: str) -> Dict:
    seed_roots = [results_root / "drivefuzz" / "seeds" / run_id, results_root / "drivefuzz" / "seeds"]
    for seed_root in seed_roots:
        for prefix in ("seed-vlad", "seed-vlad-selected"):
            rows = _load_jsonl(seed_root / f"{prefix}-{suffix}" / "mapping.jsonl")
            if rows:
                return rows[0]
    return {}


def _drivefuzz_sort_time(out_dir: Path) -> str:
    json_paths = list((out_dir / "queue").glob("*.json")) + list((out_dir / "errors").glob("*.json"))
    timestamps = []
    for path in json_paths:
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            pass
    if timestamps:
        return str(max(timestamps))
    return str(out_dir.stat().st_mtime)


def _collect_drivefuzz_runs(results_root: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    runs_root = results_root / "drivefuzz" / "runs"
    if not runs_root.exists():
        return rows
    for out_dir in sorted(runs_root.glob("*/out-drivefuzz-*")):
        if not out_dir.is_dir():
            continue
        match = DRIVEFUZZ_OUT_RE.match(out_dir.name)
        if not match:
            continue
        run_id = out_dir.parent.name
        suffix = match.group("suffix")
        mapping = _drivefuzz_mapping(results_root, run_id, suffix)
        scenario_seed = str(mapping.get("manifest_id") or suffix.rsplit("-cmd", 1)[0].replace("_seed_", "/seed_"))
        queue_paths = list((out_dir / "queue").glob("*.json"))
        error_paths = list((out_dir / "errors").glob("*.json"))
        executions = len(queue_paths)
        failures = len(error_paths)
        rows.append(
            RunRow(
                model=match.group("model"),
                baseline="DriveFuzz",
                scenario_seed=scenario_seed,
                executions=executions,
                failures=failures,
                failure_rate=_rate(failures, executions),
                run_dir=str(out_dir),
                sort_time=_drivefuzz_sort_time(out_dir),
            )
        )
    return rows


def _latest_by_model_seed_baseline(rows: Iterable[RunRow]) -> Tuple[List[RunRow], List[Tuple[RunRow, RunRow]]]:
    latest: Dict[Tuple[str, str, str], RunRow] = {}
    replaced: List[Tuple[RunRow, RunRow]] = []
    for row in rows:
        key = (row.model, row.scenario_seed, row.baseline)
        current = latest.get(key)
        if current is None or row.sort_time >= current.sort_time:
            if current is not None:
                replaced.append((current, row))
            latest[key] = row
        else:
            replaced.append((row, current))
    return list(latest.values()), replaced


def _table_rows_for_model(rows: List[RunRow]) -> List[Dict]:
    body = [
        {
            "scenario_seed": row.scenario_seed,
            "baseline": row.baseline,
            "executions": row.executions,
            "failures": row.failures,
            "failure_rate": row.failure_rate,
            "run_dir": row.run_dir,
        }
        for row in sorted(rows, key=lambda item: (item.scenario_seed, BASELINE_ORDER.get(item.baseline, 99)))
    ]

    totals: Dict[str, Dict[str, int]] = {}
    for row in rows:
        totals.setdefault(row.baseline, {"executions": 0, "failures": 0})
        totals[row.baseline]["executions"] += row.executions
        totals[row.baseline]["failures"] += row.failures

    for baseline in sorted(totals, key=lambda name: BASELINE_ORDER.get(name, 99)):
        executions = totals[baseline]["executions"]
        failures = totals[baseline]["failures"]
        body.append(
            {
                "scenario_seed": "ALL",
                "baseline": baseline,
                "executions": executions,
                "failures": failures,
                "failure_rate": _rate(failures, executions),
                "run_dir": "",
            }
        )
    return body


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario_seed", "baseline", "executions", "failures", "failure_rate", "run_dir"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[Dict]) -> None:
    lines = [
        "| Scenario/Seed | Baseline | Executions | Failures | Failure Rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        rate = f"{float(row['failure_rate']) * 100:.2f}%"
        lines.append(
            f"| {row['scenario_seed']} | {row['baseline']} | {row['executions']} | {row['failures']} | {rate} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables(results_root: Path, output_dir: Path) -> Dict[str, List[Dict]]:
    rows = _collect_local_runs(results_root) + _collect_drivefuzz_runs(results_root)
    latest_rows, replaced = _latest_by_model_seed_baseline(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    if replaced:
        warning_path = output_dir / "duplicate_runs_ignored.csv"
        with warning_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "model",
                    "scenario_seed",
                    "baseline",
                    "ignored_run_dir",
                    "kept_run_dir",
                    "ignored_sort_time",
                    "kept_sort_time",
                ],
            )
            writer.writeheader()
            for ignored, kept in replaced:
                writer.writerow(
                    {
                        "model": ignored.model,
                        "scenario_seed": ignored.scenario_seed,
                        "baseline": ignored.baseline,
                        "ignored_run_dir": ignored.run_dir,
                        "kept_run_dir": kept.run_dir,
                        "ignored_sort_time": ignored.sort_time,
                        "kept_sort_time": kept.sort_time,
                    }
                )

    by_model: Dict[str, List[RunRow]] = {}
    for row in latest_rows:
        by_model.setdefault(row.model, []).append(row)

    tables: Dict[str, List[Dict]] = {}
    all_rows: List[Dict] = []
    for model in sorted(by_model):
        table_rows = _table_rows_for_model(by_model[model])
        tables[model] = table_rows
        _write_csv(output_dir / f"{model}_rq2_table.csv", table_rows)
        _write_markdown(output_dir / f"{model}_rq2_table.md", table_rows)
        for row in table_rows:
            all_rows.append({"model": model, **row})

    if all_rows:
        fields = ["model", "scenario_seed", "baseline", "executions", "failures", "failure_rate", "run_dir"]
        with (output_dir / "rq2_all_models.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-model RQ2 tables from result directories.")
    parser.add_argument("--results-root", default="results/rq2/runs", help="Root containing vladfuzz/random/drivefuzz result directories.")
    parser.add_argument("--output-dir", default="results/rq2/tables", help="Directory for generated CSV/Markdown tables.")
    args = parser.parse_args()

    tables = build_tables(Path(args.results_root), Path(args.output_dir))
    total_rows = sum(len(rows) for rows in tables.values())
    print(f"Built RQ2 tables for {len(tables)} model(s), {total_rows} table rows.")
    print(f"Output directory: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
