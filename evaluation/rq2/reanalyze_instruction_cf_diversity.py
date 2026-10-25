"""Recompute RQ2 diversity using scoped automatic Instruction-CF attribution."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.rq2.analyze_diversity import (
    FailureCase,
    SCENE_FEATURES,
    _collect_local_failures,
    _plot_diversity_summary,
    _write_records,
    analyze_task_stratified_diversity,
    carla_weather_resolver,
    drivefuzz_scene_vector,
    encode_instructions,
    source_failure_counts,
    vladfuzz_scene_vector,
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _resolve(value, root):
    path = Path(value)
    return path if path.is_absolute() else root / path


def collect_reference(selected_runs, reference_root, previous_root):
    cases = []
    remaps = []
    rows = _csv(selected_runs)
    for row in rows:
        run = _resolve(row["run_dir"], reference_root)
        if row["baseline"] != "DriveFuzz":
            found = _collect_local_failures(
                row, run, reference_root, carla_weather_resolver
            )
        else:
            seed = Path(_json(run / "drivefuzz_metadata.json")["seed_dir"])
            if not seed.exists() and seed.is_relative_to(previous_root):
                relocated = reference_root / seed.relative_to(previous_root)
                remaps.append({"recorded": str(seed), "resolved": str(relocated)})
                seed = relocated
            with (seed / "mapping.jsonl").open(encoding="utf-8") as file:
                mappings = [json.loads(line) for line in file if line.strip()]
            if len(mappings) != 1:
                raise ValueError(f"Expected one DriveFuzz instruction mapping: {seed}")
            found = [
                FailureCase(
                    model=row["model"], method="DriveFuzz", task=row["scenario_seed"],
                    instruction=mappings[0]["instruction"],
                    scene=drivefuzz_scene_vector(_json(path)), source=path,
                )
                for path in sorted((run / "errors").glob("*.json"))
            ]
        if len(found) != int(row["failures"]):
            raise ValueError(f"Reference failure count mismatch: {run}")
        cases.extend(found)
    return rows, cases, remaps


def collect_retained(attribution_dir, repo_root):
    manifest = _json(attribution_dir / "audit_manifest.json")
    for item in manifest["inputs"]:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Attribution inputs changed: {item['path']}")
    with (attribution_dir / "retained_failures.jsonl").open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    cases = []
    seen = set()
    statics = {}
    for record in records:
        if record["label"] != "retained_failure" or len(record["failure_artifacts"]) != 1:
            raise ValueError("Expected one retained failure artifact per execution")
        run = Path(record["run_dir"])
        if run not in statics:
            config = _json(run / "metadata.json")["configuration"]
            statics[run] = _json(_resolve(config["static_scenario"], repo_root))
        result_path = run / "failures" / record["failure_artifacts"][0] / "result.json"
        if result_path in seen:
            raise ValueError(f"Duplicate retained artifact: {result_path}")
        seen.add(result_path)
        result = _json(result_path)
        instruction = result.get("mutated_instruction") or result.get("instruction")
        if instruction != record["instruction"]:
            raise ValueError(f"Instruction differs from attribution: {result_path}")
        cases.append(FailureCase(
            model=record["model"], method="Instruction-CF", task=record["task"],
            instruction=instruction,
            scene=vladfuzz_scene_vector(
                _json(result_path.with_name("scenario_config.json")), statics[run],
                weather_resolver=carla_weather_resolver,
            ),
            source=result_path,
        ))
    counts = source_failure_counts(cases)
    for row in manifest["model_summary"]:
        if counts[row["model"]]["Instruction-CF"] != row["retained_failures"]:
            raise ValueError(f"Retained count mismatch: {row['model']}")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    reference = args.reference_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output}")

    old_dir = reference / "results/rq2/diversity"
    selected = reference / "results/rq2/paper/selected_runs.csv"
    config = _json(old_dir / "diagnostics.json")
    settings = {key: config[key] for key in (
        "scene_weight", "max_per_method_task", "bootstrap_iterations", "random_seed"
    )}
    if list(SCENE_FEATURES) != config["scene_features"]:
        raise ValueError("Scenario features differ from reference analysis")
    rows, old_cases, remaps = collect_reference(selected, reference, root)
    cache_files = sorted(old_dir.glob("instruction_embeddings_*.npz"))
    if len(cache_files) != 1:
        raise ValueError("Expected one reference embedding cache")
    with np.load(cache_files[0], allow_pickle=False) as cache:
        if str(cache["model"].item()) != config["embedding_model"]:
            raise ValueError("Reference encoder mismatch")
        lookup = dict(zip(cache["texts"].tolist(), cache["embeddings"].copy()))
    old_embeddings = np.asarray([lookup[case.instruction] for case in old_cases])
    reproduced, _, _ = analyze_task_stratified_diversity(old_cases, old_embeddings, **settings)
    old_summary = _csv(old_dir / "summary.csv")
    old_index = {(r["model"], r["method"]): r for r in old_summary}
    max_error = max(
        abs(row[key] - float(old_index[row["model"], row["method"]][key]))
        for row in reproduced
        for key in ("scene_diversity", "instruction_diversity", "joint_diversity", "joint_ci_low", "joint_ci_high")
    )
    if max_error > 1e-10:
        raise ValueError(f"Reference results cannot be reproduced: max error {max_error}")
    print(f"Reference reproduced; maximum difference {max_error:.3g}", flush=True)

    new_cf = collect_retained(args.attribution_dir.resolve(), root)
    groups = {}
    for case in new_cf:
        groups.setdefault((case.model, case.task), []).append(case)
    for group in groups.values():
        group.sort(key=lambda case: str(case.source))
    old_groups = {}
    for case in old_cases:
        old_groups.setdefault((case.model, case.task, case.method), []).append(case)
    cases = []
    new_selection = []
    task_rows = _csv(args.attribution_dir / "summary_by_task.csv")
    new_tasks = {(r["model"], r["task"]): r for r in task_rows}
    for row in rows:
        identity = row["model"], row["scenario_seed"]
        if row["baseline"] == "Instruction-CF":
            cases.extend(groups.get(identity, []))
            updated = new_tasks[identity]
            new_selection.append({**row, "run_dir": updated["run_dir"],
                                  "executions": updated["executions"],
                                  "failures": updated["retained_failures"],
                                  "failure_rate": float(updated["corrected_failure_rate_percent"]) / 100})
        else:
            cases.extend(old_groups.get((*identity, row["baseline"]), []))
            new_selection.append({**row, "run_dir": str(_resolve(row["run_dir"], reference))})
    if sum(case.method == "Instruction-CF" for case in cases) != len(new_cf):
        raise ValueError("Reference task selection does not cover all retained cases")
    output.mkdir(parents=True)
    missing = sorted({case.instruction for case in cases} - set(lookup))
    print(f"Loaded {len(cases)} failures; encoding {len(missing)} new unique instructions on CPU", flush=True)
    if missing:
        vectors = encode_instructions(
            missing, model_name=config["embedding_model"], device="cpu", batch_size=32,
            cache_dir=args.hf_cache, output_dir=output,
        )
        lookup.update(zip(missing, vectors))
    embeddings = np.asarray([lookup[case.instruction] for case in cases])
    summary, comparisons, diagnostics = analyze_task_stratified_diversity(cases, embeddings, **settings)
    diagnostics.update(
        scene_features=list(SCENE_FEATURES), embedding_model=config["embedding_model"],
        source_failure_counts=source_failure_counts(cases),
        attribution_manifest=str(args.attribution_dir / "audit_manifest.json"),
        attribution_scope="Retained failures from departure-only semantic-change attribution.",
        reference_reproduction_max_error=max_error, path_relocations=remaps,
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            selected, old_dir / "summary.csv", old_dir / "diagnostics.json", cache_files[0],
            args.attribution_dir / "audit_manifest.json",
            args.attribution_dir / "retained_failures.jsonl",
        )},
    )
    changes = [{
        "model": row["model"], "method": row["method"],
        "old_tasks": int(old_index[row["model"], row["method"]]["comparable_tasks"]),
        "new_tasks": row["comparable_tasks"],
        "old_joint_diversity": float(old_index[row["model"], row["method"]]["joint_diversity"]),
        "new_joint_diversity": row["joint_diversity"],
        "change": row["joint_diversity"] - float(old_index[row["model"], row["method"]]["joint_diversity"]),
    } for row in summary]
    _write_records(output / "selected_runs.csv", new_selection)
    _write_records(output / "summary.csv", summary)
    _write_records(output / "comparisons.csv", comparisons)
    _write_records(output / "changes_from_previous.csv", changes)
    _write_records(output / "reference_reproduced.csv", reproduced)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    _plot_diversity_summary(summary, output / "rq2-diversity.pdf", settings["scene_weight"])
    print(json.dumps({"summary": summary, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
