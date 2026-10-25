"""Apply explicit semantic-change rules to departure-only Instruction-CF failures."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from vladfuzz_runtime.instruction_cf_audit import TASK_MISMATCH_FALSE_POSITIVE
from vladfuzz_runtime.instruction_counterfactuals import (
    CounterfactualVariant,
    InstructionFamily,
    Maneuver,
    build_counterfactual_batch,
)


RULE_VERSION = "departure-only-semantic-change-v1"
DIRECTIONAL_MANEUVERS = {
    Maneuver.TURN_LEFT,
    Maneuver.TURN_RIGHT,
    Maneuver.CHANGE_LANE_LEFT,
    Maneuver.CHANGE_LANE_RIGHT,
}
COUNT_FIELDS = (
    "executions",
    "raw_failures",
    "departure_only",
    "task_mismatch_false_positives",
    "retained_failures",
)


def triggered_checks(record):
    """Use enabled oracle outcomes, not raw flags such as disabled speeding."""
    checks = set()
    for node in record["detailed_log"]:
        events = node["execution_result"].get("oracle_events", {})
        if "triggered_checks" not in events:
            raise ValueError("Execution lacks explicit triggered oracle checks")
        enabled = events.get("enabled_checks", {})
        for check in events["triggered_checks"]:
            if enabled.get(check, True):
                checks.add(check)
    if record["failed"] and not checks:
        raise ValueError("Failed execution has no enabled triggered oracle checks")
    if not record["failed"] and checks:
        raise ValueError("Passing execution has triggered oracle checks")
    return checks


def attribute_record(record, variant: CounterfactualVariant):
    expected = {
        "family": variant.family.value,
        "template_id": variant.template_id,
        "instruction": variant.instruction,
        "source_maneuvers": [m.value for m in variant.source_maneuvers],
        "maneuvers": [m.value for m in variant.maneuvers],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Recorded variant differs from generator: {key}")

    checks = triggered_checks(record)
    departure_only = record["failed"] and checks == {"out_of_bounds"}
    rule = None
    changed = []
    if departure_only and variant.family == InstructionFamily.MISLEADING:
        changed = [
            {"position": i, "source": old.value, "variant": new.value}
            for i, (old, new) in enumerate(
                zip(variant.source_maneuvers, variant.maneuvers)
            )
            if old != new
        ]
        if not changed:
            raise ValueError("Misleading variant does not replace a navigation action")
        rule = "misleading_navigation_action_replaced"
    elif departure_only and variant.family == InstructionFamily.AMBIGUITY:
        changed = [
            {"position": i, "source": maneuver.value, "variant": "direction_omitted"}
            for i, maneuver in enumerate(variant.source_maneuvers)
            if maneuver in DIRECTIONAL_MANEUVERS
        ]
        if changed:
            rule = "ambiguity_turn_or_lane_change_direction_omitted"

    label = "passed"
    if record["failed"]:
        label = TASK_MISMATCH_FALSE_POSITIVE if rule else "retained_failure"
    return {
        "execution_index": record["execution_index"],
        "scenario_index": record["scenario_index"],
        "variant_index": record["variant_index"],
        "family": variant.family.value,
        "template_id": variant.template_id,
        "source_instruction": record["source_instruction"],
        "instruction": record["instruction"],
        "source_maneuvers": expected["source_maneuvers"],
        "maneuvers": expected["maneuvers"],
        "triggered_checks": sorted(checks),
        "departure_only": bool(departure_only),
        "label": label,
        "rule": rule,
        "semantic_changes": changed,
        "failure_artifacts": record.get("failure_artifacts", []),
    }


def _counts(row):
    failed = row["label"] != "passed"
    false_positive = row["label"] == TASK_MISMATCH_FALSE_POSITIVE
    return Counter(
        executions=1,
        raw_failures=int(failed),
        departure_only=int(row["departure_only"]),
        task_mismatch_false_positives=int(false_positive),
        retained_failures=int(failed and not false_positive),
    )


def _summary_row(counts, **identity):
    total = counts["executions"]
    return {
        **identity,
        **{key: counts[key] for key in COUNT_FIELDS},
        "raw_failure_rate_percent": round(100 * counts["raw_failures"] / total, 6),
        "corrected_failure_rate_percent": round(
            100 * counts["retained_failures"] / total, 6
        ),
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_root, output_dir):
    input_root = Path(input_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite derived output: {output_dir}")
    records = []
    task_rows = []
    model_counts = {}
    family_counts = {}
    input_hashes = []
    for metadata_path in sorted(input_root.glob("*/*/metadata.json")):
        run = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata["configuration"]
        model, task = config["vla_model"], config["manifest_id"]
        execution_path = run / "execution_results.jsonl"
        for path in (metadata_path, execution_path):
            input_hashes.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        counts = Counter()
        cache = {}
        seen = set()
        with execution_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["execution_index"] in seen:
                    raise ValueError(f"Duplicate execution index: {execution_path}")
                seen.add(record["execution_index"])
                if len(record["detailed_log"]) != 1 or record["failure_count"] != int(record["failed"]):
                    raise ValueError(f"Expected one node per counterfactual: {execution_path}")
                if record["source_instruction"] != config["instruction"]:
                    raise ValueError(f"Source instruction differs from configuration: {run}")
                scenario = record["scenario_index"]
                if scenario not in cache:
                    variants = build_counterfactual_batch(
                        config["instruction"],
                        k=config["k_per_family"],
                        random_seed=config["random_seed"],
                        scenario_index=scenario,
                    )
                    cache[scenario] = {
                        (v.family.value, v.template_id): v for v in variants
                    }
                variant = cache[scenario][(record["family"], record["template_id"])]
                row = attribute_record(record, variant)
                row.update(model=model, task=task, run_dir=str(run), rule_version=RULE_VERSION)
                records.append(row)
                delta = _counts(row)
                counts.update(delta)
                model_counts.setdefault(model, Counter()).update(delta)
                family_counts.setdefault((model, row["family"]), Counter()).update(delta)
        if counts["executions"] != metadata["total_simulations_executed"]:
            raise ValueError(f"Execution count differs from metadata: {run}")
        if counts["raw_failures"] != metadata["total_raw_oracle_positives"]:
            raise ValueError(f"Raw failure count differs from metadata: {run}")
        task_rows.append(_summary_row(counts, model=model, task=task, run_dir=str(run)))

    if not records:
        raise ValueError(f"No execution records found: {input_root}")
    identities = [(row["model"], row["task"]) for row in task_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("More than one run per model/task; select one campaign explicitly")
    model_rows = [
        _summary_row(counts, model=model)
        for model, counts in sorted(model_counts.items())
    ]
    family_rows = [
        _summary_row(counts, model=model, family=family)
        for (model, family), counts in sorted(family_counts.items())
    ]
    output_dir.mkdir(parents=True)
    for filename, label in (
        ("execution_attribution.jsonl", None),
        ("task_mismatch_false_positives.jsonl", TASK_MISMATCH_FALSE_POSITIVE),
        ("retained_failures.jsonl", "retained_failure"),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as file:
            for row in records:
                if label is None or row["label"] == label:
                    file.write(json.dumps(row, ensure_ascii=True) + "\n")
    _write_csv(output_dir / "summary_by_model.csv", model_rows)
    _write_csv(output_dir / "summary_by_task.csv", task_rows)
    _write_csv(output_dir / "summary_by_family.csv", family_rows)
    manifest = {
        "rule_version": RULE_VERSION,
        "scope": "Only failures whose enabled triggered checks equal [out_of_bounds].",
        "false_positive_rules": [
            "misleading_navigation_action_replaced",
            "ambiguity_turn_or_lane_change_direction_omitted",
        ],
        "retention_policy": "Retain all other oracle-positive executions under this scoped rule.",
        "rate_denominator": "All executions, including task-mismatch inputs.",
        "interpretation": "Rule-based attribution, not a trajectory-based causal proof or full input-validity audit.",
        "input_root": str(input_root),
        "run_count": len(task_rows),
        "model_summary": model_rows,
        "inputs": input_hashes,
        "generator_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "vladfuzz_runtime/instruction_counterfactuals.py").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return model_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(args.input_root, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
