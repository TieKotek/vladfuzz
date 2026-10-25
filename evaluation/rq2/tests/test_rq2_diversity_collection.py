import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq2.analyze_diversity import collect_failure_cases


class RQ2DiversityCollectionTests(unittest.TestCase):
    @staticmethod
    def _write_local_failure(run, name, instruction):
        failure = run / "failures" / name
        failure.mkdir(parents=True)
        (failure / "result.json").write_text(
            json.dumps({"mutated_instruction": instruction}), encoding="utf-8"
        )
        (failure / "scenario_config.json").write_text(
            json.dumps(
                {
                    "ego_car": {"spawn_point_index": 0},
                    "npc_vehicles": [{"spawn_point_index": 1}],
                    "weather": "ClearNoon",
                }
            ),
            encoding="utf-8",
        )

    def test_collects_all_three_methods_from_the_frozen_run_list(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            static_path = root / "static.json"
            static_path.write_text(
                json.dumps(
                    {
                        "scenario_center": {"x": 0.0, "y": 0.0},
                        "spawn_points": [
                            {"x": 0.0, "y": 0.0, "yaw": 0.0},
                            {"x": 5.0, "y": 0.0, "yaw": 0.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows = []
            for method, instruction in (
                ("VLAD-Fuzz", "Turn left."),
                ("Instruction-CF", "Take a left."),
            ):
                run = root / method
                failure = run / "failures" / "failure_1"
                failure.mkdir(parents=True)
                (run / "metadata.json").write_text(
                    json.dumps({"configuration": {"static_scenario": str(static_path)}}),
                    encoding="utf-8",
                )
                (failure / "result.json").write_text(
                    json.dumps({"mutated_instruction": instruction}), encoding="utf-8"
                )
                (failure / "scenario_config.json").write_text(
                    json.dumps(
                        {
                            "ego_car": {"spawn_point_index": 0},
                            "npc_vehicles": [{"spawn_point_index": 1}],
                            "weather": "ClearNoon",
                        }
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "model": "simlingo",
                        "scenario_seed": "town/seed_1",
                        "baseline": method,
                        "failures": "1",
                        "run_dir": str(run),
                    }
                )

            drive_run = root / "DriveFuzz"
            (drive_run / "errors").mkdir(parents=True)
            seed_dir = root / "drive_seed"
            seed_dir.mkdir()
            (seed_dir / "mapping.jsonl").write_text(
                json.dumps({"instruction": "Continue straight."}) + "\n",
                encoding="utf-8",
            )
            (drive_run / "drivefuzz_metadata.json").write_text(
                json.dumps({"seed_dir": str(seed_dir)}), encoding="utf-8"
            )
            (drive_run / "errors" / "error.json").write_text(
                json.dumps(
                    {
                        "seed": {"sp_x": 0.0, "sp_y": 0.0, "yaw": 0.0},
                        "actors": [],
                        "puddles": [],
                        "weather": {},
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                {
                    "model": "simlingo",
                    "scenario_seed": "town/seed_1",
                    "baseline": "DriveFuzz",
                    "failures": "1",
                    "run_dir": str(drive_run),
                }
            )

            selected = root / "selected.csv"
            with selected.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            cases = collect_failure_cases(
                selected,
                root,
                weather_resolver=lambda _: [0.0] * 8,
            )

            self.assertEqual(len(cases), 3)
            self.assertEqual(
                {(case.method, case.instruction) for case in cases},
                {
                    ("VLAD-Fuzz", "Turn left."),
                    ("Instruction-CF", "Take a left."),
                    ("DriveFuzz", "Continue straight."),
                },
            )

    def test_excludes_audited_task_mismatch_from_instruction_cf_cases(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            static_path = root / "static.json"
            static_path.write_text(
                json.dumps(
                    {
                        "scenario_center": {"x": 0.0, "y": 0.0},
                        "spawn_points": [
                            {"x": 0.0, "y": 0.0, "yaw": 0.0},
                            {"x": 5.0, "y": 0.0, "yaw": 0.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run = root / "Instruction-CF"
            self._write_local_failure(run, "valid_failure", "Take a left.")
            self._write_local_failure(run, "task_mismatch", "Override: turn right.")
            metadata = {
                "total_failures_detected": 2,
                "total_raw_oracle_positives": 2,
                "total_provisionally_valid_failures": 1,
                "total_pending_failure_attribution": 1,
                "failure_attribution_schema_version": 1,
                "configuration": {"static_scenario": str(static_path)},
            }
            (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            execution_rows = [
                {
                    "failed": True,
                    "failure_count": 1,
                    "failure_attribution_status": "valid_failure",
                    "failure_artifacts": ["valid_failure"],
                    "elapsed_seconds": 1.0,
                },
                {
                    "failed": True,
                    "failure_count": 1,
                    "failure_attribution_status": "pending_manual_review",
                    "audit_id": "audit-1",
                    "failure_artifacts": ["task_mismatch"],
                    "elapsed_seconds": 2.0,
                },
            ]
            (run / "execution_results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in execution_rows),
                encoding="utf-8",
            )
            (run / "false_positive_audit_candidates.jsonl").write_text(
                json.dumps({"audit_id": "audit-1"}) + "\n",
                encoding="utf-8",
            )
            (run / "false_positive_audit_annotations.jsonl").write_text(
                json.dumps(
                    {
                        "audit_id": "audit-1",
                        "label": "task_mismatch_false_positive",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selected = root / "selected.csv"
            with selected.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("model", "scenario_seed", "baseline", "failures", "run_dir"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "simlingo",
                        "scenario_seed": "town/seed_1",
                        "baseline": "Instruction-CF",
                        "failures": "1",
                        "run_dir": str(run),
                    }
                )

            cases = collect_failure_cases(
                selected,
                root,
                weather_resolver=lambda _: [0.0] * 8,
            )

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].instruction, "Take a left.")


if __name__ == "__main__":
    unittest.main()
