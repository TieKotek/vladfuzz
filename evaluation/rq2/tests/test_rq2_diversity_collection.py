import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq2.analyze_diversity import collect_failure_cases


class RQ2DiversityCollectionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
