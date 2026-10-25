import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.rq2.auto_attribute_instruction_cf import analyze
from evaluation.rq2.reanalyze_instruction_cf_diversity import collect_retained
from evaluation.rq2.tests import test_auto_attribute_instruction_cf as attribution_fixtures


class RetainedDiversityInputTests(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory)
        run = root / "runs" / "simlingo" / "run"
        run.mkdir(parents=True)
        fixture = attribution_fixtures.AutoAttributionTests()
        records = [fixture.record(family)[0] for family in ("misleading", "ambiguity", "noise")]
        static = root / "static.json"
        static.write_text(json.dumps({"scenario_center": {"x": 0, "y": 0},
                                      "spawn_points": [{"x": 0, "y": 0, "yaw": 0}]}))
        metadata = {
            "configuration": {"vla_model": "simlingo", "manifest_id": "task/seed_1",
                              "instruction": fixture.source, "k_per_family": 8,
                              "random_seed": 0, "static_scenario": str(static)},
            "total_simulations_executed": 3, "total_raw_oracle_positives": 3,
        }
        (run / "metadata.json").write_text(json.dumps(metadata))
        for i, row in enumerate(records, 1):
            row["execution_index"] = i
            row["failure_artifacts"] = [f"failure_{i}"]
            failure = run / "failures" / f"failure_{i}"
            failure.mkdir(parents=True)
            (failure / "result.json").write_text(json.dumps({"instruction": row["instruction"]}))
            (failure / "scenario_config.json").write_text(json.dumps({
                "ego_car": {"spawn_point_index": 0}, "npc_vehicles": [], "weather": "ClearNoon",
            }))
        (run / "execution_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
        output = root / "attribution"
        analyze(root / "runs", output)
        return root, run, output

    @patch("evaluation.rq2.reanalyze_instruction_cf_diversity.carla_weather_resolver", return_value=[0] * 8)
    def test_only_retained_artifacts_enter_diversity(self, _resolver):
        with TemporaryDirectory() as tmp:
            root, _, output = self.prepare(tmp)
            cases = collect_retained(output, root)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].source.parent.name, "failure_3")
            self.assertEqual(cases[0].method, "Instruction-CF")

    def test_changed_raw_input_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root, run, output = self.prepare(tmp)
            with (run / "execution_results.jsonl").open("a") as file:
                file.write("\n")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                collect_retained(output, root)

    @patch("evaluation.rq2.reanalyze_instruction_cf_diversity.carla_weather_resolver", return_value=[0] * 8)
    def test_mismatched_artifact_instruction_is_rejected(self, _resolver):
        with TemporaryDirectory() as tmp:
            root, run, output = self.prepare(tmp)
            (run / "failures/failure_3/result.json").write_text(json.dumps({"instruction": "Mismatch"}))
            with self.assertRaisesRegex(ValueError, "Instruction differs"):
                collect_retained(output, root)


if __name__ == "__main__":
    unittest.main()
