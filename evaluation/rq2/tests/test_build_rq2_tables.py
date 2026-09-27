import csv
import json
import tempfile
import unittest
from pathlib import Path


class RQ2TableBuilderTest(unittest.TestCase):
    def test_builds_per_model_tables_with_latest_runs_and_summary_rows(self):
        from evaluation.rq2.build_tables import build_tables

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"

            self._write_local_run(
                results / "vladfuzz" / "simlingo" / "old",
                model="simlingo",
                method="vlad_fuzz_cmd1",
                manifest_id="town01/seed_1",
                executions=10,
                failures=1,
                start_time="2026-07-01T00:00:00",
            )
            self._write_local_run(
                results / "vladfuzz" / "simlingo" / "new",
                model="simlingo",
                method="vlad_fuzz_cmd1",
                manifest_id="town01/seed_1",
                executions=20,
                failures=5,
                start_time="2026-07-02T00:00:00",
            )
            self._write_local_run(
                results / "random" / "simlingo" / "run",
                model="simlingo",
                method="random_cmd1",
                manifest_id="town01/seed_1",
                executions=40,
                failures=4,
                start_time="2026-07-03T00:00:00",
            )
            self._write_local_run(
                results / "instruction_counterfactual" / "simlingo" / "run",
                model="simlingo",
                method="instruction_counterfactual",
                manifest_id="town01/seed_1",
                executions=24,
                failures=6,
                start_time="2026-07-04T00:00:00",
            )
            self._write_drivefuzz_run(
                results,
                run_id="simlingo_drive",
                model="simlingo",
                suffix="town01_seed_1-cmd1",
                manifest_id="town01/seed_1",
                queue_count=5,
                error_count=2,
            )

            output_dir = root / "tables"
            tables = build_tables(results, output_dir)

            self.assertIn("simlingo", tables)
            rows = tables["simlingo"]
            self.assertEqual(
                [(row["scenario_seed"], row["baseline"], row["executions"], row["failures"], row["failure_rate"]) for row in rows],
                [
                    ("town01/seed_1", "VLAD-Fuzz", 20, 5, 0.25),
                    ("town01/seed_1", "Random", 40, 4, 0.1),
                    ("town01/seed_1", "DriveFuzz", 5, 2, 0.4),
                    ("town01/seed_1", "Instruction-CF", 24, 6, 0.25),
                    ("ALL", "VLAD-Fuzz", 20, 5, 0.25),
                    ("ALL", "Random", 40, 4, 0.1),
                    ("ALL", "DriveFuzz", 5, 2, 0.4),
                    ("ALL", "Instruction-CF", 24, 6, 0.25),
                ],
            )

            csv_path = output_dir / "simlingo_rq2_table.csv"
            with csv_path.open(newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))
            self.assertEqual(csv_rows[-1]["scenario_seed"], "ALL")
            self.assertEqual(csv_rows[-1]["baseline"], "Instruction-CF")
            self.assertFalse((output_dir / "instruction_counterfactual_rq2_table.csv").exists())

    def _write_local_run(self, path, *, model, method, manifest_id, executions, failures, start_time):
        path.mkdir(parents=True)
        metadata = {
            "start_time": start_time,
            "total_simulations_executed": executions,
            "total_failures_detected": failures,
            "configuration": {
                "vla_model": model,
                "method": method,
                "manifest_id": manifest_id,
            },
        }
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def _write_drivefuzz_run(self, results, *, run_id, model, suffix, manifest_id, queue_count, error_count):
        out_dir = results / "drivefuzz" / "runs" / run_id / f"out-drivefuzz-{model}-{suffix}"
        queue_dir = out_dir / "queue"
        error_dir = out_dir / "errors"
        queue_dir.mkdir(parents=True)
        error_dir.mkdir(parents=True)
        for index in range(queue_count):
            (queue_dir / f"1_1_{index}_100{index}.json").write_text("{}", encoding="utf-8")
        for index in range(error_count):
            (error_dir / f"1_1_{index}_100{index}.json").write_text(
                json.dumps({"events": {"crash": True}}),
                encoding="utf-8",
            )

        seed_dir = results / "drivefuzz" / "seeds" / run_id / f"seed-vlad-{suffix}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "mapping.jsonl").write_text(
            json.dumps({"manifest_id": manifest_id}) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
