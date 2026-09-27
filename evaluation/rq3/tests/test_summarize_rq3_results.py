import csv
import json
from pathlib import Path
import tempfile
import unittest


from evaluation.rq3.summarize_results import summarize_rq3


class RQ3SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.results = self.root / "results"
        self.output = self.root / "summary"
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(
            '\n'.join(
                json.dumps({"id": manifest_id})
                for manifest_id in ("town01/seed_1", "town02/seed_1")
            )
            + '\n',
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _write_run(
        self,
        configuration,
        manifest_id,
        timestamp,
        executions,
        failures,
        categories,
        checkpoints=None,
    ):
        roots = {
            "full": self.results / "rq2" / "runs" / "vladfuzz" / "simlingo",
            "global_only": self.results / "rq3" / "runs" / "global_only" / "simlingo",
            "local_only": self.results / "rq3" / "runs" / "local_only" / "simlingo",
            "random_scenario_local": self.results / "rq3" / "runs" / "random_scenario_local" / "simlingo",
        }
        methods = {
            "full": "vlad_fuzz_cmd1",
            "global_only": "global_only",
            "local_only": "local_only",
            "random_scenario_local": "random_scenario_local",
        }
        run_dir = roots[configuration] / f"{configuration}_{manifest_id.replace('/', '__')}_{timestamp}"
        run_dir.mkdir(parents=True)
        metadata = {
            "start_time": timestamp,
            "total_simulations_executed": executions,
            "total_failures_detected": failures,
            "failure_analysis": categories,
            "hourly_checkpoints": checkpoints or [],
            "configuration": {
                "method": methods[configuration],
                "vla_model": "simlingo",
                "manifest_id": manifest_id,
                "time_budget_seconds": 14400,
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return run_dir

    @staticmethod
    def _read_csv(path):
        with path.open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def test_latest_runs_produce_per_seed_aggregate_category_and_hourly_outputs(self):
        checkpoints = [
            {"hour": 1, "executions": 4, "failures": 1},
            {"hour": 2, "executions": 10, "failures": 2},
        ]
        for seed in ("town01/seed_1", "town02/seed_1"):
            self._write_run("full", seed, "2026-01-01T00:00:00", 10, 2, {"Collision": 2}, checkpoints)
            self._write_run("global_only", seed, "2026-01-01T00:00:00", 8, 1, {"Lane Invasion": 1}, checkpoints)
            self._write_run("local_only", seed, "2026-01-01T00:00:00", 12, 3, {"speed_limit_exceeded": 3}, checkpoints)

        # This older duplicate must not be included.
        self._write_run("full", "town01/seed_1", "2025-01-01T00:00:00", 100, 100, {"Collision": 100})

        summarize_rq3(self.results, self.manifest, self.output)

        per_seed = self._read_csv(self.output / "rq3_simlingo_per_seed.csv")
        self.assertEqual(len(per_seed), 6)
        full_seed = next(
            row for row in per_seed
            if row["configuration"] == "Full" and row["scenario_seed"] == "town01/seed_1"
        )
        self.assertEqual(full_seed["executions"], "10")
        self.assertEqual(full_seed["failure_rate"], "0.2")

        aggregate = self._read_csv(self.output / "rq3_simlingo_aggregate.csv")
        full_total = next(row for row in aggregate if row["configuration"] == "Full")
        self.assertEqual(full_total["executions"], "20")
        self.assertEqual(full_total["failures"], "4")
        self.assertEqual(full_total["median_failure_rate"], "0.2")

        categories = self._read_csv(self.output / "rq3_simlingo_failure_categories.csv")
        full_collision = next(
            row for row in categories
            if row["configuration"] == "Full" and row["failure_category"] == "Collision"
        )
        self.assertEqual(full_collision["count"], "4")
        local_speed = next(
            row for row in categories
            if row["configuration"] == "Local-only" and row["failure_category"] == "Speed Limit Exceeded"
        )
        self.assertEqual(local_speed["count"], "6")

        hourly = self._read_csv(self.output / "rq3_simlingo_hourly.csv")
        full_hour_two = next(
            row for row in hourly
            if row["configuration"] == "Full" and row["hour"] == "2"
        )
        self.assertEqual(full_hour_two["executions"], "20")
        self.assertEqual(full_hour_two["failures"], "4")

        warnings = self._read_csv(self.output / "warnings.csv")
        self.assertTrue(any(row["warning_type"] == "duplicate_run_ignored" for row in warnings))

    def test_paired_statistics_and_missing_configuration_warning_are_reported(self):
        for configuration, failures in (("full", 4), ("global_only", 1), ("local_only", 2)):
            self._write_run(
                configuration,
                "town01/seed_1",
                "2026-01-01T00:00:00",
                10,
                failures,
                {"Collision": failures},
            )
        for configuration, failures in (("full", 6), ("global_only", 2)):
            self._write_run(
                configuration,
                "town02/seed_1",
                "2026-01-01T00:00:00",
                10,
                failures,
                {"Collision": failures},
            )

        summarize_rq3(self.results, self.manifest, self.output)

        paired = self._read_csv(self.output / "rq3_simlingo_paired_statistics.csv")
        full_global_rate = next(
            row for row in paired
            if row["comparison"] == "Full - Global-only" and row["metric"] == "failure_rate"
        )
        self.assertEqual(full_global_rate["paired_seeds"], "2")
        self.assertEqual(full_global_rate["wins"], "2")
        self.assertEqual(full_global_rate["median_difference"], "0.35")

        warnings = self._read_csv(self.output / "warnings.csv")
        missing = [row for row in warnings if row["warning_type"] == "missing_run"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["scenario_seed"], "town02/seed_1")
        self.assertEqual(missing[0]["configuration"], "Local-only")

    def test_random_scenario_local_is_included_when_results_exist(self):
        self._write_run(
            "full",
            "town01/seed_1",
            "2026-01-01T00:00:00",
            10,
            4,
            {"Collision": 4},
        )
        self._write_run(
            "random_scenario_local",
            "town01/seed_1",
            "2026-01-01T00:00:00",
            12,
            3,
            {"Lane Invasion": 3},
        )

        summarize_rq3(self.results, self.manifest, self.output)

        per_seed = self._read_csv(self.output / "rq3_simlingo_per_seed.csv")
        random_row = next(
            row for row in per_seed
            if row["configuration"] == "Random-scenario Local"
        )
        self.assertEqual(random_row["executions"], "12")
        self.assertEqual(random_row["failures"], "3")

        paired = self._read_csv(self.output / "rq3_simlingo_paired_statistics.csv")
        comparison = next(
            row for row in paired
            if row["comparison"] == "Full - Random-scenario Local"
            and row["metric"] == "failure_rate"
        )
        self.assertEqual(comparison["paired_seeds"], "1")


if __name__ == "__main__":
    unittest.main()
