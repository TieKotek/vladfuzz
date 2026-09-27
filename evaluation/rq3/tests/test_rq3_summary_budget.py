import json
from pathlib import Path
import tempfile
import unittest


from evaluation.rq3.summarize_results import summarize_rq3


class RQ3SummaryBudgetTests(unittest.TestCase):
    def test_newer_pilot_does_not_replace_formal_four_hour_run(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            results = root / "results"
            manifest = root / "manifest.jsonl"
            output = root / "summary"
            manifest.write_text(json.dumps({"id": "town01/seed_1"}) + "\n", encoding="utf-8")

            model_root = results / "rq2" / "runs" / "vladfuzz" / "simlingo"
            for name, start_time, budget, executions, failures in (
                ("formal", "2026-01-01T00:00:00", 14400, 20, 4),
                ("newer-pilot", "2026-02-01T00:00:00", 300, 100, 100),
            ):
                run_dir = model_root / name
                run_dir.mkdir(parents=True)
                metadata = {
                    "start_time": start_time,
                    "total_simulations_executed": executions,
                    "total_failures_detected": failures,
                    "failure_analysis": {"Collision": failures},
                    "configuration": {
                        "method": "vlad_fuzz_cmd1",
                        "vla_model": "simlingo",
                        "manifest_id": "town01/seed_1",
                        "time_budget_seconds": budget,
                    },
                }
                (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            summarize_rq3(results, manifest, output)

            per_seed = (output / "rq3_simlingo_per_seed.csv").read_text(encoding="utf-8")
            self.assertIn(",20,4,0.2,", per_seed)
            self.assertNotIn(",100,100,1.0,", per_seed)
            warnings = (output / "warnings.csv").read_text(encoding="utf-8")
            self.assertIn("budget_mismatch_ignored", warnings)
            self.assertIn("300", warnings)


if __name__ == "__main__":
    unittest.main()
